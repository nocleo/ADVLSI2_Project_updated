"""B7 complete-layout inference, validation-only policy selection, and recovery."""

from __future__ import annotations

import hashlib
import gzip
import pickle
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import klayout.db as db
import numpy as np
import torch
from PIL import Image, ImageDraw

from generate_training_dataset_scripts.generate_localization_dataset import (
    CoverageGrid,
    LocalizationConfig,
    _box_to_nm,
    _edge_orientation,
    _edge_to_nm,
    _nm_per_dbu,
    _to_dbu,
    _to_nm,
    extract_exact_violations,
    extract_rdb_violations,
    load_m1_region,
    rasterize_registered_tile,
)
from training.dataset_manifest import sha256_file
from training.full_layout_stitching import (
    DeploymentPolicy,
    add_layout_coordinates,
    aggregate_proxy_metrics,
    proxy_component_vector_matches,
    public_component,
    stitch_components,
    tile_components,
)
from training.multitask_unet import MultiTaskUNet


DEFAULT_SEGMENTATION_THRESHOLDS = tuple(round(index / 10, 1) for index in range(1, 10))
DEFAULT_CLASSIFICATION_THRESHOLDS = tuple(round(index / 10, 1) for index in range(1, 10))
B7_1_CLASSIFICATION_THRESHOLDS = tuple(
    sorted(
        set(DEFAULT_CLASSIFICATION_THRESHOLDS)
        | {round(index / 100, 2) for index in range(80, 100)}
    )
)
DEFAULT_MINIMUM_AREAS = (1, 4, 9, 16)
DEFAULT_MERGE_GAPS = (0, 1, 2)
POLICY_SELECTION_OBJECTIVES = ("f1", "precision_at_recall")


def _threshold_key(value: float) -> str:
    return f"{value:.3f}"


def _tile_id(layout_name: str, ix: int, iy: int) -> str:
    return f"{layout_name}__x{ix:05d}_y{iy:05d}"


def load_ensemble(
    checkpoint_root: Path,
    seeds: Sequence[int],
    device: torch.device,
    base_channels: int = 16,
) -> tuple[list[torch.nn.Module], list[dict[str, Any]]]:
    models = []
    records = []
    for seed in seeds:
        checkpoint = checkpoint_root / f"seed_{seed}" / "best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing B6.2 checkpoint: {checkpoint}")
        model = MultiTaskUNet(base_channels=base_channels).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        models.append(model)
        records.append(
            {
                "seed": int(seed),
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
            }
        )
    return models, records


def load_layout_variant(
    layout_name: str,
    layout_path: Path,
    report_path: Path | None,
    config: LocalizationConfig,
    variant: str,
) -> dict[str, Any]:
    layout, metal_region, nm_per_dbu = load_m1_region(layout_path)
    bbox_nm = _box_to_nm(metal_region.bbox(), nm_per_dbu)
    grid = CoverageGrid.from_bbox(bbox_nm, config.stride_nm)
    if report_path is not None:
        vectors, polygons = extract_rdb_violations(
            report_path,
            layout.dbu,
            nm_per_dbu,
            layout_name,
            grid,
            config.min_rule_nm,
        )
        geometry_source = "klayout_rdb"
    else:
        vectors, polygons = extract_exact_violations(
            metal_region,
            nm_per_dbu,
            layout_name,
            grid,
            config.min_rule_nm,
        )
        geometry_source = "region_space_check"
    marker_region = db.Region()
    for polygon in polygons.values():
        marker_region.insert(polygon)
    return {
        "layout": layout,
        "layout_name": layout_name,
        "variant": variant,
        "layout_path": layout_path,
        "report_path": report_path,
        "metal_region": metal_region,
        "marker_region": marker_region,
        "nm_per_dbu": nm_per_dbu,
        "grid": grid,
        "layout_bbox_nm": list(map(int, bbox_nm)),
        "vectors": vectors,
        "geometry_source": geometry_source,
    }


def _ensemble_batch(
    models: Sequence[torch.nn.Module], images: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(images.astype(np.float32, copy=False)).unsqueeze(1).to(device)
    mask_probabilities = []
    class_probabilities = []
    with torch.no_grad():
        for model in models:
            segmentation, classification = model(tensor)
            mask_probabilities.append(torch.sigmoid(segmentation)[:, 0])
            class_probabilities.append(torch.softmax(classification, dim=1)[:, 1])
    masks = torch.stack(mask_probabilities).mean(dim=0).cpu().numpy().astype(np.float16)
    classes = torch.stack(class_probabilities).mean(dim=0).cpu().numpy().astype(np.float32)
    return masks, classes


def scan_full_layout(
    variant: dict[str, Any],
    models: Sequence[torch.nn.Module],
    device: torch.device,
    thresholds: Sequence[float] = DEFAULT_SEGMENTATION_THRESHOLDS,
    batch_size: int = 32,
    max_tiles: int | None = None,
) -> dict[str, Any]:
    """Run every central output in the coverage grid and keep sparse components."""

    config = LocalizationConfig()
    grid: CoverageGrid = variant["grid"]
    all_indices = list(grid.each_index())
    if max_tiles is not None:
        all_indices = all_indices[:max_tiles]
    vectors_by_owner: dict[tuple[int, int], list[str]] = defaultdict(list)
    for vector in variant["vectors"]:
        vectors_by_owner[tuple(map(int, vector["owner_index"]))].append(
            str(vector["violation_id"])
        )

    components = {_threshold_key(threshold): [] for threshold in thresholds}
    tile_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        # CUDA launches are asynchronous.  Synchronize at both timing
        # boundaries so B7.2 measures the complete user-visible scan rather
        # than only the CPU time spent enqueueing GPU kernels.
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for start in range(0, len(all_indices), batch_size):
        batch_indices = all_indices[start : start + batch_size]
        images = []
        targets = []
        for ix, iy in batch_indices:
            image, target = rasterize_registered_tile(
                variant["metal_region"],
                variant["marker_region"],
                grid.input_box(ix, iy, config.halo_nm),
                grid.output_box(ix, iy),
                config,
                variant["nm_per_dbu"],
            )
            images.append(image)
            targets.append(target)
        probabilities, class_probabilities = _ensemble_batch(
            models, np.stack(images), device
        )
        for offset, (ix, iy) in enumerate(batch_indices):
            tile_id = _tile_id(variant["layout_name"], ix, iy)
            owner_ids = sorted(vectors_by_owner.get((ix, iy), []))
            tile_record = {
                "tile_id": tile_id,
                "grid_index": [ix, iy],
                "class_probability": float(class_probabilities[offset]),
                "maximum_mask_probability": float(probabilities[offset].max()),
                "target_mask_pixels": int(targets[offset].sum()),
                "owner_violation_ids": owner_ids,
            }
            tile_records.append(tile_record)
            for threshold in thresholds:
                components[_threshold_key(threshold)].extend(
                    tile_components(
                        probabilities[offset],
                        threshold,
                        [ix, iy],
                        tile_id,
                        float(class_probabilities[offset]),
                    )
                )
            diagnostic = {
                **tile_record,
                "image": images[offset],
                "target": targets[offset],
                "probability": probabilities[offset].astype(np.float32),
            }
            diagnostics.append(diagnostic)
            diagnostics.sort(
                key=lambda item: (
                    bool(item["owner_violation_ids"]),
                    item["maximum_mask_probability"],
                ),
                reverse=True,
            )
            diagnostics = diagnostics[:8]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    bbox = variant["layout_bbox_nm"]
    area_mm2 = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / 1e12
    return {
        "layout": variant["layout_name"],
        "variant": variant["variant"],
        "layout_path": str(variant["layout_path"].resolve()),
        "layout_sha256": sha256_file(variant["layout_path"]),
        "report_sha256": sha256_file(variant["report_path"])
        if variant["report_path"] is not None
        else None,
        "geometry_source": variant["geometry_source"],
        "grid": {
            "x0_nm": grid.x0_nm,
            "y0_nm": grid.y0_nm,
            "nx": grid.nx,
            "ny": grid.ny,
            "stride_nm": grid.stride_nm,
        },
        "layout_bbox_nm": bbox,
        "layout_area_mm2": area_mm2,
        "tile_count": len(all_indices),
        "full_grid_tile_count": grid.nx * grid.ny,
        "complete_scan": max_tiles is None or len(all_indices) == grid.nx * grid.ny,
        "vectors": variant["vectors"],
        "tiles": tile_records,
        "components_by_threshold": components,
        "diagnostics": diagnostics,
        "runtime_seconds": elapsed,
        "tiles_per_second": len(all_indices) / elapsed if elapsed else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "_variant": variant,
    }


def components_for_policy(scan: dict[str, Any], policy: DeploymentPolicy) -> list[dict[str, Any]]:
    raw = scan["components_by_threshold"][_threshold_key(policy.segmentation_threshold)]
    stitched = stitch_components(raw, policy)
    grid = scan["grid"]
    return add_layout_coordinates(
        stitched, [grid["x0_nm"], grid["y0_nm"]], LocalizationConfig().nm_per_pixel
    )


def proxy_metrics_for_policy(scan: dict[str, Any], policy: DeploymentPolicy) -> dict[str, Any]:
    components = components_for_policy(scan, policy)
    metrics = proxy_component_vector_matches(
        components, scan["vectors"], policy.recovery_radius_nm
    )
    # Policy sweeps need only counts/ratios. Omitting per-component match maps
    # keeps the validation artifact compact across hundreds of candidates.
    metrics.pop("detected_violation_ids", None)
    metrics.pop("component_to_violation_ids", None)
    metrics["layout"] = scan["layout"]
    metrics["variant"] = scan["variant"]
    metrics["components_before_merge"] = len(
        scan["components_by_threshold"][_threshold_key(policy.segmentation_threshold)]
    )
    metrics["components_after_merge"] = len(components)
    return metrics


def select_validation_policy(
    validation_scans: Sequence[dict[str, Any]],
    segmentation_thresholds: Sequence[float] = DEFAULT_SEGMENTATION_THRESHOLDS,
    classification_thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    minimum_areas: Sequence[int] = DEFAULT_MINIMUM_AREAS,
    merge_gaps: Sequence[int] = DEFAULT_MERGE_GAPS,
    minimum_recall: float = 0.85,
    selection_objective: str = "f1",
) -> dict[str, Any]:
    """Select in two validation-only stages to limit post-processing overfit."""

    if selection_objective not in POLICY_SELECTION_OBJECTIVES:
        raise ValueError(
            f"selection_objective must be one of {POLICY_SELECTION_OBJECTIVES}"
        )

    segmentation_sweep = []
    for segmentation_threshold in segmentation_thresholds:
        policy = DeploymentPolicy(segmentation_threshold, 0.0, 1, 0)
        metrics = aggregate_proxy_metrics(
            [proxy_metrics_for_policy(scan, policy) for scan in validation_scans]
        )
        segmentation_sweep.append({"policy": policy.to_dict(), "metrics": metrics})
    eligible_segmentation = [
        item for item in segmentation_sweep if item["metrics"]["violation_recall"] >= minimum_recall
    ]
    segmentation_pool = eligible_segmentation or segmentation_sweep
    selected_segmentation = max(
        segmentation_pool,
        key=lambda item: (
            item["metrics"]["f1"],
            item["metrics"]["component_precision"],
            item["metrics"]["violation_recall"],
            item["policy"]["segmentation_threshold"],
        ),
    )["policy"]["segmentation_threshold"]

    postprocess_sweep = []
    for classification_threshold in classification_thresholds:
        for minimum_area in minimum_areas:
            for merge_gap in merge_gaps:
                policy = DeploymentPolicy(
                    selected_segmentation,
                    classification_threshold,
                    minimum_area,
                    merge_gap,
                )
                per_layout = [
                    proxy_metrics_for_policy(scan, policy) for scan in validation_scans
                ]
                metrics = aggregate_proxy_metrics(per_layout)
                postprocess_sweep.append(
                    {"policy": policy.to_dict(), "metrics": metrics, "per_layout": per_layout}
                )
    eligible = [
        item for item in postprocess_sweep if item["metrics"]["violation_recall"] >= minimum_recall
    ]
    pool = eligible or postprocess_sweep
    if selection_objective == "precision_at_recall":
        selection_key = lambda item: (
            item["metrics"]["component_precision"],
            item["metrics"]["f1"],
            item["metrics"]["violation_recall"],
            -item["metrics"]["false_component_count"],
            item["policy"]["classification_threshold"],
            item["policy"]["minimum_component_area_px"],
            -item["policy"]["merge_gap_px"],
        )
    else:
        selection_key = lambda item: (
            item["metrics"]["f1"],
            item["metrics"]["component_precision"],
            item["metrics"]["violation_recall"],
            -item["metrics"]["false_component_count"],
            item["policy"]["classification_threshold"],
            item["policy"]["minimum_component_area_px"],
            -item["policy"]["merge_gap_px"],
        )
    selected = max(pool, key=selection_key)
    return {
        "selection_split": "validation_layout_families_only",
        "two_stage_selection": True,
        "selection_objective": selection_objective,
        "minimum_violation_recall": minimum_recall,
        "passed_recall_constraint": bool(eligible),
        "selected_policy": selected["policy"],
        "selected_validation_proxy_metrics": selected["metrics"],
        "segmentation_sweep": segmentation_sweep,
        "postprocess_sweep": postprocess_sweep,
    }


def _canonical_edge(edge: Sequence[int | float]) -> tuple[int, int, int, int]:
    first = (int(round(edge[0])), int(round(edge[1])))
    second = (int(round(edge[2])), int(round(edge[3])))
    return (*first, *second) if first <= second else (*second, *first)


def canonical_pair_key(
    edge1: Sequence[int | float], edge2: Sequence[int | float]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    first = _canonical_edge(edge1)
    second = _canonical_edge(edge2)
    return (first, second) if first <= second else (second, first)


def _pair_record(pair: db.EdgePair, nm_per_dbu: float) -> dict[str, Any]:
    edge1 = _edge_to_nm(pair.first, nm_per_dbu)
    edge2 = _edge_to_nm(pair.second, nm_per_dbu)
    midpoint = [
        int(round((edge1[0] + edge1[2] + edge2[0] + edge2[2]) / 4)),
        int(round((edge1[1] + edge1[3] + edge2[1] + edge2[3]) / 4)),
    ]
    polygon = pair.polygon(0)
    key = canonical_pair_key(edge1, edge2)
    return {
        "candidate_id": "m1.2:" + hashlib.sha256(repr(key).encode()).hexdigest()[:16],
        "rule": "m1.2",
        "rule_min_nm": 140,
        "spacing_nm": _to_nm(pair.distance(), nm_per_dbu),
        "deficit_nm": 140 - _to_nm(pair.distance(), nm_per_dbu),
        "edge1_nm": edge1,
        "edge2_nm": edge2,
        "midpoint_nm": midpoint,
        "bbox_nm": list(_box_to_nm(polygon.bbox(), nm_per_dbu)),
        "orientation": _edge_orientation(pair.first),
        "_pair_key": key,
    }


def recover_exact_candidates(
    variant: dict[str, Any],
    components: Sequence[dict[str, Any]],
    policy: DeploymentPolicy,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Run local KLayout spacing checks only around model-proposed components."""

    metal_region: db.Region = variant["metal_region"]
    nm_per_dbu = float(variant["nm_per_dbu"])
    candidates: dict[
        tuple[tuple[int, int, int, int], tuple[int, int, int, int]], dict[str, Any]
    ] = {}
    component_to_candidates: dict[str, list[str]] = defaultdict(list)
    components_by_id = {str(item["component_id"]): item for item in components}
    for component in components:
        left, bottom, right, top = map(float, component["bbox_nm"])
        margin = policy.recovery_radius_nm + 140.0
        roi = db.Box(
            _to_dbu(left - margin, nm_per_dbu),
            _to_dbu(bottom - margin, nm_per_dbu),
            _to_dbu(right + margin, nm_per_dbu),
            _to_dbu(top + margin, nm_per_dbu),
        )
        try:
            local_metal = metal_region.interacting(db.Region(roi))
        except (AttributeError, TypeError):
            local_metal = metal_region & db.Region(roi)
        pairs = local_metal.space_check(_to_dbu(140, nm_per_dbu))
        for pair in pairs.each():
            record = _pair_record(pair, nm_per_dbu)
            x_nm, y_nm = record["midpoint_nm"]
            if not (
                left - policy.recovery_radius_nm
                <= x_nm
                <= right + policy.recovery_radius_nm
                and bottom - policy.recovery_radius_nm
                <= y_nm
                <= top + policy.recovery_radius_nm
            ):
                continue
            key = record["_pair_key"]
            if key not in candidates:
                record["source_component_ids"] = []
                candidates[key] = record
            candidate = candidates[key]
            candidate["source_component_ids"].append(component["component_id"])
            component_to_candidates[component["component_id"]].append(
                candidate["candidate_id"]
            )
    for candidate in candidates.values():
        candidate["source_component_ids"] = sorted(set(candidate["source_component_ids"]))
        sources = [components_by_id[item] for item in candidate["source_component_ids"]]
        candidate["mean_component_confidence"] = float(
            np.mean([item["mean_confidence"] for item in sources])
        )
        candidate["max_component_confidence"] = max(
            float(item["max_confidence"]) for item in sources
        )
    return (
        sorted(candidates.values(), key=lambda item: item["candidate_id"]),
        {key: sorted(set(value)) for key, value in component_to_candidates.items()},
    )


def _nearest_ground_truth(
    candidate: dict[str, Any],
    vectors: Sequence[dict[str, Any]],
    used: set[str],
    tolerance_nm: float = 16.0,
) -> str | None:
    key = candidate["_pair_key"]
    direct = [
        vector
        for vector in vectors
        if str(vector["violation_id"]) not in used
        and canonical_pair_key(vector["edge1_nm"], vector["edge2_nm"]) == key
    ]
    if direct:
        return str(direct[0]["violation_id"])
    midpoint = np.asarray(candidate["midpoint_nm"], dtype=float)
    nearby = []
    for vector in vectors:
        violation_id = str(vector["violation_id"])
        if violation_id in used:
            continue
        distance = float(np.linalg.norm(midpoint - np.asarray(vector["midpoint_nm"], dtype=float)))
        if distance <= tolerance_nm:
            nearby.append((distance, violation_id))
    return min(nearby)[1] if nearby else None


def _slice_recall(
    vectors: Sequence[dict[str, Any]], detected_ids: set[str], grid: dict[str, Any]
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"detected": 0, "total": 0})
    for vector in vectors:
        violation_id = str(vector["violation_id"])
        deficit = float(vector["deficit_nm"])
        severity = "near_threshold" if deficit <= 20 else "medium" if deficit <= 60 else "severe"
        ix, iy = map(int, vector["owner_index"])
        left = grid["x0_nm"] + ix * grid["stride_nm"]
        bottom = grid["y0_nm"] + iy * grid["stride_nm"]
        right = left + grid["stride_nm"]
        top = bottom + grid["stride_nm"]
        x_nm, y_nm = map(float, vector["midpoint_nm"])
        boundary_distance = min(x_nm - left, right - x_nm, y_nm - bottom, top - y_nm)
        boundary = "boundary" if boundary_distance <= 140 else "interior"
        for key in (f"severity:{severity}", f"boundary:{boundary}"):
            counts[key]["total"] += 1
            counts[key]["detected"] += int(violation_id in detected_ids)
    return {
        key: {
            **value,
            "recall": value["detected"] / value["total"] if value["total"] else 0.0,
        }
        for key, value in sorted(counts.items())
    }


def evaluate_exact_layout(scan: dict[str, Any], policy: DeploymentPolicy) -> dict[str, Any]:
    started = time.perf_counter()
    raw_components = scan["components_by_threshold"][
        _threshold_key(policy.segmentation_threshold)
    ]
    gated_component_count = sum(
        float(component["class_probability"]) >= policy.classification_threshold
        for component in raw_components
    )
    components = components_for_policy(scan, policy)
    candidates, component_to_candidates = recover_exact_candidates(
        scan["_variant"], components, policy
    )
    used_vectors: set[str] = set()
    matched_candidates = 0
    candidate_to_violation: dict[str, str] = {}
    for candidate in candidates:
        violation_id = _nearest_ground_truth(candidate, scan["vectors"], used_vectors)
        if violation_id is not None:
            used_vectors.add(violation_id)
            candidate_to_violation[candidate["candidate_id"]] = violation_id
            matched_candidates += 1
    true_components = 0
    for component in components:
        candidate_ids = component_to_candidates.get(component["component_id"], [])
        if any(candidate_id in candidate_to_violation for candidate_id in candidate_ids):
            true_components += 1
    false_components = len(components) - true_components
    component_precision = true_components / len(components) if components else 0.0
    violation_recall = len(used_vectors) / len(scan["vectors"]) if scan["vectors"] else 1.0
    f1 = (
        2 * component_precision * violation_recall / (component_precision + violation_recall)
        if component_precision + violation_recall
        else 0.0
    )
    exact_precision = matched_candidates / len(candidates) if candidates else (1.0 if not scan["vectors"] else 0.0)

    predicted_tiles = {tile for component in components for tile in component["source_tile_ids"]}
    dirty_target_tiles = {
        str(tile["tile_id"]) for tile in scan["tiles"] if int(tile["target_mask_pixels"]) > 0
    }
    false_positive_tiles = len(predicted_tiles - dirty_target_tiles)
    elapsed = time.perf_counter() - started
    public_candidates = []
    for candidate in candidates:
        item = {key: value for key, value in candidate.items() if not key.startswith("_")}
        item["matched_violation_id"] = candidate_to_violation.get(candidate["candidate_id"])
        public_candidates.append(item)
    public_components = []
    for component in components:
        item = public_component(component)
        item["candidate_ids"] = component_to_candidates.get(component["component_id"], [])
        public_components.append(item)
    return {
        "layout": scan["layout"],
        "variant": scan["variant"],
        "complete_scan": scan["complete_scan"],
        "layout_area_mm2": scan["layout_area_mm2"],
        "tiles_scanned": scan["tile_count"],
        "ground_truth_violation_count": len(scan["vectors"]),
        "detected_unique_violation_count": len(used_vectors),
        "violation_recall": violation_recall,
        "candidate_component_count": len(components),
        "true_component_count": true_components,
        "false_component_count": false_components,
        "component_precision": component_precision,
        "component_f1": f1,
        "exact_candidate_count": len(candidates),
        "matched_exact_candidate_count": matched_candidates,
        "exact_candidate_precision": exact_precision,
        "false_detections_per_mm2": false_components / scan["layout_area_mm2"]
        if scan["layout_area_mm2"]
        else None,
        "predicted_positive_tile_count": len(predicted_tiles),
        "false_positive_tile_count": false_positive_tiles,
        "false_positive_tiles_per_million_scanned": false_positive_tiles
        * 1_000_000
        / scan["tile_count"]
        if scan["tile_count"]
        else 0.0,
        "clean_layout_incorrectly_flagged": bool(not scan["vectors"] and components),
        "components_at_mask_threshold": len(raw_components),
        "components_after_classification_gate_before_merging": gated_component_count,
        "components_after_merging": len(components),
        "recall_slices": _slice_recall(scan["vectors"], used_vectors, scan["grid"]),
        "policy": policy.to_dict(),
        "runtime": {
            "scan_seconds": scan["runtime_seconds"],
            "recovery_and_evaluation_seconds": elapsed,
            "end_to_end_seconds": scan["runtime_seconds"] + elapsed,
            "tiles_per_second": scan["tiles_per_second"],
            "peak_cuda_memory_bytes": scan["peak_cuda_memory_bytes"],
            "peak_process_rss_kib": scan["peak_process_rss_kib"],
        },
        "components": public_components,
        "exact_candidates": public_candidates,
        "missed_violation_ids": sorted(
            set(str(vector["violation_id"]) for vector in scan["vectors"]) - used_vectors
        ),
    }


def aggregate_exact_layouts(layouts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_vectors = sum(item["ground_truth_violation_count"] for item in layouts)
    detected = sum(item["detected_unique_violation_count"] for item in layouts)
    components = sum(item["candidate_component_count"] for item in layouts)
    true_components = sum(item["true_component_count"] for item in layouts)
    false_components = sum(item["false_component_count"] for item in layouts)
    exact_candidates = sum(item["exact_candidate_count"] for item in layouts)
    matched_exact_candidates = sum(
        item["matched_exact_candidate_count"] for item in layouts
    )
    total_area = sum(item["layout_area_mm2"] for item in layouts)
    total_tiles = sum(item["tiles_scanned"] for item in layouts)
    false_tiles = sum(item["false_positive_tile_count"] for item in layouts)
    precision = true_components / components if components else 0.0
    recall = detected / total_vectors if total_vectors else 1.0
    return {
        "layout_count": len(layouts),
        "total_layout_area_mm2": total_area,
        "tiles_scanned": total_tiles,
        "ground_truth_violation_count": total_vectors,
        "detected_unique_violation_count": detected,
        "violation_recall": recall,
        "candidate_component_count": components,
        "true_component_count": true_components,
        "false_component_count": false_components,
        "component_precision": precision,
        "component_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "exact_candidate_count": exact_candidates,
        "matched_exact_candidate_count": matched_exact_candidates,
        "exact_candidate_precision": matched_exact_candidates / exact_candidates
        if exact_candidates
        else (1.0 if total_vectors == 0 else 0.0),
        "false_detections_per_mm2": false_components / total_area if total_area else None,
        "false_positive_tile_count": false_tiles,
        "false_positive_tiles_per_million_scanned": false_tiles * 1_000_000 / total_tiles
        if total_tiles
        else 0.0,
        "clean_layouts_incorrectly_flagged": sum(
            item["clean_layout_incorrectly_flagged"] for item in layouts
        ),
        "end_to_end_seconds": sum(item["runtime"]["end_to_end_seconds"] for item in layouts),
    }


def render_four_panel(scan: dict[str, Any], policy: DeploymentPolicy, output_path: Path) -> None:
    """Adapt the teammate notebook's useful input/GT/probability/overlay panel."""

    if not scan["diagnostics"]:
        return
    sample = scan["diagnostics"][0]
    image = (sample["image"] * 255).astype(np.uint8)
    central = image[20:180, 20:180]
    target = (sample["target"] > 0).astype(np.uint8)
    probability = np.clip(sample["probability"] * 255, 0, 255).astype(np.uint8)
    prediction = sample["probability"] >= policy.segmentation_threshold

    panels = [
        Image.fromarray(central, mode="L").convert("RGB"),
        Image.fromarray(target * 255, mode="L").convert("RGB"),
        Image.fromarray(probability, mode="L").convert("RGB"),
        Image.fromarray(central, mode="L").convert("RGB"),
    ]
    overlay = np.asarray(panels[3]).copy()
    overlay[target.astype(bool)] = [0, 220, 80]
    overlay[prediction] = [255, 80, 80]
    overlay[np.logical_and(target, prediction)] = [255, 220, 0]
    panels[3] = Image.fromarray(overlay, mode="RGB")
    canvas = Image.new("RGB", (4 * 180, 205), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(
        zip(panels, ("Input", "Ground truth", "Probability", "GT / prediction"))
    ):
        canvas.paste(panel.resize((160, 160)), (index * 180, 25))
        draw.text((index * 180 + 4, 5), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_threshold_tradeoff(selection: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    segmentation = selection["segmentation_sweep"]
    segmentation_thresholds = [
        item["policy"]["segmentation_threshold"] for item in segmentation
    ]
    for metric, label in (
        ("component_precision", "Component precision"),
        ("violation_recall", "Violation recall"),
        ("f1", "F1"),
    ):
        axes[0].plot(
            segmentation_thresholds,
            [item["metrics"][metric] for item in segmentation],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="Segmentation threshold", ylabel="Validation metric")

    by_classification: dict[float, dict[str, Any]] = {}
    recall_floor = float(selection["minimum_violation_recall"])
    objective = selection.get("selection_objective", "f1")
    for item in selection["postprocess_sweep"]:
        threshold = float(item["policy"]["classification_threshold"])
        eligible = item["metrics"]["violation_recall"] >= recall_floor
        key = (
            int(eligible),
            item["metrics"]["component_precision"]
            if objective == "precision_at_recall"
            else item["metrics"]["f1"],
            item["metrics"]["f1"],
        )
        previous = by_classification.get(threshold)
        if previous is None or key > previous["_plot_key"]:
            by_classification[threshold] = {**item, "_plot_key": key}
    classification_thresholds = sorted(by_classification)
    for metric, label in (
        ("component_precision", "Component precision"),
        ("violation_recall", "Violation recall"),
        ("f1", "F1"),
    ):
        axes[1].plot(
            classification_thresholds,
            [by_classification[value]["metrics"][metric] for value in classification_thresholds],
            marker="o",
            label=label,
        )
    selected_threshold = selection["selected_policy"]["classification_threshold"]
    axes[1].axvline(selected_threshold, color="black", linestyle="--", alpha=0.55)
    axes[1].axhline(recall_floor, color="tab:red", linestyle=":", alpha=0.55)
    axes[1].set(xlabel="Classification threshold")
    for axis in axes:
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title(
        f"{objective}; recall floor={recall_floor:.2f}; selected={selected_threshold:.2f}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def json_ready_scan(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in scan.items()
        if key
        not in {
            "vectors",
            "tiles",
            "components_by_threshold",
            "diagnostics",
            "_variant",
        }
    }


def save_scan_cache(
    path: Path, scan: dict[str, Any], checkpoint_signature: str
) -> None:
    """Persist a trusted, resume-only sparse scan cache outside the Git result."""

    payload = {
        "schema_version": 1,
        "checkpoint_signature": checkpoint_signature,
        "layout_sha256": scan["layout_sha256"],
        "report_sha256": scan["report_sha256"],
        "scan": {key: value for key, value in scan.items() if key != "_variant"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=4) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_scan_cache(
    path: Path,
    variant: dict[str, Any],
    checkpoint_signature: str,
    require_complete: bool,
) -> dict[str, Any] | None:
    """Reuse only a cache tied to the exact layout, report, and checkpoints."""

    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        expected_report = (
            sha256_file(variant["report_path"])
            if variant["report_path"] is not None
            else None
        )
        if (
            payload.get("schema_version") != 1
            or payload.get("checkpoint_signature") != checkpoint_signature
            or payload.get("layout_sha256") != sha256_file(variant["layout_path"])
            or payload.get("report_sha256") != expected_report
        ):
            return None
        scan = payload["scan"]
        if require_complete and not scan.get("complete_scan", False):
            return None
        scan["_variant"] = variant
        return scan
    except (EOFError, OSError, pickle.PickleError, KeyError, TypeError, ValueError):
        return None
