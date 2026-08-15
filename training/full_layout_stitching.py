"""Sparse full-layout component stitching and B7 deployment-policy metrics.

The B6 model emits one non-overlapping 160 x 160 central mask per tile.  This
module keeps those masks sparse: connected components are converted to global
pixel coordinates, joined across tile boundaries, and mapped back to physical
layout coordinates without allocating a full-layout raster.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Small dependency-light 8-connected component implementation."""

    binary = mask.astype(bool, copy=False)
    try:
        import cv2  # type: ignore

        count, labels, statistics, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        return labels.astype(np.int32, copy=False), [
            {"id": identifier, "area": int(statistics[identifier, cv2.CC_STAT_AREA])}
            for identifier in range(1, count)
        ]
    except ImportError:
        pass
    labels = np.zeros(binary.shape, dtype=np.int32)
    components: list[dict[str, Any]] = []
    identifier = 0
    height, width = binary.shape
    for start_row, start_column in np.argwhere(binary):
        if labels[start_row, start_column]:
            continue
        identifier += 1
        stack = [(int(start_row), int(start_column))]
        labels[start_row, start_column] = identifier
        pixels: list[tuple[int, int]] = []
        while stack:
            row, column = stack.pop()
            pixels.append((row, column))
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor_row = row + row_delta
                    neighbor_column = column + column_delta
                    if not (
                        0 <= neighbor_row < height and 0 <= neighbor_column < width
                    ):
                        continue
                    if binary[neighbor_row, neighbor_column] and not labels[
                        neighbor_row, neighbor_column
                    ]:
                        labels[neighbor_row, neighbor_column] = identifier
                        stack.append((neighbor_row, neighbor_column))
        components.append({"id": identifier, "area": len(pixels)})
    return labels, components


@dataclass(frozen=True)
class DeploymentPolicy:
    """Validation-selected policy applied unchanged to complete layouts."""

    segmentation_threshold: float
    classification_threshold: float
    minimum_component_area_px: int
    merge_gap_px: int
    recovery_radius_nm: float = 140.0

    def validate(self) -> None:
        if not 0.0 < self.segmentation_threshold < 1.0:
            raise ValueError("segmentation_threshold must be between zero and one")
        if not 0.0 <= self.classification_threshold < 1.0:
            raise ValueError("classification_threshold must be in [0, 1)")
        if self.minimum_component_area_px < 1:
            raise ValueError("minimum_component_area_px must be positive")
        if self.merge_gap_px < 0:
            raise ValueError("merge_gap_px cannot be negative")
        if self.recovery_radius_nm <= 0:
            raise ValueError("recovery_radius_nm must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def tile_components(
    mask_probability: np.ndarray,
    threshold: float,
    grid_index: Sequence[int],
    tile_id: str,
    class_probability: float,
    output_px: int = 160,
) -> list[dict[str, Any]]:
    """Convert one output mask into sparse global-pixel components.

    Global Y grows upward, matching layout coordinates. Raster row zero is the
    physical top edge, so the row transform is deliberately inverted.
    """

    if mask_probability.shape != (output_px, output_px):
        raise ValueError(
            f"Unexpected output mask for {tile_id}: {mask_probability.shape}"
        )
    ix, iy = map(int, grid_index)
    labels, local_components = _connected_components(mask_probability >= threshold)
    result: list[dict[str, Any]] = []
    for component in local_components:
        rows, columns = np.nonzero(labels == component["id"])
        global_x = ix * output_px + columns
        global_y = iy * output_px + (output_px - 1 - rows)
        pixels = np.column_stack((global_y, global_x)).astype(np.int32, copy=False)
        probabilities = mask_probability[rows, columns].astype(np.float32, copy=False)
        result.append(
            {
                "component_id": f"{tile_id}:c{int(component['id']):04d}",
                "tile_id": tile_id,
                "grid_index": [ix, iy],
                "class_probability": float(class_probability),
                "area_pixels": int(len(pixels)),
                "mean_confidence": float(probabilities.mean()),
                "max_confidence": float(probabilities.max()),
                "_pixels_yx": pixels,
            }
        )
    return result


def _pixel_sets_touch(
    first: np.ndarray, second: np.ndarray, merge_gap_px: int
) -> bool:
    """Return whether components are 8-connected after the declared closing gap."""

    # Separate per-tile connected components whose Chebyshev distance is one
    # are one global 8-connected object. merge_gap_px adds an explicit number
    # of empty pixels that may be bridged beyond that baseline adjacency.
    maximum_distance = merge_gap_px + 1
    first_min = first.min(axis=0)
    first_max = first.max(axis=0)
    second_min = second.min(axis=0)
    second_max = second.max(axis=0)
    if np.any(first_min > second_max + maximum_distance) or np.any(
        second_min > first_max + maximum_distance
    ):
        return False

    small, large = (first, second) if len(first) <= len(second) else (second, first)
    large_set = {(int(row), int(column)) for row, column in large}
    for row, column in small:
        for row_delta in range(-maximum_distance, maximum_distance + 1):
            for column_delta in range(-maximum_distance, maximum_distance + 1):
                if (int(row + row_delta), int(column + column_delta)) in large_set:
                    return True
    return False


def stitch_components(
    components: Sequence[dict[str, Any]],
    policy: DeploymentPolicy,
    output_px: int = 160,
) -> list[dict[str, Any]]:
    """Gate, merge, and area-filter sparse components across central outputs."""

    policy.validate()
    active = [
        component
        for component in components
        if float(component["class_probability"]) >= policy.classification_threshold
    ]
    if not active:
        return []

    by_tile: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, component in enumerate(active):
        by_tile[tuple(map(int, component["grid_index"]))].append(index)

    disjoint = _DisjointSet(len(active))
    tile_radius = 1 + math.ceil(policy.merge_gap_px / output_px)
    for first_index, first in enumerate(active):
        ix, iy = map(int, first["grid_index"])
        for neighbor_x in range(ix - tile_radius, ix + tile_radius + 1):
            for neighbor_y in range(iy - tile_radius, iy + tile_radius + 1):
                for second_index in by_tile.get((neighbor_x, neighbor_y), []):
                    if second_index <= first_index:
                        continue
                    if _pixel_sets_touch(
                        first["_pixels_yx"],
                        active[second_index]["_pixels_yx"],
                        policy.merge_gap_px,
                    ):
                        disjoint.union(first_index, second_index)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, component in enumerate(active):
        groups[disjoint.find(index)].append(component)

    stitched: list[dict[str, Any]] = []
    for members in groups.values():
        pixels = np.unique(
            np.concatenate([member["_pixels_yx"] for member in members]), axis=0
        )
        if len(pixels) < policy.minimum_component_area_px:
            continue
        areas = np.asarray([member["area_pixels"] for member in members], dtype=float)
        mean_confidences = np.asarray(
            [member["mean_confidence"] for member in members], dtype=float
        )
        stitched.append(
            {
                "component_id": "merged:" + min(
                    str(member["component_id"]) for member in members
                ),
                "source_component_ids": sorted(
                    str(member["component_id"]) for member in members
                ),
                "source_tile_ids": sorted({str(member["tile_id"]) for member in members}),
                "area_pixels": int(len(pixels)),
                "mean_confidence": float(np.average(mean_confidences, weights=areas)),
                "max_confidence": max(float(member["max_confidence"]) for member in members),
                "class_probability_mean": float(
                    np.mean([member["class_probability"] for member in members])
                ),
                "bbox_global_px": [
                    int(pixels[:, 1].min()),
                    int(pixels[:, 0].min()),
                    int(pixels[:, 1].max()) + 1,
                    int(pixels[:, 0].max()) + 1,
                ],
                "centroid_global_px": [
                    float(pixels[:, 1].mean() + 0.5),
                    float(pixels[:, 0].mean() + 0.5),
                ],
                "_pixels_yx": pixels,
            }
        )
    return sorted(stitched, key=lambda item: item["component_id"])


def add_layout_coordinates(
    components: Iterable[dict[str, Any]],
    grid_origin_nm: Sequence[int],
    nm_per_pixel: float = 8.0,
) -> list[dict[str, Any]]:
    """Attach physical centroid and half-open bbox coordinates to components."""

    x0_nm, y0_nm = map(float, grid_origin_nm)
    result: list[dict[str, Any]] = []
    for component in components:
        item = dict(component)
        left, bottom, right, top = item["bbox_global_px"]
        centroid_x, centroid_y = item["centroid_global_px"]
        item["bbox_nm"] = [
            x0_nm + left * nm_per_pixel,
            y0_nm + bottom * nm_per_pixel,
            x0_nm + right * nm_per_pixel,
            y0_nm + top * nm_per_pixel,
        ]
        item["centroid_nm"] = [
            x0_nm + centroid_x * nm_per_pixel,
            y0_nm + centroid_y * nm_per_pixel,
        ]
        result.append(item)
    return result


def public_component(component: dict[str, Any]) -> dict[str, Any]:
    """Remove internal pixel payload before JSON/JSONL export."""

    return {key: value for key, value in component.items() if not key.startswith("_")}


def proxy_component_vector_matches(
    components: Sequence[dict[str, Any]],
    vectors: Sequence[dict[str, Any]],
    tolerance_nm: float,
) -> dict[str, Any]:
    """Validation-only spatial proxy used to select a deployment policy.

    Exact KLayout recovery is deliberately run only after the policy is frozen.
    A component may cover multiple nearby violations; every vector is counted
    once, while each component is a true proposal when it covers at least one.
    """

    component_to_vectors: dict[str, list[str]] = {}
    detected: set[str] = set()
    for component in components:
        left, bottom, right, top = map(float, component["bbox_nm"])
        matches = []
        for vector in vectors:
            x_nm, y_nm = map(float, vector["midpoint_nm"])
            if (
                left - tolerance_nm <= x_nm <= right + tolerance_nm
                and bottom - tolerance_nm <= y_nm <= top + tolerance_nm
            ):
                violation_id = str(vector["violation_id"])
                matches.append(violation_id)
                detected.add(violation_id)
        component_to_vectors[str(component["component_id"])] = sorted(matches)

    true_components = sum(bool(values) for values in component_to_vectors.values())
    false_components = len(components) - true_components
    false_negative = len(vectors) - len(detected)
    precision = true_components / len(components) if components else 0.0
    recall = len(detected) / len(vectors) if vectors else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "component_count": len(components),
        "true_component_count": true_components,
        "false_component_count": false_components,
        "detected_violation_count": len(detected),
        "total_violation_count": len(vectors),
        "component_precision": precision,
        "violation_recall": recall,
        "f1": f1,
        "detected_violation_ids": sorted(detected),
        "component_to_violation_ids": component_to_vectors,
        "false_negative_count": false_negative,
    }


def aggregate_proxy_metrics(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    component_count = sum(int(item["component_count"]) for item in items)
    true_components = sum(int(item["true_component_count"]) for item in items)
    false_components = sum(int(item["false_component_count"]) for item in items)
    detected = sum(int(item["detected_violation_count"]) for item in items)
    total = sum(int(item["total_violation_count"]) for item in items)
    precision = true_components / component_count if component_count else 0.0
    recall = detected / total if total else 1.0
    return {
        "component_count": component_count,
        "true_component_count": true_components,
        "false_component_count": false_components,
        "detected_violation_count": detected,
        "total_violation_count": total,
        "component_precision": precision,
        "violation_recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }
