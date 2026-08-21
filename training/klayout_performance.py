"""Exact KLayout m1.2 timing and correctness helpers for the B7.2 audit.

The benchmark deliberately uses the same ``Region.space_check`` operation as
the project's Python ground-truth fallback and B7 local exact-recovery stage.
It therefore compares the learned pipeline with the exact geometry operation
that defines the labels, rather than with a weakened approximation.
"""

from __future__ import annotations

import math
import platform
import resource
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import klayout.db as db
import klayout.rdb as rdb

from training.dataset_manifest import sha256_file


M1_LAYER = (68, 20)
M1_2_MINIMUM_UM = 0.140


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""

    if not values:
        raise ValueError("Cannot summarize an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    numeric = [float(value) for value in values]
    return {
        "samples": len(numeric),
        "minimum": min(numeric),
        "median": statistics.median(numeric),
        "mean": statistics.fmean(numeric),
        "p95": percentile(numeric, 0.95),
        "maximum": max(numeric),
    }


def _edge_key(edge: db.Edge) -> tuple[int, int, int, int]:
    endpoints = sorted(
        ((int(edge.p1.x), int(edge.p1.y)), (int(edge.p2.x), int(edge.p2.y)))
    )
    return (*endpoints[0], *endpoints[1])


def canonical_pair_key(pair: db.EdgePair) -> tuple[int, ...]:
    edges = sorted((_edge_key(pair.first), _edge_key(pair.second)))
    return tuple(value for edge in edges for value in edge)


def pair_keys(pairs: Iterable[db.EdgePair]) -> set[tuple[int, ...]]:
    return {canonical_pair_key(pair) for pair in pairs}


def report_pair_keys(report_path: Path, layout_dbu: float) -> set[tuple[int, ...]]:
    """Load canonical integer-DBU m1.2 edge pairs from an RDB file."""

    report = rdb.ReportDatabase("B7.2 expected")
    report.load(str(report_path))
    result: set[tuple[int, ...]] = set()
    for item in report.each_item():
        category = report.category_by_id(item.category_id())
        if category.name().replace("'", "").strip() != "m1.2":
            continue
        for value in item.each_value():
            if not value.is_edge_pair():
                raise ValueError(f"Non-edge-pair m1.2 item in {report_path}")
            result.add(canonical_pair_key(value.edge_pair().to_itype(layout_dbu)))
    return result


def _load_m1(layout_path: Path) -> tuple[db.Layout, db.Cell, db.Region, dict[str, Any]]:
    started = time.perf_counter()
    layout = db.Layout()
    layout.read(str(layout_path))
    parse_seconds = time.perf_counter() - started
    top = layout.top_cell()
    if top is None:
        raise ValueError(f"Layout has no top cell: {layout_path}")
    layer_index = layout.find_layer(*M1_LAYER)
    if layer_index is None or int(layer_index) < 0:
        raise ValueError(f"Layout has no M1 layer {M1_LAYER}: {layout_path}")

    materialize_started = time.perf_counter()
    metal = db.Region(top.begin_shapes_rec(layer_index))
    metal.merge()
    materialize_seconds = time.perf_counter() - materialize_started
    bbox = top.bbox()
    width_um = float(bbox.width()) * float(layout.dbu)
    height_um = float(bbox.height()) * float(layout.dbu)
    metadata = {
        "dbu_um": float(layout.dbu),
        "top_cell": str(top.name),
        "hierarchy_depth": int(top.hierarchy_levels()),
        "m1_polygon_count_after_merge": int(metal.count()),
        "bbox_dbu": [int(bbox.left), int(bbox.bottom), int(bbox.right), int(bbox.top)],
        "layout_area_mm2": width_um * height_um / 1_000_000.0,
        "parse_seconds": parse_seconds,
        "m1_materialize_seconds": materialize_seconds,
    }
    return layout, top, metal, metadata


def _run_rule(metal: db.Region, minimum_spacing_dbu: int) -> list[db.EdgePair]:
    violations = metal.space_check(minimum_spacing_dbu)
    return list(violations.each())


def _write_report(
    report_path: Path,
    top_name: str,
    layout_dbu: float,
    pairs: Sequence[db.EdgePair],
) -> None:
    report = rdb.ReportDatabase("B7.2 exact m1.2")
    category = report.create_category(None, "m1.2")
    cell = report.create_cell(top_name)
    for pair in pairs:
        item = report.create_item(cell.rdb_id(), category.rdb_id())
        item.add_value(
            db.DEdgePair(
                pair.first.to_dtype(layout_dbu), pair.second.to_dtype(layout_dbu)
            )
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(report_path))


def fresh_batch_run(layout_path: Path, report_path: Path) -> dict[str, Any]:
    """Read a layout, run exact m1.2, and emit a report with phase timings."""

    total_started = time.perf_counter()
    layout, top, metal, metadata = _load_m1(layout_path)
    minimum_spacing_dbu = round(M1_2_MINIMUM_UM / float(layout.dbu))
    rule_started = time.perf_counter()
    pairs = _run_rule(metal, minimum_spacing_dbu)
    rule_seconds = time.perf_counter() - rule_started
    report_started = time.perf_counter()
    _write_report(report_path, str(top.name), float(layout.dbu), pairs)
    report_seconds = time.perf_counter() - report_started
    total_seconds = time.perf_counter() - total_started
    return {
        **metadata,
        "rule_seconds": rule_seconds,
        "report_seconds": report_seconds,
        "total_seconds": total_seconds,
        "violation_count": len(pairs),
        "pair_keys": pair_keys(pairs),
    }


def _bbox_from_pair_key(key: tuple[int, ...], margin_dbu: int) -> db.Box:
    xs = key[0::2]
    ys = key[1::2]
    return db.Box(
        min(xs) - margin_dbu,
        min(ys) - margin_dbu,
        max(xs) + margin_dbu,
        max(ys) + margin_dbu,
    )


def _local_rule(
    metal: db.Region, roi: db.Box, minimum_spacing_dbu: int
) -> set[tuple[int, ...]]:
    try:
        local_metal = metal.interacting(db.Region(roi))
    except (AttributeError, TypeError):
        local_metal = metal & db.Region(roi)
    return pair_keys(_run_rule(local_metal, minimum_spacing_dbu))


def benchmark_layout(
    layout_path: Path,
    output_report: Path,
    *,
    expected_report: Path | None = None,
    repeats: int = 5,
    warmups: int = 1,
    incremental_samples: int = 20,
    incremental_window_um: float = 10.0,
) -> dict[str, Any]:
    """Benchmark batch and already-loaded local checking for one layout."""

    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups cannot be negative")
    layout_path = layout_path.resolve()
    if not layout_path.is_file():
        raise FileNotFoundError(layout_path)

    with tempfile.TemporaryDirectory(prefix="b7_2_klayout_") as temp_dir:
        temporary_report = Path(temp_dir) / "warmup.rdb"
        for _ in range(warmups):
            fresh_batch_run(layout_path, temporary_report)

        runs = []
        for index in range(repeats):
            report = output_report if index == repeats - 1 else Path(temp_dir) / f"run_{index}.rdb"
            runs.append(fresh_batch_run(layout_path, report))

    reference_keys = runs[-1].pop("pair_keys")
    for run in runs[:-1]:
        if run.pop("pair_keys") != reference_keys:
            raise AssertionError("KLayout returned non-deterministic m1.2 edge pairs")

    layout, _, metal, loaded_metadata = _load_m1(layout_path)
    minimum_spacing_dbu = round(M1_2_MINIMUM_UM / float(layout.dbu))
    for _ in range(warmups):
        _run_rule(metal, minimum_spacing_dbu)
    loaded_rule_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        current = pair_keys(_run_rule(metal, minimum_spacing_dbu))
        loaded_rule_samples.append(time.perf_counter() - started)
        if current != reference_keys:
            raise AssertionError("Loaded-layout rule result differs from batch result")

    expected_keys = (
        report_pair_keys(expected_report, float(layout.dbu))
        if expected_report is not None
        else reference_keys
    )
    intersection = reference_keys & expected_keys
    expected_recall = len(intersection) / len(expected_keys) if expected_keys else 1.0
    expected_precision = len(intersection) / len(reference_keys) if reference_keys else 1.0

    margin_dbu = max(
        minimum_spacing_dbu,
        round(incremental_window_um / float(layout.dbu) / 2.0),
    )
    target_keys = sorted(expected_keys or reference_keys)[:incremental_samples]
    if not target_keys and incremental_samples:
        bbox = metal.bbox()
        center = db.Point((bbox.left + bbox.right) // 2, (bbox.bottom + bbox.top) // 2)
        synthetic = (
            center.x,
            center.y,
            center.x,
            center.y,
            center.x,
            center.y,
            center.x,
            center.y,
        )
        target_keys = [synthetic]
    incremental_times: list[float] = []
    incremental_target_hits = 0
    for target in target_keys:
        roi = _bbox_from_pair_key(target, margin_dbu)
        for repeat in range(repeats):
            started = time.perf_counter()
            local_keys = _local_rule(metal, roi, minimum_spacing_dbu)
            incremental_times.append(time.perf_counter() - started)
            if repeat == 0 and (target in local_keys or target not in expected_keys):
                incremental_target_hits += 1

    phase_names = (
        "parse_seconds",
        "m1_materialize_seconds",
        "rule_seconds",
        "report_seconds",
        "total_seconds",
    )
    return {
        "schema_version": 1,
        "engine": "KLayout Python Region.space_check",
        "rule": "m1.2",
        "minimum_spacing_um": M1_2_MINIMUM_UM,
        "layout_path": str(layout_path),
        "layout_sha256": sha256_file(layout_path),
        "output_report": str(output_report.resolve()),
        "expected_report": str(expected_report.resolve()) if expected_report else None,
        "expected_report_sha256": sha256_file(expected_report) if expected_report else None,
        "klayout_version": getattr(db, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repeats": repeats,
        "warmups": warmups,
        "metadata": {
            key: value
            for key, value in loaded_metadata.items()
            if not key.endswith("_seconds")
        },
        "violation_count": len(reference_keys),
        "correctness": {
            "expected_violation_count": len(expected_keys),
            "matching_pair_count": len(intersection),
            "pair_recall_vs_cached_report": expected_recall,
            "pair_precision_vs_cached_report": expected_precision,
            "exact_pair_set_match": reference_keys == expected_keys,
        },
        "batch_fresh_layout": {
            phase: summarize_samples([float(run[phase]) for run in runs])
            for phase in phase_names
        },
        "loaded_layout_rule_only": summarize_samples(loaded_rule_samples),
        "interactive_incremental": {
            "window_um": incremental_window_um,
            "sampled_regions": len(target_keys),
            "target_pairs_recovered": incremental_target_hits,
            "timing_seconds": summarize_samples(incremental_times) if incremental_times else None,
        },
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
