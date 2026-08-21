#!/usr/bin/env python3
"""Benchmark exact KLayout m1.2 against the frozen B7.1 CNN pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import layout_oas
from scripts.build_b6_localization_dataset import ensure_layout_intermediates
from training.klayout_performance import benchmark_layout
from training.localization_dataset import load_layout_splits


DEFAULT_B7_OUTPUT = PROJECT_ROOT / "results" / "b7_full_layout"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "b7_2_klayout_benchmark"
DEFAULT_REGISTRY = PROJECT_ROOT / "data" / "layout_registry.json"
DEFAULT_PROTOCOLS = PROJECT_ROOT / "data" / "evaluation_protocols.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _resolve_variant(
    layout: str,
    variant: str,
    b7_output: Path,
    prepare_missing: bool,
) -> tuple[Path, Path | None]:
    if variant == "source":
        path = layout_oas(layout)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, None
    if variant != "injected":
        raise ValueError(f"Unknown layout variant: {variant}")
    cache = b7_output / "layout_cache" / layout
    path = cache / f"{layout}_M1_m1_2_Marked.gds"
    report = cache / "sky130_drc.txt"
    if not (path.is_file() and report.is_file()) and prepare_missing:
        generated_layout, generated_report = ensure_layout_intermediates(layout, seed=42)
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_layout, path)
        shutil.copy2(generated_report, report)
    if not path.is_file() or not report.is_file():
        raise FileNotFoundError(
            "Missing injected B7 layout/report cache for "
            f"{layout}: expected {path} and {report}. Reuse the authoritative "
            "B7 output directory or pass --prepare-missing."
        )
    return path, report


def _load_cnn_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen CNN summary: {path}. Run B7.1 on the comparison "
            "machine before declaring a speed result."
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("phase") != "B7.1" or summary.get("status") != "complete":
        raise ValueError(f"Expected a complete B7.1 summary, got {path}")
    return summary


def _slice_no_misses(
    b7_output: Path, records: Sequence[dict[str, Any]], slice_name: str
) -> bool | None:
    total = 0
    detected = 0
    for record in records:
        result_path = (
            b7_output
            / "layouts"
            / record["group"]
            / f"{record['variant']}__{record['layout']}"
            / "result.json"
        )
        if not result_path.is_file():
            return None
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result.get("ground_truth_violation_count", 0)) == 0:
            continue
        values = result.get("recall_slices", {}).get(slice_name)
        if values is None:
            return None
        total += int(values["total"])
        detected += int(values["detected"])
    return detected == total if total else None


def _group_comparison(
    group: str,
    records: Sequence[dict[str, Any]],
    cnn: dict[str, Any],
    b7_output: Path,
) -> dict[str, Any]:
    selected = [record for record in records if record["group"] == group]
    klayout_median = sum(
        float(record["benchmark"]["batch_fresh_layout"]["total_seconds"]["median"])
        for record in selected
    )
    klayout_p95_bound = sum(
        float(record["benchmark"]["batch_fresh_layout"]["total_seconds"]["p95"])
        for record in selected
    )
    cnn_key = "validation" if group == "validation" else "development_confirmation"
    cnn_group = cnn[cnn_key]
    cnn_seconds = float(cnn_group["end_to_end_seconds"])
    recall = float(cnn_group["violation_recall"])
    near_threshold = _slice_no_misses(b7_output, selected, "severity:near_threshold")
    severe = _slice_no_misses(b7_output, selected, "severity:severe")
    quality_checks = {
        "violation_recall_at_least_0_995": recall >= 0.995,
        "no_near_threshold_misses": near_threshold,
        "no_registered_severe_slice_misses": severe,
        "no_clean_layout_false_alarm": int(cnn_group["clean_layouts_incorrectly_flagged"]) == 0,
        "exact_pair_precision_is_one": float(cnn_group["exact_candidate_precision"]) == 1.0,
    }
    quality_passed = all(value is True for value in quality_checks.values())
    cnn_speedup = klayout_median / cnn_seconds if cnn_seconds else None
    return {
        "layout_variants": len(selected),
        "klayout_batch_median_seconds": klayout_median,
        "klayout_batch_conservative_p95_seconds": klayout_p95_bound,
        "cnn_recorded_end_to_end_seconds": cnn_seconds,
        "cnn_device": cnn.get("runtime", {}).get("device"),
        "comparison_boundary": {
            "klayout": "layout parse + M1 materialization + exact rule + RDB write",
            "cnn": (
                "rasterization + inference + stitching + local exact recovery; "
                "excludes model/layout load and result serialization"
            ),
            "bias": "favors_cnn",
        },
        "cnn_timing_repetitions": 1,
        "cnn_p95_available": False,
        "cnn_speedup_over_klayout": cnn_speedup,
        "klayout_speedup_over_cnn": cnn_seconds / klayout_median if klayout_median else None,
        "cnn_violation_recall": recall,
        "quality_checks": quality_checks,
        "quality_gate_passed": quality_passed,
        "speed_gate_passed": bool(cnn_speedup is not None and cnn_speedup >= 2.0),
        "p95_gate_evaluable": False,
        "hard_gate_passed": False,
        "decision_reason": (
            "Quality gate failed; stop detector tuning and proceed to B8.0."
            if not quality_passed
            else "A repeated synchronized CNN run is required before a latency claim."
        ),
    }


def _render_readme(summary: dict[str, Any]) -> str:
    rows = []
    for group, item in summary["comparison"].items():
        rows.append(
            f"| {group} | {item['klayout_batch_median_seconds']:.3f} | "
            f"{item['cnn_recorded_end_to_end_seconds']:.3f} | "
            f"{item['cnn_violation_recall']:.3%} | "
            f"{'PASS' if item['hard_gate_passed'] else 'FAIL'} |"
        )
    return f"""# B7.2 KLayout competitiveness audit

Status: **{summary['status']}**. Overall hard gate:
**{'PASS' if summary['hard_gate_passed'] else 'FAIL'}**.

This audit runs the exact KLayout `Region.space_check(140 nm)` operation used
by the project label/recovery path on the same source and injected layouts as
B7.1. KLayout remains the correctness oracle.

| Group | KLayout batch median (s) | CNN recorded end-to-end (s) | CNN recall | Gate |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The KLayout number includes layout parsing, recursive M1 materialization,
exact rule execution, and RDB writing. Phase timings and an already-loaded
incremental-region scenario are in `per_layout.jsonl`.

The imported B7.1 summary contains one CNN timing observation. This is enough
to reject the branch when the registered 99.5% recall gate fails, but it is not
enough for a p95 latency claim. If quality unexpectedly passes on a future
frozen model, run at least five synchronized CNN repetitions on the same host.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_layout_splits(args.registry, args.protocols, args.protocol)
    group_layouts = {
        "validation": splits["validation"],
        "development_confirmation": splits["test"],
    }
    records: list[dict[str, Any]] = []
    for group in args.groups:
        for layout in group_layouts[group]:
            for variant in args.variants:
                layout_path, expected_report = _resolve_variant(
                    layout, variant, args.b7_output_dir, args.prepare_missing
                )
                print(f"[B7.2] KLayout {group}::{variant}::{layout}", flush=True)
                benchmark = benchmark_layout(
                    layout_path,
                    args.output_dir / "reports" / f"{group}__{variant}__{layout}.rdb",
                    expected_report=expected_report,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    incremental_samples=args.incremental_samples,
                    incremental_window_um=args.incremental_window_um,
                )
                records.append(
                    {
                        "group": group,
                        "layout": layout,
                        "variant": variant,
                        "benchmark": benchmark,
                    }
                )
    _write_jsonl(args.output_dir / "per_layout.jsonl", records)

    cnn = _load_cnn_summary(args.cnn_summary)
    comparison = {
        group: _group_comparison(group, records, cnn, args.b7_output_dir)
        for group in args.groups
    }
    all_pair_sets_match = all(
        record["benchmark"]["correctness"]["exact_pair_set_match"]
        for record in records
    )
    expected_count = sum(len(group_layouts[group]) for group in args.groups) * len(args.variants)
    complete = len(records) == expected_count
    summary = {
        "schema_version": 1,
        "phase": "B7.2",
        "status": "complete" if complete else "partial",
        "protocol": args.protocol,
        "groups": list(args.groups),
        "variants": list(args.variants),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "same_host_required": True,
        "positive_latency_claim_requires_equalized_boundaries": True,
        "cnn_summary": str(args.cnn_summary.resolve()),
        "layout_variant_count": len(records),
        "all_klayout_pairs_match_cached_reports": all_pair_sets_match,
        "comparison": comparison,
        "hard_gate_passed": bool(
            complete
            and all_pair_sets_match
            and all(item["hard_gate_passed"] for item in comparison.values())
        ),
    }
    _write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "README.md").write_text(_render_readme(summary), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b7-output-dir", type=Path, default=DEFAULT_B7_OUTPUT)
    parser.add_argument("--cnn-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--protocol", default="unseen_layout_v1")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("validation", "development_confirmation"),
        default=("validation", "development_confirmation"),
    )
    parser.add_argument(
        "--variants", nargs="+", choices=("source", "injected"), default=("source", "injected")
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--incremental-samples", type=int, default=20)
    parser.add_argument("--incremental-window-um", type=float, default=10.0)
    parser.add_argument("--prepare-missing", action="store_true")
    args = parser.parse_args(argv)
    if args.cnn_summary is None:
        args.cnn_summary = args.b7_output_dir / "summary.json"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
