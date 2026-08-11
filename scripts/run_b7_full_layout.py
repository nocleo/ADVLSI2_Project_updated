#!/usr/bin/env python3
"""Run B7 full-layout stitching and exact-coordinate recovery."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from project_paths import layout_oas
from scripts.build_b6_localization_dataset import ensure_layout_intermediates
from training.full_layout_evaluation import (
    DeploymentPolicy,
    aggregate_exact_layouts,
    evaluate_exact_layout,
    json_ready_scan,
    load_scan_cache,
    load_ensemble,
    load_layout_variant,
    render_four_panel,
    render_threshold_tradeoff,
    runtime_metadata,
    scan_full_layout,
    save_scan_cache,
    select_validation_policy,
)
from training.localization_dataset import load_layout_splits
from training.runtime_device import DEVICE_CHOICES, select_device


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "b7_full_layout"
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "results" / "b6_multitask_unet"
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


def _slug(layout: str, variant: str) -> str:
    return f"{variant}__{layout}"


def _prepare_variants(
    layout: str, include_source: bool, layout_cache_root: Path
) -> list[dict[str, Any]]:
    from generate_training_dataset_scripts.generate_localization_dataset import (
        LocalizationConfig,
    )

    config = LocalizationConfig()
    cached_layout_dir = layout_cache_root / layout
    cached_injected = cached_layout_dir / f"{layout}_M1_m1_2_Marked.gds"
    cached_report = cached_layout_dir / "sky130_drc.txt"
    if not (cached_injected.is_file() and cached_report.is_file()):
        injected, report = ensure_layout_intermediates(layout, seed=42)
        cached_layout_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(injected, cached_injected)
        shutil.copy2(report, cached_report)
    variants = [
        load_layout_variant(layout, cached_injected, cached_report, config, "injected")
    ]
    if include_source:
        variants.append(
            load_layout_variant(layout, layout_oas(layout), None, config, "source")
        )
    return variants


def _scan_split(
    layouts: Sequence[str],
    models: Sequence[torch.nn.Module],
    device: torch.device,
    batch_size: int,
    max_tiles: int | None,
    include_source: bool,
    output_dir: Path,
    checkpoint_signature: str,
) -> list[dict[str, Any]]:
    scans = []
    for layout in layouts:
        for variant in _prepare_variants(
            layout, include_source, output_dir / "layout_cache"
        ):
            cache_path = (
                output_dir
                / "scan_cache"
                / f"{_slug(layout, variant['variant'])}.pkl.gz"
            )
            cached = load_scan_cache(
                cache_path,
                variant,
                checkpoint_signature,
                require_complete=max_tiles is None,
            )
            if cached is not None and max_tiles is not None:
                expected_tiles = min(
                    max_tiles, variant["grid"].nx * variant["grid"].ny
                )
                if int(cached["tile_count"]) != expected_tiles:
                    cached = None
            if cached is not None:
                print(f"[B7] reusing scan cache {variant['variant']}::{layout}", flush=True)
                scans.append(cached)
                continue
            print(f"[B7] scanning {variant['variant']}::{layout}", flush=True)
            scan = scan_full_layout(
                variant,
                models,
                device,
                batch_size=batch_size,
                max_tiles=max_tiles,
            )
            save_scan_cache(cache_path, scan, checkpoint_signature)
            print(
                json.dumps(
                    {
                        "layout": layout,
                        "variant": variant["variant"],
                        "tiles": scan["tile_count"],
                        "vectors": len(scan["vectors"]),
                        "seconds": scan["runtime_seconds"],
                    }
                ),
                flush=True,
            )
            scans.append(scan)
    return scans


def _evaluate_and_export(
    scans: Sequence[dict[str, Any]],
    policy: DeploymentPolicy,
    output_dir: Path,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for scan in scans:
        result = evaluate_exact_layout(scan, policy)
        slug = _slug(scan["layout"], scan["variant"])
        layout_dir = output_dir / "layouts" / split / slug
        _write_json(layout_dir / "result.json", result)
        _write_jsonl(layout_dir / "components.jsonl", result["components"])
        _write_jsonl(layout_dir / "exact_candidates.jsonl", result["exact_candidates"])
        _write_json(layout_dir / "scan.json", json_ready_scan(scan))
        render_four_panel(scan, policy, layout_dir / "four_panel.png")
        print(
            json.dumps(
                {
                    "layout": scan["layout"],
                    "variant": scan["variant"],
                    "recall": result["violation_recall"],
                    "component_precision": result["component_precision"],
                    "false_components": result["false_component_count"],
                }
            ),
            flush=True,
        )
        results.append(result)
    aggregate = aggregate_exact_layouts(results)
    _write_json(output_dir / f"{split}_summary.json", {"aggregate": aggregate, "layouts": results})
    return results, aggregate


def _acceptance(
    validation: dict[str, Any],
    development: dict[str, Any],
    complete: bool,
) -> dict[str, Any]:
    checks = {
        "complete_full_grid_scans": complete,
        "validation_violation_recall_at_least_0_85": validation["violation_recall"] >= 0.85,
        "development_violation_recall_at_least_0_85": development["violation_recall"] >= 0.85,
        "development_component_precision_at_least_0_80": development[
            "component_precision"
        ]
        >= 0.80,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _render_readme(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    development = summary["development_confirmation"]
    policy = summary["deployment_policy"]
    status = "accepted" if summary["acceptance"]["passed"] else "not accepted"
    return f"""# B7 full-layout stitching and exact-coordinate recovery

Status: **{status}**.

The three B6.2 checkpoints are averaged at inference. The deployment policy was
selected only on complete validation-family layouts, including their natural
clean-source variants, and then frozen for development confirmation. The B9
final holdout remains unopened.

## Frozen deployment policy

- Segmentation threshold: `{policy['segmentation_threshold']}`
- Classification threshold: `{policy['classification_threshold']}`
- Minimum merged component area: `{policy['minimum_component_area_px']}` pixels
- Fragment merge gap: `{policy['merge_gap_px']}` pixels
- Local exact-recovery radius: `{policy['recovery_radius_nm']}` nm

## Full-layout results

| Metric | Validation | Development confirmation |
|---|---:|---:|
| Unique violation recall | {_percent(validation['violation_recall'])} | {_percent(development['violation_recall'])} |
| Candidate-component precision | {_percent(validation['component_precision'])} | {_percent(development['component_precision'])} |
| Component F1 | {_percent(validation['component_f1'])} | {_percent(development['component_f1'])} |
| Recovered exact-pair precision | {_percent(validation['exact_candidate_precision'])} | {_percent(development['exact_candidate_precision'])} |
| False detections / mm2 | {validation['false_detections_per_mm2']:.2f} | {development['false_detections_per_mm2']:.2f} |
| False-positive tiles / million | {validation['false_positive_tiles_per_million_scanned']:.1f} | {development['false_positive_tiles_per_million_scanned']:.1f} |
| Clean layouts incorrectly flagged | {validation['clean_layouts_incorrectly_flagged']} | {development['clean_layouts_incorrectly_flagged']} |

Every exported proposal contains its stitched component centroid/bounding box,
mean and maximum confidence, exact M1 edge pair, measured spacing, deficit, and
source tile/component IDs. Exact pairs are recovered with a local KLayout
`m1.2` query around model proposals; the CNN is still only a candidate generator
and does not replace sign-off DRC.

## Execution decision

The official path uses batched, non-overlapping central-output tiles. A single
fully-convolutional pass is not numerically equivalent because the accepted
multi-task model includes global pooling for the tile classification gate and a
fixed central crop. Recomputing halo features is therefore retained in B7 to
preserve the accepted B6.2 outputs; architectural throughput changes remain a
separate controlled experiment.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    models, checkpoints = load_ensemble(
        args.checkpoint_dir, args.seeds, device, args.base_channels
    )
    checkpoint_signature = ":".join(item["sha256"] for item in checkpoints)
    splits = load_layout_splits(args.registry, args.protocols, args.protocol)
    validation_layouts = splits["validation"]
    development_layouts = splits["test"]
    if args.validation_layouts:
        validation_layouts = list(args.validation_layouts)
    if args.development_layouts:
        development_layouts = list(args.development_layouts)

    validation_scans = _scan_split(
        validation_layouts,
        models,
        device,
        args.batch_size,
        args.max_tiles_per_layout,
        not args.no_source_variants,
        args.output_dir,
        checkpoint_signature,
    )
    selection = select_validation_policy(validation_scans)
    _write_json(args.output_dir / "validation_policy_selection.json", selection)
    render_threshold_tradeoff(selection, args.output_dir / "validation_threshold_tradeoff.png")
    policy = DeploymentPolicy(**selection["selected_policy"])
    validation_results, validation_aggregate = _evaluate_and_export(
        validation_scans, policy, args.output_dir, "validation"
    )
    del validation_scans

    development_scans = _scan_split(
        development_layouts,
        models,
        device,
        args.batch_size,
        args.max_tiles_per_layout,
        not args.no_source_variants,
        args.output_dir,
        checkpoint_signature,
    )
    development_results, development_aggregate = _evaluate_and_export(
        development_scans, policy, args.output_dir, "development_confirmation"
    )
    complete = all(
        item["complete_scan"] for item in validation_results + development_results
    )
    acceptance = _acceptance(validation_aggregate, development_aggregate, complete)
    summary = {
        "schema_version": 1,
        "phase": "B7",
        "status": "complete" if complete else "smoke_only",
        "official_result": bool(complete),
        "protocol": args.protocol,
        "selection_split": "validation_layout_families_only",
        "development_confirmation_is_final_holdout": False,
        "untouched_b9_final_holdout_used": False,
        "ensemble": {
            "method": "arithmetic mean of three B6.2 probability outputs",
            "checkpoints": checkpoints,
        },
        "deployment_policy": policy.to_dict(),
        "validation_policy_proxy": selection["selected_validation_proxy_metrics"],
        "validation": validation_aggregate,
        "development_confirmation": development_aggregate,
        "acceptance": acceptance,
        "runtime": runtime_metadata(device),
        "execution": {
            "selected": "batched_nonoverlapping_central_tiles",
            "fully_convolutional_equivalent": False,
            "reason": "global classification pooling and fixed central output crop make a full-layout pass non-equivalent",
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "README.md").write_text(
        _render_readme(summary), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--protocol", default="unseen_layout_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--max-tiles-per-layout", type=int)
    parser.add_argument("--no-source-variants", action="store_true")
    parser.add_argument("--validation-layouts", nargs="+")
    parser.add_argument("--development-layouts", nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
