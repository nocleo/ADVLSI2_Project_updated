"""Run B3.2 scheduler/early-stopping and B3.3 threshold calibration.

B3.1 remains an immutable negative result.  This extension changes one factor
at a time on ``unseen_layout_v1`` validation data.  Frozen-test evaluation is
unlocked only when the predeclared quality/efficiency/stability gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_b3_optimization import (  # noqa: E402
    BASELINE_LEARNING_RATE,
    BASELINE_OPTIMIZER,
    DEFAULT_SEEDS,
    MODEL_SOURCE,
    SEARCH_PROTOCOL,
    TEST_METRICS,
    load_json,
    metric_stats,
    selected_validation_metrics,
    sha256_file,
    summarize_test_runs,
)

PROTOCOLS = ("tile_random_reference", "unseen_layout_v1")
QUALITY_TOLERANCE = 0.005
MIN_EPOCH_REDUCTION = 0.25
SCHEDULER = {"name": "plateau", "factor": 0.5, "patience": 3, "min_lr": 1e-5}
EARLY_STOPPING = {"patience": 5, "min_delta": 1e-4}


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda run: int(run["seed"]))
    metrics = [selected_validation_metrics(run) for run in ordered]
    return {
        "seeds": [int(run["seed"]) for run in ordered],
        "epochs_completed": metric_stats([float(run.get("epochs_completed", run["epochs"])) for run in ordered]),
        "best_epochs": [int(run["best_epoch"]) for run in ordered],
        "validation": {
            name: metric_stats([float(metric[name]) for metric in metrics])
            for name in ("accuracy", "precision", "recall", "f1")
        },
        "per_seed": [
            {
                "seed": int(run["seed"]),
                "f1": float(metric["f1"]),
                "recall": float(metric["recall"]),
                "epochs_completed": int(run.get("epochs_completed", run["epochs"])),
            }
            for run, metric in zip(ordered, metrics)
        ],
    }


def quality_preserved(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return all(
        float(candidate["validation"][name]["mean"])
        >= float(baseline["validation"][name]["mean"]) - QUALITY_TOLERANCE
        for name in ("accuracy", "recall", "f1")
    )


def b2_runs(args: argparse.Namespace, protocol: str) -> list[dict[str, Any]]:
    runs = []
    for seed in args.seeds:
        run = load_json(args.b2_results / "runs" / f"{protocol}_seed_{seed}.json")
        if run.get("protocol") != protocol or int(run.get("seed", -1)) != seed:
            raise ValueError(f"Invalid B2 run metadata for {protocol}, seed {seed}")
        run["_checkpoint_path"] = str(args.b2_checkpoints / f"{protocol}_seed_{seed}.pth")
        runs.append(run)
    return runs


def train_command(
    args: argparse.Namespace,
    protocol: str,
    seed: int,
    checkpoint: Path,
    metrics: Path,
    scheduler: str,
    early_stopping_patience: int,
    skip_test: bool,
) -> list[str]:
    command = [
        str(args.python), str(args.trainer),
        "--dataset", str(args.dataset), "--manifest", str(args.manifest),
        "--protocol", protocol, "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--learning-rate", str(BASELINE_LEARNING_RATE),
        "--optimizer", BASELINE_OPTIMIZER, "--weight-decay", "0",
        "--scheduler", scheduler, "--scheduler-factor", str(SCHEDULER["factor"]),
        "--scheduler-patience", str(SCHEDULER["patience"]),
        "--scheduler-min-lr", str(SCHEDULER["min_lr"]),
        "--early-stopping-patience", str(early_stopping_patience),
        "--early-stopping-min-delta", str(EARLY_STOPPING["min_delta"] if early_stopping_patience else 0),
        "--seed", str(seed), "--output", str(checkpoint), "--metrics", str(metrics),
    ]
    if skip_test:
        command.append("--skip-test")
    if args.cpu:
        command.append("--cpu")
    return command


def run_recipe(
    args: argparse.Namespace,
    protocol: str,
    recipe_id: str,
    scheduler: str,
    early_stopping_patience: int,
    skip_test: bool,
) -> list[dict[str, Any]]:
    runs = []
    for seed in args.seeds:
        stem = f"{protocol}__{recipe_id}__seed_{seed}"
        checkpoint = args.output_dir / "extension" / "checkpoints" / f"{stem}.pth"
        metrics = args.output_dir / "extension" / "runs" / f"{stem}.json"
        command = train_command(args, protocol, seed, checkpoint, metrics, scheduler, early_stopping_patience, skip_test)
        if args.dry_run:
            print(shlex.join(command))
            continue
        expected = {
            "protocol": protocol,
            "seed": seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "RMSprop",
            "learning_rate": BASELINE_LEARNING_RATE,
            "scheduler": scheduler,
            "early_stopping_patience": early_stopping_patience,
            "test_evaluated": not skip_test,
            "manifest_id": args.manifest_data["manifest_id"],
            "dataset_archive_sha256": args.dataset_sha256,
            "trainer_source_sha256": sha256_file(args.trainer),
        }
        if metrics.exists() and not args.force:
            run = load_json(metrics)
        else:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            metrics.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            run = load_json(metrics)
        mismatches = {key: (value, run.get(key)) for key, value in expected.items() if run.get(key) != value}
        if mismatches:
            raise ValueError(f"Run mismatch for {stem}: {mismatches}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != run.get("weights_sha256"):
            raise ValueError(f"Checkpoint mismatch for {stem}")
        if skip_test and any(metric in run for metric in TEST_METRICS):
            raise ValueError(f"Validation-only run leaked test metrics: {stem}")
        run["_checkpoint_path"] = str(checkpoint)
        runs.append(run)
    return runs


def calibrate_runs(
    args: argparse.Namespace,
    protocol: str,
    recipe_id: str,
    runs: list[dict[str, Any]],
    recall_floor: float,
) -> list[dict[str, Any]]:
    outputs = []
    for run in sorted(runs, key=lambda item: int(item["seed"])):
        seed = int(run["seed"])
        output = args.output_dir / "extension" / "calibration" / f"{protocol}__{recipe_id}__seed_{seed}.json"
        command = [
            str(args.python), str(args.calibrator), "--dataset", str(args.dataset),
            "--manifest", str(args.manifest), "--protocol", protocol,
            "--checkpoint", str(run["_checkpoint_path"]), "--output", str(output),
            "--seed", str(seed), "--batch-size", str(args.batch_size),
            "--recall-floor", str(recall_floor),
        ]
        if args.cpu:
            command.append("--cpu")
        if args.dry_run:
            print(shlex.join(command))
            continue
        if not Path(run["_checkpoint_path"]).is_file():
            raise FileNotFoundError(f"Missing checkpoint: {run['_checkpoint_path']}")
        if not output.exists() or args.force:
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        result = load_json(output)
        if result.get("split") != "validation" or result.get("test_evaluated") is not False:
            raise ValueError(f"Calibration artifact is not validation-only: {output}")
        outputs.append(result)
    return outputs


def calibration_summary(calibrations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "thresholds": [float(item["selected_threshold"]) for item in calibrations],
        "default": {
            name: metric_stats([float(item["default_metrics"][name]) for item in calibrations])
            for name in ("accuracy", "recall", "f1")
        },
        "selected": {
            name: metric_stats([float(item["selected_metrics"][name]) for item in calibrations])
            for name in ("accuracy", "recall", "f1")
        },
    }


def evaluate_runs(
    args: argparse.Namespace,
    protocol: str,
    recipe_id: str,
    runs: list[dict[str, Any]],
    calibrations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outputs = []
    by_seed = {int(item["seed"]): item for item in calibrations}
    for run in sorted(runs, key=lambda item: int(item["seed"])):
        seed = int(run["seed"])
        threshold = float(by_seed[seed]["selected_threshold"])
        output = args.output_dir / "extension" / "test" / f"{protocol}__{recipe_id}__seed_{seed}.json"
        command = [
            str(args.python), str(args.evaluator), "--dataset", str(args.dataset),
            "--manifest", str(args.manifest), "--protocol", protocol,
            "--checkpoint", str(run["_checkpoint_path"]), "--metrics", str(output),
            "--seed", str(seed), "--batch-size", str(args.batch_size),
            "--threshold", str(threshold),
        ]
        if args.cpu:
            command.append("--cpu")
        if not output.exists() or args.force:
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        result = load_json(output)
        if float(result.get("decision_threshold", -1)) != threshold:
            raise ValueError(f"Threshold mismatch in {output}")
        outputs.append(result)
    return outputs


def run_extension(args: argparse.Namespace) -> dict[str, Any] | None:
    baseline_unseen_runs = b2_runs(args, SEARCH_PROTOCOL)
    baseline = summarize_runs(baseline_unseen_runs)
    scheduler_runs = run_recipe(args, SEARCH_PROTOCOL, "plateau_full", "plateau", 0, True)
    if args.dry_run:
        print("# Early-stopping, calibration, and frozen-test commands depend on validation selection.")
        return None
    scheduler_summary = summarize_runs(scheduler_runs)
    scheduler_selected = quality_preserved(scheduler_summary, baseline) and (
        float(scheduler_summary["validation"]["f1"]["mean"])
        >= float(baseline["validation"]["f1"]["mean"])
    )
    selected_full_runs = scheduler_runs if scheduler_selected else baseline_unseen_runs
    selected_scheduler = "plateau" if scheduler_selected else "none"
    selected_full_id = "plateau_full" if scheduler_selected else "b2_baseline"

    early_runs = run_recipe(
        args, SEARCH_PROTOCOL, f"{selected_scheduler}_early_stop",
        selected_scheduler, int(EARLY_STOPPING["patience"]), True,
    )
    early_summary = summarize_runs(early_runs)
    epoch_reduction = 1.0 - float(early_summary["epochs_completed"]["mean"]) / args.epochs
    early_quality = quality_preserved(early_summary, baseline)
    early_selected = early_quality and (
        epoch_reduction >= MIN_EPOCH_REDUCTION
        or float(early_summary["validation"]["f1"]["mean"])
        > float((scheduler_summary if scheduler_selected else baseline)["validation"]["f1"]["mean"])
    )
    final_runs = early_runs if early_selected else selected_full_runs
    final_recipe = f"{selected_scheduler}_early_stop" if early_selected else selected_full_id
    final_summary = early_summary if early_selected else (scheduler_summary if scheduler_selected else baseline)
    recall_floor = max(0.0, float(baseline["validation"]["recall"]["mean"]) - QUALITY_TOLERANCE)
    unseen_calibrations = calibrate_runs(args, SEARCH_PROTOCOL, final_recipe, final_runs, recall_floor)
    calibration = calibration_summary(unseen_calibrations)
    calibrated_quality = all(
        float(calibration["selected"][name]["mean"])
        >= float(baseline["validation"][name]["mean"]) - QUALITY_TOLERANCE
        for name in ("accuracy", "recall", "f1")
    )
    quality_gain = float(calibration["selected"]["f1"]["mean"]) > float(baseline["validation"]["f1"]["mean"])
    recall_gain = float(calibration["selected"]["recall"]["mean"]) > float(baseline["validation"]["recall"]["mean"])
    lower_variance = (
        float(calibration["selected"]["f1"]["sample_stddev"])
        < float(baseline["validation"]["f1"]["sample_stddev"])
        and calibrated_quality
    )
    efficiency_gain = early_selected and epoch_reduction >= MIN_EPOCH_REDUCTION
    benefits = {
        "quality_gain": quality_gain,
        "recall_gain": recall_gain,
        "lower_seed_variance": lower_variance,
        "at_least_25_percent_fewer_epochs": efficiency_gain,
    }
    validation_passed = calibrated_quality and any(benefits.values())
    summary: dict[str, Any] = {
        "phase": "B3 extension", "experiment": "scheduler_early_stopping_threshold",
        "test_used_for_selection": False, "manifest_id": args.manifest_data["manifest_id"],
        "dataset_archive_sha256": args.dataset_sha256, "seeds": list(args.seeds),
        "epochs": args.epochs, "quality_tolerance": QUALITY_TOLERANCE,
        "minimum_epoch_reduction": MIN_EPOCH_REDUCTION,
        "baseline_validation": baseline, "scheduler_validation": scheduler_summary,
        "scheduler_selected": scheduler_selected, "early_stopping_validation": early_summary,
        "epoch_reduction": epoch_reduction, "early_stopping_selected": early_selected,
        "selected_recipe": final_recipe, "selected_validation": final_summary,
        "threshold_calibration": calibration, "benefits": benefits,
        "validation_gate_passed": validation_passed, "frozen_test": {},
        "source_hashes": {
            "trainer": sha256_file(args.trainer), "calibrator": sha256_file(args.calibrator),
            "evaluator": sha256_file(args.evaluator), "model": sha256_file(MODEL_SOURCE),
        },
    }
    if not validation_passed:
        summary["status"] = "failed_validation"
        return summary

    unseen_test = evaluate_runs(args, SEARCH_PROTOCOL, final_recipe, final_runs, unseen_calibrations)
    if final_recipe == "b2_baseline":
        tile_runs = b2_runs(args, "tile_random_reference")
    else:
        tile_runs = run_recipe(
            args, "tile_random_reference", final_recipe, selected_scheduler,
            int(EARLY_STOPPING["patience"]) if early_selected else 0, True,
        )
    tile_baseline = summarize_runs(b2_runs(args, "tile_random_reference"))
    tile_recall_floor = max(0.0, float(tile_baseline["validation"]["recall"]["mean"]) - QUALITY_TOLERANCE)
    tile_calibrations = calibrate_runs(args, "tile_random_reference", final_recipe, tile_runs, tile_recall_floor)
    tile_test = evaluate_runs(args, "tile_random_reference", final_recipe, tile_runs, tile_calibrations)
    summary["threshold_calibration_by_protocol"] = {
        "unseen_layout_v1": calibration,
        "tile_random_reference": calibration_summary(tile_calibrations),
    }
    frozen = {
        "unseen_layout_v1": summarize_test_runs(unseen_test),
        "tile_random_reference": summarize_test_runs(tile_test),
    }
    regressions = []
    b2_summary = load_json(args.b2_results / "summary.json")["protocol_results"]
    for protocol in PROTOCOLS:
        baseline_metrics = b2_summary[protocol]["metrics"]
        for metric in ("test_accuracy", "test_recall", "test_f1"):
            if float(frozen[protocol]["metrics"][metric]["mean"]) < float(baseline_metrics[metric]["mean"]) - QUALITY_TOLERANCE:
                regressions.append(f"{protocol}: {metric} regressed beyond tolerance")
    summary["frozen_test"] = frozen
    summary["frozen_test_regressions"] = regressions
    summary["status"] = "passed" if not regressions else "failed_confirmation"
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    def pct(stats: dict[str, Any]) -> str:
        return f"{100 * float(stats['mean']):.2f}% +/- {100 * float(stats['sample_stddev']):.2f}%"
    lines = [
        "# B3.2/B3.3 Extension", "", f"Status: **{summary['status'].upper()}**  ",
        "Selection used validation data only; frozen tests were locked until the validation gate passed.", "",
        "| Stage | Validation dirty F1 | Validation dirty recall | Mean epochs |", "|---|---:|---:|---:|",
    ]
    for name, key in (("B2 baseline", "baseline_validation"), ("Plateau scheduler", "scheduler_validation"), ("Early stopping", "early_stopping_validation")):
        value = summary[key]
        lines.append(f"| {name} | {pct(value['validation']['f1'])} | {pct(value['validation']['recall'])} | {float(value['epochs_completed']['mean']):.1f} |")
    selected = summary["threshold_calibration"]["selected"]
    lines += ["", f"Selected recipe: `{summary['selected_recipe']}`.", f"Calibrated validation dirty F1: {pct(selected['f1'])}.", "", "Benefits:"]
    lines += [f"- {name.replace('_', ' ')}: **{value}**" for name, value in summary["benefits"].items()]
    if summary.get("frozen_test"):
        lines += ["", "## Frozen-test confirmation", "", "| Protocol | Accuracy | Dirty recall | Dirty F1 |", "|---|---:|---:|---:|"]
        for protocol, result in summary["frozen_test"].items():
            metrics = result["metrics"]
            lines.append(f"| `{protocol}` | {pct(metrics['test_accuracy'])} | {pct(metrics['test_recall'])} | {pct(metrics['test_f1'])} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "training_datasets" / "combined_training_dataset.zip")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "b1_current_audit" / "manifest.json")
    parser.add_argument("--b2-results", type=Path, default=PROJECT_ROOT / "results" / "b2_baselines")
    parser.add_argument("--b2-checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "b3_optimization")
    parser.add_argument("--trainer", type=Path, default=PROJECT_ROOT / "training" / "train_classifier.py")
    parser.add_argument("--calibrator", type=Path, default=PROJECT_ROOT / "training" / "calibrate_classifier_threshold.py")
    parser.add_argument("--evaluator", type=Path, default=PROJECT_ROOT / "training" / "evaluate_classifier.py")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or args.epochs < 1 or args.batch_size < 1:
        parser.error("seeds must be unique; epochs and batch size must be positive")
    args.manifest_data = load_json(args.manifest)
    args.dataset_sha256 = sha256_file(args.dataset)
    if args.dataset_sha256 != args.manifest_data["dataset"]["archive_sha256"]:
        parser.error("dataset archive hash does not match manifest")
    return args


def main() -> None:
    args = parse_args()
    summary = run_extension(args)
    if summary is None:
        return
    summary["configuration_id"] = hashlib.sha256(json.dumps({
        key: summary[key] for key in ("experiment", "manifest_id", "dataset_archive_sha256", "seeds", "epochs")
    }, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "extension_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "EXTENSION_README.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "selected_recipe": summary["selected_recipe"]}, indent=2))


if __name__ == "__main__":
    main()
