"""Run B3's pre-registered optimizer and learning-rate experiment.

The search uses only ``unseen_layout_v1`` train/validation data.  It first
compares RMSprop and Adam at 1e-3, then sweeps 3e-4/1e-3/3e-3 for the selected
optimizer.  The frozen test splits are evaluated only after a configuration is
selected from mean validation dirty F1 across seeds 42/43/44, with validation
dirty recall constrained not to fall below B2.
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
MODEL_SOURCE = PROJECT_ROOT / "run_inference_pc_optimized" / "define_cnn_model.py"
DEFAULT_SEEDS = (42, 43, 44)
SEARCH_PROTOCOL = "unseen_layout_v1"
CONFIRMATION_PROTOCOLS = ("tile_random_reference", "unseen_layout_v1")
BASELINE_OPTIMIZER = "rmsprop"
BASELINE_LEARNING_RATE = 0.001
LEARNING_RATES = (0.0003, 0.001, 0.003)
OPTIMIZER_LABELS = {"rmsprop": "RMSprop", "adam": "Adam"}
TEST_METRICS = ("test_accuracy", "test_precision", "test_recall", "test_f1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty metric series")
    return {
        "runs": len(values),
        "mean": statistics.fmean(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def portable_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def learning_rate_id(value: float) -> str:
    return format(value, ".8g").replace(".", "p").replace("-", "m")


def candidate_id(optimizer: str, learning_rate: float) -> str:
    return f"{optimizer}_lr_{learning_rate_id(learning_rate)}"


def selected_validation_metrics(run: dict[str, Any]) -> dict[str, Any]:
    direct = run.get("best_validation_metrics")
    if isinstance(direct, dict):
        return direct
    best_epoch = int(run["best_epoch"])
    matches = [
        record
        for record in run.get("history", [])
        if int(record.get("epoch", -1)) == best_epoch
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Run does not contain exactly one selected validation epoch {best_epoch}"
        )
    record = matches[0]
    return {
        "loss": record["validation_loss"],
        "accuracy": record["validation_accuracy"],
        "precision": record["validation_precision"],
        "recall": record["validation_recall"],
        "f1": record["validation_f1"],
        "predicted_class_counts": record["validation_predicted_class_counts"],
        "confusion_matrix": record["validation_confusion_matrix"],
    }


def summarize_validation_candidate(
    optimizer: str,
    learning_rate: float,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda run: int(run["seed"]))
    metrics = [selected_validation_metrics(run) for run in ordered]
    collapsed_seeds = [
        int(run["seed"])
        for run, metric in zip(ordered, metrics)
        if any(
            int(metric.get("predicted_class_counts", {}).get(name, 0)) == 0
            for name in ("clean", "dirty")
        )
    ]
    return {
        "candidate_id": candidate_id(optimizer, learning_rate),
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "seeds": [int(run["seed"]) for run in ordered],
        "best_epochs": [int(run["best_epoch"]) for run in ordered],
        "collapsed_seeds": collapsed_seeds,
        "validation": {
            name: metric_stats([float(metric[name]) for metric in metrics])
            for name in ("loss", "accuracy", "precision", "recall", "f1")
        },
        "per_seed": [
            {
                "seed": int(run["seed"]),
                "best_epoch": int(run["best_epoch"]),
                "validation_f1": float(metric["f1"]),
                "validation_recall": float(metric["recall"]),
            }
            for run, metric in zip(ordered, metrics)
        ],
    }


def select_candidate(
    candidates: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_recall = float(baseline["validation"]["recall"]["mean"])
    eligible = [
        candidate
        for candidate in candidates
        if not candidate["collapsed_seeds"]
        and float(candidate["validation"]["recall"]["mean"]) >= baseline_recall
    ]
    if not eligible:
        raise ValueError("No candidate preserves baseline validation dirty recall")
    return max(
        eligible,
        key=lambda candidate: (
            float(candidate["validation"]["f1"]["mean"]),
            float(candidate["validation"]["recall"]["mean"]),
            candidate["candidate_id"],
        ),
    )


def paired_seed_wins(candidate: dict[str, Any], baseline: dict[str, Any]) -> int:
    baseline_by_seed = {
        int(item["seed"]): float(item["validation_f1"])
        for item in baseline["per_seed"]
    }
    return sum(
        float(item["validation_f1"]) > baseline_by_seed[int(item["seed"])]
        for item in candidate["per_seed"]
    )


def baseline_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs = []
    for seed in args.seeds:
        path = args.b2_results / "runs" / f"{SEARCH_PROTOCOL}_seed_{seed}.json"
        run = load_json(path)
        expected = {
            "protocol": SEARCH_PROTOCOL,
            "seed": seed,
            "manifest_id": args.manifest_data["manifest_id"],
            "dataset_archive_sha256": args.dataset_sha256,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": BASELINE_LEARNING_RATE,
            "optimizer": "RMSprop",
        }
        mismatches = {
            key: {"expected": value, "actual": run.get(key)}
            for key, value in expected.items()
            if run.get(key) != value
        }
        if mismatches:
            raise ValueError(f"B2 baseline mismatch in {path}: {mismatches}")
        runs.append(run)
    return runs


def build_train_command(
    args: argparse.Namespace,
    protocol: str,
    optimizer: str,
    learning_rate: float,
    seed: int,
    checkpoint: Path,
    metrics: Path,
    skip_test: bool,
) -> list[str]:
    command = [
        str(args.python),
        str(args.trainer),
        "--dataset",
        str(args.dataset),
        "--manifest",
        str(args.manifest),
        "--protocol",
        protocol,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(learning_rate),
        "--optimizer",
        optimizer,
        "--weight-decay",
        "0",
        "--seed",
        str(seed),
        "--output",
        str(checkpoint),
        "--metrics",
        str(metrics),
    ]
    if skip_test:
        command.append("--skip-test")
    if args.cpu:
        command.append("--cpu")
    return command


def expected_train_config(
    args: argparse.Namespace,
    protocol: str,
    optimizer: str,
    learning_rate: float,
    seed: int,
    skip_test: bool,
) -> dict[str, Any]:
    return {
        "dataset_archive_sha256": args.dataset_sha256,
        "manifest_id": args.manifest_data["manifest_id"],
        "protocol": protocol,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "augmentation": "Manhattan rotations and reflections (training only)",
        "model": "NCSU_DRCNN",
        "model_source_sha256": sha256_file(MODEL_SOURCE),
        "trainer_source_sha256": sha256_file(args.trainer),
        "optimizer": OPTIMIZER_LABELS[optimizer],
        "test_evaluated": not skip_test,
    }


def validate_train_run(
    run: dict[str, Any],
    expected: dict[str, Any],
    checkpoint: Path,
) -> None:
    mismatches = {
        key: {"expected": value, "actual": run.get(key)}
        for key, value in expected.items()
        if run.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Training run configuration mismatch: {mismatches}")
    selected = selected_validation_metrics(run)
    counts = selected.get("predicted_class_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("Selected validation checkpoint is missing class counts")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    if sha256_file(checkpoint) != run.get("weights_sha256"):
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    if expected["test_evaluated"]:
        missing = sorted(set(TEST_METRICS) - run.keys())
        if missing:
            raise ValueError(f"Confirmed run is missing test metrics: {missing}")
    elif any(name in run for name in TEST_METRICS):
        raise ValueError("Search-only run contains frozen test metrics")


def run_training_candidate(
    args: argparse.Namespace,
    protocol: str,
    optimizer: str,
    learning_rate: float,
    skip_test: bool,
) -> list[dict[str, Any]]:
    phase = "search" if skip_test else "confirmation"
    candidate = candidate_id(optimizer, learning_rate)
    runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        stem = f"{protocol}__{candidate}__seed_{seed}"
        checkpoint = args.output_dir / phase / "checkpoints" / f"{stem}.pth"
        metrics = args.output_dir / phase / "runs" / f"{stem}.json"
        command = build_train_command(
            args,
            protocol,
            optimizer,
            learning_rate,
            seed,
            checkpoint,
            metrics,
            skip_test,
        )
        if args.dry_run:
            print(shlex.join(command))
            continue
        expected = expected_train_config(
            args, protocol, optimizer, learning_rate, seed, skip_test
        )
        if metrics.exists() and not args.force:
            run = load_json(metrics)
            validate_train_run(run, expected, checkpoint)
            print(f"Reusing verified {phase} run {stem}", flush=True)
        else:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            metrics.parent.mkdir(parents=True, exist_ok=True)
            print(f"Running {phase} run {stem}", flush=True)
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            run = load_json(metrics)
            validate_train_run(run, expected, checkpoint)
        run["_checkpoint_path"] = str(checkpoint)
        run["_metrics_path"] = str(metrics)
        runs.append(run)
    return runs


def build_evaluation_command(
    args: argparse.Namespace,
    checkpoint: Path,
    metrics: Path,
    seed: int,
) -> list[str]:
    command = [
        str(args.python),
        str(args.evaluator),
        "--dataset",
        str(args.dataset),
        "--manifest",
        str(args.manifest),
        "--protocol",
        SEARCH_PROTOCOL,
        "--checkpoint",
        str(checkpoint),
        "--metrics",
        str(metrics),
        "--seed",
        str(seed),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.cpu:
        command.append("--cpu")
    return command


def evaluate_selected_unseen(
    args: argparse.Namespace,
    search_runs: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for run in sorted(search_runs, key=lambda item: int(item["seed"])):
        seed = int(run["seed"])
        checkpoint = Path(run["_checkpoint_path"])
        metrics = (
            args.output_dir
            / "confirmation"
            / "evaluations"
            / f"{SEARCH_PROTOCOL}__{candidate['candidate_id']}__seed_{seed}.json"
        )
        command = build_evaluation_command(args, checkpoint, metrics, seed)
        if metrics.exists() and not args.force:
            result = load_json(metrics)
        else:
            metrics.parent.mkdir(parents=True, exist_ok=True)
            print(f"Evaluating selected unseen-layout checkpoint for seed {seed}")
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            result = load_json(metrics)
        expected = {
            "dataset_archive_sha256": args.dataset_sha256,
            "manifest_id": args.manifest_data["manifest_id"],
            "protocol": SEARCH_PROTOCOL,
            "split": "test",
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        mismatches = {
            key: {"expected": value, "actual": result.get(key)}
            for key, value in expected.items()
            if result.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Checkpoint evaluation mismatch: {mismatches}")
        results.append(result)
    return results


def summarize_test_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seeds": sorted(int(run["seed"]) for run in runs),
        "metrics": {
            metric: metric_stats([float(run[metric]) for run in runs])
            for metric in TEST_METRICS
        },
    }


def b2_test_baselines(args: argparse.Namespace) -> dict[str, Any]:
    summary = load_json(args.b2_results / "summary.json")
    if summary["configuration"]["manifest_id"] != args.manifest_data["manifest_id"]:
        raise ValueError("B2 summary uses a different manifest")
    return summary["protocol_results"]


def test_regressions(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    issues = []
    for metric in ("test_accuracy", "test_recall", "test_f1"):
        candidate_mean = float(candidate["metrics"][metric]["mean"])
        baseline_mean = float(baseline["metrics"][metric]["mean"])
        if candidate_mean < baseline_mean:
            issues.append(
                f"{metric} regressed from {baseline_mean:.6f} to {candidate_mean:.6f}"
            )
    return issues


def run_experiment(args: argparse.Namespace) -> dict[str, Any] | None:
    baseline_run_set = baseline_runs(args)
    baseline = summarize_validation_candidate(
        BASELINE_OPTIMIZER,
        BASELINE_LEARNING_RATE,
        baseline_run_set,
    )

    adam_runs = run_training_candidate(
        args,
        SEARCH_PROTOCOL,
        "adam",
        BASELINE_LEARNING_RATE,
        skip_test=True,
    )
    if args.dry_run:
        print("# Later learning-rate and confirmation commands depend on validation selection.")
        return None
    stage1_candidates = [
        baseline,
        summarize_validation_candidate(
            "adam", BASELINE_LEARNING_RATE, adam_runs
        ),
    ]
    stage1_selected = select_candidate(stage1_candidates, baseline)
    selected_optimizer = str(stage1_selected["optimizer"])
    search_runs: dict[str, list[dict[str, Any]]] = {
        baseline["candidate_id"]: baseline_run_set,
        candidate_id("adam", BASELINE_LEARNING_RATE): adam_runs,
    }

    stage2_candidates = []
    for learning_rate in LEARNING_RATES:
        identifier = candidate_id(selected_optimizer, learning_rate)
        if identifier not in search_runs:
            search_runs[identifier] = run_training_candidate(
                args,
                SEARCH_PROTOCOL,
                selected_optimizer,
                learning_rate,
                skip_test=True,
            )
        stage2_candidates.append(
            summarize_validation_candidate(
                selected_optimizer,
                learning_rate,
                search_runs[identifier],
            )
        )
    selected = select_candidate(stage2_candidates, baseline)
    wins = paired_seed_wins(selected, baseline)
    search_issues = []
    if float(selected["validation"]["f1"]["mean"]) <= float(
        baseline["validation"]["f1"]["mean"]
    ):
        search_issues.append("selected mean validation dirty F1 did not improve over B2")
    if wins < 2:
        search_issues.append(
            f"selected validation dirty F1 improved on only {wins}/{len(args.seeds)} paired seeds"
        )

    summary: dict[str, Any] = {
        "phase": "B3",
        "experiment": "optimizer_then_learning_rate",
        "selection_protocol": SEARCH_PROTOCOL,
        "selection_metric": "mean validation dirty F1",
        "test_used_for_selection": False,
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": 0.0,
        "manifest_id": args.manifest_data["manifest_id"],
        "dataset_archive_sha256": args.dataset_sha256,
        "model": "NCSU_DRCNN",
        "model_source_sha256": sha256_file(MODEL_SOURCE),
        "trainer_source_sha256": sha256_file(args.trainer),
        "evaluator_source_sha256": sha256_file(args.evaluator),
        "baseline_validation": baseline,
        "optimizer_stage": {
            "candidates": stage1_candidates,
            "selected": stage1_selected["candidate_id"],
        },
        "learning_rate_stage": {
            "candidates": stage2_candidates,
            "selected": selected["candidate_id"],
        },
        "selected_configuration": selected,
        "paired_seed_f1_wins": wins,
        "search_acceptance_issues": search_issues,
        "confirmation": {},
    }
    if search_issues:
        summary["status"] = "failed_search"
        return summary

    selected_runs = search_runs[selected["candidate_id"]]
    unseen_runs = evaluate_selected_unseen(args, selected_runs, selected)
    tile_runs = run_training_candidate(
        args,
        "tile_random_reference",
        str(selected["optimizer"]),
        float(selected["learning_rate"]),
        skip_test=False,
    )
    confirmation = {
        "unseen_layout_v1": summarize_test_runs(unseen_runs),
        "tile_random_reference": summarize_test_runs(tile_runs),
    }
    baselines = b2_test_baselines(args)
    confirmation_issues = []
    for protocol in CONFIRMATION_PROTOCOLS:
        protocol_issues = test_regressions(confirmation[protocol], baselines[protocol])
        confirmation[protocol]["b2_baseline"] = baselines[protocol]["metrics"]
        confirmation[protocol]["acceptance_issues"] = protocol_issues
        confirmation_issues.extend(f"{protocol}: {issue}" for issue in protocol_issues)
    summary["confirmation"] = confirmation
    summary["confirmation_acceptance_issues"] = confirmation_issues
    summary["status"] = "passed" if not confirmation_issues else "failed_confirmation"
    return summary


def percent(stats: dict[str, Any]) -> str:
    return f"{100 * float(stats['mean']):.2f}% +/- {100 * float(stats['sample_stddev']):.2f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# B3 Training Optimization",
        "",
        f"Status: **{str(summary['status']).upper()}**  ",
        f"Manifest: `{summary['manifest_id']}`  ",
        "Selection: unseen-layout validation dirty F1 only; frozen test data was not used.",
        "",
        "## Optimizer stage (learning rate 0.001)",
        "",
        "| Candidate | Validation dirty F1 | Validation dirty recall |",
        "|---|---:|---:|",
    ]
    for candidate in summary["optimizer_stage"]["candidates"]:
        lines.append(
            f"| `{candidate['candidate_id']}` | "
            f"{percent(candidate['validation']['f1'])} | "
            f"{percent(candidate['validation']['recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Learning-rate stage",
            "",
            "| Candidate | Validation dirty F1 | Validation dirty recall |",
            "|---|---:|---:|",
        ]
    )
    for candidate in summary["learning_rate_stage"]["candidates"]:
        lines.append(
            f"| `{candidate['candidate_id']}` | "
            f"{percent(candidate['validation']['f1'])} | "
            f"{percent(candidate['validation']['recall'])} |"
        )
    lines.extend(
        [
            "",
            f"Selected: `{summary['selected_configuration']['candidate_id']}`; "
            f"paired validation-F1 wins: {summary['paired_seed_f1_wins']}/{len(summary['seeds'])}.",
        ]
    )
    if summary["search_acceptance_issues"]:
        lines.extend(["", "## Search acceptance issues", ""])
        lines.extend(f"- {issue}" for issue in summary["search_acceptance_issues"])
    if summary.get("confirmation"):
        lines.extend(
            [
                "",
                "## Frozen test confirmation",
                "",
                "| Protocol | Accuracy | Dirty recall | Dirty F1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for protocol, result in summary["confirmation"].items():
            metrics = result["metrics"]
            lines.append(
                f"| `{protocol}` | {percent(metrics['test_accuracy'])} | "
                f"{percent(metrics['test_recall'])} | {percent(metrics['test_f1'])} |"
            )
        if summary.get("confirmation_acceptance_issues"):
            lines.extend(["", "## Confirmation acceptance issues", ""])
            lines.extend(
                f"- {issue}" for issue in summary["confirmation_acceptance_issues"]
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "training_datasets" / "combined_training_dataset.zip",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "b1_current_audit" / "manifest.json",
    )
    parser.add_argument(
        "--b2-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "b2_baselines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "b3_optimization",
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        default=PROJECT_ROOT / "training" / "train_classifier.py",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=PROJECT_ROOT / "training" / "evaluate_classifier.py",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    args.manifest_data = load_json(args.manifest)
    args.dataset_sha256 = sha256_file(args.dataset)
    if args.dataset_sha256 != args.manifest_data["dataset"]["archive_sha256"]:
        parser.error("dataset archive hash does not match the B1 manifest")
    return args


def main() -> None:
    args = parse_args()
    summary = run_experiment(args)
    if summary is None:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: summary[key]
        for key in (
            "experiment",
            "selection_protocol",
            "selection_metric",
            "seeds",
            "epochs",
            "batch_size",
            "weight_decay",
            "manifest_id",
            "dataset_archive_sha256",
            "model_source_sha256",
            "trainer_source_sha256",
            "evaluator_source_sha256",
        )
    }
    summary["configuration_id"] = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "selected": summary["selected_configuration"]["candidate_id"]}, indent=2))


if __name__ == "__main__":
    main()
