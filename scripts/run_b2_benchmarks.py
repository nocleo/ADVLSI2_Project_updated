"""Run and aggregate the frozen B2 dual classification baselines.

The runner deliberately changes only the training seed. Dataset, manifest,
protocol definitions, model, optimizer, and hyperparameters remain fixed so
the two evaluation tracks are comparable and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = PROJECT_ROOT / "run_inference_pc_optimized" / "define_cnn_model.py"
DEFAULT_PROTOCOLS = ("tile_random_reference", "unseen_layout_v1")
DEFAULT_SEEDS = (42, 43, 44)
AGGREGATE_METRICS = (
    "test_loss",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def run_name(protocol: str, seed: int) -> str:
    return f"{protocol}_seed_{seed}"


def build_command(
    args: argparse.Namespace,
    protocol: str,
    seed: int,
    weights_path: Path,
    metrics_path: Path,
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
        str(args.learning_rate),
        "--seed",
        str(seed),
        "--output",
        str(weights_path),
        "--metrics",
        str(metrics_path),
    ]
    if args.cpu:
        command.append("--cpu")
    if args.no_augmentation:
        command.append("--no-augmentation")
    return command


def expected_run_config(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    dataset_sha256: str,
    protocol: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "dataset_archive_sha256": dataset_sha256,
        "manifest_id": manifest["manifest_id"],
        "protocol": protocol,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "augmentation": (
            "none"
            if args.no_augmentation
            else "Manhattan rotations and reflections (training only)"
        ),
        "model": "NCSU_DRCNN",
        "model_source_sha256": sha256_file(MODEL_SOURCE),
        "trainer_source_sha256": sha256_file(args.trainer),
        "optimizer": "RMSprop",
    }


def validate_run(metrics: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": value, "actual": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Run configuration mismatch: {json.dumps(mismatches, sort_keys=True)}")

    required = {
        "best_epoch",
        "best_validation_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_predicted_class_counts",
        "test_confusion_matrix",
        "test_per_layout",
        "runtime",
        "weights_sha256",
        "history",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise ValueError(f"Run metrics are missing required fields: {missing}")

    best_epoch = int(metrics["best_epoch"])
    history = metrics.get("history", [])
    best_records = [record for record in history if int(record.get("epoch", -1)) == best_epoch]
    if len(best_records) != 1:
        raise ValueError(f"Run history does not contain exactly one best epoch {best_epoch}")
    validation_counts = best_records[0].get("validation_predicted_class_counts")
    if not isinstance(validation_counts, dict):
        raise ValueError("Selected checkpoint is missing validation predicted-class counts")


def run_acceptance_issues(metrics: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    test_counts = metrics["test_predicted_class_counts"]
    if any(
        int(test_counts.get(class_name, 0)) == 0
        for class_name in ("clean", "dirty")
    ):
        issues.append(f"test predictor collapsed to one class: {test_counts}")

    best_epoch = int(metrics["best_epoch"])
    best_record = next(
        record for record in metrics["history"] if int(record["epoch"]) == best_epoch
    )
    validation_counts = best_record["validation_predicted_class_counts"]
    if any(
        int(validation_counts.get(class_name, 0)) == 0
        for class_name in ("clean", "dirty")
    ):
        issues.append(
            f"selected checkpoint collapsed on validation: {validation_counts}"
        )
    return issues


def validate_checkpoint(metrics: dict[str, Any], weights_path: Path) -> None:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {weights_path}")
    actual_hash = sha256_file(weights_path)
    if actual_hash != metrics["weights_sha256"]:
        raise ValueError(
            f"Checkpoint hash mismatch for {weights_path}: "
            f"expected {metrics['weights_sha256']}, got {actual_hash}"
        )


def normalize_run_paths(
    metrics: dict[str, Any], output_dir: Path, weights_path: Path
) -> None:
    """Keep a run directory portable when it is copied from Colab or Drive."""

    try:
        metrics["weights"] = weights_path.resolve().relative_to(
            output_dir.resolve()
        ).as_posix()
    except ValueError:
        metrics["weights"] = portable_path(weights_path)


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


def summarize_per_layout(runs: list[dict[str, Any]]) -> dict[str, Any]:
    layouts = sorted(
        {
            layout
            for run in runs
            for layout in run.get("test_per_layout", {})
        }
    )
    summary: dict[str, Any] = {}
    for layout in layouts:
        available = [
            run["test_per_layout"][layout]
            for run in runs
            if layout in run.get("test_per_layout", {})
        ]
        sample_counts = {
            int(result["samples"])
            for result in available
            if "samples" in result
        }
        if len(sample_counts) > 1:
            raise ValueError(f"Per-layout sample count changed across runs for {layout}")
        metric_summary = {
            metric: metric_stats(
                [float(result[metric.removeprefix("test_")]) for result in available]
            )
            for metric in AGGREGATE_METRICS
        }
        summary[layout] = {
            "samples": next(iter(sample_counts)) if sample_counts else None,
            **metric_summary,
        }
    return summary


def pooled_confusion_matrix(runs: list[dict[str, Any]]) -> list[list[int]]:
    pooled = [[0, 0], [0, 0]]
    for run in runs:
        matrix = run["test_confusion_matrix"]
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError(f"Invalid binary confusion matrix: {matrix}")
        for row in range(2):
            for column in range(2):
                pooled[row][column] += int(matrix[row][column])
    return pooled


def summarize_runs(
    runs: list[dict[str, Any]],
    manifest: dict[str, Any],
    dataset_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_protocol: dict[str, list[dict[str, Any]]] = {protocol: [] for protocol in args.protocols}
    for run in runs:
        by_protocol[run["protocol"]].append(run)

    protocols: dict[str, Any] = {}
    for protocol, protocol_runs in by_protocol.items():
        if len(protocol_runs) != len(args.seeds):
            raise ValueError(
                f"Protocol {protocol!r} has {len(protocol_runs)} runs; expected {len(args.seeds)}"
            )
        protocols[protocol] = {
            "seeds": sorted(int(run["seed"]) for run in protocol_runs),
            "metrics": {
                metric: metric_stats([float(run[metric]) for run in protocol_runs])
                for metric in AGGREGATE_METRICS
            },
            "best_epochs": [int(run["best_epoch"]) for run in protocol_runs],
            "pooled_confusion_matrix": pooled_confusion_matrix(protocol_runs),
            "per_layout": summarize_per_layout(protocol_runs),
        }

    runtimes = {
        json.dumps(run["runtime"], sort_keys=True)
        for run in runs
    }
    if len(runtimes) != 1:
        raise ValueError("B2 runs used different runtime environments")
    devices = sorted({str(run["device"]) for run in runs})
    if len(devices) != 1:
        raise ValueError(f"B2 runs used different devices: {devices}")
    runtime = json.loads(next(iter(runtimes)))
    repository_commits = sorted(
        {
            str(run["repository_commit"])
            for run in runs
            if run.get("repository_commit") is not None
        }
    )
    if len(repository_commits) > 1:
        raise ValueError(f"B2 runs used different repository commits: {repository_commits}")

    configuration = {
        "manifest_id": manifest["manifest_id"],
        "dataset_archive_sha256": dataset_sha256,
        "protocols": list(args.protocols),
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "model": "NCSU_DRCNN",
        "model_source_sha256": sha256_file(MODEL_SOURCE),
        "trainer_source_sha256": sha256_file(args.trainer),
        "optimizer": "RMSprop",
        "augmentation": (
            "none"
            if args.no_augmentation
            else "Manhattan rotations and reflections (training only)"
        ),
        "device": devices[0],
        "runtime": runtime,
    }
    config_id = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    acceptance_issues = [
        {
            "protocol": run["protocol"],
            "seed": int(run["seed"]),
            "issues": list(run.get("_acceptance_issues", [])),
        }
        for run in sorted(runs, key=lambda item: (item["protocol"], item["seed"]))
        if run.get("_acceptance_issues")
    ]
    return {
        "phase": "B2",
        "status": "passed" if not acceptance_issues else "failed",
        "acceptance_issues": acceptance_issues,
        "configuration_id": config_id,
        "configuration": configuration,
        "repository_commit": repository_commits[0] if repository_commits else None,
        "protocol_results": protocols,
        "run_metrics": [
            {
                "protocol": run["protocol"],
                "seed": run["seed"],
                "metrics": run["_metrics_path"],
                "weights": run["_weights_path"],
                "weights_sha256": run["weights_sha256"],
            }
            for run in sorted(runs, key=lambda item: (item["protocol"], item["seed"]))
        ],
    }


def percent(mean: float, stddev: float) -> str:
    return f"{100 * mean:.2f}% ± {100 * stddev:.2f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# B2 Dual Classification Baselines",
        "",
        f"Status: **{summary['status'].upper()}**  ",
        f"Configuration: `{summary['configuration_id'][:12]}`  ",
        f"Manifest: `{summary['configuration']['manifest_id']}`",
        "",
        "| Protocol | Accuracy | Dirty precision | Dirty recall | Dirty F1 | Seeds |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for protocol, result in summary["protocol_results"].items():
        metrics = result["metrics"]
        formatted = [
            percent(metrics[name]["mean"], metrics[name]["sample_stddev"])
            for name in ("test_accuracy", "test_precision", "test_recall", "test_f1")
        ]
        lines.append(
            f"| `{protocol}` | {' | '.join(formatted)} | "
            f"{', '.join(map(str, result['seeds']))} |"
        )

    for protocol, result in summary["protocol_results"].items():
        lines.extend(
            [
                "",
                f"## {protocol}: per-layout test metrics",
                "",
                "| Layout | Samples | Accuracy | Dirty recall | Dirty F1 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for layout, metrics in result["per_layout"].items():
            fields = []
            for name in ("test_accuracy", "test_recall", "test_f1"):
                fields.append(percent(metrics[name]["mean"], metrics[name]["sample_stddev"]))
            samples = metrics["samples"] if metrics["samples"] is not None else "n/a"
            lines.append(f"| `{layout}` | {samples} | {' | '.join(fields)} |")

        matrix = result["pooled_confusion_matrix"]
        lines.extend(
            [
                "",
                "Pooled test confusion matrix across seeds:",
                "",
                "| Actual / predicted | Clean | Dirty |",
                "|---|---:|---:|",
                f"| Clean | {matrix[0][0]} | {matrix[0][1]} |",
                f"| Dirty | {matrix[1][0]} | {matrix[1][1]} |",
            ]
        )
    if summary["acceptance_issues"]:
        lines.extend(["", "## Acceptance issues", ""])
        for failed_run in summary["acceptance_issues"]:
            for issue in failed_run["issues"]:
                lines.append(
                    f"- `{failed_run['protocol']}` seed {failed_run['seed']}: {issue}"
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "b2_baselines",
    )
    parser.add_argument("--protocols", nargs="+", default=list(DEFAULT_PROTOCOLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--trainer",
        type=Path,
        default=PROJECT_ROOT / "training" / "train_classifier.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    if len(args.protocols) != len(set(args.protocols)):
        raise ValueError("Protocol names must be unique")
    invalid_protocols = [
        protocol
        for protocol in args.protocols
        if re.fullmatch(r"[A-Za-z0-9_.-]+", protocol) is None
    ]
    if invalid_protocols:
        raise ValueError(f"Protocol names are not safe output identifiers: {invalid_protocols}")
    manifest = load_json(args.manifest)
    dataset_sha256 = sha256_file(args.dataset)
    if dataset_sha256 != manifest["dataset"]["archive_sha256"]:
        raise ValueError("Dataset archive hash does not match the B1 manifest")
    missing_protocols = sorted(set(args.protocols) - set(manifest["protocols"]))
    if missing_protocols:
        raise ValueError(f"Manifest does not define protocols: {missing_protocols}")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Training seeds must be unique")

    runs: list[dict[str, Any]] = []
    for protocol in args.protocols:
        for seed in args.seeds:
            name = run_name(protocol, seed)
            weights_path = args.output_dir / "checkpoints" / f"{name}.pth"
            metrics_path = args.output_dir / "runs" / f"{name}.json"
            command = build_command(args, protocol, seed, weights_path, metrics_path)
            if args.dry_run:
                print(shlex.join(command))
                continue

            expected = expected_run_config(
                args, manifest, dataset_sha256, protocol, seed
            )
            if metrics_path.exists() and not args.force:
                metrics = load_json(metrics_path)
                validate_run(metrics, expected)
                validate_checkpoint(metrics, weights_path)
                print(f"Reusing verified run {name}", flush=True)
            else:
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Running {name}", flush=True)
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)
                metrics = load_json(metrics_path)
                validate_run(metrics, expected)
                validate_checkpoint(metrics, weights_path)
            normalize_run_paths(metrics, args.output_dir, weights_path)
            metrics["_acceptance_issues"] = run_acceptance_issues(metrics)
            metrics_path.write_text(
                json.dumps(
                    {
                        key: value
                        for key, value in metrics.items()
                        if not key.startswith("_")
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                metrics["_metrics_path"] = metrics_path.resolve().relative_to(
                    args.output_dir.resolve()
                ).as_posix()
                metrics["_weights_path"] = weights_path.resolve().relative_to(
                    args.output_dir.resolve()
                ).as_posix()
            except ValueError:
                metrics["_metrics_path"] = portable_path(metrics_path)
                metrics["_weights_path"] = portable_path(weights_path)
            runs.append(metrics)

    if args.dry_run:
        return

    summary = summarize_runs(runs, manifest, dataset_sha256, args)
    summary_path = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "README.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {"summary": str(summary_path), "status": summary["status"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
