"""Run B4's pre-registered compact-architecture experiment.

Candidate selection uses only the frozen unseen-layout validation split.  The
runner creates frozen-test evaluations and tile-random confirmation runs only
after the validation gate passes.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_b2_benchmarks import metric_stats, pooled_confusion_matrix, summarize_per_layout
from training.dataset_manifest import sha256_file


SEEDS = (42, 43, 44)
BASELINE_MODEL = "NCSU_DRCNN"
COMPACT_MODEL = "CompactBNPool"
COMPACT_MODEL_SOURCE = PROJECT_ROOT / "training" / "classifier_models.py"
SEARCH_PROTOCOL = "unseen_layout_v1"
REFERENCE_PROTOCOL = "tile_random_reference"
QUALITY_TOLERANCE = 0.005
MAX_LATENCY_RATIO = 1.5
VALIDATION_METRICS = ("accuracy", "precision", "recall", "f1")
TEST_METRICS = ("test_accuracy", "test_precision", "test_recall", "test_f1")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def training_name(protocol: str, seed: int) -> str:
    return f"{protocol}__{COMPACT_MODEL}__seed_{seed}"


def checkpoint_path(args: argparse.Namespace, protocol: str, seed: int) -> Path:
    return args.output_dir / "checkpoints" / f"{training_name(protocol, seed)}.pth"


def training_metrics_path(args: argparse.Namespace, protocol: str, seed: int) -> Path:
    return args.output_dir / "validation" / f"{training_name(protocol, seed)}.json"


def test_metrics_path(args: argparse.Namespace, protocol: str, seed: int) -> Path:
    return args.output_dir / "test" / f"{training_name(protocol, seed)}.json"


def build_training_command(
    args: argparse.Namespace, protocol: str, seed: int
) -> list[str]:
    return [
        str(args.python), str(args.trainer),
        "--dataset", str(args.dataset),
        "--manifest", str(args.manifest),
        "--protocol", protocol,
        "--model", COMPACT_MODEL,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--optimizer", "rmsprop",
        "--seed", str(seed),
        "--output", str(checkpoint_path(args, protocol, seed)),
        "--metrics", str(training_metrics_path(args, protocol, seed)),
        "--skip-test",
    ] + (["--cpu"] if args.cpu else [])


def build_evaluation_command(
    args: argparse.Namespace, protocol: str, seed: int
) -> list[str]:
    return [
        str(args.python), str(args.evaluator),
        "--dataset", str(args.dataset),
        "--manifest", str(args.manifest),
        "--protocol", protocol,
        "--model", COMPACT_MODEL,
        "--checkpoint", str(checkpoint_path(args, protocol, seed)),
        "--metrics", str(test_metrics_path(args, protocol, seed)),
        "--seed", str(seed),
        "--batch-size", str(args.batch_size),
        "--threshold", "0.5",
    ] + (["--cpu"] if args.cpu else [])


def build_benchmark_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python), str(args.benchmark),
        "--models", BASELINE_MODEL, COMPACT_MODEL,
        "--output", str(args.output_dir / "architecture_benchmark.json"),
        "--warmup", str(args.benchmark_warmup),
        "--repeats", str(args.benchmark_repeats),
        "--threads", "1",
    ]


def selected_validation_metrics(run: dict[str, Any]) -> dict[str, Any]:
    """Return selected validation metrics from either trainer result schema.

    B2 artifacts predate the top-level ``best_validation_metrics`` field and
    store the selected metrics only in the history record for ``best_epoch``.
    B4 artifacts contain the newer convenience field.  Supporting both keeps
    the checked-in B2 baseline authoritative and resume-safe.
    """
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


def summarize_validation(runs: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [selected_validation_metrics(run) for run in runs]
    return {
        "seeds": [int(run["seed"]) for run in runs],
        "metrics": {
            name: metric_stats([float(metrics[name]) for metrics in selected])
            for name in VALIDATION_METRICS
        },
        "best_epochs": [int(run["best_epoch"]) for run in runs],
        "per_seed": [
            {
                "seed": int(run["seed"]),
                **{
                    name: float(metrics[name])
                    for name in VALIDATION_METRICS
                },
            }
            for run, metrics in zip(runs, selected)
        ],
    }


def paired_wins(candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> int:
    baseline_by_seed = {
        int(run["seed"]): float(run[metric]) for run in baseline["per_seed"]
    }
    return sum(
        float(run[metric]) > baseline_by_seed[int(run["seed"])]
        for run in candidate["per_seed"]
    )


def validation_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    f1_wins = paired_wins(candidate, baseline, "f1")
    checks = {
        "mean_f1_improved": candidate["metrics"]["f1"]["mean"] > baseline["metrics"]["f1"]["mean"],
        "paired_f1_wins_at_least_two": f1_wins >= 2,
        "accuracy_preserved": candidate["metrics"]["accuracy"]["mean"] >= baseline["metrics"]["accuracy"]["mean"] - QUALITY_TOLERANCE,
        "recall_preserved": candidate["metrics"]["recall"]["mean"] >= baseline["metrics"]["recall"]["mean"] - QUALITY_TOLERANCE,
        "no_collapsed_seed": all(
            all(int(count) > 0 for count in run["best_validation_metrics"]["predicted_class_counts"].values())
            for run in candidate["_runs"]
        ),
    }
    return {"passed": all(checks.values()), "paired_f1_wins": f1_wins, "checks": checks}


def summarize_test(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seeds": [int(run["seed"]) for run in runs],
        "metrics": {
            name: metric_stats([float(run[name]) for run in runs])
            for name in TEST_METRICS
        },
        "per_seed": [
            {"seed": int(run["seed"]), **{name: float(run[name]) for name in TEST_METRICS}}
            for run in runs
        ],
        "pooled_confusion_matrix": pooled_confusion_matrix(runs),
        "per_layout": summarize_per_layout(runs),
    }


def frozen_test_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    f1_wins = paired_wins(candidate, baseline, "test_f1")
    recall_wins = paired_wins(candidate, baseline, "test_recall")
    checks = {
        "mean_f1_improved": candidate["metrics"]["test_f1"]["mean"] > baseline["metrics"]["test_f1"]["mean"],
        "mean_recall_improved": candidate["metrics"]["test_recall"]["mean"] > baseline["metrics"]["test_recall"]["mean"],
        "paired_f1_wins_at_least_two": f1_wins >= 2,
        "paired_recall_wins_at_least_two": recall_wins >= 2,
        "accuracy_preserved": candidate["metrics"]["test_accuracy"]["mean"] >= baseline["metrics"]["test_accuracy"]["mean"] - QUALITY_TOLERANCE,
    }
    return {
        "passed": all(checks.values()),
        "paired_f1_wins": f1_wins,
        "paired_recall_wins": recall_wins,
        "checks": checks,
    }


def reference_test_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    checks = {
        name.removeprefix("test_") + "_preserved":
        candidate["metrics"][name]["mean"] >= baseline["metrics"][name]["mean"] - QUALITY_TOLERANCE
        for name in ("test_accuracy", "test_recall", "test_f1")
    }
    return {"passed": all(checks.values()), "checks": checks}


def cost_gate(benchmark: dict[str, Any]) -> dict[str, Any]:
    baseline = benchmark["models"][BASELINE_MODEL]
    candidate = benchmark["models"][COMPACT_MODEL]
    torch_ratio = candidate["pytorch_cpu_batch1"]["median_ms"] / baseline["pytorch_cpu_batch1"]["median_ms"]
    onnx_ratio = candidate["onnx_cpu_batch1"]["median_ms"] / baseline["onnx_cpu_batch1"]["median_ms"]
    checks = {
        "fewer_parameters": candidate["parameters"] < baseline["parameters"],
        "smaller_state_dict": candidate["state_dict_bytes"] < baseline["state_dict_bytes"],
        "pytorch_latency_at_most_1_5x": torch_ratio <= MAX_LATENCY_RATIO,
        "onnx_latency_at_most_1_5x": onnx_ratio <= MAX_LATENCY_RATIO,
    }
    return {
        "passed": all(checks.values()),
        "pytorch_latency_ratio": torch_ratio,
        "onnx_latency_ratio": onnx_ratio,
        "checks": checks,
    }


def expected_training(args: argparse.Namespace, protocol: str, seed: int) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    return {
        "dataset_archive_sha256": sha256_file(args.dataset),
        "manifest_id": manifest["manifest_id"],
        "protocol": protocol,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "RMSprop",
        "weight_decay": 0.0,
        "scheduler": "none",
        "early_stopping_patience": 0,
        "model": COMPACT_MODEL,
        "model_source_sha256": sha256_file(COMPACT_MODEL_SOURCE),
        "trainer_source_sha256": sha256_file(args.trainer),
        "test_evaluated": False,
        "augmentation": "Manhattan rotations and reflections (training only)",
    }


def validate_fields(result: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Artifact configuration mismatch: {json.dumps(mismatches, sort_keys=True)}")


def obtain_training(args: argparse.Namespace, protocol: str, seed: int) -> dict[str, Any]:
    metrics_path = training_metrics_path(args, protocol, seed)
    weights = checkpoint_path(args, protocol, seed)
    if args.force or not metrics_path.exists():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        weights.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(build_training_command(args, protocol, seed), check=True, cwd=PROJECT_ROOT)
    result = load_json(metrics_path)
    validate_fields(result, expected_training(args, protocol, seed))
    if not weights.is_file() or sha256_file(weights) != result["weights_sha256"]:
        raise ValueError(f"Missing or invalid checkpoint: {weights}")
    return result


def obtain_evaluation(args: argparse.Namespace, protocol: str, seed: int) -> dict[str, Any]:
    metrics_path = test_metrics_path(args, protocol, seed)
    if args.force or not metrics_path.exists():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(build_evaluation_command(args, protocol, seed), check=True, cwd=PROJECT_ROOT)
    result = load_json(metrics_path)
    expected = expected_training(args, protocol, seed)
    validate_fields(result, {
        key: value for key, value in expected.items()
        if key in {"dataset_archive_sha256", "manifest_id", "protocol", "seed", "batch_size", "model", "model_source_sha256"}
    })
    if float(result.get("decision_threshold", -1)) != 0.5:
        raise ValueError("B4 frozen evaluation must use the B2 threshold 0.5")
    if sha256_file(checkpoint_path(args, protocol, seed)) != result["checkpoint_sha256"]:
        raise ValueError("B4 evaluation checkpoint hash mismatch")
    return result


def baseline_validation(args: argparse.Namespace) -> dict[str, Any]:
    runs = [load_json(args.b2_results / "runs" / f"{SEARCH_PROTOCOL}_seed_{seed}.json") for seed in args.seeds]
    summary = summarize_validation(runs)
    summary["_runs"] = runs
    return summary


def baseline_test(args: argparse.Namespace, protocol: str) -> dict[str, Any]:
    runs = [load_json(args.b2_results / "runs" / f"{protocol}_seed_{seed}.json") for seed in args.seeds]
    return summarize_test(runs)


def clean_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def artifact_record(
    args: argparse.Namespace, protocol: str, seed: int, run: dict[str, Any]
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "seed": seed,
        "validation_metrics": training_metrics_path(args, protocol, seed).relative_to(args.output_dir).as_posix(),
        "checkpoint": checkpoint_path(args, protocol, seed).relative_to(args.output_dir).as_posix(),
        "checkpoint_sha256": run["weights_sha256"],
    }


def render_report(summary: dict[str, Any]) -> str:
    baseline = summary["validation"]["baseline"]["metrics"]
    candidate = summary["validation"]["candidate"]["metrics"]
    lines = [
        "# B4 Compact Architecture Experiment", "",
        f"Status: **{summary['status'].upper()}**  ",
        f"Frozen tests unlocked: **{summary['test_unlocked']}**", "",
        "## Validation-only selection", "",
        "| Model | Accuracy | Dirty recall | Dirty F1 |",
        "|---|---:|---:|---:|",
        f"| `{BASELINE_MODEL}` | {100*baseline['accuracy']['mean']:.2f}% | {100*baseline['recall']['mean']:.2f}% | {100*baseline['f1']['mean']:.2f}% |",
        f"| `{COMPACT_MODEL}` | {100*candidate['accuracy']['mean']:.2f}% | {100*candidate['recall']['mean']:.2f}% | {100*candidate['f1']['mean']:.2f}% |",
    ]
    if summary["test_unlocked"]:
        lines.extend(["", "## Frozen-test confirmation", "", "| Protocol | Accuracy | Dirty recall | Dirty F1 |", "|---|---:|---:|---:|"])
        for protocol, result in summary["frozen_test"].items():
            metrics = result["candidate"]["metrics"]
            lines.append(
                f"| `{protocol}` | {100*metrics['test_accuracy']['mean']:.2f}% | "
                f"{100*metrics['test_recall']['mean']:.2f}% | {100*metrics['test_f1']['mean']:.2f}% |"
            )
    gate_failures = [name for name, passed in summary["validation"]["gate"]["checks"].items() if not passed]
    if summary["test_unlocked"]:
        gate_failures.extend(
            f"unseen:{name}" for name, passed in summary["frozen_test"][SEARCH_PROTOCOL]["gate"]["checks"].items() if not passed
        )
        gate_failures.extend(
            f"reference:{name}" for name, passed in summary["frozen_test"][REFERENCE_PROTOCOL]["gate"]["checks"].items() if not passed
        )
        gate_failures.extend(
            f"cost:{name}" for name, passed in summary["cost_gate"]["checks"].items() if not passed
        )
    lines.extend(["", "## Decision", ""])
    if gate_failures:
        lines.append("Candidate rejected by: " + ", ".join(f"`{name}`" for name in gate_failures) + ".")
    else:
        lines.append("Candidate passed the pre-registered validation, frozen-test, and inference-cost gates.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(render_report(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "training_datasets" / "combined_training_dataset.zip")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "b1_current_audit" / "manifest.json")
    parser.add_argument("--b2-results", type=Path, default=PROJECT_ROOT / "results" / "b2_baselines")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "b4_architecture")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--trainer", type=Path, default=PROJECT_ROOT / "training" / "train_classifier.py")
    parser.add_argument("--evaluator", type=Path, default=PROJECT_ROOT / "training" / "evaluate_classifier.py")
    parser.add_argument("--benchmark", type=Path, default=PROJECT_ROOT / "scripts" / "benchmark_classifier_architectures.py")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"B4 seeds are frozen to {SEEDS}")
    if args.epochs != 30 or args.batch_size != 32 or args.learning_rate != 0.001:
        raise ValueError("B4 must preserve the B2 epoch, batch-size, and learning-rate recipe")
    manifest = load_json(args.manifest)
    if sha256_file(args.dataset) != manifest["dataset"]["archive_sha256"]:
        raise ValueError("Dataset archive hash does not match the frozen B1 manifest")

    benchmark_command = build_benchmark_command(args)
    search_commands = [build_training_command(args, SEARCH_PROTOCOL, seed) for seed in args.seeds]
    if args.dry_run:
        for command in [benchmark_command, *search_commands]:
            print(shlex.join(command))
        return

    benchmark_path = args.output_dir / "architecture_benchmark.json"
    if args.force or not benchmark_path.exists():
        subprocess.run(benchmark_command, check=True, cwd=PROJECT_ROOT)
    benchmark = load_json(benchmark_path)

    baseline = baseline_validation(args)
    candidate_runs = [obtain_training(args, SEARCH_PROTOCOL, seed) for seed in args.seeds]
    candidate = summarize_validation(candidate_runs)
    candidate["_runs"] = candidate_runs
    gate = validation_gate(candidate, baseline)
    summary: dict[str, Any] = {
        "phase": "B4",
        "experiment": "compact_batchnorm_global_pooling",
        "status": "validation_passed" if gate["passed"] else "validation_rejected",
        "test_unlocked": bool(gate["passed"]),
        "configuration": {
            "candidate": COMPACT_MODEL,
            "baseline": BASELINE_MODEL,
            "manifest_id": manifest["manifest_id"],
            "dataset_archive_sha256": sha256_file(args.dataset),
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "RMSprop",
            "learning_rate": args.learning_rate,
            "augmentation": "Manhattan rotations and reflections (training only)",
            "decision_threshold": 0.5,
            "quality_tolerance": QUALITY_TOLERANCE,
            "max_latency_ratio": MAX_LATENCY_RATIO,
        },
        "validation": {
            "baseline": clean_summary(baseline),
            "candidate": clean_summary(candidate),
            "gate": gate,
        },
        "architecture_benchmark": benchmark,
        "repository_commit": repository_commit(),
        "source_hashes": {
            "model": sha256_file(COMPACT_MODEL_SOURCE),
            "trainer": sha256_file(args.trainer),
            "evaluator": sha256_file(args.evaluator),
            "runner": sha256_file(Path(__file__)),
            "benchmark": sha256_file(args.benchmark),
        },
        "run_artifacts": [
            artifact_record(args, SEARCH_PROTOCOL, seed, run)
            for seed, run in zip(args.seeds, candidate_runs)
        ],
    }
    if not gate["passed"]:
        write_outputs(args, summary)
        return

    unseen_evaluations = [obtain_evaluation(args, SEARCH_PROTOCOL, seed) for seed in args.seeds]
    tile_training = [obtain_training(args, REFERENCE_PROTOCOL, seed) for seed in args.seeds]
    tile_evaluations = [obtain_evaluation(args, REFERENCE_PROTOCOL, seed) for seed in args.seeds]
    frozen: dict[str, Any] = {}
    for protocol, runs in ((SEARCH_PROTOCOL, unseen_evaluations), (REFERENCE_PROTOCOL, tile_evaluations)):
        baseline_result = baseline_test(args, protocol)
        candidate_result = summarize_test(runs)
        protocol_gate = (
            frozen_test_gate(candidate_result, baseline_result)
            if protocol == SEARCH_PROTOCOL
            else reference_test_gate(candidate_result, baseline_result)
        )
        frozen[protocol] = {
            "baseline": baseline_result,
            "candidate": candidate_result,
            "gate": protocol_gate,
        }
    inference_gate = cost_gate(benchmark)
    summary["frozen_test"] = frozen
    summary["cost_gate"] = inference_gate
    summary["run_artifacts"].extend(
        artifact_record(args, REFERENCE_PROTOCOL, seed, run)
        for seed, run in zip(args.seeds, tile_training)
    )
    summary["test_artifacts"] = [
        {
            "protocol": protocol,
            "seed": seed,
            "metrics": test_metrics_path(args, protocol, seed).relative_to(args.output_dir).as_posix(),
        }
        for protocol in (SEARCH_PROTOCOL, REFERENCE_PROTOCOL)
        for seed in args.seeds
    ]
    summary["status"] = (
        "passed"
        if (
            frozen[SEARCH_PROTOCOL]["gate"]["passed"]
            and frozen[REFERENCE_PROTOCOL]["gate"]["passed"]
            and inference_gate["passed"]
        )
        else "rejected"
    )
    write_outputs(args, summary)


if __name__ == "__main__":
    main()
