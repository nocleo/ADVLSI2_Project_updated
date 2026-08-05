"""Run the pre-registered B5 B2+B4 probability-ensemble experiment.

Only validation predictions are created during candidate selection.  Frozen
test predictions are exported after the validation gate passes, so candidate
weights cannot be chosen from test behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_manifest import parse_sample_path, sha256_file


B2_MODEL = "NCSU_DRCNN"
B4_MODEL = "CompactBNPool"
MODELS = (B2_MODEL, B4_MODEL)
SEARCH_PROTOCOL = "unseen_layout_v1"
REFERENCE_PROTOCOL = "tile_random_reference"
SEEDS = (42, 43, 44)
B2_WEIGHTS = (0.25, 0.50, 0.75)
THRESHOLD = 0.5
QUALITY_TOLERANCE = 0.005


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "runs": len(values),
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Labels and predictions must be non-empty and aligned")
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "samples": len(labels),
        "accuracy": (tn + tp) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "predicted_class_counts": {"clean": predictions.count(0), "dirty": predictions.count(1)},
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def checkpoint_path(args: argparse.Namespace, model: str, protocol: str, seed: int) -> Path:
    if model == B2_MODEL:
        return args.b2_checkpoints / f"{protocol}_seed_{seed}.pth"
    return args.b4_checkpoints / f"{protocol}__{B4_MODEL}__seed_{seed}.pth"


def prediction_path(
    args: argparse.Namespace, model: str, protocol: str, split: str, seed: int
) -> Path:
    return args.output_dir / "predictions" / split / f"{protocol}__{model}__seed_{seed}.json"


def build_prediction_command(
    args: argparse.Namespace, model: str, protocol: str, split: str, seed: int
) -> list[str]:
    command = [
        str(args.python),
        str(args.exporter),
        "--dataset", str(args.dataset),
        "--manifest", str(args.manifest),
        "--protocol", protocol,
        "--split", split,
        "--model", model,
        "--checkpoint", str(checkpoint_path(args, model, protocol, seed)),
        "--output", str(prediction_path(args, model, protocol, split, seed)),
        "--seed", str(seed),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    if args.cpu:
        command.append("--cpu")
    return command


def expected_checkpoint_hashes(args: argparse.Namespace) -> dict[tuple[str, str, int], str]:
    hashes: dict[tuple[str, str, int], str] = {}
    for protocol in (SEARCH_PROTOCOL, REFERENCE_PROTOCOL):
        for seed in args.seeds:
            b2 = load_json(args.b2_results / "runs" / f"{protocol}_seed_{seed}.json")
            hashes[(B2_MODEL, protocol, seed)] = str(b2["weights_sha256"])

    b4 = load_json(args.b4_results / "summary.json")
    if b4.get("status") != "rejected" or not b4.get("test_unlocked"):
        raise ValueError("B5 requires the completed, rejected B4 experiment")
    for artifact in b4["run_artifacts"]:
        hashes[(B4_MODEL, str(artifact["protocol"]), int(artifact["seed"]))] = str(
            artifact["checkpoint_sha256"]
        )
    expected = {
        (model, protocol, seed)
        for model in MODELS
        for protocol in (SEARCH_PROTOCOL, REFERENCE_PROTOCOL)
        for seed in args.seeds
    }
    if set(hashes) != expected:
        raise ValueError("Authoritative B2/B4 summaries do not cover every B5 checkpoint")
    return hashes


def validate_prediction_artifact(
    result: dict[str, Any],
    *,
    args: argparse.Namespace,
    model: str,
    protocol: str,
    split: str,
    seed: int,
    expected_paths: list[str],
    expected_checkpoint_sha256: str,
) -> None:
    expected = {
        "phase": "B5",
        "dataset_archive_sha256": sha256_file(args.dataset),
        "manifest_id": load_json(args.manifest)["manifest_id"],
        "protocol": protocol,
        "split": split,
        "seed": seed,
        "model": model,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "samples": len(expected_paths),
        "batch_size": args.batch_size,
    }
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Prediction artifact mismatch: {json.dumps(mismatches, sort_keys=True)}")
    records = result.get("records", [])
    actual_paths = [record.get("path") for record in records]
    if actual_paths != expected_paths:
        raise ValueError("Prediction records do not exactly match the frozen manifest order")
    for record in records:
        probability = float(record["dirty_probability"])
        if int(record["label"]) not in (0, 1) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Prediction artifact contains an invalid label or probability")


def obtain_predictions(
    args: argparse.Namespace,
    model: str,
    protocol: str,
    split: str,
    seed: int,
    expected_paths: list[str],
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    output = prediction_path(args, model, protocol, split, seed)
    checkpoint = checkpoint_path(args, model, protocol, seed)
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise ValueError(f"Missing or hash-invalid authoritative checkpoint: {checkpoint}")
    if args.force or not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            build_prediction_command(args, model, protocol, split, seed),
            check=True,
            cwd=PROJECT_ROOT,
        )
    result = load_json(output)
    validate_prediction_artifact(
        result,
        args=args,
        model=model,
        protocol=protocol,
        split=split,
        seed=seed,
        expected_paths=expected_paths,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    return result


def aligned_records(
    b2: dict[str, Any], b4: dict[str, Any]
) -> list[tuple[str, int, float, float]]:
    if len(b2["records"]) != len(b4["records"]):
        raise ValueError("B2/B4 prediction counts differ")
    aligned: list[tuple[str, int, float, float]] = []
    for left, right in zip(b2["records"], b4["records"]):
        if left["path"] != right["path"] or int(left["label"]) != int(right["label"]):
            raise ValueError("B2/B4 predictions are not sample-aligned")
        aligned.append(
            (
                str(left["path"]),
                int(left["label"]),
                float(left["dirty_probability"]),
                float(right["dirty_probability"]),
            )
        )
    return aligned


def score_records(
    records: list[tuple[str, int, float, float]], b2_weight: float
) -> dict[str, Any]:
    labels = [label for _, label, _, _ in records]
    probabilities = [
        b2_weight * b2_probability + (1.0 - b2_weight) * b4_probability
        for _, _, b2_probability, b4_probability in records
    ]
    predictions = [int(probability >= THRESHOLD) for probability in probabilities]
    metrics = classification_metrics(labels, predictions)

    by_layout: dict[str, tuple[list[int], list[int]]] = {}
    for (path, label, _, _), prediction in zip(records, predictions):
        _, layout, _, _ = parse_sample_path(path)
        labels_out, predictions_out = by_layout.setdefault(layout, ([], []))
        labels_out.append(label)
        predictions_out.append(prediction)
    metrics["per_layout"] = {
        layout: classification_metrics(layout_labels, layout_predictions)
        for layout, (layout_labels, layout_predictions) in sorted(by_layout.items())
    }
    return metrics


def single_model_score(
    records: list[tuple[str, int, float, float]], model: str
) -> dict[str, Any]:
    weight = 1.0 if model == B2_MODEL else 0.0
    return score_records(records, weight)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("accuracy", "precision", "recall", "f1")
    layouts = sorted({layout for run in runs for layout in run["per_layout"]})
    return {
        "seeds": [int(run["seed"]) for run in runs],
        "metrics": {
            name: metric_stats([float(run[name]) for run in runs]) for name in names
        },
        "per_seed": [
            {
                "seed": int(run["seed"]),
                **{name: float(run[name]) for name in names},
                "predicted_class_counts": run["predicted_class_counts"],
                "confusion_matrix": run["confusion_matrix"],
            }
            for run in runs
        ],
        "pooled_confusion_matrix": [
            [sum(int(run["confusion_matrix"][row][column]) for run in runs) for column in range(2)]
            for row in range(2)
        ],
        "per_layout": {
            layout: {
                name: metric_stats(
                    [float(run["per_layout"][layout][name]) for run in runs if layout in run["per_layout"]]
                )
                for name in names
            }
            for layout in layouts
        },
    }


def score_model_runs(
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]], model: str
) -> dict[str, Any]:
    runs = []
    for seed, b2, b4 in pairs:
        run = single_model_score(aligned_records(b2, b4), model)
        run["seed"] = seed
        runs.append(run)
    return summarize_runs(runs)


def score_ensemble_runs(
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]], b2_weight: float
) -> dict[str, Any]:
    runs = []
    for seed, b2, b4 in pairs:
        run = score_records(aligned_records(b2, b4), b2_weight)
        run["seed"] = seed
        runs.append(run)
    summary = summarize_runs(runs)
    summary["b2_weight"] = b2_weight
    summary["b4_weight"] = 1.0 - b2_weight
    summary["decision_threshold"] = THRESHOLD
    return summary


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if [float(candidate["b2_weight"]) for candidate in candidates] != list(B2_WEIGHTS):
        raise ValueError(f"B5 candidates are frozen to {B2_WEIGHTS}")
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["metrics"]["f1"]["mean"]),
            float(candidate["metrics"]["accuracy"]["mean"]),
            -abs(float(candidate["b2_weight"]) - 0.5),
            float(candidate["b2_weight"]),
        ),
    )


def paired_wins(candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> int:
    baseline_by_seed = {
        int(run["seed"]): float(run[metric]) for run in baseline["per_seed"]
    }
    return sum(
        float(run[metric]) > baseline_by_seed[int(run["seed"])]
        for run in candidate["per_seed"]
    )


def validation_gate(
    candidate: dict[str, Any], b2: dict[str, Any], b4: dict[str, Any]
) -> dict[str, Any]:
    stronger = b4 if b4["metrics"]["f1"]["mean"] >= b2["metrics"]["f1"]["mean"] else b2
    f1_wins = paired_wins(candidate, stronger, "f1")
    checks = {
        "mean_f1_beats_both_single_models": candidate["metrics"]["f1"]["mean"] > max(
            b2["metrics"]["f1"]["mean"], b4["metrics"]["f1"]["mean"]
        ),
        "paired_f1_wins_vs_stronger_model_at_least_two": f1_wins >= 2,
        "accuracy_preserved_vs_stronger_model": candidate["metrics"]["accuracy"]["mean"] >= stronger["metrics"]["accuracy"]["mean"] - QUALITY_TOLERANCE,
        "recall_preserved_vs_stronger_model": candidate["metrics"]["recall"]["mean"] >= stronger["metrics"]["recall"]["mean"] - QUALITY_TOLERANCE,
        "no_collapsed_seed": all(
            all(int(count) > 0 for count in run["predicted_class_counts"].values())
            for run in candidate["per_seed"]
        ),
    }
    return {"passed": all(checks.values()), "paired_f1_wins": f1_wins, "checks": checks}


def frozen_unseen_gate(candidate: dict[str, Any], b2: dict[str, Any]) -> dict[str, Any]:
    accuracy_wins = paired_wins(candidate, b2, "accuracy")
    f1_wins = paired_wins(candidate, b2, "f1")
    checks = {
        "mean_accuracy_improved": candidate["metrics"]["accuracy"]["mean"] > b2["metrics"]["accuracy"]["mean"],
        "mean_f1_improved": candidate["metrics"]["f1"]["mean"] > b2["metrics"]["f1"]["mean"],
        "paired_accuracy_wins_at_least_two": accuracy_wins >= 2,
        "paired_f1_wins_at_least_two": f1_wins >= 2,
        "recall_preserved": candidate["metrics"]["recall"]["mean"] >= b2["metrics"]["recall"]["mean"] - QUALITY_TOLERANCE,
    }
    return {
        "passed": all(checks.values()),
        "paired_accuracy_wins": accuracy_wins,
        "paired_f1_wins": f1_wins,
        "checks": checks,
    }


def reference_gate(candidate: dict[str, Any], b2: dict[str, Any]) -> dict[str, Any]:
    checks = {
        f"{name}_preserved": candidate["metrics"][name]["mean"] >= b2["metrics"][name]["mean"] - QUALITY_TOLERANCE
        for name in ("accuracy", "recall", "f1")
    }
    return {"passed": all(checks.values()), "checks": checks}


def disagreement_summary(
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    totals = {"both_correct": 0, "b2_only_correct": 0, "b4_only_correct": 0, "both_wrong": 0}
    by_label = {"clean": {key: 0 for key in totals}, "dirty": {key: 0 for key in totals}}
    for _, b2, b4 in pairs:
        for _, label, b2_probability, b4_probability in aligned_records(b2, b4):
            b2_correct = int(b2_probability >= THRESHOLD) == label
            b4_correct = int(b4_probability >= THRESHOLD) == label
            key = (
                "both_correct" if b2_correct and b4_correct else
                "b2_only_correct" if b2_correct else
                "b4_only_correct" if b4_correct else
                "both_wrong"
            )
            totals[key] += 1
            by_label["dirty" if label else "clean"][key] += 1
    return {"pooled_across_seeds": totals, "by_true_label": by_label}


def clean_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(candidate))


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# B5 Validation-Selected B2+B4 Ensemble", "",
        f"Status: **{summary['status']}**  ",
        f"Frozen tests unlocked: **{summary['test_unlocked']}**", "",
        "## Validation-only search", "",
        "| Candidate | Accuracy | Dirty recall | Dirty F1 |",
        "|---|---:|---:|---:|",
    ]
    validation = summary["validation"]
    for name, result in ((B2_MODEL, validation["single_models"][B2_MODEL]), (B4_MODEL, validation["single_models"][B4_MODEL])):
        metrics = result["metrics"]
        lines.append(f"| `{name}` | {100*metrics['accuracy']['mean']:.2f}% | {100*metrics['recall']['mean']:.2f}% | {100*metrics['f1']['mean']:.2f}% |")
    for candidate in validation["candidates"]:
        metrics = candidate["metrics"]
        marker = " (selected)" if candidate["b2_weight"] == validation["selected"]["b2_weight"] else ""
        lines.append(f"| B2 {candidate['b2_weight']:.2f} / B4 {candidate['b4_weight']:.2f}{marker} | {100*metrics['accuracy']['mean']:.2f}% | {100*metrics['recall']['mean']:.2f}% | {100*metrics['f1']['mean']:.2f}% |")
    lines.extend(["", "## Decision", ""])
    if not summary["test_unlocked"]:
        failed = [name for name, passed in validation["gate"]["checks"].items() if not passed]
        lines.append("Validation gate rejected the ensemble: " + ", ".join(f"`{name}`" for name in failed) + ".")
    else:
        for protocol, result in summary["frozen_test"].items():
            metrics = result["ensemble"]["metrics"]
            lines.append(f"- `{protocol}` ensemble: accuracy {100*metrics['accuracy']['mean']:.2f}%, recall {100*metrics['recall']['mean']:.2f}%, F1 {100*metrics['f1']['mean']:.2f}%. Gate passed: **{result['gate']['passed']}**.")
        lines.append("")
        lines.append("The ensemble is accepted only if both frozen protocols pass; otherwise B2 remains the classifier baseline.")
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
    parser.add_argument("--b4-results", type=Path, default=PROJECT_ROOT / "results" / "b4_architecture")
    parser.add_argument("--b2-checkpoints", type=Path, required=True)
    parser.add_argument("--b4-checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "b5_ensemble")
    parser.add_argument("--exporter", type=Path, default=PROJECT_ROOT / "training" / "export_classifier_predictions.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--b2-weights", nargs="+", type=float, default=list(B2_WEIGHTS))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"B5 seeds are frozen to {SEEDS}")
    if tuple(args.b2_weights) != B2_WEIGHTS:
        raise ValueError(f"B5 B2 weights are frozen to {B2_WEIGHTS}")
    if args.threshold != THRESHOLD or args.batch_size != 32:
        raise ValueError("B5 threshold and batch size are frozen to B2/B4")

    manifest = load_json(args.manifest)
    if sha256_file(args.dataset) != manifest["dataset"]["archive_sha256"]:
        raise ValueError("Dataset archive hash does not match the frozen B1 manifest")

    validation_commands = [
        build_prediction_command(args, model, SEARCH_PROTOCOL, "validation", seed)
        for model in MODELS
        for seed in args.seeds
    ]
    if args.dry_run:
        for command in validation_commands:
            print(shlex.join(command))
        return

    checkpoint_hashes = expected_checkpoint_hashes(args)
    validation_paths = manifest["protocols"][SEARCH_PROTOCOL]["splits"]["validation"]
    validation_pairs = []
    for seed in args.seeds:
        predictions = {
            model: obtain_predictions(
                args, model, SEARCH_PROTOCOL, "validation", seed,
                validation_paths, checkpoint_hashes[(model, SEARCH_PROTOCOL, seed)]
            )
            for model in MODELS
        }
        validation_pairs.append((seed, predictions[B2_MODEL], predictions[B4_MODEL]))

    b2_validation = score_model_runs(validation_pairs, B2_MODEL)
    b4_validation = score_model_runs(validation_pairs, B4_MODEL)
    candidates = [score_ensemble_runs(validation_pairs, weight) for weight in args.b2_weights]
    selected = select_candidate(candidates)
    gate = validation_gate(selected, b2_validation, b4_validation)
    summary: dict[str, Any] = {
        "phase": "B5",
        "experiment": "validation_selected_b2_b4_probability_blend",
        "status": "validation_passed" if gate["passed"] else "validation_rejected",
        "test_unlocked": bool(gate["passed"]),
        "configuration": {
            "models": list(MODELS),
            "manifest_id": manifest["manifest_id"],
            "dataset_archive_sha256": sha256_file(args.dataset),
            "seeds": list(args.seeds),
            "candidate_b2_weights": list(args.b2_weights),
            "candidate_b4_weights": [1.0 - weight for weight in args.b2_weights],
            "decision_threshold": args.threshold,
            "selection_metric": "mean unseen-layout validation dirty F1",
            "quality_tolerance": QUALITY_TOLERANCE,
        },
        "validation": {
            "single_models": {B2_MODEL: b2_validation, B4_MODEL: b4_validation},
            "candidates": candidates,
            "selected": clean_candidate(selected),
            "gate": gate,
            "disagreement": disagreement_summary(validation_pairs),
        },
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "exporter": sha256_file(args.exporter),
        },
    }
    if not gate["passed"]:
        write_outputs(args, summary)
        return

    frozen: dict[str, Any] = {}
    for protocol in (SEARCH_PROTOCOL, REFERENCE_PROTOCOL):
        test_paths = manifest["protocols"][protocol]["splits"]["test"]
        pairs = []
        for seed in args.seeds:
            predictions = {
                model: obtain_predictions(
                    args, model, protocol, "test", seed,
                    test_paths, checkpoint_hashes[(model, protocol, seed)]
                )
                for model in MODELS
            }
            pairs.append((seed, predictions[B2_MODEL], predictions[B4_MODEL]))
        b2_test = score_model_runs(pairs, B2_MODEL)
        b4_test = score_model_runs(pairs, B4_MODEL)
        ensemble = score_ensemble_runs(pairs, float(selected["b2_weight"]))
        protocol_gate = frozen_unseen_gate(ensemble, b2_test) if protocol == SEARCH_PROTOCOL else reference_gate(ensemble, b2_test)
        frozen[protocol] = {
            "single_models": {B2_MODEL: b2_test, B4_MODEL: b4_test},
            "ensemble": ensemble,
            "gate": protocol_gate,
            "disagreement": disagreement_summary(pairs),
        }
    summary["frozen_test"] = frozen
    summary["status"] = "accepted" if all(result["gate"]["passed"] for result in frozen.values()) else "rejected"
    write_outputs(args, summary)


if __name__ == "__main__":
    main()
