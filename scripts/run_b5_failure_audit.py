"""Run the B5.2 path-aligned classifier failure-slice audit.

The audit is deliberately restricted to training and validation predictions.
It joins authoritative B2/B4 probabilities with the frozen B1 manifest, reports
calibration and error slices, and checks whether an observed mechanism repeats
across families and seeds. Exact DRC geometry is accepted only through an
optional annotation JSONL; absent fields are reported as unavailable rather
than inferred from raster pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
B2_MODEL = "NCSU_DRCNN"
B4_MODEL = "CompactBNPool"
MODELS = (B2_MODEL, B4_MODEL)
DEFAULT_PROTOCOLS = ("unseen_layout_v1", "tile_random_reference")
DEFAULT_SPLITS = ("train", "validation")
DEFAULT_SEEDS = (42, 43, 44)
THRESHOLD = 0.5
MIN_SLICE_SUPPORT = 50
MIN_FAMILY_SUPPORT = 20
MEANINGFUL_ERROR_GAP = 0.05
EPSILON = 1e-12

DENSITY_BINS = (
    ("0.03-0.15", 0.03, 0.15),
    ("0.15-0.30", 0.15, 0.30),
    ("0.30-0.45", 0.30, 0.45),
    ("0.45-0.60", 0.45, 0.60),
    ("0.60-0.85", 0.60, 0.8500001),
)

GEOMETRY_FIELDS = (
    "violation_count",
    "edge_orientation",
    "edge_length_nm",
    "spacing_deficit_nm",
    "nearby_shape_count",
    "distance_to_tile_boundary_nm",
    "distance_to_supervised_boundary_nm",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_layout(path: str) -> str:
    parts = PurePosixPath(path).parts
    if len(parts) != 2 or parts[0] not in {"clean", "dirty"}:
        raise ValueError(f"Unexpected sample path: {path!r}")
    filename = parts[1]
    if not filename.endswith(".npy") or "_tile_" not in filename:
        raise ValueError(f"Unexpected sample filename: {filename!r}")
    return filename[:-4].rsplit("_tile_", 1)[0]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def density_bin(value: float) -> str:
    for name, lower, upper in DENSITY_BINS:
        if lower <= value < upper:
            return name
    return "outside-audited-range"


def prediction_path(
    root: Path, protocol: str, split: str, model: str, seed: int
) -> Path:
    return root / split / f"{protocol}__{model}__seed_{seed}.json"


def checkpoint_path(root: Path, model: str, protocol: str, seed: int) -> Path:
    if model == B2_MODEL:
        return root / f"{protocol}_seed_{seed}.pth"
    return root / f"{protocol}__{B4_MODEL}__seed_{seed}.pth"


def expected_checkpoint_hashes(
    b2_results: Path, b4_results: Path, protocols: Iterable[str], seeds: Iterable[int]
) -> dict[tuple[str, str, int], str]:
    hashes: dict[tuple[str, str, int], str] = {}
    for protocol in protocols:
        for seed in seeds:
            artifact = load_json(b2_results / "runs" / f"{protocol}_seed_{seed}.json")
            hashes[(B2_MODEL, protocol, seed)] = str(artifact["weights_sha256"])
    b4_summary = load_json(b4_results / "summary.json")
    for artifact in b4_summary["run_artifacts"]:
        key = (B4_MODEL, str(artifact["protocol"]), int(artifact["seed"]))
        hashes[key] = str(artifact["checkpoint_sha256"])
    expected = {
        (model, protocol, seed)
        for model in MODELS
        for protocol in protocols
        for seed in seeds
    }
    missing = sorted(expected - set(hashes))
    if missing:
        raise ValueError(f"Authoritative checkpoint hashes are missing: {missing}")
    return hashes


def export_missing_predictions(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    hashes: dict[tuple[str, str, int], str],
) -> None:
    if args.dataset is None or args.b2_checkpoints is None or args.b4_checkpoints is None:
        raise ValueError(
            "--export-missing requires --dataset, --b2-checkpoints, and --b4-checkpoints"
        )
    for protocol in args.protocols:
        for split in args.splits:
            expected_paths = manifest["protocols"][protocol]["splits"][split]
            for seed in args.seeds:
                for model in MODELS:
                    output = prediction_path(args.predictions, protocol, split, model, seed)
                    if output.is_file() and not args.force:
                        continue
                    checkpoint_root = (
                        args.b2_checkpoints if model == B2_MODEL else args.b4_checkpoints
                    )
                    checkpoint = checkpoint_path(checkpoint_root, model, protocol, seed)
                    expected_hash = hashes[(model, protocol, seed)]
                    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_hash:
                        raise ValueError(
                            f"Missing or hash-invalid authoritative checkpoint: {checkpoint}"
                        )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        str(args.python),
                        str(args.exporter),
                        "--dataset", str(args.dataset),
                        "--manifest", str(args.manifest),
                        "--protocol", protocol,
                        "--split", split,
                        "--model", model,
                        "--checkpoint", str(checkpoint),
                        "--output", str(output),
                        "--seed", str(seed),
                        "--batch-size", str(args.batch_size),
                        "--device", args.device,
                    ]
                    if args.cpu:
                        command.append("--cpu")
                    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
                    exported = load_json(output)
                    if [item["path"] for item in exported["records"]] != expected_paths:
                        raise ValueError(f"Exporter lost manifest order for {output}")


def load_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_path = str(record.get("path", ""))
            if not sample_path or sample_path in annotations:
                raise ValueError(
                    f"Invalid or duplicate annotation path at line {line_number}: {sample_path!r}"
                )
            annotations[sample_path] = record
    return annotations


def validate_prediction_artifact(
    artifact: dict[str, Any],
    *,
    manifest: dict[str, Any],
    protocol: str,
    split: str,
    model: str,
    seed: int,
    expected_paths: list[str],
    dataset_sha256: str | None,
    checkpoint_sha256: str | None,
) -> None:
    expected: dict[str, Any] = {
        "manifest_id": manifest["manifest_id"],
        "protocol": protocol,
        "split": split,
        "model": model,
        "seed": seed,
        "samples": len(expected_paths),
    }
    if dataset_sha256 is not None:
        expected["dataset_archive_sha256"] = dataset_sha256
    if checkpoint_sha256 is not None:
        expected["checkpoint_sha256"] = checkpoint_sha256
    mismatches = {
        key: {"expected": value, "actual": artifact.get(key)}
        for key, value in expected.items()
        if artifact.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Prediction artifact metadata mismatch: {json.dumps(mismatches, sort_keys=True)}"
        )
    records = artifact.get("records", [])
    if [record.get("path") for record in records] != expected_paths:
        raise ValueError("Prediction artifact does not match manifest path order")
    for record in records:
        probability = float(record["dirty_probability"])
        if int(record["label"]) not in (0, 1) or not 0.0 <= probability <= 1.0:
            raise ValueError("Prediction artifact contains an invalid label/probability")


def join_records(
    b2: dict[str, Any],
    b4: dict[str, Any],
    manifest_records: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    protocol: str,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for left, right in zip(b2["records"], b4["records"]):
        if left["path"] != right["path"] or int(left["label"]) != int(right["label"]):
            raise ValueError("B2/B4 prediction records are not path-aligned")
        path = str(left["path"])
        if path not in manifest_records:
            raise ValueError(f"Prediction path is absent from manifest: {path}")
        metadata = manifest_records[path]
        label = int(left["label"])
        b2_probability = float(left["dirty_probability"])
        b4_probability = float(right["dirty_probability"])
        b2_prediction = int(b2_probability >= THRESHOLD)
        b4_prediction = int(b4_probability >= THRESHOLD)
        record: dict[str, Any] = {
            "protocol": protocol,
            "split": split,
            "seed": seed,
            "path": path,
            "label": label,
            "source_layout": metadata.get("source_layout", parse_layout(path)),
            "layout_family": metadata.get("layout_family", parse_layout(path)),
            "metal_density": float(metadata["metal_density"]),
            "density_bin": density_bin(float(metadata["metal_density"])),
            "b2_probability": b2_probability,
            "b4_probability": b4_probability,
            "b2_prediction": b2_prediction,
            "b4_prediction": b4_prediction,
            "b2_correct": b2_prediction == label,
            "b4_correct": b4_prediction == label,
            "model_disagreement": b2_prediction != b4_prediction,
        }
        if b2_prediction == label and b4_prediction == label:
            record["paired_outcome"] = "both_correct"
        elif b2_prediction == label:
            record["paired_outcome"] = "b2_only_correct"
        elif b4_prediction == label:
            record["paired_outcome"] = "b4_only_correct"
        else:
            record["paired_outcome"] = "both_wrong"
        annotation = annotations.get(path, {})
        for field in GEOMETRY_FIELDS:
            if field in annotation:
                record[field] = annotation[field]
        joined.append(record)
    return joined


def calibration_bins(labels: list[int], probabilities: list[float]) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        selected = [
            (label, probability)
            for label, probability in zip(labels, probabilities)
            if lower <= probability < upper or (index == 9 and probability == 1.0)
        ]
        if not selected:
            continue
        bins.append(
            {
                "range": [lower, upper],
                "samples": len(selected),
                "mean_probability": statistics.mean(value for _, value in selected),
                "dirty_fraction": statistics.mean(label for label, _ in selected),
            }
        )
    return bins


def model_metrics(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    prefix = "b2" if model == B2_MODEL else "b4"
    labels = [int(record["label"]) for record in records]
    probabilities = [float(record[f"{prefix}_probability"]) for record in records]
    predictions = [int(record[f"{prefix}_prediction"]) for record in records]
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tn + tp) / len(labels) if labels else 0.0
    bins = calibration_bins(labels, probabilities)
    ece = sum(
        item["samples"] / len(labels)
        * abs(item["mean_probability"] - item["dirty_fraction"])
        for item in bins
    ) if labels else 0.0
    clipped = [min(1.0 - EPSILON, max(EPSILON, value)) for value in probabilities]
    return {
        "samples": len(labels),
        "accuracy": accuracy,
        "accuracy_ci95": wilson_interval(tn + tp, len(labels)),
        "precision": precision,
        "recall": recall,
        "recall_ci95": wilson_interval(tp, tp + fn),
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "specificity_ci95": wilson_interval(tn, tn + fp),
        "error_rate": (fp + fn) / len(labels) if labels else 0.0,
        "error_rate_ci95": wilson_interval(fp + fn, len(labels)),
        "brier_score": statistics.mean(
            (probability - label) ** 2 for label, probability in zip(labels, probabilities)
        ) if labels else 0.0,
        "negative_log_likelihood": -statistics.mean(
            label * math.log(probability) + (1 - label) * math.log(1 - probability)
            for label, probability in zip(labels, clipped)
        ) if labels else 0.0,
        "expected_calibration_error_10bin": ece,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "calibration_bins": bins,
    }


def grouped_metrics(
    records: list[dict[str, Any]], key: str, model: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if key in record:
            groups[str(record[key])].append(record)
    return {
        value: {
            **model_metrics(selected, model),
            "families": sorted({str(record["layout_family"]) for record in selected}),
        }
        for value, selected in sorted(groups.items())
    }


def disagreement_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, Counter[str]] = {
        "clean": Counter(),
        "dirty": Counter(),
    }
    by_family: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"clean": Counter(), "dirty": Counter()}
    )
    for record in records:
        label = "dirty" if record["label"] else "clean"
        outcome = str(record["paired_outcome"])
        by_label[label][outcome] += 1
        by_family[str(record["layout_family"])][label][outcome] += 1
    return {
        "overall": dict(Counter(str(record["paired_outcome"]) for record in records)),
        "by_label": {label: dict(counts) for label, counts in by_label.items()},
        "by_family": {
            family: {label: dict(counts) for label, counts in labels.items()}
            for family, labels in sorted(by_family.items())
        },
    }


def density_candidates(run_records: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    prefix = "b2" if model == B2_MODEL else "b4"
    candidates: list[dict[str, Any]] = []
    for label in (0, 1):
        label_records = [record for record in run_records if record["label"] == label]
        for bin_name, _, _ in DENSITY_BINS:
            selected = [record for record in label_records if record["density_bin"] == bin_name]
            complement = [record for record in label_records if record["density_bin"] != bin_name]
            if len(selected) < MIN_SLICE_SUPPORT or len(complement) < MIN_SLICE_SUPPORT:
                continue
            selected_errors = sum(not record[f"{prefix}_correct"] for record in selected)
            complement_errors = sum(not record[f"{prefix}_correct"] for record in complement)
            selected_rate = selected_errors / len(selected)
            complement_rate = complement_errors / len(complement)
            selected_ci = wilson_interval(selected_errors, len(selected))
            complement_ci = wilson_interval(complement_errors, len(complement))
            if selected_rate - complement_rate < MEANINGFUL_ERROR_GAP:
                continue
            repeating_families: list[str] = []
            for family in sorted({str(record["layout_family"]) for record in label_records}):
                family_slice = [
                    record for record in selected if record["layout_family"] == family
                ]
                family_rest = [
                    record for record in complement if record["layout_family"] == family
                ]
                if min(len(family_slice), len(family_rest)) < MIN_FAMILY_SUPPORT:
                    continue
                slice_rate = sum(not record[f"{prefix}_correct"] for record in family_slice) / len(family_slice)
                rest_rate = sum(not record[f"{prefix}_correct"] for record in family_rest) / len(family_rest)
                if slice_rate - rest_rate >= MEANINGFUL_ERROR_GAP:
                    repeating_families.append(family)
            candidates.append(
                {
                    "model": model,
                    "label": "dirty" if label else "clean",
                    "density_bin": bin_name,
                    "support": len(selected),
                    "error_rate": selected_rate,
                    "error_rate_ci95": selected_ci,
                    "complement_error_rate": complement_rate,
                    "complement_error_rate_ci95": complement_ci,
                    "error_rate_gap": selected_rate - complement_rate,
                    "intervals_separated": selected_ci[0] > complement_ci[1],
                    "repeating_families": repeating_families,
                    "family_gate_passed": (
                        len(repeating_families) >= 2
                        and selected_ci[0] > complement_ci[1]
                    ),
                }
            )
    return candidates


def disagreement_gate(validation_runs: list[dict[str, Any]]) -> dict[str, Any]:
    evidence: dict[str, dict[str, set[int]]] = {
        "dirty_b2_advantage": defaultdict(set),
        "clean_b4_advantage": defaultdict(set),
    }
    for run in validation_runs:
        seed = int(run["seed"])
        for family, labels in run["disagreement"]["by_family"].items():
            dirty = labels["dirty"]
            clean = labels["clean"]
            if dirty.get("b2_only_correct", 0) > dirty.get("b4_only_correct", 0):
                evidence["dirty_b2_advantage"][family].add(seed)
            if clean.get("b4_only_correct", 0) > clean.get("b2_only_correct", 0):
                evidence["clean_b4_advantage"][family].add(seed)
    dirty_families = sorted(
        family for family, seeds in evidence["dirty_b2_advantage"].items() if len(seeds) >= 2
    )
    clean_families = sorted(
        family for family, seeds in evidence["clean_b4_advantage"].items() if len(seeds) >= 2
    )
    return {
        "criterion": (
            "B2 uniquely corrects more dirty samples and B4 uniquely corrects more clean "
            "samples in at least two families, each repeated in at least two seeds."
        ),
        "dirty_b2_advantage_families": dirty_families,
        "clean_b4_advantage_families": clean_families,
        "passed": len(dirty_families) >= 2 and len(clean_families) >= 2,
    }


def feature_availability(
    records: list[dict[str, Any]], annotations_supplied: bool
) -> dict[str, Any]:
    availability: dict[str, Any] = {
        "layout_family": {"available_records": len(records), "source": "B1 manifest"},
        "metal_density": {"available_records": len(records), "source": "B1 manifest"},
    }
    for field in GEOMETRY_FIELDS:
        available = sum(field in record for record in records)
        availability[field] = {
            "available_records": available,
            "source": "exact DRC geometry annotation JSONL" if available else None,
            "status": "available" if available == len(records) else "partial" if available else "unavailable",
        }
    availability["geometry_annotations_supplied"] = annotations_supplied
    return availability


def build_summary(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    all_records: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
    annotations_supplied: bool,
) -> dict[str, Any]:
    validation_runs = [run for run in run_summaries if run["split"] == "validation"]
    run_index = {
        (run["protocol"], run["split"], int(run["seed"])): run
        for run in run_summaries
    }
    generalization_gaps: list[dict[str, Any]] = []
    if "train" in args.splits and "validation" in args.splits:
        for protocol in args.protocols:
            for seed in args.seeds:
                train = run_index[(protocol, "train", seed)]
                validation = run_index[(protocol, "validation", seed)]
                generalization_gaps.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        "models": {
                            model: {
                                metric: (
                                    train["models"][model][metric]
                                    - validation["models"][model][metric]
                                )
                                for metric in ("accuracy", "recall", "f1")
                            }
                            for model in MODELS
                        },
                    }
                )
    density_evidence = [
        {
            "protocol": run["protocol"],
            "split": run["split"],
            "seed": run["seed"],
            "candidate": candidate,
        }
        for run in validation_runs
        for candidate in run["density_candidates"]
        if candidate["family_gate_passed"]
    ]
    disagreement_by_protocol = {
        protocol: disagreement_gate(
            [run for run in validation_runs if run["protocol"] == protocol]
        )
        for protocol in args.protocols
    }
    disagreement_both = all(
        item["passed"] for item in disagreement_by_protocol.values()
    )
    geometry_complete = all(
        all(field in record for field in GEOMETRY_FIELDS) for record in all_records
    ) if all_records else False
    recommendations: list[str] = []
    if disagreement_both:
        recommendations.append(
            "Pre-register one recall-constrained disagreement gate for B5.3; the class-conditional complementarity repeats across both validation protocols."
        )
    if density_evidence:
        recommendations.append(
            "Review the replicated density slices before choosing B5.3; they meet the pre-registered support, family, and effect-size gates."
        )
    if not geometry_complete:
        recommendations.append(
            "Do not claim boundary, orientation, or severity mechanisms from B5.2. Preserve B2 unless validation-only measured evidence selects B5.3, and generate exact edge-pair annotations in B6.1."
        )
    if not disagreement_both and not density_evidence:
        recommendations.append(
            "No measured B5.2 slice passes the evidence gate; close classifier-only tuning, retain B2, and proceed to B6.1."
        )
    return {
        "phase": "B5.2",
        "status": "complete_with_geometry" if geometry_complete else "complete_measured_features_geometry_unavailable",
        "manifest_id": manifest["manifest_id"],
        "protocols": list(args.protocols),
        "splits": list(args.splits),
        "seeds": list(args.seeds),
        "decision_threshold": THRESHOLD,
        "evidence_gate": {
            "minimum_slice_support": MIN_SLICE_SUPPORT,
            "minimum_per_family_support": MIN_FAMILY_SUPPORT,
            "meaningful_error_rate_gap": MEANINGFUL_ERROR_GAP,
            "minimum_repeating_families": 2,
            "minimum_repeating_seeds": 2,
        },
        "feature_availability": feature_availability(all_records, annotations_supplied),
        "runs": run_summaries,
        "train_minus_validation_gaps": generalization_gaps,
        "eligible_density_slices": density_evidence,
        "disagreement_gate_by_protocol": disagreement_by_protocol,
        "disagreement_gate_passed_on_both_validation_protocols": disagreement_both,
        "recommendations": recommendations,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# B5.2 Classifier Failure-Slice Audit",
        "",
        f"Status: **{summary['status']}**",
        "",
        "This report uses training/validation predictions only. Existing test splits and the new final holdout are not read.",
        "",
        "## Validation overview",
        "",
        "| Protocol | Seed | Model | Accuracy | Dirty recall | Dirty F1 | Brier | ECE |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        if run["split"] != "validation":
            continue
        for model in MODELS:
            metrics = run["models"][model]
            lines.append(
                f"| `{run['protocol']}` | {run['seed']} | `{model}` | "
                f"{metrics['accuracy']:.2%} | {metrics['recall']:.2%} | "
                f"{metrics['f1']:.2%} | {metrics['brier_score']:.4f} | "
                f"{metrics['expected_calibration_error_10bin']:.4f} |"
            )
    lines.extend(["", "## Evidence gates", ""])
    for protocol, result in summary["disagreement_gate_by_protocol"].items():
        lines.append(
            f"- `{protocol}` class-conditional disagreement gate: **{'pass' if result['passed'] else 'fail'}**; "
            f"B2-dirty families={len(result['dirty_b2_advantage_families'])}, "
            f"B4-clean families={len(result['clean_b4_advantage_families'])}."
        )
    lines.append(
        f"- Replicated density slices passing support/family/effect gates: **{len(summary['eligible_density_slices'])}**."
    )
    if summary["train_minus_validation_gaps"]:
        lines.extend(
            [
                "",
                "## Train-minus-validation gaps",
                "",
                "Positive values indicate worse generalization than non-augmented training performance.",
                "",
                "| Protocol | Seed | Model | Accuracy gap | Recall gap | F1 gap |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for item in summary["train_minus_validation_gaps"]:
            for model in MODELS:
                gaps = item["models"][model]
                lines.append(
                    f"| `{item['protocol']}` | {item['seed']} | `{model}` | "
                    f"{gaps['accuracy']:.2%} | {gaps['recall']:.2%} | "
                    f"{gaps['f1']:.2%} |"
                )
    lines.extend(["", "## Feature availability", ""])
    for field, status in summary["feature_availability"].items():
        if field == "geometry_annotations_supplied":
            continue
        lines.append(
            f"- `{field}`: {status.get('status', 'available')} "
            f"({status['available_records']} records; source: {status.get('source') or 'not supplied'})."
        )
    lines.extend(["", "## Decision", ""])
    lines.extend(f"- {item}" for item in summary["recommendations"])
    lines.extend(
        [
            "",
            "The machine-readable report is `summary.json`; path-aligned audit records are in `records.jsonl`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if any(split not in DEFAULT_SPLITS for split in args.splits):
        raise ValueError("B5.2 may read only train and validation splits")
    manifest = load_json(args.manifest)
    for protocol in args.protocols:
        if protocol not in manifest["protocols"]:
            raise ValueError(f"Protocol is absent from manifest: {protocol}")
    manifest_records = {record["path"]: record for record in manifest["records"]}
    annotations = load_annotations(args.geometry_annotations)
    dataset_sha256 = sha256_file(args.dataset) if args.dataset is not None else None
    hashes: dict[tuple[str, str, int], str] | None = None
    if args.export_missing or args.verify_checkpoint_hashes:
        hashes = expected_checkpoint_hashes(
            args.b2_results, args.b4_results, args.protocols, args.seeds
        )
    if args.export_missing:
        assert hashes is not None
        export_missing_predictions(args, manifest, hashes)

    all_records: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for protocol in args.protocols:
        for split in args.splits:
            expected_paths = manifest["protocols"][protocol]["splits"][split]
            for seed in args.seeds:
                artifacts: dict[str, dict[str, Any]] = {}
                for model in MODELS:
                    path = prediction_path(args.predictions, protocol, split, model, seed)
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing prediction artifact: {path}")
                    artifact = load_json(path)
                    checkpoint_hash = hashes[(model, protocol, seed)] if hashes else None
                    validate_prediction_artifact(
                        artifact,
                        manifest=manifest,
                        protocol=protocol,
                        split=split,
                        model=model,
                        seed=seed,
                        expected_paths=expected_paths,
                        dataset_sha256=dataset_sha256,
                        checkpoint_sha256=checkpoint_hash,
                    )
                    artifacts[model] = artifact
                records = join_records(
                    artifacts[B2_MODEL],
                    artifacts[B4_MODEL],
                    manifest_records,
                    annotations,
                    protocol=protocol,
                    split=split,
                    seed=seed,
                )
                all_records.extend(records)
                run_summaries.append(
                    {
                        "protocol": protocol,
                        "split": split,
                        "seed": seed,
                        "models": {
                            model: {
                                **model_metrics(records, model),
                                "by_layout_family": grouped_metrics(records, "layout_family", model),
                                "by_density_bin": grouped_metrics(records, "density_bin", model),
                            }
                            for model in MODELS
                        },
                        "disagreement": disagreement_summary(records),
                        "density_candidates": [
                            candidate
                            for model in MODELS
                            for candidate in density_candidates(records, model)
                        ],
                    }
                )
    summary = build_summary(
        args, manifest, all_records, run_summaries, bool(args.geometry_annotations)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "summary.json", summary)
    records_path = args.output_dir / "records.jsonl"
    temporary = records_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(records_path)
    (args.output_dir / "README.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocols", nargs="+", default=list(DEFAULT_PROTOCOLS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--geometry-annotations", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--export-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-checkpoint-hashes", action="store_true")
    parser.add_argument("--b2-checkpoints", type=Path)
    parser.add_argument("--b4-checkpoints", type=Path)
    parser.add_argument(
        "--b2-results", type=Path, default=PROJECT_ROOT / "results" / "b2_baselines"
    )
    parser.add_argument(
        "--b4-results", type=Path, default=PROJECT_ROOT / "results" / "b4_architecture"
    )
    parser.add_argument(
        "--exporter",
        type=Path,
        default=PROJECT_ROOT / "training" / "export_classifier_predictions.py",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({
        "status": summary["status"],
        "disagreement_gate_passed": summary["disagreement_gate_passed_on_both_validation_protocols"],
        "eligible_density_slices": len(summary["eligible_density_slices"]),
        "recommendations": summary["recommendations"],
    }, indent=2))


if __name__ == "__main__":
    main()
