"""Evaluate one frozen classifier checkpoint on a manifest test split.

This command is separate from training so B3 hyperparameter candidates can be
selected from validation metrics before any frozen test data is evaluated.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "run_inference_pc_optimized"))

from training.train_classifier import (
    DRCDataset,
    classification_metrics,
    load_protocol,
    parse_sample_path,
    portable_path,
    repository_commit,
    sha256_file,
)
from training.classifier_models import (
    BASELINE_MODEL,
    MODEL_NAMES,
    build_classifier,
    model_source_path,
)


def evaluate_at_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, object]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = count = 0
    labels_out: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            dirty_probability = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            count += labels.size(0)
            labels_out.extend(labels.cpu().tolist())
            predictions.extend((dirty_probability >= threshold).long().cpu().tolist())
    return {
        "samples": count,
        "loss": total_loss / count,
        **classification_metrics(labels_out, predictions),
    }


def per_layout_metrics_at_threshold(
    model: torch.nn.Module,
    data_root: Path,
    paths: list[str],
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> dict[str, dict[str, object]]:
    by_layout: dict[str, list[str]] = {}
    for path in paths:
        _, layout, _, _ = parse_sample_path(path)
        by_layout.setdefault(layout, []).append(path)
    return {
        layout: evaluate_at_threshold(
            model,
            DataLoader(DRCDataset(data_root, augment=False, sample_paths=layout_paths), batch_size=batch_size),
            device,
            threshold,
        )
        for layout, layout_paths in sorted(by_layout.items())
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    manifest, splits = load_protocol(args.manifest, args.dataset, args.protocol)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_classifier(args.model).to(device)
    model.load_state_dict(state)

    with tempfile.TemporaryDirectory(prefix="advlsi-evaluation-") as temp_dir:
        data_root = Path(temp_dir)
        with zipfile.ZipFile(args.dataset) as archive:
            archive.extractall(data_root)
        test_set = DRCDataset(
            data_root,
            augment=False,
            sample_paths=splits["test"],
        )
        metrics = evaluate_at_threshold(
            model,
            DataLoader(test_set, batch_size=args.batch_size),
            device,
            args.threshold,
        )
        per_layout = per_layout_metrics_at_threshold(
            model,
            data_root,
            splits["test"],
            args.batch_size,
            device,
            args.threshold,
        )

    result: dict[str, object] = {
        "dataset": portable_path(args.dataset),
        "dataset_archive_sha256": sha256_file(args.dataset),
        "manifest": portable_path(args.manifest),
        "manifest_id": manifest["manifest_id"],
        "protocol": args.protocol,
        "split": "test",
        "samples": metrics["samples"],
        "seed": args.seed,
        "device": str(device),
        "model": args.model,
        "model_source_sha256": sha256_file(model_source_path(args.model)),
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "checkpoint": portable_path(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "batch_size": args.batch_size,
        "decision_threshold": args.threshold,
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_predicted_class_counts": metrics["predicted_class_counts"],
        "test_confusion_matrix": metrics["confusion_matrix"],
        "test_per_layout": per_layout,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "repository_commit": repository_commit(),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


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
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, default=BASELINE_MODEL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between zero and one")
    return args


if __name__ == "__main__":
    evaluate_checkpoint(parse_args())
