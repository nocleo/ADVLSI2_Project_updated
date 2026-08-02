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
    MODEL_SOURCE,
    evaluate,
    load_protocol,
    per_layout_test_metrics,
    portable_path,
    repository_commit,
    sha256_file,
)
from run_inference_pc_optimized.define_cnn_model import NCSU_DRCNN


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    manifest, splits = load_protocol(args.manifest, args.dataset, args.protocol)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = NCSU_DRCNN().to(device)
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
        metrics = evaluate(
            model,
            DataLoader(test_set, batch_size=args.batch_size),
            device,
        )
        per_layout = per_layout_test_metrics(
            model,
            data_root,
            splits["test"],
            args.batch_size,
            device,
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
        "model": "NCSU_DRCNN",
        "model_source_sha256": sha256_file(MODEL_SOURCE),
        "evaluation_source_sha256": sha256_file(Path(__file__)),
        "checkpoint": portable_path(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "batch_size": args.batch_size,
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    return args


if __name__ == "__main__":
    evaluate_checkpoint(parse_args())
