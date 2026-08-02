"""Select a dirty-class decision threshold from one frozen validation split only."""

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

from run_inference_pc_optimized.define_cnn_model import NCSU_DRCNN
from training.train_classifier import (
    DRCDataset,
    MODEL_SOURCE,
    load_protocol,
    portable_path,
    repository_commit,
    sha256_file,
)
from training.threshold_calibration import classification_metrics, select_threshold


def calibrate(args: argparse.Namespace) -> dict[str, object]:
    manifest, splits = load_protocol(args.manifest, args.dataset, args.protocol)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = NCSU_DRCNN().to(device)
    model.load_state_dict(state)
    labels_out: list[int] = []
    probabilities: list[float] = []
    with tempfile.TemporaryDirectory(prefix="advlsi-calibration-") as temp_dir:
        data_root = Path(temp_dir)
        with zipfile.ZipFile(args.dataset) as archive:
            archive.extractall(data_root)
        loader = DataLoader(
            DRCDataset(data_root, augment=False, sample_paths=splits["validation"]),
            batch_size=args.batch_size,
        )
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                logits = model(inputs.to(device))
                probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
                labels_out.extend(labels.tolist())
    threshold, selected = select_threshold(labels_out, probabilities, args.recall_floor)
    default_metrics = classification_metrics(
        labels_out, [int(probability >= 0.5) for probability in probabilities]
    )
    result: dict[str, object] = {
        "dataset": portable_path(args.dataset),
        "dataset_archive_sha256": sha256_file(args.dataset),
        "manifest": portable_path(args.manifest),
        "manifest_id": manifest["manifest_id"],
        "protocol": args.protocol,
        "split": "validation",
        "test_evaluated": False,
        "samples": len(labels_out),
        "seed": args.seed,
        "model": "NCSU_DRCNN",
        "model_source_sha256": sha256_file(MODEL_SOURCE),
        "calibration_source_sha256": sha256_file(Path(__file__)),
        "checkpoint": portable_path(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "recall_floor": args.recall_floor,
        "default_threshold": 0.5,
        "default_metrics": default_metrics,
        "selected_threshold": threshold,
        "selected_metrics": selected,
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "repository_commit": repository_commit(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recall-floor", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.recall_floor <= 1:
        parser.error("batch size must be positive and recall floor in [0, 1]")
    return args


if __name__ == "__main__":
    calibrate(parse_args())
