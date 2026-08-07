"""Export path-aligned classifier probabilities for one manifest split.

The B5 ensemble runner keeps validation selection separate from frozen-test
evaluation.  This command performs inference only; it never selects an
ensemble weight or reads another model's predictions. B5.2 also uses it for
non-augmented training predictions during failure analysis.
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

from training.classifier_models import MODEL_NAMES, build_classifier, model_source_path
from training.runtime_device import DEVICE_CHOICES, select_device
from training.train_classifier import (
    DRCDataset,
    load_protocol,
    portable_path,
    repository_commit,
    sha256_file,
)


def export_predictions(args: argparse.Namespace) -> dict[str, object]:
    manifest, splits = load_protocol(args.manifest, args.dataset, args.protocol)
    if args.split not in {"train", "validation", "test"}:
        raise ValueError("Prediction split must be train, validation, or test")

    device = select_device(args.device, args.cpu)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_classifier(args.model).to(device)
    model.load_state_dict(state)
    model.eval()

    with tempfile.TemporaryDirectory(prefix="advlsi-b5-predictions-") as temp_dir:
        data_root = Path(temp_dir)
        with zipfile.ZipFile(args.dataset) as archive:
            archive.extractall(data_root)
        dataset = DRCDataset(
            data_root,
            augment=False,
            sample_paths=splits[args.split],
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        probabilities: list[float] = []
        labels_out: list[int] = []
        with torch.no_grad():
            for inputs, labels in loader:
                logits = model(inputs.to(device))
                probabilities.extend(
                    torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                )
                labels_out.extend(labels.tolist())

    if not (
        len(dataset.relative_paths) == len(labels_out) == len(probabilities)
    ):
        raise RuntimeError("Prediction export lost sample alignment")

    result: dict[str, object] = {
        "phase": "B5",
        "dataset": portable_path(args.dataset),
        "dataset_archive_sha256": sha256_file(args.dataset),
        "manifest": portable_path(args.manifest),
        "manifest_id": manifest["manifest_id"],
        "protocol": args.protocol,
        "split": args.split,
        "seed": args.seed,
        "model": args.model,
        "checkpoint": portable_path(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_source_sha256": sha256_file(model_source_path(args.model)),
        "samples": len(probabilities),
        "batch_size": args.batch_size,
        "device": str(device),
        "records": [
            {
                "path": path,
                "label": int(label),
                "dirty_probability": float(probability),
            }
            for path, label, probability in zip(
                dataset.relative_paths, labels_out, probabilities
            )
        ],
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "repository_commit": repository_commit(),
        "exporter_source_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("model", "protocol", "split", "seed", "samples", "device")}, indent=2))
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
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
        help="Manifest split to export. B5.2 uses train/validation only.",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export_predictions(parse_args())
