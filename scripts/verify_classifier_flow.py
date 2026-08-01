"""Fast verification of dataset -> training -> ONNX -> inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "run_inference_pc_optimized"))

from define_cnn_model import NCSU_DRCNN


def dataset_samples(dataset_zip: Path, limit: int = 4) -> np.ndarray:
    with zipfile.ZipFile(dataset_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith(".npy")]
        classes = {Path(name).parts[0] for name in names}
        if classes != {"clean", "dirty"}:
            raise AssertionError(f"Expected clean/dirty directories, found {sorted(classes)}")
        arrays = [np.load(BytesIO(archive.read(name))).astype(np.float32) for name in names[:limit]]
    if any(array.shape != (200, 200) for array in arrays):
        raise AssertionError("Every classifier tile must have shape 200x200")
    return np.stack(arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "training_datasets" / "combined_training_dataset.zip",
    )
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    tiles = dataset_samples(args.dataset)
    with tempfile.TemporaryDirectory(prefix="advlsi-verify-") as temp_dir:
        temp = Path(temp_dir)
        weights = temp / "weights.pth"
        metrics = temp / "metrics.json"
        onnx_path = temp / "model.onnx"

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "training" / "train_classifier.py"),
                "--dataset", str(args.dataset),
                "--output", str(weights),
                "--metrics", str(metrics),
                "--epochs", "1",
                "--max-samples", str(args.samples),
                "--batch-size", "16",
                "--cpu",
            ],
            check=True,
        )

        model = NCSU_DRCNN()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        torch.onnx.export(
            model,
            torch.from_numpy(tiles[:1, None]),
            onnx_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        logits = session.run(None, {session.get_inputs()[0].name: tiles[:, None]})[0]
        if logits.shape != (len(tiles), 2) or not np.isfinite(logits).all():
            raise AssertionError(f"Unexpected inference output: {logits.shape}")

        result = json.loads(metrics.read_text(encoding="utf-8"))
        required_metrics = {
            "best_epoch",
            "best_validation_loss",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_predicted_class_counts",
            "test_confusion_matrix",
        }
        missing = required_metrics - result.keys()
        if missing:
            raise AssertionError(f"Training metrics are missing fields: {sorted(missing)}")
        if "training only" not in result["augmentation"]:
            raise AssertionError(f"Unexpected augmentation mode: {result['augmentation']}")
        result.update({"onnx_output_shape": list(logits.shape), "status": "passed"})
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
