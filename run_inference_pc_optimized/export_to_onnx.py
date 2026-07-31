import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import MODEL_WEIGHTS_PTH, MODEL_WEIGHTS_ONNX

from define_cnn_model import NCSU_DRCNN


def export_model(weights_path=MODEL_WEIGHTS_PTH, onnx_path=MODEL_WEIGHTS_ONNX):
    weights_path = Path(weights_path)
    onnx_path = Path(onnx_path)

    print("[*] Loading PyTorch model...")
    model = NCSU_DRCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 1, 200, 200)

    print("[*] Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        dynamo=False,
    )
    print(f"[*] Done! ONNX model successfully saved to '{onnx_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained DRC classifier weights to ONNX.")
    parser.add_argument("--weights", type=Path, default=MODEL_WEIGHTS_PTH)
    parser.add_argument("--output", type=Path, default=MODEL_WEIGHTS_ONNX)
    args = parser.parse_args()
    export_model(args.weights, args.output)
