"""Benchmark classifier size and paired CPU/ONNX batch-one latency."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import onnx
import onnxruntime as ort
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.classifier_models import MODEL_NAMES, build_classifier, model_source_path
from training.dataset_manifest import sha256_file
from training.train_classifier import portable_path, repository_commit


def timed_ms(action, warmup: int, repeats: int) -> dict[str, float | int]:
    for _ in range(warmup):
        action()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
    }


def benchmark_model(
    name: str, warmup: int, repeats: int, threads: int
) -> dict[str, object]:
    torch.manual_seed(0)
    model = build_classifier(name).cpu().eval()
    sample = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    with tempfile.TemporaryDirectory(prefix="advlsi-b4-benchmark-") as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint = temp_root / f"{name}.pth"
        onnx_path = temp_root / f"{name}.onnx"
        torch.save(model.state_dict(), checkpoint)
        torch.onnx.export(
            model,
            sample,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(onnx_path))

        with torch.inference_mode():
            pytorch_latency = timed_ms(lambda: model(sample), warmup, repeats)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        sample_numpy = sample.numpy()
        onnx_latency = timed_ms(
            lambda: session.run(None, {"input": sample_numpy}), warmup, repeats
        )
        return {
            "model": name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "state_dict_bytes": checkpoint.stat().st_size,
            "onnx_bytes": onnx_path.stat().st_size,
            "model_source_sha256": sha256_file(model_source_path(name)),
            "pytorch_cpu_batch1": pytorch_latency,
            "onnx_cpu_batch1": onnx_latency,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("warmup must be non-negative; repeats and threads must be positive")

    torch.set_num_threads(args.threads)
    results = {
        "phase": "B4",
        "input_shape": [1, 1, 200, 200],
        "threads": args.threads,
        "models": {
            name: benchmark_model(name, args.warmup, args.repeats, args.threads)
            for name in args.models
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
        },
        "repository_commit": repository_commit(),
        "benchmark_source": portable_path(Path(__file__)),
        "benchmark_source_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
