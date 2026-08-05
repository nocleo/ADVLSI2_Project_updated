"""Resume-safe local launcher for B4 on an Apple-silicon Mac."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "b4_architecture_mps",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of Apple MPS")
    args = parser.parse_args()

    if not args.cpu and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS is unavailable. Confirm that this is an Apple-silicon Mac and "
            "that the virtual environment contains a current PyTorch build."
        )

    device = "cpu" if args.cpu else "mps"
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"B4 training device: {device}", flush=True)
    print(f"Resume-safe output: {args.output_dir.resolve()}", flush=True)
    print(
        "The first epoch prints epoch_seconds and estimated_remaining_seconds.",
        flush=True,
    )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_b4_architecture.py"),
        "--python",
        sys.executable,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.cpu:
        command.append("--cpu")
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)


if __name__ == "__main__":
    main()
