#!/usr/bin/env python3
"""Repository entry point for the B6.2 multi-task U-Net experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_multitask_unet import main


if __name__ == "__main__":
    raise SystemExit(main())
