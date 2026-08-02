"""Shared PyTorch device selection for CUDA, Apple Metal, and CPU runtimes."""

from __future__ import annotations

import torch


DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def select_device(requested: str = "auto", force_cpu: bool = False) -> torch.device:
    """Resolve a requested backend and fail clearly when it is unavailable."""

    if force_cpu:
        requested = "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    elif requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    elif requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but PyTorch cannot access the Apple GPU. "
            "Use an Apple-silicon Mac with a current macOS/PyTorch build, or select CPU."
        )
    return torch.device(requested)
