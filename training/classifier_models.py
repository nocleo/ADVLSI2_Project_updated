"""Classifier architecture registry for controlled architecture experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from run_inference_pc_optimized.define_cnn_model import NCSU_DRCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL = "NCSU_DRCNN"
COMPACT_MODEL = "CompactBNPool"
MODEL_NAMES = (BASELINE_MODEL, COMPACT_MODEL)


class CompactBNPool(nn.Module):
    """Compact CNN with normalization and spatially invariant global pooling.

    Average pooling captures global metal density while max pooling preserves a
    strong response to sparse local spacing defects.  The concatenated pooled
    representation avoids the baseline's large, location-specific dense layer.
    """

    def __init__(self) -> None:
        super().__init__()
        channels = (1, 16, 24, 48, 64)
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
        self.features = nn.Sequential(*blocks)
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(channels[-1] * 2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        average = self.average_pool(features).flatten(1)
        maximum = self.maximum_pool(features).flatten(1)
        return self.classifier(torch.cat((average, maximum), dim=1))


def build_classifier(name: str) -> nn.Module:
    if name == BASELINE_MODEL:
        return NCSU_DRCNN()
    if name == COMPACT_MODEL:
        return CompactBNPool()
    raise ValueError(f"Unsupported classifier architecture: {name}")


def model_source_path(name: str) -> Path:
    if name == BASELINE_MODEL:
        return PROJECT_ROOT / "run_inference_pc_optimized" / "define_cnn_model.py"
    if name == COMPACT_MODEL:
        return Path(__file__)
    raise ValueError(f"Unsupported classifier architecture: {name}")
