"""Small shared-encoder U-Net for B6.2 segmentation and classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )
        self.convolution = DoubleConv(output_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = F.interpolate(
                inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.convolution(torch.cat((skip, inputs), dim=1))


class MultiTaskUNet(nn.Module):
    """Predict a central mask and tile label from one shared 200x200 encoder."""

    def __init__(self, base_channels: int = 16, output_size: int = 160) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        self.output_size = output_size
        self.encoder1 = DoubleConv(1, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.decoder3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.segmentation_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.maximum_pool = nn.AdaptiveMaxPool2d(1)
        self.classification_head = nn.Linear(base_channels * 16, 2)

    def _central_crop(self, tensor: torch.Tensor) -> torch.Tensor:
        height, width = tensor.shape[-2:]
        if self.output_size > height or self.output_size > width:
            raise ValueError(
                f"Output crop {self.output_size} exceeds decoder shape {(height, width)}"
            )
        top = (height - self.output_size) // 2
        left = (width - self.output_size) // 2
        return tensor[..., top : top + self.output_size, left : left + self.output_size]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))
        decoded = self.decoder3(bottleneck, encoder3)
        decoded = self.decoder2(decoded, encoder2)
        decoded = self.decoder1(decoded, encoder1)
        segmentation = self._central_crop(self.segmentation_head(decoded))
        average = self.average_pool(bottleneck).flatten(1)
        maximum = self.maximum_pool(bottleneck).flatten(1)
        classification = self.classification_head(torch.cat((average, maximum), dim=1))
        return segmentation, classification


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        score = (2 * intersection + self.epsilon) / (denominator + self.epsilon)
        return 1 - score.mean()


class MultiTaskLoss(nn.Module):
    """Pre-registered B6.2 loss: weighted BCE + Dice + 0.25 x CE."""

    def __init__(self, positive_weight: float, classification_weight: float = 0.25) -> None:
        super().__init__()
        if positive_weight <= 0 or classification_weight < 0:
            raise ValueError("Loss weights must be positive")
        self.register_buffer("positive_weight", torch.tensor([positive_weight]))
        self.classification_weight = classification_weight
        self.dice = SoftDiceLoss()
        self.classification = nn.CrossEntropyLoss()

    def forward(
        self,
        segmentation_logits: torch.Tensor,
        classification_logits: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bce = F.binary_cross_entropy_with_logits(
            segmentation_logits, masks, pos_weight=self.positive_weight
        )
        dice = self.dice(segmentation_logits, masks)
        classification = self.classification(classification_logits, labels)
        total = bce + dice + self.classification_weight * classification
        return total, {"bce": bce, "dice": dice, "classification": classification}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
