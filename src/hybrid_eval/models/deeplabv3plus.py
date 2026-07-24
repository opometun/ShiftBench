"""DeepLabV3+ with a domain-neutral ImageNet-pretrained ResNet-101 encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.resnet import ResNet
from torchvision.models.segmentation.deeplabv3 import ASPP


DEEPLABV3PLUS_BACKBONE_WEIGHTS = ResNet101_Weights.IMAGENET1K_V2


class ResNetFeatureBackbone(nn.Module):
    """Expose the low- and high-level ResNet features used by DeepLabV3+."""

    def __init__(self, resnet: ResNet) -> None:
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        low_level = self.layer1(features)
        features = self.layer2(low_level)
        features = self.layer3(features)
        high_level = self.layer4(features)
        return {"low": low_level, "out": high_level}


class DeepLabV3PlusHead(nn.Module):
    """ASPP plus the low-level feature decoder that distinguishes V3+ from V3."""

    def __init__(
        self,
        *,
        high_level_channels: int,
        low_level_channels: int,
        num_classes: int,
        atrous_rates: Sequence[int] = (6, 12, 18),
    ) -> None:
        super().__init__()
        self.aspp = ASPP(high_level_channels, atrous_rates, out_channels=256)
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        for module in (self.low_level_projection, self.decoder):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        low_level = self.low_level_projection(features["low"])
        high_level = self.aspp(features["out"])
        high_level = F.interpolate(
            high_level,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.decoder(torch.cat((high_level, low_level), dim=1))


class DeepLabV3Plus(nn.Module):
    """Semantic-segmentation model returning torchvision-compatible output."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.classifier(self.backbone(images))
        logits = F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"out": logits}


def get_deeplabv3plus_resnet101(
    num_classes: int = 19, pretrained: bool = True
) -> DeepLabV3Plus:
    """Build DeepLabV3+ with an ImageNet-only ResNet-101 encoder.

    The segmentation decoder is always initialized from scratch. This avoids
    leaking Cityscapes supervision into experiments that evaluate on Cityscapes.
    """
    weights = DEEPLABV3PLUS_BACKBONE_WEIGHTS if pretrained else None
    resnet = resnet101(
        weights=weights,
        replace_stride_with_dilation=(False, False, True),
    )
    backbone = ResNetFeatureBackbone(resnet)
    classifier = DeepLabV3PlusHead(
        high_level_channels=2048,
        low_level_channels=256,
        num_classes=num_classes,
    )
    return DeepLabV3Plus(backbone, classifier)


__all__ = [
    "DEEPLABV3PLUS_BACKBONE_WEIGHTS",
    "DeepLabV3Plus",
    "DeepLabV3PlusHead",
    "ResNetFeatureBackbone",
    "get_deeplabv3plus_resnet101",
]
