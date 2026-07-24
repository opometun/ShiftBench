"""Memory-bounded metrics for semantic segmentation."""

from __future__ import annotations

import math

import torch


def _flatten_probabilities(probs: torch.Tensor, targets: torch.Tensor | None = None):
    if probs.ndim == 4:
        probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
    elif probs.ndim != 2:
        raise ValueError("probs must have shape (N, C) or (B, C, H, W)")
    if targets is not None:
        targets = targets.reshape(-1)
        if targets.numel() != probs.shape[0]:
            raise ValueError("targets and probabilities contain different pixel counts")
    return probs, targets


def calculate_ece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
    ignore_index: int = 255,
) -> float:
    """Calculate expected calibration error for segmentation probabilities."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    probs, targets = _flatten_probabilities(probs, targets)
    valid = targets != ignore_index
    if not torch.any(valid):
        return 0.0
    probs = probs[valid]
    targets = targets[valid]
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)
    boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    ece = torch.zeros((), dtype=torch.float64, device=probs.device)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = confidences.gt(lower) & confidences.le(upper)
        if torch.any(in_bin):
            proportion = in_bin.float().mean()
            accuracy = accuracies[in_bin].float().mean()
            confidence = confidences[in_bin].mean()
            ece += torch.abs(confidence - accuracy).double() * proportion.double()
    return float(ece.item())


def calculate_predictive_entropy(
    probs: torch.Tensor,
    targets: torch.Tensor | None = None,
    ignore_index: int = 255,
    eps: float = 1e-8,
) -> float:
    """Calculate mean predictive entropy, excluding ignored pixels when supplied."""
    probs, targets = _flatten_probabilities(probs, targets)
    if targets is not None:
        probs = probs[targets != ignore_index]
    if probs.shape[0] == 0:
        return 0.0
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(eps)), dim=1)
    return float(entropy.mean().item())


class StreamingSegmentationMetrics:
    """Accumulate mIoU, ECE, and entropy without retaining whole datasets."""

    def __init__(self, num_classes: int, num_bins: int = 15, ignore_index: int = 255) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.bin_count = torch.zeros(num_bins, dtype=torch.float64)
        self.bin_confidence = torch.zeros(num_bins, dtype=torch.float64)
        self.bin_accuracy = torch.zeros(num_bins, dtype=torch.float64)
        self.entropy_sum = 0.0
        self.pixel_count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 4 or targets.ndim != 3:
            raise ValueError("Expected logits (B, C, H, W) and targets (B, H, W)")
        if logits.shape[1] != self.num_classes or logits.shape[0] != targets.shape[0] or logits.shape[2:] != targets.shape[1:]:
            raise ValueError("Logits and targets have incompatible shapes")

        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        valid = targets != self.ignore_index
        valid &= targets.ge(0) & targets.lt(self.num_classes)
        if not torch.any(valid):
            return

        valid_targets = targets[valid].to(torch.int64)
        valid_predictions = predictions[valid].to(torch.int64)
        encoded = valid_targets * self.num_classes + valid_predictions
        confusion = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += confusion.reshape(self.num_classes, self.num_classes).cpu()

        valid_confidences = confidences[valid]
        valid_accuracies = valid_predictions.eq(valid_targets)
        bin_indices = torch.clamp((valid_confidences * self.num_bins).ceil().long() - 1, 0, self.num_bins - 1)
        for index in range(self.num_bins):
            in_bin = bin_indices == index
            if torch.any(in_bin):
                count = int(in_bin.sum().item())
                self.bin_count[index] += count
                self.bin_confidence[index] += valid_confidences[in_bin].double().sum().cpu()
                self.bin_accuracy[index] += valid_accuracies[in_bin].double().sum().cpu()

        valid_probs = probs.permute(0, 2, 3, 1)[valid]
        entropy = -torch.sum(valid_probs * torch.log(valid_probs.clamp_min(1e-8)), dim=1)
        self.entropy_sum += float(entropy.double().sum().item())
        self.pixel_count += int(valid_targets.numel())

    def compute(self) -> dict[str, object]:
        intersection = self.confusion.diag().double()
        union = self.confusion.sum(dim=0).double() + self.confusion.sum(dim=1).double() - intersection
        class_ious = [
            float(intersection[index] / union[index]) if union[index] > 0 else math.nan
            for index in range(self.num_classes)
        ]
        valid_ious = [value for value in class_ious if not math.isnan(value)]
        miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

        total = float(self.bin_count.sum().item())
        ece = 0.0
        if total:
            occupied = self.bin_count > 0
            mean_confidence = self.bin_confidence[occupied] / self.bin_count[occupied]
            mean_accuracy = self.bin_accuracy[occupied] / self.bin_count[occupied]
            ece = float((torch.abs(mean_confidence - mean_accuracy) * self.bin_count[occupied] / total).sum().item())

        return {
            "mIoU": miou,
            "ece": ece,
            "entropy": self.entropy_sum / self.pixel_count if self.pixel_count else 0.0,
            "class_ious": {str(index): value for index, value in enumerate(class_ious) if not math.isnan(value)},
            "valid_pixels": self.pixel_count,
        }
