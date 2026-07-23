"""Frozen DINOv3 image encoder.

Added for the SADGE appearance metric, but usable as a general backend. Pools
via pooler_output rather than DINOv2's CLS-token slice — a different summary
of the image, so features from the two are not interchangeable.

Features are returned unnormalized, like DINOv2. SADGE's cosine similarity
normalizes at the point of use, where it belongs to the metric, not the
encoder.

Requires the optional 'features' dependencies (torch, transformers, pillow).
"""

from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def load_frozen_dinov3(
    device: torch.device,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[AutoImageProcessor, AutoModel]:
    """Load a DINOv3 encoder in eval mode with gradients disabled.

    Args:
        device: Device to place the model on.
        model_name: Hugging Face model id.

    Returns:
        The matching image processor and the frozen model.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return processor, model


def extract_features(
    processor: AutoImageProcessor,
    model: AutoModel,
    image_batch: list,
    device: torch.device,
) -> torch.Tensor:
    """Embed a batch of PIL images as (batch_size, hidden_size) features.

    Takes the model's pooler_output as the image-level summary.
    """
    inputs = processor(images=image_batch, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.pooler_output
