"""Frozen DINOv2 image encoder.

The model name and the choice of pooled token below define what every
downstream distance in this benchmark actually measures. Changing either
changes all reported numbers, so they live here rather than in a script.

Requires the optional 'features' dependencies (torch, transformers, pillow).
"""

from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL_NAME = "facebook/dinov2-base"


def get_device() -> torch.device:
    """Best available torch device, preferring CUDA then Apple MPS."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frozen_dinov2(
    device: torch.device,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[AutoImageProcessor, AutoModel]:
    """Load a DINOv2 encoder in eval mode with gradients disabled.

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

    Takes the CLS token, index 0 of the sequence, as the image-level summary
    rather than mean-pooling the patch tokens.
    """
    inputs = processor(images=image_batch, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0]
