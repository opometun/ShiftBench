"""Frozen StreetCLIP image encoder.

StreetCLIP is a CLIP ViT-L/14 fine-tuned for geolocalization, so it is pooled
differently from DINOv2: the image embedding comes from the projection head via
get_image_features, not from a hidden-state token.

Embeddings are L2 normalized by default. CLIP is trained with a cosine
objective, so only direction carries meaning and the vector magnitude is an
arbitrary axis that would otherwise leak into every distance we compute. This
is a deliberate difference from the DINOv2 backend, which is not normalized.

Distances computed over this encoder are NOT comparable to distances computed
over any other encoder.

Requires the optional 'features' dependencies (torch, transformers, pillow).
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, CLIPModel

DEFAULT_MODEL_NAME = "geolocal/StreetCLIP"


def load_frozen_streetclip(
    device: torch.device,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[AutoProcessor, CLIPModel]:
    """Load a StreetCLIP encoder in eval mode with gradients disabled.

    Args:
        device: Device to place the model on.
        model_name: Hugging Face model id.

    Returns:
        The matching processor and the frozen model.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return processor, model


def extract_features(
    processor: AutoProcessor,
    model: CLIPModel,
    image_batch: list,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """Embed a batch of PIL images as (batch_size, projection_dim) features.

    Args:
        processor: Processor paired with the model.
        model: Frozen StreetCLIP model.
        image_batch: PIL images, already converted to RGB.
        device: Device the model lives on.
        normalize: L2 normalize each embedding. See the module docstring for
            why this defaults to True; turning it off changes every distance.

    Returns:
        Image embeddings from the projection head.
    """
    inputs = processor(images=image_batch, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.inference_mode():
        features = model.get_image_features(**inputs)

        # transformers 4.x returned the projected embedding tensor directly
        # from get_image_features; 5.x returns a BaseModelOutputWithPooling,
        # and calling .norm() on it raises AttributeError.
        #
        # Rather than guess which field of that object holds the embedding,
        # rebuild it from CLIP's documented architecture: pool the vision
        # tower, then apply the projection head. That is exactly what
        # get_image_features does internally, so both library versions yield
        # identical vectors instead of quietly different ones.
        #
        # (Do not reuse the 5.x object's pooler_output here -- it is already
        # projected to projection_dim, so feeding it through visual_projection
        # again fails with a 768-vs-1024 shape mismatch.)
        if not isinstance(features, torch.Tensor):
            vision_outputs = model.vision_model(**inputs)
            features = model.visual_projection(vision_outputs.pooler_output)

    if normalize:
        features = features / features.norm(dim=-1, keepdim=True)
    return features
