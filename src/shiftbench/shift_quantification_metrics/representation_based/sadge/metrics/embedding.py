# This file's contents were taken from sadge/metrics/embedding.py of https://github.com/SADGE-metric/sadge-reproduction.git repository  on July 19th, 2026
# FOLLOWING ADJUSTMENTS WERE MADE BY US:
# - removed the alternative appearance-based embedding models CLIP ViT-L/14, SigLIP-SO400M, SAM3 vision encoder, DINOv2 ViT-L/14.
# - delegated model loading and feature extraction to shiftbench.features.dinov3,
#   the project's shared encoder backend (same model id and pooler_output pooling;
#   inference now runs under torch.inference_mode instead of torch.no_grad).

"""Cosine similarity in pretrained image embedding space DINOv3 ViT-L/16."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as torch_functional
from PIL import Image

from shiftbench.features.dinov3 import extract_features, load_frozen_dinov3


class DinoV3Metric:
    """DINOv3 ViT-L/16 cosine similarity (pooler output). Higher is better."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        print("Loading DINOv3...")
        self._processor, self._model = load_frozen_dinov3(self.device)

    def compute(self, real_path: Path, syn_path: Path) -> float:
        self._load()
        images = [
            Image.open(real_path).convert("RGB"),
            Image.open(syn_path).convert("RGB"),
        ]
        embeddings = extract_features(self._processor, self._model, images, self.device)
        embeddings = torch_functional.normalize(embeddings, p=2, dim=-1)
        return float((embeddings[0] @ embeddings[1]).item())

    @property
    def higher_is_better(self) -> bool:
        return True
