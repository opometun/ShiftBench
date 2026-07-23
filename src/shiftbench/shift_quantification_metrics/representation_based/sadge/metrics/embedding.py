# This file's contents were taken from sadge/metrics/embedding.py of https://github.com/SADGE-metric/sadge-reproduction.git repository  on July 19th, 2026
# FOLLOWING ADJUSTMENTS WERE MADE BY US: 
# - removed the alternative appearance-based embedding models CLIP ViT-L/14, SigLIP-SO400M, SAM3 vision encoder, DINOv2 ViT-L/14.

"""Cosine similarity in pretrained image embedding space DINOv3 ViT-L/16."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as torch_functional
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
)


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
        self._processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
        self._model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").to(self.device).eval()

    def compute(self, real_path: Path, syn_path: Path) -> float:
        self._load()
        inputs = self._processor(
            images=[
                Image.open(real_path).convert("RGB"),
                Image.open(syn_path).convert("RGB"),
            ],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            embeddings = self._model(**inputs).pooler_output
            embeddings = torch_functional.normalize(embeddings, p=2, dim=-1)
        return float((embeddings[0] @ embeddings[1]).item())

    @property
    def higher_is_better(self) -> bool:
        return True
