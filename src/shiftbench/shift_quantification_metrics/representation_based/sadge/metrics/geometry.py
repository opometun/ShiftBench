# This file's contents were taken from sadge/metrics/geometry.py of https://github.com/SADGE-metric/sadge-reproduction.git repository  on July 19th, 2026
# FOLLOWING ADJUSTMENTS WERE MADE BY US:
# - removed the alternative geometric inlier-count metrics SuperPoint+LightGlue and LoFTR
# - adding MAST3R_DIR directly
# - fail with a clear message when the MASt3R git submodule is not initialized

"""Geometric inlier-count metric MASt3R (mutual nearest neighbour matches), 
which returns the number of inliers after USAC-MAGSAC fundamental-matrix verification. 
Higher is better."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional

MAST3R_DIR = Path(__file__).resolve().parent / "mast3r"


def _verify_with_magsac(pts1: np.ndarray, pts2: np.ndarray) -> int:
    if len(pts1) < 8:
        return 0
    try:
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.USAC_MAGSAC, 3.0, 0.99, 1000)
    except cv2.error:
        return 0
    return int(mask.sum()) if mask is not None else 0


class GeoGapMetric:
    """MASt3R inlier count after mutual NN matching."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._mast3r = None

    def _load(self) -> None:
        if self._mast3r is not None:
            return
        if not (MAST3R_DIR / "mast3r" / "model.py").exists():
            raise RuntimeError(
                "MASt3R submodule is not initialized; run "
                "'git submodule update --init --recursive' in the repository "
                "root, then retry."
            )
        # MASt3R is a git submodule; expose its package roots before import.
        if str(MAST3R_DIR) not in sys.path:
            sys.path.insert(0, str(MAST3R_DIR))
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from mast3r.model import AsymmetricMASt3R

        print("Loading MASt3R...")
        self._mast3r = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        ).to(self.device).eval()

    @torch.inference_mode()
    def compute(self, real_path: Path, syn_path: Path) -> int:
        self._load()
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        imgs = load_images([str(real_path), str(syn_path)], size=512, verbose=False)
        out = inference([tuple(imgs)], self._mast3r, self.device, batch_size=1, verbose=False)
        desc1 = out["pred1"]["desc"].squeeze(0).to(self.device)
        desc2 = out["pred2"]["desc"].squeeze(0).to(self.device)
        if max(desc1.shape[0], desc1.shape[1]) > 256:
            desc1 = torch_functional.interpolate(
                desc1.permute(2, 0, 1)[None], size=(256, 256), mode="bilinear",
            )[0].permute(1, 2, 0)
            desc2 = torch_functional.interpolate(
                desc2.permute(2, 0, 1)[None], size=(256, 256), mode="bilinear",
            )[0].permute(1, 2, 0)

        d1 = desc1.reshape(-1, desc1.shape[-1])
        d2 = desc2.reshape(-1, desc2.shape[-1])
        scores = d1 @ d2.t()
        top12 = scores.max(1).indices
        top21 = scores.max(0).indices
        mutual = top21[top12] == torch.arange(len(top12), device=self.device)
        matches = torch.where(mutual)[0]
        if len(matches) < 8:
            return 0

        w = desc1.shape[1]
        scale = 512.0 / w
        pts1 = torch.stack([matches % w, matches // w], dim=1).cpu().numpy() * scale
        pts2 = torch.stack([top12[mutual] % w, top12[mutual] // w], dim=1).cpu().numpy() * scale
        return _verify_with_magsac(pts1, pts2)

    @property
    def higher_is_better(self) -> bool:
        return True
