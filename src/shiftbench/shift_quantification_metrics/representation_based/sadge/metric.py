# This file's contents were taken from sadge/metric.py of https://github.com/SADGE-metric/sadge-reproduction.git repository on July 19th, 2026
# FOLLOWING ADJUSTMENTS WERE MADE BY US: 
# - removed config_path
# - hard-coded the officially fitted SADGE parameters (a, b, c, geo_mean, geo_std, app_mean, app_std)
# - added METRICS, BEST_FN, set_seed, build_query_candidates, and run_metrics
# - added option to process directory or list of paths via _normalize_to_paths()

"""The SADGE fusion class.

SADGE = a * G + b * A + c * G * A,
where G = z(log1p(geo_inliers)) and A = z(app_value)."""
from __future__ import annotations

import random
import torch
import numpy as np
from pathlib import Path

from shiftbench.shift_quantification_metrics.representation_based.sadge.metrics.embedding import DinoV3Metric
from shiftbench.shift_quantification_metrics.representation_based.sadge.metrics.geometry import GeoGapMetric


# The best geometric-appearance-metric-pair used by SADGE
# (metric_name, metric_cls, needs_device)
METRICS = [
    ("dinov3_sim", DinoV3Metric, True),
    ("geo_inliers", GeoGapMetric, True),
]

# Direction of "best" per metric, used to keep the strongest score across the K candidates per query
BEST_FN = {
    "dinov3_sim": max,
    "geo_inliers": max,
}


# set_seed is only required for replicating our study (scripts/run_sadge.py)
# not used when running SADGE via an experiment config (src/shiftbench/experiments/run.py)
def set_seed(seed:int=42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SADGE:
    def __init__(
        self,
        app_metric: str | None = None,
        params: tuple[float, ...] | None = None,
        geo_mean: float | None = None,
        geo_std: float | None = None,
        app_mean: float | None = None,
        app_std: float | None = None,
    ) -> None:
        self.app_metric = app_metric or "dinov3_sim"
        self.params = params or (0.0, 1.8548, 1.3399)
        self.geo_mean = geo_mean or 7.9420
        self.geo_std = geo_std or 1.7384
        self.app_mean = app_mean or 0.6359
        self.app_std = app_std or 0.1918

    def _z_transform(self, geo_inliers: float, app_value: float) -> tuple[float, float]:
        """Standardize appearance and geometry metrics with z-score normalization."""
        geo_z = (np.log1p(max(0.0, float(geo_inliers))) - self.geo_mean) / self.geo_std
        app_z = (float(app_value) - self.app_mean) / self.app_std
        return geo_z, app_z

    def __call__(self, geo_inliers: float, app_value: float) -> float:
        """Compute the SADGE score."""
        geo_z, app_z = self._z_transform(geo_inliers, app_value)
        a, b, c = self.params[0], self.params[1], self.params[2]
        return float(a * geo_z + b * app_z + c * geo_z * app_z)


def _normalize_to_paths(x):
    """Accept either a directory Path or a list of paths."""

    # Case 1: folder directory containing all images
    if isinstance(x, (str, Path)) and Path(x).is_dir():
        return sorted(
            [p for p in Path(x).iterdir()
             if p.is_file() and p.suffix.lower() == ".png"]
        )

    # Case 2: list of all img file paths
    if isinstance(x, (list, tuple)):
        return [Path(p) for p in x if Path(p).is_file()]

    raise TypeError(f"Unsupported input type for image paths: {type(x)}")


def build_query_candidates(train_input, inference_input, K:int) -> list[tuple[Path, list[Path]]]:
    """Select K train dataset images randomly for each inference dataset image."""
    # Collect all image paths
    train_img_paths = _normalize_to_paths(train_input)
    inference_img_paths = _normalize_to_paths(inference_input)

    query_candidates = []

    # For each image in the inference dataset, sample K candidates from the hybrid training dataset
    for img in inference_img_paths:
        if len(train_img_paths) <= K:
            candidates = train_img_paths
        else:
            candidates = random.sample(train_img_paths, K)

        # Append query tuple: (real_image_path, list_of_candidates)
        query_candidates.append((img, candidates))

    return query_candidates


# run_metrics is called from either run_sadge.py (study replication)
# or from pairwise.py -> quantify_benchmark_shift() (run SADGE via experiment config)
def run_metrics(query_candidates:list, device:torch.device) -> tuple[float, float]:
    """Get the dataset-level metric scores."""
    results = {}

    for metric_name, metric_cls, needs_device in METRICS:
        metric = metric_cls(device) if needs_device else metric_cls()
        best_vals = []

        for inference_img_path, candidates in query_candidates:
            # Compute similarity between inference image and each candidate image of the hybrid dataset
            vals = []
            for train_img_path in candidates:
                val = metric.compute(inference_img_path, train_img_path)
                vals.append(val)
            # Save the similarity score of the best candidate for current inference image
            best_vals.append(BEST_FN[metric_name](vals))

        # Get the average similarity across all inference images as dataset-level metric score
        results[metric_name] = float(np.mean(best_vals))

        # Cleanup memory
        del metric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    G = results["geo_inliers"]
    A = results["dinov3_sim"]
    return G, A
