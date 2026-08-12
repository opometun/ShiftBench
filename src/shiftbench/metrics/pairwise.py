"""Pairwise metrics: compare two image directories/datasets with no per-dataset summary.

These are the metrics that cannot summarize one dataset alone: 
they match images across the two datasets, so both must be present at once 
and nothing lands in the .npz artifact path.

Functions on this file are only called when running SADGE via an experiment configuration.
Study reproducibility code structures the contents of quantify_benchmark_shift() individually.
"""

from __future__ import annotations

import torch

# Recorded next to the SADGE score in run artifacts. K is the candidate count
# build_query_candidates samples per query image; higher_is_better is flagged
# because SADGE is a fused SIMILARITY — unlike every distance here, a LARGER
# value means the datasets are MORE alike.
SADGE_PARAMS = {
    "K": 10,
    "appearance_metric": "dinov3_sim",
    "geometry_metric": "mast3r_inliers",
    "higher_is_better": True,
}


def sadge_pairwise(train_input, inference_input) -> float:
    """SADGE score between two image directories/datasets.

    train_input stores the location of dataset A images (either a list or img paths, or a folder directory containing all images), 
    inference_input stores the location of dataset B images (either a list or img paths, or a folder directory containing all images), 
    matching quantify_benchmark_shift's argument order.

    Heavy imports are deferred: needs the '[sadge]' extras and the MASt3R git
    submodule (initialized recursively).
    """
    from shiftbench.features.device import get_device
    from shiftbench.shift_quantification_metrics.representation_based.sadge.metric import (
        _normalize_to_paths, 
    )

    train_paths = _normalize_to_paths(train_input)
    infer_paths = _normalize_to_paths(inference_input)

    if len(train_paths) == 0 or len(infer_paths) == 0:
        raise ValueError("SADGE cannot run on empty image sets.")

    return float(
        quantify_benchmark_shift(train_paths, infer_paths, get_device())
    )


def quantify_benchmark_shift(train_input, inference_input, device:torch.device) -> float: 
    """Compute SADGE-based distribution shift between two datasets."""
    from shiftbench.shift_quantification_metrics.representation_based.sadge.metric import (
        build_query_candidates,
        run_metrics,
        SADGE,
    )
    query_candidates = build_query_candidates(train_input, inference_input, K=SADGE_PARAMS["K"])
    G, A = run_metrics(query_candidates, device)
    _check_calibration(geo_inliers=G, app_similarity=A)
    sadge = SADGE()
    return sadge(G, A)


def _check_calibration(geo_inliers, app_similarity):
    import warnings
    import numpy as np
    from shiftbench.shift_quantification_metrics.representation_based.sadge.metric import SADGE
    geo_inliers, app_similarity = np.array(geo_inliers), np.array(app_similarity)

    # Compute own calibration stats
    geo_log = np.log1p(geo_inliers)
    geo_mean = geo_log.mean()
    geo_std  = geo_log.std()
    app_mean = app_similarity.mean()
    app_std  = app_similarity.std()

    # SADGE defaults
    sadge_default = SADGE()
    geo_mean_default = sadge_default.geo_mean
    geo_std_default = sadge_default.geo_std
    app_mean_default = sadge_default.app_mean
    app_std_default  = sadge_default.app_std

    # Raise warnings if stats differ too much 
    # Mean differs from default more than one default standard deviation
    if abs(geo_mean - geo_mean_default) > geo_std_default:
        warnings.warn(
            f"Geometry mean differs significantly from SADGE calibration.\n"
            f"  SADGE default: {geo_mean_default:.4f}, std: {geo_std_default:.4f}\n"
            f"  Your stats:    {geo_mean:.4f}, std: {geo_std:.4f}\n"
            "Consider recalibration.",
            category=UserWarning,
        )

    if abs(app_mean - app_mean_default) > app_std_default:
        warnings.warn(
            f"Appearance mean differs significantly from SADGE calibration.\n"
            f"  SADGE default: {app_mean_default:.4f}, std: {app_std_default:.4f}\n"
            f"  Your stats:    {app_mean:.4f}, std: {app_std:.4f}\n"
            "Consider recalibration.",
            category=UserWarning,
        )

    # Variance differs from default (less than half or more than double the calibrated variance)
    if geo_std < 0.5 * geo_std_default or geo_std > 2.0 * geo_std_default:
        warnings.warn(
            f"Geometry std ({geo_std}) differs strongly from SADGE calibration ({geo_std_default}). "
            f"  SADGE default: {geo_mean_default:.4f}, std: {geo_std_default:.4f}\n"
            f"  Your stats:    {geo_mean:.4f}, std: {geo_std:.4f}\n"
            "Consider recalibration.",
            category=UserWarning,
        )

    if app_std < 0.5 * app_std_default or app_std > 2.0 * app_std_default:
        warnings.warn(
            f"Appearance std ({app_std}) differs strongly from SADGE calibration ({app_std_default}). "
            f"  SADGE default: {app_mean_default:.4f}, std: {app_std_default:.4f}\n"
            f"  Your stats:    {app_mean:.4f}, std: {app_std:.4f}\n"
            "Consider recalibration.",
            category=UserWarning,
        )

    # Geometry collapse check (ranking may invert)
    geo_z_default = (geo_log - geo_mean_default) / geo_std_default
    frac_low = np.mean(geo_z_default < -2)

    if frac_low > 0.3:
        warnings.warn(
            f"{frac_low*100:.1f}% of geometry z-scores are < -2 under SADGE calibration. "
            "Consider recalibration.",
            category=UserWarning,
        )