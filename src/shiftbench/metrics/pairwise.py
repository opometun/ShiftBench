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
    query_candidates = build_query_candidates(train_input, inference_input, K=10)
    G, A = run_metrics(query_candidates, device)
    sadge = SADGE()
    return sadge(G, A)