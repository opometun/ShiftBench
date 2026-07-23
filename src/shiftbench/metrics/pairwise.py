"""Pairwise metrics: compare two image directories with no per-dataset summary.

These are the metrics that cannot summarize one dataset alone (see the Phase 0
decision in docs/integration-plan.md): they match images across the two
datasets, so both must be present at once and nothing lands in the .npz
artifact path.
"""

from __future__ import annotations

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


def sadge_pairwise(image_dir_a: str, image_dir_b: str) -> float:
    """SADGE score between two image directories.

    image_dir_a plays the training-set role, image_dir_b the inference-set
    role, matching quantify_benchmark_shift's argument order.

    Heavy imports are deferred: needs the '[sadge]' extras and the MASt3R git
    submodule (initialized recursively).
    """
    from pathlib import Path

    from shiftbench.features.device import get_device
    from shiftbench.shift_quantification_metrics.representation_based.sadge.metric import (
        quantify_benchmark_shift,
    )

    return float(
        quantify_benchmark_shift(Path(image_dir_a), Path(image_dir_b), get_device())
    )
