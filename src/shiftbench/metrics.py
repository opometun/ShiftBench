"""Distribution-shift metrics over frozen-encoder feature sets.

Each dataset is summarized as a Gaussian over its embeddings: a mean vector
(where the point cloud sits) and a covariance matrix (how it is spread). The
distances below compare two such summaries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.linalg


CHUNK_SIZE = 4096


def compute_mean(embeddings: np.ndarray) -> np.ndarray:
    """Centroid of an (n_samples, n_features) embedding set.

    Accumulates in float64 even for float32 features, because summing many
    thousands of samples in float32 loses precision.
    """
    return embeddings.mean(axis=0, dtype=np.float64)


def compute_covariance(
    embeddings: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Feature covariance of an (n_samples, n_features) embedding set.

    Equivalent to np.cov(embeddings, rowvar=False) but accumulated a chunk at
    a time. np.cov upcasts the whole array to float64 in one go, which doubles
    peak memory and defeats a memory-mapped input; this reads chunk_size rows
    at a time, so a large feature array never has to be resident at once.

    Args:
        embeddings: Feature array, possibly memory-mapped.
        chunk_size: Rows converted to float64 at a time.

    Returns:
        Covariance matrix of shape (n_features, n_features), using the
        unbiased n_samples - 1 normalization.
    """
    n_samples, n_features = embeddings.shape
    mu = compute_mean(embeddings)

    accumulator = np.zeros((n_features, n_features), dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        block = np.asarray(embeddings[start : start + chunk_size], dtype=np.float64)
        block -= mu
        accumulator += block.T @ block

    return accumulator / (n_samples - 1)


def centroid_distance(mu_a: np.ndarray, mu_b: np.ndarray) -> float:
    """Euclidean distance between two dataset centroids.

    Ignores spread entirely, so two datasets with matching means score 0 even
    if one collapsed to a single point. Use frechet_distance to catch that.
    """
    return float(np.linalg.norm(mu_a - mu_b))


def frechet_distance(
    mu_a: np.ndarray,
    sigma_a: np.ndarray,
    mu_b: np.ndarray,
    sigma_b: np.ndarray,
) -> float:
    """Frechet (2-Wasserstein) distance between two Gaussian summaries.

    Raises:
        ValueError: If the matrix square root comes back meaningfully complex,
            which means the inputs were not usable covariance matrices.
    """
    mean_term = (mu_a - mu_b) @ (mu_a - mu_b)

    covmean = scipy.linalg.sqrtm(sigma_a @ sigma_b)
    # sqrtm on a product of covariances picks up a small imaginary part from
    # rounding. Tiny is expected and dropped; large means the input was bad.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("sqrtm returned significant imaginary values")
        covmean = covmean.real

    d2 = mean_term + sigma_a.trace() + sigma_b.trace() - 2.0 * covmean.trace()
    # Near-identical distributions can land just below zero from rounding.
    d2 = max(d2, 0.0)
    return float(np.sqrt(d2))


@dataclass(frozen=True)
class Metric:
    """One distance, callable through a signature shared by all of them.

    Attributes:
        name: Short name used on the command line.
        uses_covariance: Whether the distance looks at spread at all.
        compute: Takes (mu_a, sigma_a, mu_b, sigma_b), returns a distance.
            Metrics that ignore covariance still accept it, so callers do not
            have to branch on which metric they picked.
    """

    name: str
    uses_covariance: bool
    compute: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]


def _centroid_from_stats(
    mu_a: np.ndarray,
    sigma_a: np.ndarray,
    mu_b: np.ndarray,
    sigma_b: np.ndarray,
) -> float:
    """Adapt centroid_distance to the shared four-argument signature."""
    return centroid_distance(mu_a, mu_b)


METRICS: dict[str, Metric] = {
    "centroid": Metric(
        name="centroid",
        uses_covariance=False,
        compute=_centroid_from_stats,
    ),
    "frechet": Metric(
        name="frechet",
        uses_covariance=True,
        compute=frechet_distance,
    ),
}


def get_metric(name: str) -> Metric:
    """Look up a distance by name.

    Raises:
        ValueError: If the name is not a registered metric.
    """
    try:
        return METRICS[name]
    except KeyError:
        available = ", ".join(sorted(METRICS))
        raise ValueError(f"Unknown metric '{name}'. Available: {available}") from None
