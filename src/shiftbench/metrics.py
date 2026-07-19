"""Distribution-shift metrics over frozen-encoder feature sets.

Each dataset is summarized as a Gaussian over its embeddings: a mean vector
(where the point cloud sits) and a covariance matrix (how it is spread). The
distances below compare two such summaries.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def compute_mean(embeddings: np.ndarray) -> np.ndarray:
    """Centroid of an (n_samples, n_features) embedding set."""
    return embeddings.mean(axis=0)


def compute_covariance(embeddings: np.ndarray) -> np.ndarray:
    """Feature covariance of an (n_samples, n_features) embedding set.

    rowvar=False because our variables are the columns, not the rows.
    """
    return np.cov(embeddings, rowvar=False)


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
