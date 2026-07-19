"""Compute and save feature distribution statistics."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.metrics import compute_covariance, compute_mean  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_path")
    args = parser.parse_args()
    embeddings = np.load(args.embeddings_path, mmap_mode="r")
    mu = compute_mean(embeddings)
    sigma = compute_covariance(embeddings)
    np.savez("feature_stats.npz", mu=mu, sigma=sigma)


if __name__ == "__main__":
    main()
