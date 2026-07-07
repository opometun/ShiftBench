"""Compute and save feature distribution statistics."""

import argparse

import numpy as np


def compute_mean(embeddings):
    return embeddings.mean(axis=0)


def compute_covariance(embeddings):
    return np.cov(embeddings, rowvar=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_path")
    args = parser.parse_args()
    embeddings = np.load(args.embeddings_path, mmap_mode="r")
    mu = embeddings.mean(axis=0)
    centered = embeddings - mu
    sigma = centered.T @ centered / (embeddings.shape[0] - 1)
    np.savez("feature_stats.npz", mu=mu, sigma=sigma)


if __name__ == "__main__":
    main()
