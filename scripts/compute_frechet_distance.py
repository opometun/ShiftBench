"""Compute Fréchet distance between two feature distributions."""

import argparse

import numpy as np
import scipy.linalg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_a_path")
    parser.add_argument("stats_b_path")
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path to write the computed Fréchet distance.",
    )
    args = parser.parse_args()
    stats_a = np.load(args.stats_a_path)
    stats_b = np.load(args.stats_b_path)
    mu_a = stats_a["mu"]
    mu_b = stats_b["mu"]
    sigma_a = stats_a["sigma"]
    sigma_b = stats_b["sigma"]
    mean_term = (mu_a - mu_b) @ (mu_a - mu_b)
    covmean = scipy.linalg.sqrtm(sigma_a @ sigma_b)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("sqrtm returned significant imaginary values")
        covmean = covmean.real
    d2 = mean_term + sigma_a.trace() + sigma_b.trace() - 2.0 * covmean.trace()
    d2 = max(d2, 0.0)
    distance = np.sqrt(d2)
    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as file:
            file.write(f"{distance}\n")
    else:
        print(distance)


if __name__ == "__main__":
    main()
