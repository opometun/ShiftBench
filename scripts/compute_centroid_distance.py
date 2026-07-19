"""Compute centroid distance between two feature distributions."""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_a_path")
    parser.add_argument("stats_b_path")
    args = parser.parse_args()
    stats_a = np.load(args.stats_a_path)
    stats_b = np.load(args.stats_b_path)
    mu_a = stats_a["mu"]
    mu_b = stats_b["mu"]
    distance = np.linalg.norm(mu_a - mu_b)
    print(distance)


if __name__ == "__main__":
    main()
