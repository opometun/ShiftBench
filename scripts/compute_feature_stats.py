"""Compute and save feature distribution statistics."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.metrics import compute_covariance, compute_mean  # noqa: E402
from shiftbench.provenance import UNKNOWN, read_feature_provenance  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a feature array as a mean and covariance."
    )
    parser.add_argument("embeddings_path")
    parser.add_argument(
        "output_path",
        help="Path to write the output .npz stats file.",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="Override the encoder recorded in the embeddings sidecar.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model id recorded in the embeddings sidecar.",
    )
    args = parser.parse_args()

    provenance = read_feature_provenance(args.embeddings_path)
    encoder = args.encoder or provenance.get("encoder", UNKNOWN)
    model = args.model or provenance.get("model", UNKNOWN)
    if encoder == UNKNOWN:
        print(
            f"warning: no provenance sidecar for {args.embeddings_path}; these "
            "stats cannot be checked against another encoder's",
            file=sys.stderr,
        )

    embeddings = np.load(args.embeddings_path, mmap_mode="r")
    mu = compute_mean(embeddings)
    sigma = compute_covariance(embeddings)
    np.savez(args.output_path, mu=mu, sigma=sigma, encoder=encoder, model=model)
    print(f"{args.output_path} <- {encoder} ({model})")


if __name__ == "__main__":
    main()
