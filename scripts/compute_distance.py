"""Compute a distance between two feature distributions."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.metrics import METRICS, get_metric  # noqa: E402
from shiftbench.provenance import require_same_encoder  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two feature distribution stats files."
    )
    parser.add_argument("stats_a_path")
    parser.add_argument("stats_b_path")
    parser.add_argument(
        "--metric",
        required=True,
        choices=sorted(METRICS),
        help=(
            "Which distance to compute. Required because the metrics measure "
            "different things and the result does not say which one it is."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path to write the computed distance. Defaults to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metric = get_metric(args.metric)

    stats_a = np.load(args.stats_a_path)
    stats_b = np.load(args.stats_b_path)
    require_same_encoder(args.stats_a_path, stats_a, args.stats_b_path, stats_b)

    distance = metric.compute(
        stats_a["mu"],
        stats_a["sigma"],
        stats_b["mu"],
        stats_b["sigma"],
    )

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as file:
            file.write(f"{distance}\n")
    else:
        print(distance)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
