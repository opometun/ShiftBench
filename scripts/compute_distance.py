"""Compute a distance between two feature distributions."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.metrics import METRICS, get_metric  # noqa: E402
from shiftbench.provenance import (  # noqa: E402
    UNKNOWN,
    describe_summary,
    require_comparable,
)


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
        help="Path to .txt file to write the computed distance. Defaults to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_path is not None:
        output_path = Path(args.output_path)
        if output_path.suffix != ".txt":
            output_path = output_path.with_suffix(".txt")
        if output_path.exists():
            warnings.warn(
                f"Output file already exists and will be overwritten: {output_path}",
                category=UserWarning,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    metric = get_metric(args.metric)

    if metric.compare is None:
        raise ValueError(
            f"Metric '{metric.name}' is pairwise and does not compare "
            "precomputed summaries"
        )

    stats_a = np.load(args.stats_a_path)
    stats_b = np.load(args.stats_b_path)
    require_comparable(args.stats_a_path, stats_a, args.stats_b_path, stats_b)

    kind, _ = describe_summary(stats_a)
    if kind != UNKNOWN and kind != metric.summary:
        raise ValueError(
            f"Metric '{metric.name}' expects '{metric.summary}' summaries, "
            f"these files hold '{kind}'"
        )

    distance = metric.compare(stats_a, stats_b)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(f"{distance}\n")
    else:
        print(distance)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
