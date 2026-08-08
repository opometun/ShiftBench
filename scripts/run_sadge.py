"""Compute the SADGE score for ONE mixture against the real inference set.

SADGE is the benchmark the study's other metrics are measured against. It is
pairwise rather than summarisable: it samples K training images per inference
image and scores each pair with MASt3R geometry plus DINOv3 appearance, so
both datasets must be present at once and nothing lands in the .npz artifact
path the other metrics use.

One mixture per invocation, so scripts/hpc/sadge.sbatch can run them as an
array -- nine tasks of ~1-1.5h each rather than one 9-13h job that loses
everything if it fails.

Cost: 975 inference images x K=10 candidates = 9,750 MASt3R forward passes per
mixture. --max-queries subsamples the inference set if that is too slow; the
estimate gets noisier but stays unbiased.

Unlike every other metric here SADGE is a SIMILARITY: a HIGHER score means the
datasets are more alike. scripts/analyze_correlation.py knows this via
SIMILARITY_METRICS and flips the sign when ranking predictors.

Writes results/shift/sadge/<mixture>.json. Merge with scripts/merge_sadge.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixture", help="e.g. gta100")
    parser.add_argument(
        "--materialized-root",
        default="data/study/streetViewData_materialized",
        help="Holds <mixture>/img, written by scripts/materialize_train_split.py",
    )
    parser.add_argument(
        "--inference-dir",
        default="data/study/streetViewData/test/cityscapes/img",
        help="Real inference images; must match the split used for the other metrics",
    )
    parser.add_argument("--output-dir", default="results/shift/sadge")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help=(
            "Use only this many inference images. Cuts runtime proportionally "
            "at the cost of a noisier estimate. Applies the same subsample to "
            "every mixture (fixed seed), so scores stay comparable."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "SADGE samples K candidates per query at random. metric.py defines "
            "set_seed but quantify_benchmark_shift never calls it, so without "
            "this the score is not reproducible between runs."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_dir = Path(args.materialized_root) / args.mixture / "img"
    inference_dir = Path(args.inference_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{args.mixture}.json"

    if destination.exists() and not args.force:
        print(f"{destination} exists; use --force to recompute")
        return 0
    for directory in (train_dir, inference_dir):
        if not directory.is_dir():
            raise SystemExit(
                f"Missing {directory}. For the training side run:\n"
                f"  python scripts/materialize_train_split.py "
                f"data/study/{args.mixture}.csv "
                f"{Path(args.materialized_root) / args.mixture}"
            )

    n_train = len(list(train_dir.glob("*.png")))
    n_infer = len(list(inference_dir.glob("*.png")))
    print(f"mixture={args.mixture}  train={n_train} imgs  inference={n_infer} imgs")

    # Seed before anything samples. SADGE's own set_seed is never invoked by
    # quantify_benchmark_shift, so this is what makes the run reproducible.
    from shiftbench.shift_quantification_metrics.representation_based.sadge.metric import (
        set_seed,
    )

    set_seed(args.seed)

    # Subsampling is done by pointing SADGE at a directory of symlinks rather
    # than by patching its query loop, so the upstream code stays untouched.
    query_dir = inference_dir
    if args.max_queries is not None and args.max_queries < n_infer:
        query_dir = out_dir / f"_queries_{args.max_queries}"
        if not query_dir.is_dir():
            query_dir.mkdir(parents=True, exist_ok=True)
            chosen = random.Random(args.seed).sample(
                sorted(inference_dir.glob("*.png")), args.max_queries
            )
            for path in chosen:
                link = query_dir / path.name
                if not link.exists():
                    link.symlink_to(path.resolve())
        print(f"subsampled queries: {len(list(query_dir.glob('*.png')))}")

    # quantify_benchmark_shift returns only the fused score, which is not enough
    # to interpret it. SADGE reduces to A_z * (b + c*G_z) because its fitted 'a'
    # is 0, so the multiplier -- and hence the SIGN of the whole metric --
    # inverts once G_z < -b/c (about 253 geometric inliers, against a
    # calibration mean near 2,812). Without G and A there is no way to tell an
    # inverted score from a genuinely dissimilar dataset. Calling the internals
    # directly records both raw components and their z-scores.
    from shiftbench.metrics.pairwise import SADGE_PARAMS
    from shiftbench.features.device import get_device
    from shiftbench.shift_quantification_metrics.representation_based.sadge.metric import (
        SADGE,
        build_query_candidates,
        run_metrics,
    )

    started = time.time()
    device = get_device()
    query_candidates = build_query_candidates(train_dir, query_dir, K=SADGE_PARAMS["K"])
    geo_inliers, app_similarity = run_metrics(query_candidates, device)
    fusion = SADGE()
    geo_z, app_z = fusion._z_transform(geo_inliers, app_similarity)
    score = fusion(geo_inliers, app_similarity)
    elapsed = time.time() - started

    b, c = fusion.params[1], fusion.params[2]
    multiplier = b + c * geo_z
    # Cast explicitly: _z_transform goes through np.log1p, so these are numpy
    # scalars. np.float64 subclasses float and serialises fine, but np.bool_
    # does NOT subclass bool and json.dumps rejects it with
    # "Object of type bool is not JSON serializable" -- after the full 39-minute
    # computation has already run.
    inverted = bool(multiplier < 0)

    payload = {
        "mixture": args.mixture,
        "sadge": float(score),
        # Raw dataset-level components, before fusion.
        "geo_inliers": float(geo_inliers),
        "app_similarity": float(app_similarity),
        # z-scores against SADGE's fitted constants, and the resulting sign.
        "geo_z": float(geo_z),
        "app_z": float(app_z),
        "fusion_multiplier": float(multiplier),
        "sign_inverted": inverted,
        "calibration": {
            "geo_mean": fusion.geo_mean, "geo_std": fusion.geo_std,
            "app_mean": fusion.app_mean, "app_std": fusion.app_std,
            "params_abc": list(fusion.params),
            "inlier_count_at_calibration_mean": float(np.expm1(fusion.geo_mean)),
            "inlier_count_where_sign_flips": float(
                np.expm1(-b / c * fusion.geo_std + fusion.geo_mean)
            ),
        },
        "elapsed_seconds": elapsed,
        "train_images": n_train,
        "query_images": len(list(query_dir.glob("*.png"))),
        "seed": args.seed,
        "params": SADGE_PARAMS,
    }
    if inverted:
        print(
            f"WARNING: fusion multiplier is {multiplier:.4f} (negative).\n"
            f"  Mean best-inlier count {geo_inliers:.1f} is far below SADGE's\n"
            f"  calibration regime, so the fused score is SIGN-INVERTED:\n"
            f"  more similar datasets receive LOWER scores. Use geo_inliers and\n"
            f"  app_similarity directly rather than the fused value."
        )
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"SADGE({args.mixture}) = {score:.6f}   [{elapsed/60:.1f} min]")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
