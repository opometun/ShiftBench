"""Merge per-mixture SADGE scores into distances.json.

scripts/run_sadge.py writes one JSON per mixture so array tasks never contend
over a shared file. This folds them into results/shift/distances.json under the
key 'sadge', which is what scripts/analyze_correlation.py reads.

SADGE is a similarity, not a distance -- analyze_correlation.py's
SIMILARITY_METRICS set already accounts for that when applying the expected
sign, so nothing is inverted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sadge-dir", default="results/shift/sadge")
    parser.add_argument("--distances", default="results/shift/distances.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sadge_dir = Path(args.sadge_dir)
    distances_path = Path(args.distances)

    if not distances_path.is_file():
        raise SystemExit(f"Missing {distances_path}; run run_shift_metrics.py first.")
    distances = json.loads(distances_path.read_text(encoding="utf-8"))

    merged, missing, no_components, inverted = 0, [], [], []
    for mixture in distances:
        source = sadge_dir / f"{mixture}.json"
        if not source.is_file():
            missing.append(mixture)
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        distances[mixture]["sadge"] = float(payload["sadge"])

        # Also expose the two components SADGE fuses. Both are similarities
        # (higher = the datasets are more alike) and both rank the mixtures
        # sensibly on their own, whereas the fused score does not: every
        # mixture in this study falls below SADGE's fitted calibration means
        # for appearance (0.6359) and geometry (~2812 inliers), which puts the
        # fusion term (b + c*geo_z) near or past its zero crossing at 253
        # inliers and can flip the sign of the result.
        if "app_similarity" in payload:
            distances[mixture]["sadge_appearance"] = float(payload["app_similarity"])
            distances[mixture]["sadge_geometry"] = float(payload["geo_inliers"])
            if payload.get("sign_inverted"):
                inverted.append(mixture)
            print(f"  {mixture:28s} sadge {payload['sadge']:+.4f}   "
                  f"app {payload['app_similarity']:.4f}   "
                  f"geo {payload['geo_inliers']:7.1f}"
                  f"{'   [fusion sign inverted]' if payload.get('sign_inverted') else ''}")
        else:
            no_components.append(mixture)
            print(f"  {mixture:28s} sadge {payload['sadge']:+.4f}   "
                  f"(components absent -- produced before instrumentation)")
        merged += 1

    if missing:
        print(f"\nNo SADGE score yet for: {', '.join(sorted(missing))}")
        print("Those mixtures will simply lack the metric in the correlation;")
        print("analyze_correlation.py skips metrics that are not present for all.")

    if no_components:
        print(f"\n{len(no_components)} mixture(s) lack the component fields: "
              f"{', '.join(sorted(no_components))}")
        print("Re-run them so sadge_appearance / sadge_geometry are complete:")
        print("  sbatch --export=ALL,SADGE_FORCE=1 scripts/hpc/sadge.sbatch")
        print("Until then those two metrics are dropped from the correlation,")
        print("since it only uses metrics present for every mixture.")

    if inverted:
        print(f"\nWARNING: the fused SADGE score is sign-inverted for "
              f"{len(inverted)} mixture(s):")
        print(f"  {', '.join(sorted(inverted))}")
        print("Their geometric inlier counts fall below the 253 threshold where")
        print("the fusion multiplier changes sign, so a MORE similar dataset")
        print("receives a LOWER fused score. Prefer sadge_appearance and")
        print("sadge_geometry, which rank the mixtures correctly on their own.")

    distances_path.write_text(json.dumps(distances, indent=2) + "\n", encoding="utf-8")
    print(f"\nMerged {merged} SADGE scores into {distances_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
