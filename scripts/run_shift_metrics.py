"""Compute every shift metric for all study mixtures against the inference set.

Runs the three-step pipeline (extract -> summarise -> compare) across 9 mixture
manifests and one inference manifest, producing the x-axis of the study's
research question: pre-training shift scores to correlate against the
downstream mIoU already measured by hybrid_eval.

The existing scripts are invoked as subprocesses rather than reimplemented, so
provenance sidecars and the refuse-to-compare checks behave exactly as they do
when run by hand.

Artifacts (all under --output-dir, default results/shift):
    features/<dataset>_<encoder>.npy      frozen encoder embeddings
    summaries/<dataset>_<summary>.npz     per-dataset summaries
    distances.json                        every metric value, per mixture

Everything is resumable: an artifact that already exists is not recomputed, so
a job killed by the wall clock can simply be resubmitted.

Prerequisite: scripts/make_shift_manifests.py (writes the shift_*.csv files).

SADGE is deliberately excluded. It is pairwise over image *directories* rather
than summarisable per dataset, needs the MASt3R submodule, and per the README
has never been run. scripts/run_sadge.py covers it separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MIXES = [
    "cityscapes100",
    "cityscapes75_gta25",
    "cityscapes50_gta50",
    "cityscapes25_gta75",
    "gta100",
    "cityscapes75_synscapes25",
    "cityscapes50_synscapes50",
    "cityscapes25_synscapes75",
    "synscapes100",
]

# summary kind -> extra CLI flags for compute_feature_stats.py
HISTOGRAM_SUMMARIES = {
    "color": ["--image-column", "img_path"],
    "texture": ["--image-column", "img_path"],
    "class_frequency": ["--mask-column", "mask_path", "--num-classes", "19"],
    "class_presence": ["--mask-column", "mask_path", "--num-classes", "19"],
    "scene_complexity": ["--mask-column", "mask_path", "--num-classes", "19"],
}

# Which encoder backs which named metric, per the July project update:
# FID uses DINOv2 (appearance-focused), FCD uses StreetCLIP (semantic-focused).
ENCODERS = {"dinov2": "FID", "streetclip": "FCD"}

# metric name -> summary artifact suffix it compares
HISTOGRAM_METRICS = {
    "color_js": "color",
    "texture_js": "texture",
    "class_frequency_js": "class_frequency",
    "class_presence_js": "class_presence",
    "scene_complexity_js": "scene_complexity",
}
EMBEDDING_METRICS = ["frechet", "centroid"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", default="data/study")
    parser.add_argument("--output-dir", default="results/shift")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the GPU encoder metrics; run only label- and image-based ones.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Restrict to these mixtures (default: all nine).",
    )
    return parser.parse_args(argv)


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"  $ {printable}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"FAILED: {printable}")


def capture(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"FAILED: {' '.join(command)}")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    study = Path(args.study_dir)
    out = Path(args.output_dir)
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "summaries").mkdir(parents=True, exist_ok=True)

    mixes = args.only or MIXES
    datasets = {"inference": study / "shift_inference.csv"}
    for mix in mixes:
        datasets[mix] = study / f"shift_{mix}_train.csv"
    for name, manifest in datasets.items():
        if not manifest.is_file():
            raise SystemExit(
                f"Missing {manifest}. Run scripts/make_shift_manifests.py first."
            )

    print(f"=== summarising {len(datasets)} datasets ===", flush=True)
    for name, manifest in datasets.items():
        print(f"[{name}]", flush=True)
        for summary, flags in HISTOGRAM_SUMMARIES.items():
            target = out / "summaries" / f"{name}_{summary}.npz"
            if target.exists():
                print(f"  skip {target.name} (exists)", flush=True)
                continue
            run([sys.executable, "scripts/compute_feature_stats.py",
                 str(manifest), str(target), "--summary", summary, *flags])

        if args.skip_embeddings:
            continue
        for encoder in ENCODERS:
            features = out / "features" / f"{name}_{encoder}.npy"
            if not features.exists():
                run([sys.executable, "scripts/extract_features.py", str(features),
                     "--manifest", str(manifest), "--encoder", encoder,
                     "--image-column", "img_path"])
            target = out / "summaries" / f"{name}_gaussian_{encoder}.npz"
            if target.exists():
                print(f"  skip {target.name} (exists)", flush=True)
                continue
            run([sys.executable, "scripts/compute_feature_stats.py",
                 str(features), str(target), "--summary", "gaussian"])

    print("\n=== computing distances against the inference set ===", flush=True)
    results: dict[str, dict[str, float]] = {}
    for mix in mixes:
        print(f"[{mix}]", flush=True)
        row: dict[str, float] = {}
        for metric, summary in HISTOGRAM_METRICS.items():
            value = capture([sys.executable, "scripts/compute_distance.py",
                             str(out / "summaries" / f"{mix}_{summary}.npz"),
                             str(out / "summaries" / f"inference_{summary}.npz"),
                             "--metric", metric])
            row[metric] = float(value)
            print(f"  {metric:24s} {float(value):.6f}", flush=True)

        if not args.skip_embeddings:
            for encoder, label in ENCODERS.items():
                for metric in EMBEDDING_METRICS:
                    value = capture([
                        sys.executable, "scripts/compute_distance.py",
                        str(out / "summaries" / f"{mix}_gaussian_{encoder}.npz"),
                        str(out / "summaries" / f"inference_gaussian_{encoder}.npz"),
                        "--metric", metric])
                    key = f"{metric}_{encoder}"
                    row[key] = float(value)
                    tag = f"({label})" if metric == "frechet" else ""
                    print(f"  {key:24s} {float(value):.6f} {tag}", flush=True)
        results[mix] = row

    destination = out / "distances.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
