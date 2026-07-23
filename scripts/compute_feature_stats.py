"""Summarize a dataset as a small stored artifact for distance computation.

Gaussian summaries consume an embeddings .npy; histogram summaries consume a
CSV manifest and decode the images or masks it lists. Either way the output is
a small .npz recording what produced it, so mismatched artifacts refuse to
compare later instead of yielding a plausible wrong number.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.metrics import SUMMARIES, get_summary  # noqa: E402
from shiftbench.provenance import UNKNOWN, read_feature_provenance  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input_path",
        help="Embeddings .npy (gaussian) or CSV manifest (histogram summaries).",
    )
    parser.add_argument(
        "output_path",
        help="Path to write the output .npz stats file.",
    )
    parser.add_argument(
        "--summary",
        default="gaussian",
        choices=sorted(SUMMARIES),
        help="How to reduce the dataset to a stored summary.",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="gaussian only: override the encoder recorded in the sidecar.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="gaussian only: override the model id recorded in the sidecar.",
    )
    parser.add_argument(
        "--image-column",
        default=None,
        help="image summaries: manifest column holding image paths.",
    )
    parser.add_argument(
        "--mask-column",
        default=None,
        help="mask summaries: manifest column holding mask paths (required).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="mask summaries: class count the masks are labeled with (required).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = get_summary(args.summary)
    params = dict(summary.default_params)

    if summary.input == "embeddings":
        provenance = read_feature_provenance(args.input_path)
        encoder = args.encoder or provenance.get("encoder", UNKNOWN)
        model = args.model or provenance.get("model", UNKNOWN)
        if encoder == UNKNOWN:
            print(
                f"warning: no provenance sidecar for {args.input_path}; these "
                "stats cannot be checked against another encoder's",
                file=sys.stderr,
            )
        embeddings = np.load(args.input_path, mmap_mode="r")
        data = summary.make(embeddings)
    elif summary.input == "images":
        from shiftbench.datasets.loaders import load_rgb_images
        from shiftbench.datasets.manifest import load_image_paths

        encoder = model = "n/a"
        images = load_rgb_images(load_image_paths(args.input_path, args.image_column))
        data = summary.make(images)
    else:  # masks
        if args.mask_column is None or args.num_classes is None:
            raise ValueError(
                f"summary '{summary.name}' needs --mask-column and --num-classes"
            )

        from shiftbench.datasets.loaders import load_masks
        from shiftbench.datasets.manifest import load_mask_paths

        encoder = model = "n/a"
        params["num_classes"] = args.num_classes
        masks = load_masks(load_mask_paths(args.input_path, args.mask_column))
        data = summary.make(masks, num_classes=args.num_classes)

    np.savez(
        args.output_path,
        **data,
        encoder=encoder,
        model=model,
        summary=summary.name,
        params=json.dumps(params, sort_keys=True),
    )
    print(f"{args.output_path} <- {summary.name} summary ({encoder} {model})")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
