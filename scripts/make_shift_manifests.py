"""Emit split-filtered manifests for shift quantification.

The study CSVs in data/study/ hold every split at once (2,000 train rows plus
500 validation and 975 test rows). The metric pipeline does not filter by
split -- shiftbench.datasets.manifest.load_image_paths reads every row it is
given -- so pointing a metric at cityscapes100.csv would summarise a 3,475-row
mixture rather than the 2,000-image training set.

The research question asks for the shift between a *hybrid training dataset*
and the *real inference dataset*, so this writes:

  shift_<mix>_train.csv   the 2,000 train rows of each mixture
  shift_inference.csv     the 975 held-out real Cityscapes test rows

Outputs land in data/study/ alongside the source CSVs, because manifest paths
are resolved relative to the manifest's own directory -- writing them to a
subdirectory would silently break every path.

The inference split is taken from cityscapes100.csv, but every study CSV
carries the identical validation/test rows (they differ only in their train
composition); --check-consistency verifies that rather than assuming it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

STUDY_MIXES = [
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        default="data/study",
        help="Directory holding the study CSVs (default: data/study)",
    )
    parser.add_argument(
        "--inference-split",
        default="test",
        help=(
            "Split to use as the real inference distribution. 'test' is the "
            "975 held-out Cityscapes images; 'validation' is the 500 used for "
            "early stopping during training."
        ),
    )
    parser.add_argument("--split-column", default="split")
    parser.add_argument(
        "--check-consistency",
        action="store_true",
        help="Verify every mix carries an identical inference split.",
    )
    return parser.parse_args(argv)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    study_dir = Path(args.study_dir)
    if not study_dir.is_dir():
        raise SystemExit(f"Study directory not found: {study_dir}")

    inference_signature: set[str] | None = None

    for mix in STUDY_MIXES:
        source = study_dir / f"{mix}.csv"
        if not source.is_file():
            raise SystemExit(f"Missing study CSV: {source}")
        fieldnames, rows = _read(source)

        train_rows = [r for r in rows if r[args.split_column].strip() == "train"]
        infer_rows = [
            r for r in rows if r[args.split_column].strip() == args.inference_split
        ]
        if not train_rows:
            raise SystemExit(f"{source} has no 'train' rows")
        if not infer_rows:
            raise SystemExit(f"{source} has no '{args.inference_split}' rows")

        _write(study_dir / f"shift_{mix}_train.csv", fieldnames, train_rows)
        print(f"shift_{mix}_train.csv: {len(train_rows)} train rows")

        signature = {r["img_path"] for r in infer_rows}
        if inference_signature is None:
            inference_signature = signature
            _write(study_dir / "shift_inference.csv", fieldnames, infer_rows)
            print(
                f"shift_inference.csv: {len(infer_rows)} "
                f"{args.inference_split} rows (from {mix})"
            )
        elif args.check_consistency and signature != inference_signature:
            raise SystemExit(
                f"{mix} has a different {args.inference_split} split than the "
                "first mix; the inference distribution must be identical "
                "across mixes or the shift scores are not comparable."
            )

    print("\nDone. Every mix now has a train-only manifest, plus one shared")
    print(f"inference manifest built from the '{args.inference_split}' split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
