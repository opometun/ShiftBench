"""Materialize a hybrid-mix study CSV's train split into flat img/mask dirs.

`hybrid_eval.training.train` (the SegFormer/DeepLabV3+ training entry point)
takes a single flat `--train-img-dir` / `--train-mask-dir` pair and globs every
supported image file in them; it has no notion of ShiftBench's CSV manifests.
The study CSVs in `data/study/*.csv`, on the other hand, mix two source
datasets per row (e.g. cityscapes50_gta50.csv references both
`streetViewData/train/cityscapes/...` and `streetViewData/train/gtaV/...`).

This script bridges the two: it reads a study CSV, resolves every train-split
row's image/mask path (reusing shiftbench's manifest resolver, so the
Windows-style backslash paths in the tracked CSVs resolve correctly on
POSIX), and symlinks each pair into `<output>/img/<sample_id><ext>` and
`<output>/mask/<sample_id><ext>`.

Renaming to `sample_id` (rather than keeping the original basename) is not
cosmetic: GTA-V and Synscapes both use short numeric filenames
(`00451.png`, `377.png`, ...), so merging their original basenames into one
flat directory risks silent collisions between sources. `sample_id` is
unique within a study CSV by construction (see shiftbench.datasets.schema's
duplicate-id check), so this is collision-free.

Validation and test splits are not handled here: every study config's
non-train rows are 100% Cityscapes (the held-out real inference data), so
they can be pointed at `streetViewData/validation/cityscapes/{img,mask}` and
`streetViewData/test/cityscapes/{img,mask}` directly, with no mixing and
therefore no need to materialize anything.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.datasets.manifest import resolve_image_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Study CSV, e.g. data/study/cityscapes50_gta50.csv")
    parser.add_argument(
        "output_dir",
        help="Destination root; img/ and mask/ subdirectories are created under it",
    )
    parser.add_argument(
        "--split-column", default="split", help="Column naming the split (default: split)"
    )
    parser.add_argument(
        "--split-value", default="train", help="Which split value to materialize (default: train)"
    )
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--image-column", default="img_path")
    parser.add_argument("--mask-column", default="mask_path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing symlinks instead of skipping them",
    )
    return parser.parse_args(argv)


def _link(source: Path, destination: Path, force: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        if not force:
            return
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        # Filesystems without symlink support (some network mounts, Windows
        # without privilege) fall back to a hardlink, then a copy.
        try:
            os.link(source, destination)
        except OSError:
            import shutil

            shutil.copy2(source, destination)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv_path).resolve()
    manifest_dir = str(csv_path.parent)
    output_dir = Path(args.output_dir)
    img_dir = output_dir / "img"
    mask_dir = output_dir / "mask"

    linked = 0
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = {
            args.split_column,
            args.id_column,
            args.image_column,
            args.mask_column,
        }.difference(reader.fieldnames or ())
        if missing_columns:
            raise SystemExit(
                f"{csv_path} is missing columns: {', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            if row[args.split_column].strip() != args.split_value:
                continue
            sample_id = row[args.id_column].strip()
            image_source = Path(
                resolve_image_path(manifest_dir, row[args.image_column])
            )
            mask_source = Path(
                resolve_image_path(manifest_dir, row[args.mask_column])
            )
            _link(image_source, img_dir / f"{sample_id}{image_source.suffix}", args.force)
            _link(mask_source, mask_dir / f"{sample_id}{mask_source.suffix}", args.force)
            linked += 1

    if linked == 0:
        raise SystemExit(
            f"No rows with {args.split_column}={args.split_value!r} found in {csv_path}"
        )
    print(f"Linked {linked} {args.split_value} pairs from {csv_path.name} into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
