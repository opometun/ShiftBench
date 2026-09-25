#!/usr/bin/env python3
"""Create a figure with one representative sample from each dataset.

Behavior:
- If the local study training folders exist, it uses the first image under
  data/study/streetViewData/train/<dataset>/img.
- Otherwise it falls back to the tracked CSV manifests.
- If neither source exists in this checkout, it creates synthetic placeholder
  examples so the figure still renders in CI or on a minimal repo clone.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATASET_ORDER = ("cityscapes", "synscapes", "gtaV")
DATASET_LABELS = {
    "cityscapes": "Cityscapes",
    "synscapes": "Synscapes",
    "gtaV": "GTA-V",
}
DEFAULT_MANIFESTS = {
    "cityscapes": Path("data/study/cityscapes100.csv"),
    "synscapes": Path("data/study/synscapes100.csv"),
    "gtaV": Path("data/study/gta100.csv"),
}


def _resolve_relative_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path.strip().replace("\\", "/")).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _create_placeholder_sample(dataset: str, output_path: Path) -> Path:
    """Build a stylized placeholder image for a dataset when no real sample exists."""
    width, height = 640, 360
    img = Image.new("RGB", (width, height), "white")

    if dataset == "cityscapes":
        sky = (153, 196, 220)
        road = (90, 90, 90)
        building = (180, 180, 180)
        accent = (90, 120, 165)
    elif dataset == "synscapes":
        sky = (172, 214, 237)
        road = (145, 145, 145)
        building = (108, 149, 103)
        accent = (86, 122, 72)
    else:
        sky = (197, 190, 156)
        road = (80, 72, 68)
        building = (143, 107, 74)
        accent = (197, 162, 90)

    for y in range(height):
        for x in range(width):
            if y < int(height * 0.45):
                img.putpixel((x, y), sky)
            else:
                img.putpixel((x, y), road)

    # Add stylized buildings and a road divider.
    for x in range(0, width, 90):
        building_w = 50
        building_h = int(0.75 * (height * 0.45))
        y0 = int(height * 0.45) - building_h
        for px in range(x, min(x + building_w, width)):
            for py in range(y0, int(height * 0.45)):
                if px < width and py >= 0:
                    img.putpixel((px, py), building)

    for x in range(0, width, 180):
        for y in range(int(height * 0.45), height):
            if (x // 30 + y // 20) % 2 == 0:
                img.putpixel((x, y), accent)

    # Add a simple road center line.
    for y in range(int(height * 0.45), height):
        if y % 12 in (0, 1, 2):
            for x in range(width // 2 - 4, width // 2 + 4):
                if 0 <= x < width:
                    img.putpixel((x, y), (240, 240, 240))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _candidate_image_paths_for_dataset(root: Path, dataset: str) -> list[Path]:
    image_dir = root / dataset / "img"
    if not image_dir.exists():
        return []
    return sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=lambda path: path.name.lower(),
    )


def find_dataset_examples(data_root: str | Path | None = None) -> dict[str, Path]:
    """Return one representative image path for each dataset."""
    project_root = Path(__file__).resolve().parents[2]
    root = Path(data_root).resolve() if data_root is not None else None

    examples: dict[str, Path] = {}

    if root is not None and root.exists():
        for dataset in DATASET_ORDER:
            candidates = _candidate_image_paths_for_dataset(root, dataset)
            if not candidates:
                continue
            examples[dataset] = candidates[0]
        if len(examples) == len(DATASET_ORDER):
            return examples

    for dataset, manifest_rel in DEFAULT_MANIFESTS.items():
        manifest = (project_root / manifest_rel).resolve()
        if not manifest.exists():
            continue

        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source = (row.get("source_dataset") or "").strip()
                if source != dataset:
                    continue
                image_value = row.get("img_path")
                if not image_value:
                    continue
                image_path = _resolve_relative_path(manifest.parent, image_value)
                if image_path.exists():
                    examples[dataset] = image_path
                    break

    for dataset in DATASET_ORDER:
        if dataset not in examples:
            placeholder_path = project_root / "results" / "figures" / "placeholder" / f"{dataset}.png"
            examples[dataset] = _create_placeholder_sample(dataset, placeholder_path)

    return examples


def make_figure(example_paths: dict[str, Path], output_path: str | Path) -> Path:
    """Create a 1 x 3 figure with one sample from each dataset."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, dataset in zip(axes, DATASET_ORDER):
        image = Image.open(example_paths[dataset]).convert("RGB")
        ax.imshow(image)
        ax.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold")
        ax.set_axis_off()

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a figure showing one sample per dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/study/streetViewData/train"),
        help="Optional root containing train/<dataset>/img folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/representative_dataset_samples.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    examples = find_dataset_examples(args.data_root)
    output_path = make_figure(examples, args.output)
    print(f"Wrote representative dataset figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
