"""Preparing images (inputs and masks), as well as dataset .csv for an experiment."""

import csv
import random
from pathlib import Path
from PIL import Image

from shiftbench.datasets.manifest import load_image_paths


INTERPOLATION_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def resize_images(
    input_manifest: str,
    image_column: str | None = None, 
    target_resolution: tuple[int, int] = (1440, 720),
    interpolation_method: str = "nearest", 
) -> None:
    """Resize the input images and segmentation masks to a uniform target resolution.

    Args:
        input_manifest: Path to the CSV manifest.
        image_column: Column holding image paths.
        target_resolution: The resolution to be shared among data samples. Default is the resolution of Synscapes images.
        interpolation_method: Method used for interpolation. Recommendation: use 'nearest' for segmentation masks and 'lanczos' for input images.
    """
    target_w, target_h = target_resolution
    interp = INTERPOLATION_MAP[interpolation_method]
    image_paths = load_image_paths(input_manifest, image_column)

    for path in image_paths:
        img = Image.open(path)
        w, h = img.size

        if (w, h) == (target_w, target_h):
            continue

        # Check aspect ratio
        aspect_ratio_matches = abs((w / h) - (target_w / target_h)) < 1e-3

        if aspect_ratio_matches:
            img = img.resize(target_resolution, interp)

        else:
            # Resize preserving aspect ratio
            scale_w = target_w / w
            scale_h = target_h / h
            scale = max(scale_w, scale_h)

            new_w = round(w * scale)
            new_h = round(h * scale)

            if (new_w < target_w) or (new_h < target_h):
                raise ValueError(f"Resized image ({new_w}, {new_h}) does not fill target box ({target_w}, {target_h}).")

            img = img.resize((new_w, new_h), interp)

            # Center crop
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h

            img = img.crop((left, top, right, bottom))

        img.save(path)


def collect_pairs(
    root: Path, 
    source: str, 
    split: str,
) -> list[dict]:
    """Collect (img, mask) pairs from a dataset split. 
    
    Requirements:
        - both input and mask image of a pair must share the exact same filename & must be .png files
        - input images need to be stored under this path structure: `root/split/source/img/` (e.g. `streetViewData/train/cityscapes/img/0001.png`)
        - masks need to be stored under this path structure: `root/split/source/mask/` (e.g. `streetViewData/train/cityscapes/mask/0001.png`)
    """
    img_dir = root / split / source / "img"
    mask_dir = root / split / source / "mask"

    # Directory existence checks
    if not img_dir.exists():
        raise ValueError(
            f"Image directory does not exist: {img_dir}. "
            f"Expected structure: root/{split}/{source}/img/"
        )

    if not mask_dir.exists():
        raise ValueError(
            f"Mask directory does not exist: {mask_dir}. "
            f"Expected structure: root/{split}/{source}/mask/"
        )

    # PNG-only image files
    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        raise ValueError(
            f"No PNG images found in directory: {img_dir}. "
            f"Expected PNG files for dataset '{source}' split '{split}'."
        )

    pairs = []
    for img_path in img_files:

        # Enforce PNG for images
        if img_path.suffix.lower() != ".png":
            raise ValueError(f"Image file is not a PNG: {img_path}")

        mask_path = mask_dir / img_path.name

        # Mask existence
        if not mask_path.exists():
            raise ValueError(
                f"Missing mask for image '{img_path.name}' in dataset '{source}' split '{split}'. "
                f"Expected mask at: {mask_path}"
            )

        # Enforce PNG for masks
        if mask_path.suffix.lower() != ".png":
            raise ValueError(f"Mask file for '{img_path.name}' is not a PNG: {mask_path}")

        pairs.append({
            "img_path": str(img_path),
            "mask_path": str(mask_path),
            "source_dataset": source,
            "split": split,
        })

    return pairs


def build_street_view_dataset_csv(
    datasets_root: str,
    output_csv: str,
    train_selection: dict,
    seed: int = 42,
) -> None:
    """
    Build street view dataset CSV with arbitrary mixture ratios.

    Args:
        datasets_root: Path to root folder (e.g. `../data/streetViewData/`).
        output_csv: Path to output CSV file.
        train_selection: dict mapping source -> number of samples, e.g.:
            {"cityscapes": 1500, "synscapes": 500}
        seed: Random seed for reproducibility.
    """
    random.seed(seed)

    root = Path(datasets_root)

    # Always include ALL validation + test Cityscapes
    val_pairs = collect_pairs(root, "cityscapes", "validation")
    test_pairs = collect_pairs(root, "cityscapes", "test")

    # Collect train samples per dataset
    train_pairs = []
    for source, count in train_selection.items():
        all_pairs = collect_pairs(root, source, "train")
        if count > len(all_pairs):
            raise ValueError(
                f"Requested {count} samples from {source}, but only {len(all_pairs)} available."
            )
        selected = random.sample(all_pairs, count)
        train_pairs.extend(selected)

    # Combine rows
    rows = train_pairs + val_pairs + test_pairs

    # Add sample_id
    for i, row in enumerate(rows, start=1):
        row["sample_id"] = i

    # Write CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "source_dataset",
                "mask_path",
                "img_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
