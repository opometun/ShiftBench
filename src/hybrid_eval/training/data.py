"""Dataset and preprocessing utilities for semantic segmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS: Final = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMG_MEAN: Final = [0.485, 0.456, 0.406]
IMG_STD: Final = [0.229, 0.224, 0.225]


def _image_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {root}")
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported images found in directory: {root}")
    return files


def _mask_path(mask_dir: Path, image_path: Path) -> Path:
    exact = mask_dir / image_path.name
    if exact.is_file():
        return exact
    png = mask_dir / f"{image_path.stem}.png"
    if png.is_file():
        return png
    raise FileNotFoundError(f"No mask found for {image_path.name} in {mask_dir}")


class JointTransform:
    """Apply synchronized geometric transforms to an image and label mask."""

    def __init__(
        self,
        image_size: tuple[int, int] = (512, 1024),
        crop_size: int | None = 512,
        is_train: bool = True,
        scale_range: tuple[float, float] = (1.0, 1.0),
        photometric_distortion: bool = False,
    ) -> None:
        self.image_size = image_size
        self.crop_size = crop_size
        self.is_train = is_train
        self.scale_range = scale_range
        self.photometric_distortion = photometric_distortion
        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        if min(image_size) <= 0:
            raise ValueError("image_size dimensions must be positive")
        if crop_size is not None and (crop_size <= 0 or crop_size > min(image_size)):
            raise ValueError("crop_size must be positive and no larger than image_size")
        if scale_range[0] <= 0 or scale_range[1] < scale_range[0]:
            raise ValueError("scale_range must contain positive, increasing values")

    def __call__(self, image: Image.Image, mask: Image.Image | None = None):
        scale = random.uniform(*self.scale_range) if self.is_train else 1.0
        resized = tuple(
            max(1, round(dimension * scale)) for dimension in self.image_size
        )
        image = TF.resize(image, resized, antialias=True)
        if mask is not None:
            mask = TF.resize(mask, resized, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            if mask is None:
                raise ValueError("Training transforms require a segmentation mask")
            if self.photometric_distortion:
                image = self.color_jitter(image)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if self.crop_size is not None:
                width, height = image.size
                pad_right = max(0, self.crop_size - width)
                pad_bottom = max(0, self.crop_size - height)
                if pad_right or pad_bottom:
                    padding = [0, 0, pad_right, pad_bottom]
                    image = TF.pad(image, padding, fill=0)
                    mask = TF.pad(mask, padding, fill=255)
                top, left, height, width = T.RandomCrop.get_params(
                    image,
                    output_size=(self.crop_size, self.crop_size),
                )
                image = TF.crop(image, top, left, height, width)
                mask = TF.crop(mask, top, left, height, width)

        image_tensor = TF.normalize(TF.to_tensor(image), mean=IMG_MEAN, std=IMG_STD)
        if mask is None:
            return image_tensor
        mask_tensor = torch.as_tensor(np.array(mask, copy=True), dtype=torch.long)
        if mask_tensor.ndim != 2:
            raise ValueError(
                "Segmentation masks must contain one integer class ID per pixel"
            )
        return image_tensor, mask_tensor


class SimpleSegmentationDataset(Dataset):
    """Load paired image/mask files for training and validation."""

    def __init__(
        self, img_dir: str | Path, mask_dir: str | Path, transform=None
    ) -> None:
        self.images = _image_files(img_dir)
        self.mask_dir = Path(mask_dir)
        if not self.mask_dir.is_dir():
            raise NotADirectoryError(f"Mask directory does not exist: {self.mask_dir}")
        self.masks = [_mask_path(self.mask_dir, image) for image in self.images]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        if self.transform is not None:
            return self.transform(image, mask)
        image_tensor = TF.normalize(TF.to_tensor(image), mean=IMG_MEAN, std=IMG_STD)
        mask_tensor = torch.as_tensor(np.array(mask, copy=True), dtype=torch.long)
        return image_tensor, mask_tensor


class InferenceImageDataset(Dataset):
    """Load images for prediction, optionally with masks for evaluation."""

    def __init__(
        self,
        img_dir: str | Path,
        transform: JointTransform,
        mask_dir: str | Path | None = None,
    ) -> None:
        self.images = _image_files(img_dir)
        self.transform = transform
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        if self.mask_dir is not None:
            if not self.mask_dir.is_dir():
                raise NotADirectoryError(
                    f"Mask directory does not exist: {self.mask_dir}"
                )
            self.masks = [_mask_path(self.mask_dir, image) for image in self.images]
        else:
            self.masks = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.masks is None:
            return self.transform(image), image_path.name
        mask = Image.open(self.masks[idx])
        image_tensor, mask_tensor = self.transform(image, mask)
        return image_tensor, mask_tensor, image_path.name
