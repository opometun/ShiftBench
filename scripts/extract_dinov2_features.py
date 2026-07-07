"""Scaffold for frozen DINOv2 feature extraction."""
import argparse
import csv
import os

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
IMAGE_PATH_COLUMNS = ("image_path", "file_path", "filepath", "path", "image")
BATCH_SIZE = 32


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frozen_dinov2(device):

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME)

    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return processor, model


def extract_features(processor, model, image_batch, device):
    inputs = processor(images=image_batch, return_tensors="pt")
    inputs.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen DINOv2 CLS features.")
    parser.add_argument(
        "input_manifest",
        help="Path to the input dataset or mixture manifest.",
    )
    parser.add_argument(
        "output_path",
        help="Path to write the output .npy feature array.",
    )
    return parser.parse_args()


def load_image_paths(input_manifest):
    manifest_path = os.path.abspath(os.path.expanduser(input_manifest))
    manifest_dir = os.path.dirname(manifest_path)

    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input manifest is empty: {input_manifest}") from exc

        column_by_name = {name.strip(): index for index, name in enumerate(header)}
        image_column = next(
            (
                column_by_name[column_name]
                for column_name in IMAGE_PATH_COLUMNS
                if column_name in column_by_name
            ),
            None,
        )
        if image_column is None:
            expected = ", ".join(IMAGE_PATH_COLUMNS)
            raise ValueError(
                f"Input manifest must contain an image path column: {expected}"
            )

        image_paths = []
        for row_number, row in enumerate(reader, start=2):
            if image_column >= len(row):
                raise ValueError(f"Row {row_number}: missing image path")
            image_path = os.path.expanduser(row[image_column].strip())
            if not image_path:
                raise ValueError(f"Row {row_number}: missing image path")
            if not os.path.isabs(image_path):
                image_path = os.path.join(manifest_dir, image_path)
            image_paths.append(image_path)

    return image_paths


def main():
    args = parse_args()
    device = get_device()
    processor, model = load_frozen_dinov2(device)

    image_paths = load_image_paths(args.input_manifest)
    feature_batches = []
    for start_index in range(0, len(image_paths), BATCH_SIZE):
        image_batch = []
        for image_path in image_paths[start_index : start_index + BATCH_SIZE]:
            with Image.open(image_path) as image:
                image_batch.append(image.convert("RGB"))

        feature_batches.append(extract_features(processor, model, image_batch, device).to("cpu"))
    features = torch.cat(feature_batches, dim=0)
    np.save(args.output_path, features.numpy())


if __name__ == "__main__":
    main()
