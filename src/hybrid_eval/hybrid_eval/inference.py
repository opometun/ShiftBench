"""Standalone checkpoint-driven inference for semantic-segmentation models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from hybrid_eval.models import MODEL_NAMES, build_model
from hybrid_eval.training.checkpoint import load_checkpoint
from hybrid_eval.training.data import InferenceImageDataset, JointTransform
from hybrid_eval.training.metrics import StreamingSegmentationMetrics
from hybrid_eval.training.train import forward_logits, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic-segmentation inference from a saved checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint created by hybrid_eval.training.train")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG masks and summary.json")
    parser.add_argument("--mask-dir", help="Optional ground-truth masks for evaluation")
    parser.add_argument("--model", choices=MODEL_NAMES, help="Required only for a legacy state-dict checkpoint")
    parser.add_argument("--num-classes", type=int, help="Required only for a legacy state-dict checkpoint")
    parser.add_argument("--image-height", type=int, help="Defaults to the checkpoint's training height")
    parser.add_argument("--image-width", type=int, help="Defaults to the checkpoint's training width")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    for name in ("num_classes", "image_height", "image_width"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _checkpoint_setting(payload: dict[str, Any], cli_value, key: str, default=None):
    if cli_value is not None:
        return cli_value
    if payload.get(key) is not None:
        return payload[key]
    arguments = payload.get("arguments") or {}
    return arguments.get(key, default)


def load_inference_model(
    checkpoint_path: str | Path,
    *,
    model_name: str | None = None,
    num_classes: int | None = None,
    device: torch.device,
):
    payload = load_checkpoint(checkpoint_path)
    resolved_model = _checkpoint_setting(payload, model_name, "model_name")
    resolved_classes = _checkpoint_setting(payload, num_classes, "num_classes")
    if resolved_model not in MODEL_NAMES:
        raise ValueError("Checkpoint lacks model metadata; pass --model")
    if not isinstance(resolved_classes, int) or resolved_classes <= 0:
        raise ValueError("Checkpoint lacks class-count metadata; pass --num-classes")
    model = build_model(resolved_model, num_classes=resolved_classes, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload, resolved_model, resolved_classes


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    model, payload, model_name, num_classes = load_inference_model(
        args.checkpoint,
        model_name=args.model,
        num_classes=args.num_classes,
        device=device,
    )
    image_height = int(_checkpoint_setting(payload, args.image_height, "image_height", 512))
    image_width = int(_checkpoint_setting(payload, args.image_width, "image_width", 1024))
    transform = JointTransform(image_size=(image_height, image_width), crop_size=None, is_train=False)
    dataset = InferenceImageDataset(args.image_dir, transform=transform, mask_dir=args.mask_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    accumulator = StreamingSegmentationMetrics(num_classes) if args.mask_dir else None
    dtype = np.uint8 if num_classes <= 256 else np.uint16
    processed = 0
    is_segformer = model_name == "segformer"

    for batch in loader:
        if accumulator is None:
            images, names = batch
            targets = None
        else:
            images, targets, names = batch
            targets = targets.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        logits = forward_logits(model, images, is_segformer)
        if accumulator is not None:
            accumulator.update(logits, targets)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        for prediction, name in zip(predictions, names):
            destination = masks_dir / f"{Path(name).stem}.png"
            Image.fromarray(prediction.astype(dtype, copy=False)).save(destination)
            processed += 1

    summary: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model": model_name,
        "num_classes": num_classes,
        "device": str(device),
        "image_size": [image_height, image_width],
        "images": processed,
    }
    if accumulator is not None:
        summary["metrics"] = accumulator.compute()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {processed} prediction masks to {masks_dir}")
    if "metrics" in summary:
        print(f"mIoU: {summary['metrics']['mIoU']:.4f} | ECE: {summary['metrics']['ece']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
