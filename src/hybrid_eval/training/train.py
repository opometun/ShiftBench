"""Command-line training entry point for semantic-segmentation baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hybrid_eval.models import MODEL_NAMES, build_model
from hybrid_eval.training.checkpoint import load_checkpoint, save_checkpoint
from hybrid_eval.training.data import JointTransform, SimpleSegmentationDataset
from hybrid_eval.training.metrics import StreamingSegmentationMetrics


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit or automatically selected PyTorch device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: nn.Module,
    model_name: str,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build the standardized optimizer with separate encoder and decoder rates."""
    backbone = model.segformer if model_name == "segformer" else model.backbone
    backbone_parameters = [
        parameter for parameter in backbone.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone_parameters or not head_parameters:
        raise ValueError(
            f"Could not split {model_name!r} into backbone and decoder parameters"
        )
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr, "name": "backbone"},
            {"params": head_parameters, "lr": head_lr, "name": "decoder"},
        ],
        weight_decay=weight_decay,
    )


def parse_class_weights(value: str | None, num_classes: int) -> torch.Tensor | None:
    """Parse an optional comma-separated class-weight vector."""
    if value is None:
        return None
    try:
        weights = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(
            "--class-weights must contain comma-separated numbers"
        ) from error
    if len(weights) != num_classes:
        raise ValueError(
            f"--class-weights requires {num_classes} values, received {len(weights)}"
        )
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("--class-weights values must be finite and positive")
    return torch.tensor(weights, dtype=torch.float32)


def calculate_miou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 19,
    ignore_index: int = 255,
) -> tuple[float, list[float]]:
    """Calculate mean intersection-over-union from integer predictions."""
    preds = preds.reshape(-1).to(torch.int64)
    targets = targets.reshape(-1).to(torch.int64)
    valid = targets != ignore_index
    valid &= targets.ge(0) & targets.lt(num_classes)
    valid &= preds.ge(0) & preds.lt(num_classes)
    preds, targets = preds[valid], targets[valid]
    encoded = targets * num_classes + preds
    confusion = torch.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )
    intersection = confusion.diag().double()
    union = confusion.sum(0).double() + confusion.sum(1).double() - intersection
    class_ious = [
        float(intersection[index] / union[index]) if union[index] > 0 else float("nan")
        for index in range(num_classes)
    ]
    valid_ious = [value for value in class_ious if not np.isnan(value)]
    return (float(np.mean(valid_ious)) if valid_ious else 0.0), class_ious


def forward_logits(
    model: nn.Module, images: torch.Tensor, is_segformer: bool
) -> torch.Tensor:
    """Run a model and return logits at input resolution."""
    outputs = model(pixel_values=images) if is_segformer else model(images)
    logits = outputs.logits if is_segformer else outputs["out"]
    if logits.shape[2:] != images.shape[2:]:
        logits = nn.functional.interpolate(
            logits, size=images.shape[2:], mode="bilinear", align_corners=False
        )
    return logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_segformer: bool = False,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch_index, (images, masks) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if is_segformer:
            logits = model(pixel_values=images).logits
            if logits.shape[2:] != masks.shape[1:]:
                logits = nn.functional.interpolate(
                    logits,
                    size=masks.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss = criterion(logits, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs["out"], masks)
            auxiliary = outputs.get("aux")
            if auxiliary is not None:
                loss = loss + 0.4 * criterion(auxiliary, masks)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("Training loader produced no batches")
    return total_loss / sample_count


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 19,
    is_segformer: bool = False,
) -> dict[str, Any]:
    """Evaluate a model with constant memory relative to dataset size."""
    model.eval()
    accumulator = StreamingSegmentationMetrics(num_classes=num_classes)
    total_loss = 0.0
    sample_count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if is_segformer:
            logits = model(pixel_values=images).logits
            if logits.shape[2:] != masks.shape[1:]:
                logits = nn.functional.interpolate(
                    logits, size=masks.shape[1:], mode="bilinear", align_corners=False
                )
            loss = criterion(logits, masks)
        else:
            outputs = model(images)
            logits = outputs["out"]
            loss = criterion(logits, masks)
        accumulator.update(logits, masks)
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("Validation loader produced no batches")
    metrics = accumulator.compute()
    metrics["loss"] = total_loss / sample_count
    return metrics


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Training history must be a JSON list: {path}")
    return data


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SegFormer-B2 or DeepLabV3+ on paired segmentation data"
    )
    parser.add_argument(
        "--model", required=True, choices=MODEL_NAMES, help="Model architecture"
    )
    parser.add_argument(
        "--train-img-dir", required=True, help="Directory containing training images"
    )
    parser.add_argument(
        "--train-mask-dir", required=True, help="Directory containing training masks"
    )
    parser.add_argument(
        "--val-img-dir", required=True, help="Directory containing validation images"
    )
    parser.add_argument(
        "--val-mask-dir", required=True, help="Directory containing validation masks"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max-iterations", type=int, help="Optional fixed optimizer-step budget"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--lr", type=float, help="Legacy shortcut that sets both learning rates"
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler", choices=("poly", "none"), default="poly")
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--no-photometric-distortion", action="store_true")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument(
        "--class-weights", help="Comma-separated positive weights, one per class"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument(
        "--compact-checkpoints",
        action="store_true",
        help="Omit optimizer/scheduler tensors; useful for smoke tests and inference artifacts",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", help="Resume from a checkpoint created by this script"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize the model without downloaded weights",
    )
    args = parser.parse_args(argv)
    for name in (
        "epochs",
        "batch_size",
        "image_height",
        "image_width",
        "crop_size",
        "num_classes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    for name in ("lr", "backbone_lr", "head_lr", "weight_decay", "poly_power"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_iterations is not None and args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    if args.scale_min <= 0 or args.scale_max < args.scale_min:
        parser.error(
            "--scale-min and --scale-max must define a positive, increasing range"
        )
    try:
        parse_class_weights(args.class_weights, args.num_classes)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    image_size = (args.image_height, args.image_width)
    train_dataset = SimpleSegmentationDataset(
        args.train_img_dir,
        args.train_mask_dir,
        JointTransform(
            image_size=image_size,
            crop_size=args.crop_size,
            is_train=True,
            scale_range=(args.scale_min, args.scale_max),
            photometric_distortion=not args.no_photometric_distortion,
        ),
    )
    val_dataset = SimpleSegmentationDataset(
        args.val_img_dir,
        args.val_mask_dir,
        JointTransform(image_size=image_size, crop_size=None, is_train=False),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_options
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training loader contains no batches")

    resume_payload = load_checkpoint(args.resume) if args.resume else None
    if resume_payload is not None:
        stored_model = resume_payload.get("model_name")
        stored_classes = resume_payload.get("num_classes")
        if stored_model is not None and stored_model != args.model:
            raise ValueError(
                f"Checkpoint model is {stored_model!r}, not {args.model!r}"
            )
        if stored_classes is not None and stored_classes != args.num_classes:
            raise ValueError(
                f"Checkpoint has {stored_classes} classes, not {args.num_classes}"
            )

    model = build_model(
        args.model,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained and resume_payload is None,
    )
    model.to(device)
    backbone_lr = args.lr if args.lr is not None else args.backbone_lr
    head_lr = args.lr if args.lr is not None else args.head_lr
    optimizer = build_optimizer(
        model,
        args.model,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=args.weight_decay,
    )
    total_iterations = args.max_iterations or args.epochs * len(train_loader)
    planned_epochs = math.ceil(total_iterations / len(train_loader))
    scheduler = (
        torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_iterations,
            power=args.poly_power,
        )
        if args.scheduler == "poly"
        else None
    )
    class_weights = parse_class_weights(args.class_weights, args.num_classes)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        ignore_index=255,
    )
    start_epoch, global_step, best_miou = 1, 0, -1.0
    history_path = output_dir / f"training_history_{args.model}.json"
    history: list[dict[str, Any]] = []
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        if "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            _move_optimizer_state(optimizer, device)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        global_step = int(
            resume_payload.get("global_step", (start_epoch - 1) * len(train_loader))
        )
        if scheduler is not None and "scheduler_state_dict" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        best_miou = float(resume_payload.get("best_miou", -1.0))
        history = _load_history(history_path)
        print(f"Resuming after epoch {start_epoch - 1}")

    if global_step >= total_iterations or start_epoch > planned_epochs:
        raise ValueError("Checkpoint already reached the requested training budget")

    is_segformer = args.model == "segformer"
    for epoch in range(start_epoch, planned_epochs + 1):
        batches_this_epoch = min(len(train_loader), total_iterations - global_step)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            is_segformer,
            scheduler=scheduler,
            max_batches=batches_this_epoch,
        )
        global_step += batches_this_epoch
        val_metrics = validate(
            model, val_loader, criterion, device, args.num_classes, is_segformer
        )
        epoch_history = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mIoU": val_metrics["mIoU"],
            "val_ece": val_metrics["ece"],
            "val_entropy": val_metrics["entropy"],
        }
        history.append(epoch_history)
        current_miou = float(val_metrics["mIoU"])
        improved = current_miou > best_miou
        best_miou = max(best_miou, current_miou)
        checkpoint_args = vars(args).copy()
        checkpoint_args.update(
            {
                "resolved_backbone_lr": backbone_lr,
                "resolved_head_lr": head_lr,
                "planned_epochs": planned_epochs,
                "total_iterations": total_iterations,
            }
        )
        save_checkpoint(
            output_dir / f"latest_model_{args.model}.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            model_name=args.model,
            num_classes=args.num_classes,
            best_miou=best_miou,
            metrics=val_metrics,
            arguments=checkpoint_args,
            scheduler=scheduler,
            global_step=global_step,
            include_optimizer_state=not args.compact_checkpoints,
        )
        if improved:
            save_checkpoint(
                output_dir / f"best_model_{args.model}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_name=args.model,
                num_classes=args.num_classes,
                best_miou=best_miou,
                metrics=val_metrics,
                arguments=checkpoint_args,
                scheduler=scheduler,
                global_step=global_step,
                include_optimizer_state=False,
            )
        _write_history(history_path, history)
        print(
            f"Epoch {epoch}/{planned_epochs} | step {global_step}/{total_iterations} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | mIoU {current_miou:.4f} | ECE {val_metrics['ece']:.4f}"
        )
    print(f"Training complete. Best validation mIoU: {best_miou:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
