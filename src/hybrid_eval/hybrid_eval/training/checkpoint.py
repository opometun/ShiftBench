"""Versioned checkpoint I/O shared by training and inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 2


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a trusted tensor-only checkpoint and normalize legacy state dicts."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
    except TypeError:  # PyTorch < 2.0 compatibility
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}")
    if "model_state_dict" in payload:
        return payload
    if payload and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    ):
        return {"format_version": 0, "model_state_dict": payload}
    raise ValueError(
        f"Checkpoint does not contain a model_state_dict: {checkpoint_path}"
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_name: str,
    num_classes: int,
    best_miou: float,
    metrics: dict[str, Any],
    arguments: dict[str, Any],
    scheduler: Any | None = None,
    global_step: int = 0,
    include_optimizer_state: bool = True,
) -> None:
    """Atomically save a resumable, self-describing checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": model_name,
        "num_classes": num_classes,
        "epoch": epoch,
        "global_step": global_step,
        "best_miou": best_miou,
        "metrics": metrics,
        "arguments": arguments,
        "model_state_dict": model.state_dict(),
    }
    if include_optimizer_state:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, temporary)
    os.replace(temporary, destination)
