"""Device selection shared by every feature extraction backend."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Best available torch device, preferring CUDA then Apple MPS."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
