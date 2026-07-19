"""Tracking which encoder produced a feature artifact.

A distance is only meaningful between features from the same encoder and the
same checkpoint. Nothing about a raw feature array says where it came from, so
extraction writes a small JSON sidecar next to the .npy, that provenance is
carried into the stats .npz, and the distance scripts refuse to compare two
artifacts that disagree.

Features are kept as plain .npy rather than a metadata-carrying .npz because
np.load only supports mmap_mode on .npy, and stats computation relies on it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

UNKNOWN = "unknown"


def sidecar_path(features_path: str | Path) -> Path:
    """Path of the JSON sidecar belonging to a feature array."""
    return Path(f"{features_path}.json")


def write_feature_provenance(
    features_path: str | Path,
    encoder: str,
    model: str,
) -> Path:
    """Record which encoder produced a feature array.

    Args:
        features_path: Path of the saved .npy feature array.
        encoder: Registered encoder name, for example 'dinov2'.
        model: Hugging Face model id the encoder ran.

    Returns:
        Path of the written sidecar.
    """
    path = sidecar_path(features_path)
    payload = {"encoder": encoder, "model": model}
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def read_feature_provenance(features_path: str | Path) -> dict[str, str]:
    """Read a feature array's sidecar, or empty dict if it has none.

    Feature arrays produced before provenance existed have no sidecar, which
    is not an error here. It surfaces later as an unverifiable comparison.
    """
    path = sidecar_path(features_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return {
        "encoder": str(data.get("encoder", UNKNOWN)),
        "model": str(data.get("model", UNKNOWN)),
    }


def describe(stats: Mapping[str, Any]) -> tuple[str, str]:
    """Return the (encoder, model) a loaded stats .npz was built from."""
    encoder = str(stats["encoder"]) if "encoder" in stats else UNKNOWN
    model = str(stats["model"]) if "model" in stats else UNKNOWN
    return encoder, model


def require_same_encoder(
    label_a: str,
    stats_a: Mapping[str, Any],
    label_b: str,
    stats_b: Mapping[str, Any],
) -> None:
    """Fail if two stats artifacts came from different encoders.

    Unknown provenance warns instead of failing, so stats files written before
    this existed still work.

    Raises:
        ValueError: If both sides are known and disagree.
    """
    encoder_a, model_a = describe(stats_a)
    encoder_b, model_b = describe(stats_b)

    if UNKNOWN in (encoder_a, encoder_b):
        print(
            "warning: missing encoder provenance, cannot verify that "
            f"{label_a} and {label_b} are comparable",
            file=sys.stderr,
        )
        return

    if (encoder_a, model_a) != (encoder_b, model_b):
        msg = (
            "Refusing to compare features from different encoders: "
            f"{label_a} is {encoder_a}/{model_a}, "
            f"{label_b} is {encoder_b}/{model_b}. "
            "These distances are not on the same scale."
        )
        raise ValueError(msg)
