"""Feature extraction backends, addressable by name.

Adding an encoder means adding a module next to this one and one ENCODERS
entry. Nothing in scripts/ should import a specific backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from shiftbench.features import dinov2, streetclip


@dataclass(frozen=True)
class EncoderBackend:
    """One frozen encoder and the two functions needed to run it.

    Attributes:
        name: Short name used on the command line and in run artifacts.
        default_model_name: Hugging Face id used when none is given.
        load: Takes (device, model_name), returns (processor, model).
        extract: Takes (processor, model, image_batch, device), returns a
            (batch_size, hidden_size) tensor.
    """

    name: str
    default_model_name: str
    load: Callable
    extract: Callable


ENCODERS: dict[str, EncoderBackend] = {
    "dinov2": EncoderBackend(
        name="dinov2",
        default_model_name=dinov2.DEFAULT_MODEL_NAME,
        load=dinov2.load_frozen_dinov2,
        extract=dinov2.extract_features,
    ),
    "streetclip": EncoderBackend(
        name="streetclip",
        default_model_name=streetclip.DEFAULT_MODEL_NAME,
        load=streetclip.load_frozen_streetclip,
        extract=streetclip.extract_features,
    ),
}


def get_encoder(name: str) -> EncoderBackend:
    """Look up an encoder backend by name.

    Raises:
        ValueError: If the name is not a registered encoder.
    """
    try:
        return ENCODERS[name]
    except KeyError:
        available = ", ".join(sorted(ENCODERS))
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}") from None
