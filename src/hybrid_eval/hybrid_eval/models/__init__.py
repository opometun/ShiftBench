from .deeplabv3 import get_deeplabv3_resnet101
from .deeplabv3plus import get_deeplabv3plus_resnet101
from .segformer import get_segformer_b2

PRIMARY_MODEL_NAMES = ("segformer", "deeplabv3plus")
MODEL_NAMES = (*PRIMARY_MODEL_NAMES, "deeplabv3")


def build_model(model_name: str, num_classes: int = 19, pretrained: bool = True):
    """Build one of the supported semantic-segmentation models."""
    if model_name == "segformer":
        return get_segformer_b2(num_classes=num_classes, pretrained=pretrained)
    if model_name == "deeplabv3plus":
        return get_deeplabv3plus_resnet101(
            num_classes=num_classes, pretrained=pretrained
        )
    if model_name == "deeplabv3":
        return get_deeplabv3_resnet101(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model {model_name!r}; choose one of {MODEL_NAMES}")


__all__ = [
    "MODEL_NAMES",
    "PRIMARY_MODEL_NAMES",
    "build_model",
    "get_segformer_b2",
    "get_deeplabv3plus_resnet101",
    "get_deeplabv3_resnet101",
]
