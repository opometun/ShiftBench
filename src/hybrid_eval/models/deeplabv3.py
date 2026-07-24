import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights


def get_deeplabv3_resnet101(num_classes: int = 19, pretrained: bool = True):
    """
    Returns a DeepLabV3 model with a ResNet-101 backbone.
    Adapts the output layer to map to num_classes.
    """
    if pretrained:
        weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        model = segmentation.deeplabv3_resnet101(weights=weights)
    else:
        model = segmentation.deeplabv3_resnet101(
            weights=None, weights_backbone=None, aux_loss=True
        )

    # DeepLabV3 outputs logits from the classifier head and the auxiliary head.
    # We replace the final Conv2d layer (which has 256 channels input) to project to num_classes.
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    # Replace the auxiliary classifier head final Conv2d as well
    if model.aux_classifier is not None:
        in_aux_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(
            in_aux_channels, num_classes, kernel_size=1
        )

    return model
