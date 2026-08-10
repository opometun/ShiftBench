from transformers import SegformerConfig, SegformerForSemanticSegmentation

SEGFORMER_B2_BACKBONE_CHECKPOINT = "nvidia/mit-b2"

# Dropout on the transformer's hidden (feed-forward) states. HuggingFace ships
# this at 0.0, so it must be set explicitly -- it is NOT already 0.1 by default.
# The study trains on 2,000 images with no data augmentation (no crop, no scale
# jitter, no photometric distortion), so this carries regularization load that
# augmentation would otherwise provide. Note SegFormer separately defaults
# drop_path_rate=0.1 and classifier_dropout_prob=0.1, which stay as-is.
SEGFORMER_HIDDEN_DROPOUT_PROB = 0.1


def get_segformer_b2(
    num_classes: int = 19,
    pretrained: bool = True,
    hidden_dropout_prob: float = SEGFORMER_HIDDEN_DROPOUT_PROB,
):
    """
    Returns a SegFormer-B2 model for semantic segmentation.
    Pretrained mode loads only the domain-neutral ImageNet-1k MiT-B2 encoder;
    the segmentation decoder is initialized for the requested class count.
    """
    if pretrained:
        # The config is loaded first so hidden_dropout_prob can be overridden
        # before the weights are attached to it.
        config = SegformerConfig.from_pretrained(
            SEGFORMER_B2_BACKBONE_CHECKPOINT,
            num_labels=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_B2_BACKBONE_CHECKPOINT,
            config=config,
            ignore_mismatched_sizes=True,
        )
    else:
        # MiT-B2 configuration, defined locally so --no-pretrained is fully offline.
        config = SegformerConfig(
            num_labels=num_classes,
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            num_attention_heads=[1, 2, 5, 8],
            decoder_hidden_size=768,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        model = SegformerForSemanticSegmentation(config)
    return model


__all__ = [
    "SEGFORMER_B2_BACKBONE_CHECKPOINT",
    "SEGFORMER_HIDDEN_DROPOUT_PROB",
    "get_segformer_b2",
]
