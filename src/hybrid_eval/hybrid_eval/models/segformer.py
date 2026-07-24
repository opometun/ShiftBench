from transformers import SegformerConfig, SegformerForSemanticSegmentation

SEGFORMER_B2_BACKBONE_CHECKPOINT = "nvidia/mit-b2"


def get_segformer_b2(num_classes: int = 19, pretrained: bool = True):
    """
    Returns a SegFormer-B2 model for semantic segmentation.
    Pretrained mode loads only the domain-neutral ImageNet-1k MiT-B2 encoder;
    the segmentation decoder is initialized for the requested class count.
    """
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_B2_BACKBONE_CHECKPOINT,
            num_labels=num_classes,
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
        )
        model = SegformerForSemanticSegmentation(config)
    return model


__all__ = ["SEGFORMER_B2_BACKBONE_CHECKPOINT", "get_segformer_b2"]
