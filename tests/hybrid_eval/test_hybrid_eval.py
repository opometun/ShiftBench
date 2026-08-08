from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from hybrid_eval.inference import main as inference_main
from hybrid_eval.models import PRIMARY_MODEL_NAMES
from hybrid_eval.models.deeplabv3plus import DeepLabV3Plus, DeepLabV3PlusHead
from hybrid_eval.models.segformer import SEGFORMER_B2_BACKBONE_CHECKPOINT
from hybrid_eval.training.checkpoint import load_checkpoint, save_checkpoint
from hybrid_eval.training.data import JointTransform, SimpleSegmentationDataset
from hybrid_eval.training.metrics import (
    StreamingSegmentationMetrics,
    calculate_ece,
    calculate_predictive_entropy,
)
from hybrid_eval.training.train import (
    build_optimizer,
    calculate_miou,
    parse_class_weights,
    train_one_epoch,
    validate,
)


class TinyDeepLab(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, images):
        return {"out": self.classifier(images)}


class TinyFeatureBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.low = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.high = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)

    def forward(self, images):
        low = self.low(images)
        return {"low": low, "out": self.high(low)}


def test_joint_transform_preserves_integer_mask_labels():
    image = Image.fromarray(np.full((4, 6, 3), 127, dtype=np.uint8))
    mask_array = np.array(
        [[0, 0, 1, 1, 255, 255], [0, 1, 1, 255, 255, 0]] * 2,
        dtype=np.uint8,
    )
    transform = JointTransform(image_size=(8, 12), crop_size=None, is_train=False)
    image_tensor, mask_tensor = transform(image, Image.fromarray(mask_array))
    assert image_tensor.shape == (3, 8, 12)
    assert mask_tensor.dtype == torch.long
    assert set(mask_tensor.unique().tolist()) == {0, 1, 255}


def test_masks_are_remapped_from_label_ids_to_train_ids():
    # The study's masks are stored as raw Cityscapes labelIds (0-33). Without
    # remapping, CrossEntropyLoss(num_classes=19) hard-fails on any pixel
    # labelled 19-33, so this guards the whole training pipeline.
    image = Image.fromarray(np.full((2, 4, 3), 127, dtype=np.uint8))
    # 7=road->0, 8=sidewalk->1, 26=car->13, 33=bicycle->18,
    # 0=unlabeled->255, 34=out-of-scheme->255
    mask = Image.fromarray(np.array([[7, 8, 26, 33], [0, 34, 7, 8]], dtype=np.uint8))

    transform = JointTransform(image_size=(2, 4), crop_size=None, is_train=False)
    _, mask_tensor = transform(image, mask)

    assert mask_tensor.tolist() == [[0, 1, 13, 18], [255, 255, 0, 1]]
    valid = mask_tensor[mask_tensor != 255]
    assert int(valid.max()) <= 18

    # Opting out must leave the raw labelIds untouched -- remapping twice would
    # corrupt them (trainId 0 would become 255).
    raw_transform = JointTransform(
        image_size=(2, 4), crop_size=None, is_train=False, remap_label_ids=False
    )
    _, raw_mask = raw_transform(image, mask)
    assert raw_mask.tolist() == [[7, 8, 26, 33], [0, 34, 7, 8]]


def test_training_transform_pads_scaled_masks_with_ignore_label():
    image = Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8))
    mask = Image.fromarray(np.ones((4, 4), dtype=np.uint8))
    transform = JointTransform(
        image_size=(4, 4),
        crop_size=4,
        is_train=True,
        scale_range=(0.5, 0.5),
    )
    image_tensor, mask_tensor = transform(image, mask)
    assert image_tensor.shape == (3, 4, 4)
    assert mask_tensor.shape == (4, 4)
    assert set(mask_tensor.unique().tolist()) == {1, 255}


def test_required_models_are_domain_neutral_segformer_and_deeplabv3plus():
    assert PRIMARY_MODEL_NAMES == ("segformer", "deeplabv3plus")
    assert SEGFORMER_B2_BACKBONE_CHECKPOINT == "nvidia/mit-b2"
    assert "cityscapes" not in SEGFORMER_B2_BACKBONE_CHECKPOINT.lower()


def test_deeplabv3plus_decoder_returns_input_resolution_logits():
    model = DeepLabV3Plus(
        TinyFeatureBackbone(),
        DeepLabV3PlusHead(
            high_level_channels=8,
            low_level_channels=4,
            num_classes=3,
            atrous_rates=(1, 2, 3),
        ),
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 16, 24))["out"]
    assert output.shape == (2, 3, 16, 24)


def test_optimizer_uses_separate_backbone_and_decoder_learning_rates():
    model = DeepLabV3Plus(
        TinyFeatureBackbone(),
        DeepLabV3PlusHead(
            high_level_channels=8,
            low_level_channels=4,
            num_classes=2,
            atrous_rates=(1, 2, 3),
        ),
    )
    optimizer = build_optimizer(
        model,
        "deeplabv3plus",
        backbone_lr=1e-4,
        head_lr=1e-3,
        weight_decay=0.01,
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "backbone",
        "decoder",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-4, 1e-3]
    )


def test_class_weights_require_one_positive_value_per_class():
    assert parse_class_weights("1, 2, 3", 3).tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="requires 3 values"):
        parse_class_weights("1,2", 3)
    with pytest.raises(ValueError, match="finite and positive"):
        parse_class_weights("1,0,3", 3)


def test_training_step_budget_advances_polynomial_scheduler_exactly_once():
    images = torch.randn(3, 3, 4, 4)
    masks = torch.randint(0, 2, (3, 4, 4))
    loader = DataLoader(TensorDataset(images, masks), batch_size=1)
    model = TinyDeepLab()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=1, power=0.9
    )
    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        scheduler=scheduler,
        max_batches=1,
    )
    assert loss > 0
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_dataset_pairs_jpeg_image_with_same_stem_png_mask(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_dir / "sample.jpg")
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(mask_dir / "sample.png")
    dataset = SimpleSegmentationDataset(
        image_dir,
        mask_dir,
        JointTransform(image_size=(4, 4), crop_size=None, is_train=False),
    )
    image, mask = dataset[0]
    assert image.shape == (3, 4, 4)
    assert torch.equal(mask, torch.ones((4, 4), dtype=torch.long))


def test_dataset_reports_missing_mask_during_initialization(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_dir / "missing.png")
    with pytest.raises(FileNotFoundError, match="missing.png"):
        SimpleSegmentationDataset(image_dir, mask_dir)


def test_streaming_metrics_are_exact_for_perfect_predictions():
    targets = torch.tensor([[[0, 1], [1, 255]]])
    logits = torch.tensor([[[[8.0, -8.0], [-8.0, 0.0]], [[-8.0, 8.0], [8.0, 0.0]]]])
    metrics = StreamingSegmentationMetrics(num_classes=2)
    metrics.update(logits, targets)
    result = metrics.compute()
    assert result["mIoU"] == pytest.approx(1.0)
    assert result["valid_pixels"] == 3
    assert result["ece"] < 1e-5
    assert result["entropy"] < 1e-4
    assert calculate_ece(torch.softmax(logits, 1), targets) < 1e-5
    assert calculate_predictive_entropy(torch.softmax(logits, 1), targets) < 1e-4


def test_calculate_miou_ignores_void_pixels():
    predictions = torch.tensor([[[0, 1], [0, 1]]])
    targets = torch.tensor([[[0, 1], [255, 1]]])
    mean, class_ious = calculate_miou(predictions, targets, num_classes=2)
    assert mean == pytest.approx(1.0)
    assert class_ious == pytest.approx([1.0, 1.0])


def test_train_validate_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    images = torch.randn(2, 3, 4, 4)
    masks = torch.randint(0, 2, (2, 4, 4))
    loader = DataLoader(TensorDataset(images, masks), batch_size=1)
    model = TinyDeepLab()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    train_loss = train_one_epoch(
        model, loader, optimizer, criterion, torch.device("cpu")
    )
    metrics = validate(model, loader, criterion, torch.device("cpu"), num_classes=2)
    assert train_loss > 0
    assert 0 <= metrics["mIoU"] <= 1
    assert metrics["valid_pixels"] == 32

    checkpoint_path = tmp_path / "model.pth"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        model_name="deeplabv3",
        num_classes=2,
        best_miou=float(metrics["mIoU"]),
        metrics=metrics,
        arguments={"image_height": 4, "image_width": 4},
        scheduler=scheduler,
        global_step=3,
    )
    payload = load_checkpoint(checkpoint_path)
    assert payload["format_version"] == 2
    assert payload["global_step"] == 3
    assert "scheduler_state_dict" in payload
    assert payload["model_name"] == "deeplabv3"
    assert payload["epoch"] == 1
    assert set(payload["model_state_dict"]) == set(model.state_dict())

    compact_path = tmp_path / "compact.pth"
    save_checkpoint(
        compact_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        model_name="deeplabv3",
        num_classes=2,
        best_miou=float(metrics["mIoU"]),
        metrics=metrics,
        arguments={},
        include_optimizer_state=False,
    )
    compact = load_checkpoint(compact_path)
    assert "optimizer_state_dict" not in compact
    assert "scheduler_state_dict" not in compact


def test_inference_cli_writes_masks_and_metrics(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "predictions"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_dir / "sample.png")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(mask_dir / "sample.png")

    model = TinyDeepLab()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias[:] = torch.tensor([3.0, -3.0])
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_path = tmp_path / "model.pth"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        model_name="deeplabv3",
        num_classes=2,
        best_miou=1.0,
        metrics={"mIoU": 1.0},
        arguments={"image_height": 4, "image_width": 4},
    )
    monkeypatch.setattr(
        "hybrid_eval.inference.build_model", lambda *_args, **_kwargs: TinyDeepLab()
    )
    assert (
        inference_main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--image-dir",
                str(image_dir),
                "--mask-dir",
                str(mask_dir),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                "1",
                "--num-workers",
                "0",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (output_dir / "masks" / "sample.png").is_file()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["images"] == 1
    assert summary["metrics"]["mIoU"] == pytest.approx(1.0)
