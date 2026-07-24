# Segmentation Models for Hybrid Datasets

[![Open the smoke test in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sigurius23/Segmentation-Models/blob/main/notebooks/hybrid_eval_colab_smoke_test.ipynb)

An end-to-end semantic-segmentation baseline package for studying hybrid real and synthetic urban-scene datasets. It provides the two project architectures, a shared training protocol, resumable checkpoints, standalone inference, and streaming downstream metrics.

## Included models

- **SegFormer-B2**, initialized from the domain-neutral ImageNet-1k `nvidia/mit-b2` encoder when pretrained weights are enabled.
- **DeepLabV3+-ResNet101**, implemented with an ASPP module and low-level feature decoder. Pretrained mode loads only torchvision's ImageNet ResNet-101 encoder.
- The older `deeplabv3` model key is retained only for legacy-checkpoint compatibility.

No model loader uses Cityscapes-fine-tuned weights. This avoids leaking target-domain supervision into experiments that evaluate on Cityscapes-like real data.

## What the pipeline supports

- Paired image/mask loading by filename or matching image stem
- Synchronized scaling, cropping, flipping, and photometric augmentation
- Separate backbone and decoder learning rates
- Polynomial learning-rate decay and fixed optimizer-step budgets
- Optional class-weighted cross-entropy
- Versioned, self-describing checkpoints with exact training resumption
- PNG prediction-mask export
- Streaming mIoU, expected calibration error (ECE), and predictive entropy
- CPU, CUDA, and Apple MPS execution

## Repository structure

```text
hybrid_eval/
├── models/             # SegFormer-B2 and DeepLabV3(+)
├── training/           # data, metrics, checkpoints, and training CLI
└── inference.py        # checkpoint-driven inference CLI
notebooks/
└── hybrid_eval_colab_smoke_test.ipynb
tests/
└── test_hybrid_eval.py
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Sigurius23/Segmentation-Models.git
cd Segmentation-Models
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
```

The requirements file pins the reference local ML stack. The Colab notebook keeps Colab's preinstalled compatible PyTorch/TorchVision pair and installs only missing packages.

## Dataset layout

Training and validation images must have corresponding integer-label masks. A mask may use the same filename as its image, or a `.png` file with the same stem. Label `255` is treated as ignored.

```text
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/          # optional during inference
```

Each mask must be a single-channel image containing one class ID per pixel.

## Train

SegFormer-B2:

```bash
python -m hybrid_eval.training.train \
  --model segformer \
  --train-img-dir data/train/images \
  --train-mask-dir data/train/masks \
  --val-img-dir data/val/images \
  --val-mask-dir data/val/masks \
  --num-classes 19 \
  --max-iterations 40000 \
  --output-dir output/segformer
```

DeepLabV3+-ResNet101 uses the same protocol:

```bash
python -m hybrid_eval.training.train \
  --model deeplabv3plus \
  --train-img-dir data/train/images \
  --train-mask-dir data/train/masks \
  --val-img-dir data/val/images \
  --val-mask-dir data/val/masks \
  --num-classes 19 \
  --max-iterations 40000 \
  --output-dir output/deeplabv3plus
```

Use `--no-pretrained` for a fully offline random initialization. Other useful controls include `--backbone-lr`, `--head-lr`, `--class-weights`, `--scale-min`, `--scale-max`, `--crop-size`, `--device`, and `--seed`. Run `python -m hybrid_eval.training.train --help` for the complete interface.

Training produces:

- `latest_model_<model>.pth`: resumable model, optimizer, and scheduler state
- `best_model_<model>.pth`: best validation-mIoU checkpoint
- `training_history_<model>.json`: per-epoch losses and metrics

Resume an interrupted run with `--resume output/<model>/latest_model_<model>.pth`. Add `--compact-checkpoints` when optimizer state is not needed.

## Inference and evaluation

```bash
python -m hybrid_eval.inference \
  --checkpoint output/segformer/best_model_segformer.pth \
  --image-dir data/test/images \
  --mask-dir data/test/masks \
  --output-dir output/segformer/test-predictions
```

The command writes predicted PNG masks plus `summary.json`. When `--mask-dir` is supplied, the summary also contains mIoU, per-class IoU, ECE, predictive entropy, and valid-pixel count. Without masks, inference still exports predictions.

## Colab smoke test

Open [`notebooks/hybrid_eval_colab_smoke_test.ipynb`](notebooks/hybrid_eval_colab_smoke_test.ipynb) or use the badge at the top of this page. The notebook:

1. Builds a fixed-size hybrid proxy with real-like and synthetic-like urban scenes.
2. Trains both required models with identical data and hyperparameters.
3. Restores both checkpoints in separate inference processes.
4. Compares validation/test mIoU, ECE, and predictive entropy.
5. Visualizes the domain gap and model predictions.
6. Runs the focused regression suite.

It is an engineering smoke test, not a scientific benchmark. The final study must replace the proxy data with the selected real and synthetic datasets, hold total training size constant while sweeping mixture ratios, calculate the planned pre-training shift metrics, and relate those metrics to downstream real-domain performance.

## Tests

```bash
python -m pytest tests/test_hybrid_eval.py -q
```

The tests cover model-policy invariants, the DeepLabV3+ decoder, aligned transforms, image/mask pairing, optimizer groups, training and validation steps, streaming metrics, checkpoint round trips, and standalone inference artifacts.
