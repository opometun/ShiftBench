#!/usr/bin/env bash
# Prepare the environment for SADGE. Run ONCE from a GPU compute node before
# submitting scripts/hpc/sadge.sbatch:
#
#   salloc --partition=gpu --gres=gpu:1 --constraint="A100|H100.80gb" \
#          --ntasks=1 --cpus-per-task=8 --time=01:00:00
#   srun --pty bash -l
#   bash scripts/hpc/setup_sadge.sh
#
# Every step here is something that would otherwise fail inside a queued array
# task, where nine jobs would hit it simultaneously and the logs would be
# interleaved. The last two steps also pre-cache model weights so the array
# tasks do not all race to download the same multi-GB checkpoints.
#
# The README notes that "SADGE's MASt3R runtime requirements (einops etc.) are
# not covered by the .[sadge] extra" -- this script is that gap.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MAST3R_DIR="src/shiftbench/shift_quantification_metrics/representation_based/sadge/metrics/mast3r"

echo "=== 1/5 submodules present? ==="
for required in \
  "$MAST3R_DIR/mast3r/model.py" \
  "$MAST3R_DIR/dust3r/dust3r/inference.py" \
  "$MAST3R_DIR/dust3r/croco/models/blocks.py" ; do
  if [[ ! -f "$required" ]]; then
    echo "MISSING: $required" >&2
    echo "The cluster blocks outbound git, so run this on a machine with" >&2
    echo "GitHub access and rsync the result over:" >&2
    echo "  git submodule update --init --recursive" >&2
    exit 1
  fi
  echo "  ok  $required"
done

source "$HOME/shiftbench-venv/bin/activate"
if mkdir -p "/scratch/$USER/tmp" 2>/dev/null; then
  export TMPDIR="/scratch/$USER/tmp"
else
  export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PIP_RETRIES=10 PIP_TIMEOUT=120

echo
echo "=== 2/5 installing MASt3R/dust3r runtime dependencies ==="
# Deliberately NOT dust3r/requirements.txt wholesale: that pulls gradio,
# trimesh, pyglet and tensorboard for its demo and training paths, none of
# which the inference chain in geometry.py touches. This is the subset those
# imports actually reach.
python -m pip install einops roma scikit-learn matplotlib

echo
echo "=== 3/5 import chain ==="
python - <<'PY'
import sys, pathlib
root = pathlib.Path.cwd()
mast3r = root / "src/shiftbench/shift_quantification_metrics/representation_based/sadge/metrics/mast3r"
sys.path.insert(0, str(mast3r))
import mast3r.utils.path_to_dust3r  # noqa: F401
from mast3r.model import AsymmetricMASt3R  # noqa: F401
from dust3r.inference import inference  # noqa: F401
from dust3r.utils.image import load_images  # noqa: F401
print("  ok  mast3r.model / dust3r.inference / dust3r.utils.image all import")
PY

echo
echo "=== 4/5 caching MASt3R weights (~2GB, open model) ==="
python - <<'PY'
import sys, pathlib, torch
root = pathlib.Path.cwd()
sys.path.insert(0, str(root / "src/shiftbench/shift_quantification_metrics/representation_based/sadge/metrics/mast3r"))
import mast3r.utils.path_to_dust3r  # noqa: F401
from mast3r.model import AsymmetricMASt3R
m = AsymmetricMASt3R.from_pretrained(
    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
print("  ok  MASt3R loaded |", sum(p.numel() for p in m.parameters())/1e6, "M params")
PY

echo
echo "=== 5/5 caching DINOv3 weights (GATED on HuggingFace) ==="
echo "If this fails with 401/403, the model requires accepting its licence at"
echo "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and then"
echo "authenticating:  huggingface-cli login   (or export HF_TOKEN=...)"
python - <<'PY'
from shiftbench.features.dinov3 import load_frozen_dinov3, DEFAULT_MODEL_NAME
import torch
print(f"  model: {DEFAULT_MODEL_NAME}")
processor, model = load_frozen_dinov3(torch.device("cpu"))
print("  ok  DINOv3 loaded |", sum(p.numel() for p in model.parameters())/1e6, "M params")
PY

echo
echo "All five checks passed. Submit with:"
echo "  cd $REPO_ROOT && sbatch scripts/hpc/sadge.sbatch"
