#!/usr/bin/env bash
# One-time environment setup on hpc3. Run this ONCE from an interactive
# allocation -- NOT the head node, and not from inside the job array:
#
#   salloc --partition=workq --time=01:00:00 --cpus-per-task=4 --mem-per-cpu=4G
#   srun --pty bash -l
#   bash scripts/hpc/setup_venv.sh
#
# train_array.sbatch just activates the venv this script creates; it does not
# install anything itself, so 18 array tasks starting at once never race each
# other over pip.
#
# --- Head node vs compute node --------------------------------------------
# The head node and the compute nodes have DIFFERENT software stacks. python3.11
# exists on compute nodes but not on the head node, which is why running this
# from the head node fails with "python3.11: command not found". That is also
# why the interpreter search below must happen where the job actually runs.
#
# python3.11 is preferred because pyproject.toml declares
# requires-python = ">=3.11,<3.12", so `pip install -e .` only works on 3.11.
# If only 3.12 is found, the script falls back to installing the dependencies
# directly and relying on PYTHONPATH=src (which the README documents as a
# supported alternative). That fallback is safe for training specifically:
# tomllib -- the real 3.11+ requirement -- is imported by shiftbench.config,
# which hybrid_eval's training code never touches.
#
# --- Downloads failing partway through -------------------------------------
# The compute nodes have a small /tmp, and pip stages downloads there. When it
# fills up, pip reports "Connection interrupted / not enough bytes were
# received ... This is an issue with network connectivity" -- which is wrong
# and very misleading. wget on the same URL reveals the truth:
# "No space left on device". TMPDIR is therefore redirected to $HOME below.
# The PIP_* retry settings are kept for genuinely flaky transfers.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root (this file lives in scripts/hpc/)
REPO_ROOT="$(pwd)"

# --- Locate an interpreter -------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.11 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "No python3.11 or python3.12 found on $(hostname)." >&2
  echo "If you are on the head node, get a compute node first:" >&2
  echo "  salloc --partition=workq --time=01:00:00 --cpus-per-task=4 --mem-per-cpu=4G" >&2
  echo "  srun --pty bash -l" >&2
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using $("$PYTHON_BIN" --version) at $PYTHON_BIN  (host: $(hostname))"

# --- Keep pip off the tiny /tmp on compute nodes ---------------------------
# /tmp and /var/tmp are 10MB tmpfs here; /scratch is node-local with 5.9TB.
if [[ -z "${TMPDIR_OVERRIDE:-}" ]] && mkdir -p "/scratch/$USER/tmp" 2>/dev/null; then
  export TMPDIR="/scratch/$USER/tmp"
else
  export TMPDIR="${TMPDIR_OVERRIDE:-$HOME/tmp}"
  mkdir -p "$TMPDIR"
fi
echo "TMPDIR=$TMPDIR  ($(df -h "$TMPDIR" | awk 'NR==2 {print $4}') available)"

# Retry settings, for transfers that are genuinely flaky rather than out of disk.
export PIP_RETRIES=10
export PIP_TIMEOUT=120
export PIP_RESUME_RETRIES=100

if [[ ! -d "$HOME/shiftbench-venv" ]]; then
  "$PYTHON_BIN" -m venv "$HOME/shiftbench-venv"
fi
source "$HOME/shiftbench-venv/bin/activate"
python -m pip install --upgrade pip

if [[ "$PYTHON_VERSION" == "3.11" ]]; then
  # Matches pyproject's requires-python, so the package installs properly.
  python -m pip install -e ".[train]"
else
  # 3.12 fallback: pyproject would reject the install, so pull the dependencies
  # directly. Keep in sync with [project].dependencies plus the .train extra.
  #
  # numpy/scipy are NOT optional even though this is "just training":
  # scripts/materialize_train_split.py imports shiftbench.datasets.manifest ->
  # shiftbench.config -> shiftbench.metrics -> scipy, so the array job fails at
  # the data-materialization step without them.
  echo "python3.11 not found; installing dependencies directly (PYTHONPATH mode)"
  python -m pip install \
    "numpy>=1.26" \
    "scipy>=1.11" \
    "torch>=2.2" \
    "torchvision>=0.17" \
    "transformers>=4.40" \
    "pillow>=10.0"
fi

echo
echo "venv ready at $HOME/shiftbench-venv"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torchvision, transformers; print('torchvision', torchvision.__version__, '| transformers', transformers.__version__)"

# Smoke-check the exact import chain the array job depends on, so a missing
# dependency surfaces here rather than inside a queued job.
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python -c "
from shiftbench.datasets.manifest import resolve_image_path
from hybrid_eval.models import build_model
from hybrid_eval.training.train import parse_args
print('import chain OK: shiftbench.manifest + hybrid_eval.models + training.train')
"

cat <<'NOTE'

--- If downloads keep failing ---------------------------------------------
The proxy truncates large wheels. Fallback: fetch them on your laptop (where
rsync already works) and copy them over, then install offline.

On your Mac:
  pip download --dest wheels \
    --platform manylinux_2_28_x86_64 --python-version 311 \
    --only-binary=:all: \
    "numpy>=1.26" "scipy>=1.11" "torch>=2.2" "torchvision>=0.17" \
    "transformers>=4.40" "pillow>=10.0"
  rsync -av --progress wheels/ sstueck@hpc3.rz.uos.de:~/wheels/

Then on a compute node:
  source ~/shiftbench-venv/bin/activate
  pip install --no-index --find-links ~/wheels -e ".[train]"
NOTE
