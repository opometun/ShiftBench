#!/usr/bin/env bash
#SBATCH --job-name=shift-metrics-cpu
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/shift_metrics_cpu_%j.out
#SBATCH --error=logs/shift_metrics_cpu_%j.err
#
# The image- and label-based half of the shift metrics: colour histogram, LBP
# texture, class frequency, class presence, scene complexity. All pure
# numpy/OpenCV work with no GPU involvement.
#
# Split out from shift_metrics.sbatch deliberately. That job requests a GPU for
# its whole duration, but only the two encoder passes (DINOv2, StreetCLIP) use
# one -- roughly 20 minutes out of several hours. Running the CPU half on workq
# frees a contended A100 and starts immediately, since workq usually has idle
# nodes while the gpu partition is queued.
#
# Run this FIRST, then shift_metrics.sbatch. The driver skips artifacts that
# already exist, so the GPU job will do only the embedding work and then write
# a complete distances.json covering both halves.
#
# Submit from the repo root:
#   cd <repo root> && sbatch scripts/hpc/shift_metrics_cpu.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
for required in scripts/run_shift_metrics.py data/study/shift_inference.csv; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: $REPO_ROOT is not the repo root (missing $required)" >&2
    exit 1
  fi
done
mkdir -p logs

if mkdir -p "/scratch/$USER/tmp" 2>/dev/null; then
  export TMPDIR="/scratch/$USER/tmp"
else
  export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
fi
echo "TMPDIR=$TMPDIR"

source "$HOME/shiftbench-venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python scripts/run_shift_metrics.py \
  --study-dir data/study \
  --output-dir results/shift \
  --skip-embeddings

echo
echo "CPU-side metrics done. Now submit scripts/hpc/shift_metrics.sbatch for"
echo "the DINOv2/StreetCLIP embeddings; it will skip everything above."
