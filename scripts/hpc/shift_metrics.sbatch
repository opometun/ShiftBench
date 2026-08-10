#!/usr/bin/env bash
#SBATCH --job-name=shift-metrics
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/shift_metrics_%j.out
#SBATCH --error=logs/shift_metrics_%j.err
#
# Computes every summarisable shift metric for all 9 mixtures against the
# real Cityscapes inference set -- the x-axis of the research question, to be
# correlated against the downstream mIoU that train_array.sbatch produced.
#
# Submit from the repo root:
#   cd <repo root> && sbatch scripts/hpc/shift_metrics.sbatch
#
# Prerequisites:
#   1. python scripts/make_shift_manifests.py --check-consistency
#   2. pip install -e ".[train,features,image]"   (adds opencv + scikit-image)
#
# Runtime: dominated by texture/LBP (~6 min per 2,000-image dataset) and the
# two encoder passes. 12h is generous; the driver is resumable, so a job that
# does hit the wall can simply be resubmitted and will skip finished artifacts.
#
# SADGE is NOT included -- it needs the MASt3R submodule and has never been
# run. See scripts/hpc/README_shift.md.
set -euo pipefail

# SLURM copies this script to a spool dir, so BASH_SOURCE is useless for
# locating the repo; SLURM_SUBMIT_DIR is the directory sbatch ran in.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
for required in scripts/run_shift_metrics.py data/study/shift_inference.csv; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: $REPO_ROOT is not the repo root, or manifests are missing" >&2
    echo "       (missing $required)" >&2
    echo "Run: python scripts/make_shift_manifests.py --check-consistency" >&2
    exit 1
  fi
done
mkdir -p logs

# /tmp and /var/tmp are 10MB tmpfs on the compute nodes; /scratch is node-local.
if mkdir -p "/scratch/$USER/tmp" 2>/dev/null; then
  export TMPDIR="/scratch/$USER/tmp"
else
  export TMPDIR="$HOME/tmp"
  mkdir -p "$TMPDIR"
fi
echo "TMPDIR=$TMPDIR"

source "$HOME/shiftbench-venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python scripts/run_shift_metrics.py \
  --study-dir data/study \
  --output-dir results/shift

echo
echo "Done. Distances written to results/shift/distances.json"
