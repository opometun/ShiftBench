#!/usr/bin/env bash
#SBATCH --job-name=sadge
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=1-9%2
#SBATCH --output=logs/sadge_%A_%a.out
#SBATCH --error=logs/sadge_%A_%a.err
#
# SADGE for all nine mixtures, one per array task.
#
# Deliberately an array rather than a single job: SADGE runs ~9,750 MASt3R
# forward passes per mixture (975 inference images x K=10 candidates), so a
# single job would be 9-13 hours and a failure at hour 8 would lose everything.
# Nine tasks of ~1-1.5h each, two at a time, finish in ~5h wall-clock and fail
# independently. Each task skips its own already-written result, so a
# resubmission only redoes what is missing.
#
# Prerequisites:
#   1. git submodule update --init --recursive   (on a machine with GitHub
#      access -- the cluster blocks outbound git. MASt3R AND its own nested
#      dust3r submodule are both required.)
#   2. pip install -e ".[sadge]" plus MASt3R/dust3r runtime deps, which the
#      extra does not cover -- see scripts/hpc/setup_sadge.sh
#   3. MASt3R weights pre-cached (they download from HuggingFace on first use;
#      nine tasks racing to download them at once is asking for trouble)
#
# Submit from the repo root:
#   cd <repo root> && sbatch scripts/hpc/sadge.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

MIXES=(
  cityscapes100
  cityscapes75_gta25
  cityscapes50_gta50
  cityscapes25_gta75
  gta100
  cityscapes75_synscapes25
  cityscapes50_synscapes50
  cityscapes25_synscapes75
  synscapes100
)
MIX="${MIXES[$((SLURM_ARRAY_TASK_ID - 1))]}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: mixture=${MIX}"

# /tmp and /var/tmp are 10MB tmpfs here; /scratch is node-local with 5.9TB.
if mkdir -p "/scratch/$USER/tmp" 2>/dev/null; then
  export TMPDIR="/scratch/$USER/tmp"
else
  export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
fi

source "$HOME/shiftbench-venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# The materialised symlink directory is what SADGE reads; train_array.sbatch
# already created these, but regenerate defensively (idempotent, symlinks only).
python scripts/materialize_train_split.py \
  "data/study/${MIX}.csv" \
  "data/study/streetViewData_materialized/${MIX}"

# run_sadge.py skips a mixture whose result already exists, which is what makes
# a resubmission cheap. Set SADGE_FORCE=1 to recompute anyway -- needed after a
# change to what the script records, since an existing file would otherwise be
# kept and the new fields never written:
#   sbatch --export=ALL,SADGE_FORCE=1 --array=1 scripts/hpc/sadge.sbatch
srun python scripts/run_sadge.py "$MIX" \
  --output-dir results/shift/sadge \
  ${SADGE_FORCE:+--force} \
  ${SADGE_MAX_QUERIES:+--max-queries "$SADGE_MAX_QUERIES"}

echo "Task ${SLURM_ARRAY_TASK_ID} (${MIX}) done."
