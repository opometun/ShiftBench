#!/usr/bin/env bash
#SBATCH --job-name=eval-test
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_test_%j.out
#SBATCH --error=logs/eval_test_%j.err
#
# Evaluates every trained mixture on the held-out real Cityscapes TEST split.
#
# Why this exists: training reports mIoU on the *validation* split, which was
# also used for early stopping and best-checkpoint selection, so it is
# optimistically biased and not a clean measure of downstream utility. The
# shift metrics are computed against the *test* split (see
# scripts/make_shift_manifests.py), so the correlation must use test-split
# performance or the two axes describe different distributions.
#
# Forward passes only over 975 images per mixture -- minutes, not hours.
#
# Submit from the repo root:
#   cd <repo root> && sbatch scripts/hpc/evaluate_test.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

if mkdir -p "/scratch/$USER/tmp" 2>/dev/null; then
  export TMPDIR="/scratch/$USER/tmp"
else
  export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"
fi

source "$HOME/shiftbench-venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

TEST_IMG="data/study/streetViewData/test/cityscapes/img"
TEST_MASK="data/study/streetViewData/test/cityscapes/mask"
MODEL="${MODEL:-segformer}"

# Matches train_array.sbatch: seed 42 keeps the original path, other seeds live
# under <model>_seed<N>/. Set both together, e.g.
#   sbatch --export=ALL,MODEL=segformer,SEED=43 scripts/hpc/evaluate_test.sbatch
SEED="${SEED:-42}"
if [[ "$SEED" == "42" ]]; then
  RUN_SUBDIR="${MODEL}"
else
  RUN_SUBDIR="${MODEL}_seed${SEED}"
fi
echo "model=${MODEL} seed=${SEED} -> output/<mix>/${RUN_SUBDIR}"

MIXES="cityscapes100 cityscapes75_gta25 cityscapes50_gta50 cityscapes25_gta75 gta100 cityscapes75_synscapes25 cityscapes50_synscapes50 cityscapes25_synscapes75 synscapes100"

for mix in $MIXES; do
  checkpoint="output/${mix}/${RUN_SUBDIR}/best_model_${MODEL}.pth"
  destination="output/${mix}/${RUN_SUBDIR}/test_eval"
  if [[ ! -f "$checkpoint" ]]; then
    echo "SKIP ${mix}: no checkpoint at ${checkpoint}" >&2
    continue
  fi
  if [[ -f "${destination}/summary.json" ]]; then
    echo "SKIP ${mix}: already evaluated"
    continue
  fi
  echo "=== ${mix} ==="
  python -m hybrid_eval.inference \
    --checkpoint "$checkpoint" \
    --image-dir "$TEST_IMG" \
    --mask-dir "$TEST_MASK" \
    --output-dir "$destination" \
    --batch-size 8 \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
done

echo
echo "Test-split evaluation complete."
