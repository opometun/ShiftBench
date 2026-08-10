#!/usr/bin/env bash
#SBATCH --job-name=hybrid-eval-train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# No --mem: the cluster's DefMemPerCPU is 7421MB, so 8 CPUs already yields
# ~58GB. Setting --mem explicitly would cap it lower for no benefit -- the GPU
# nodes carry 950GB (A100) to 1.95TB (H100), so memory never gates scheduling
# here. The AIP reference sbatch files likewise set no --mem.
#SBATCH --array=1-18%4
#SBATCH --output=logs/train_array_%A_%a.out
#SBATCH --error=logs/train_array_%A_%a.err
#
# One array task = one (model, dataset-mix) run from study_runs.txt.
# 9 mixes x 2 models = 18 tasks; adjust --array if study_runs.txt changes.
#
# The "%4" throttles this to 4 concurrent tasks. The gpu partition has only
# 2 A100 nodes (4 GPUs each) plus 1 H100 node, so an unthrottled 18-task array
# could occupy every A100 on the cluster for a day at a time. Raise it if the
# partition is quiet and you are in a hurry; lower it if others are waiting.
# (The AIP reference array job uses %2; their comment notes "the gpu queue is
# currently quite long", so consider dropping to %2 if you see others waiting
# in `squeue -p gpu`.)
#
# --constraint="A100|H100.80gb" follows the AIP reference sbatch: it lets the
# job land on EITHER GPU type rather than pinning to A100, which both shortens
# the queue wait and makes the 80GB H100 reachable. That matters because the
# A100s here advertise no memory suffix in sinfo, so they may be the 40GB
# variant -- on which batch size 8 at 1440x720 is not guaranteed to fit.
#
# --num-workers tracks SLURM_CPUS_PER_TASK rather than train.py's default of 2.
# At 1440x720 every sample needs a PNG decode plus a resize, so 2 workers would
# leave the GPU waiting on the data loader while 6 of the 8 reserved CPUs sat
# idle.
#
# Submit:   sbatch scripts/hpc/train_array.sbatch
# Resubmit: same command. Each task auto-resumes from its own
#           latest_model_<model>.pth if the previous attempt didn't finish
#           inside the --time budget (see the 48h cap in the HPC intro deck).
#
# Before the first submission: run scripts/hpc/setup_venv.sh once
# interactively (see that file for why it's not inlined here), and make sure
# data/study/streetViewData/ has been transferred to this checkout via
# rsync/SFTP/SSHFS -- it is gitignored and can't come from `git clone`.

set -euo pipefail

# SLURM copies this script into a spool directory before executing it, so
# ${BASH_SOURCE[0]} points at something like /var/spool/slurmd/jobNNN/slurm_script
# -- NOT at scripts/hpc/ in the repo. Deriving the repo root from it lands in
# /var/spool and every relative path breaks ("mkdir: cannot create directory
# 'logs': Permission denied"). SLURM_SUBMIT_DIR is the directory sbatch was
# invoked from, which is the reliable anchor.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# Fail loudly and early if submitted from the wrong directory, rather than
# halfway through with a confusing missing-file error.
for required in scripts/hpc/study_runs.txt scripts/materialize_train_split.py src/hybrid_eval; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: $REPO_ROOT is not the repo root (missing $required)." >&2
    echo "Submit with: cd <repo root> && sbatch scripts/hpc/train_array.sbatch" >&2
    exit 1
  fi
done

# Covers a logs/ deleted between submission and execution. It does NOT help the
# --output/--error paths above: SLURM opens those before the script runs, so
# logs/ must exist at submit time or the job dies with no log to explain it.
# logs/.gitkeep is tracked for exactly that reason.
mkdir -p logs

# /tmp and /var/tmp on the compute nodes are 10MB tmpfs. PyTorch, HuggingFace
# and pip all stage files there and fail confusingly when they fill (pip
# misreports a full disk as "connection interrupted"). /scratch is node-local
# (5.9TB, /dev/md127); prefer it over NFS $HOME because the DataLoader workers'
# temp dirs get deleted while still open, which on NFS triggers silly-rename
# and floods the log with "Device or resource busy: .nfs0000..." at shutdown.
# Node-local also means much faster worker IPC.
#
# Only ephemeral scratch goes here. Checkpoints stay under $OUTPUT_DIR on NFS,
# since /scratch is per-node and a resumed array task may land elsewhere.
if [[ -z "${TMPDIR_OVERRIDE:-}" ]] && mkdir -p "/scratch/$USER/tmp" 2>/dev/null; then
  export TMPDIR="/scratch/$USER/tmp"
else
  export TMPDIR="${TMPDIR_OVERRIDE:-$HOME/tmp}"
  mkdir -p "$TMPDIR"
fi
echo "TMPDIR=$TMPDIR"

source "$HOME/shiftbench-venv/bin/activate"

# setup_venv.sh installs the dependencies but not the package itself (hpc3 has
# no Python 3.11, which pyproject.toml requires -- see that script's header).
# Putting src/ on PYTHONPATH is the documented alternative to `pip install -e .`.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Python block-buffers stdout when it is a file rather than a TTY, so the
# per-epoch progress lines sit in a buffer for a long time and the .out log
# looks frozen while training is in fact running normally. stderr is unbuffered,
# which is why warnings appear promptly and progress does not. Force unbuffered
# output so `tail -f` on the log reflects reality.
export PYTHONUNBUFFERED=1

RUN_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/hpc/study_runs.txt)
if [[ -z "$RUN_LINE" || "$RUN_LINE" == \#* ]]; then
  echo "No run configured for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
read -r MODEL CONFIG <<< "$RUN_LINE"
echo "Array task ${SLURM_ARRAY_TASK_ID}: model=${MODEL} config=${CONFIG}"

# --- Materialize this mix's train split (idempotent; symlinks only) -------
TRAIN_DIR="data/study/streetViewData_materialized/${CONFIG}"
python scripts/materialize_train_split.py \
  "data/study/${CONFIG}.csv" \
  "$TRAIN_DIR"

VAL_IMG_DIR="data/study/streetViewData/validation/cityscapes/img"
VAL_MASK_DIR="data/study/streetViewData/validation/cityscapes/mask"

# --- Seed and output location ----------------------------------------------
# Seed 42 is the original single-seed study and keeps its historical output
# path, so re-running it never overwrites or is overwritten by a repeat.
# Any other seed writes to <model>_seed<N>/, which is what makes repeat runs
# possible at all: without it every seed would collide on the same checkpoint
# and history files.
#
# Repeats exist to measure the noise floor. The single-seed study could not
# distinguish a real 0.4-point difference between synthetic sources from
# run-to-run variation, which is why two of the four matched-ratio comparisons
# had to be scored as ties. With several seeds the spread is measured rather
# than assumed.
#
#   SEED=43 sbatch --export=ALL,SEED=43 scripts/hpc/train_array.sbatch
SEED="${SEED:-42}"
if [[ "$SEED" == "42" ]]; then
  OUTPUT_DIR="output/${CONFIG}/${MODEL}"
else
  OUTPUT_DIR="output/${CONFIG}/${MODEL}_seed${SEED}"
fi
mkdir -p "$OUTPUT_DIR"
echo "seed=${SEED}  output=${OUTPUT_DIR}"

# --- Per-model hyperparameters ---------------------------------------------
# Shared study recipe: fixed Synscapes-native 1440x720, with ALL augmentation
# disabled -- --crop-size 0, --scale-min/max 1.0, --no-photometric-distortion,
# and --no-hflip (see docs/hybrid_eval.md "Study run" section). 35 epochs with
# early stopping on val_loss (patience 5) and linear warmup (1000 steps) into
# the existing poly decay.
#
# --no-hflip matches the team's stated "no input modification" protocol. Drop
# that one flag if you decide the mild regularization from random mirroring is
# worth more than strict protocol adherence; it applies uniformly to all 9
# dataset mixes either way, so it does not bias the shift-vs-performance
# correlation.
#
# SegFormer-B2 batch size / LR are per the team's plan: try batch 8 first
# (needs ~34-40GB VRAM per that estimate -- unverified against this cluster's
# actual A100 memory; run scripts/hpc/setup_venv.sh's interactive session and
# profile one step with torch.cuda.max_memory_allocated() before trusting
# this), falling back to batch 4 if it OOMs. --lr sets one uniform learning
# rate rather than the script's default backbone/head split -- confirm that's
# what was intended before relying on it for the paper.
#
# DeepLabV3+ batch size / LR are NOT yet decided by the team (only SegFormer's
# were discussed). Left at train.py's own defaults (backbone-lr 1e-4,
# head-lr 1e-3) with a batch size guess -- revisit before the real study runs.
case "$MODEL" in
  segformer)
    BATCH_SIZE=8
    LR_FLAGS="--lr 5e-5"
    ;;
  deeplabv3plus)
    BATCH_SIZE=8
    LR_FLAGS=""   # uses train.py's default --backbone-lr/--head-lr
    ;;
  *)
    echo "Unhandled model '$MODEL'" >&2
    exit 1
    ;;
esac

RESUME_FLAGS=()
LATEST_CHECKPOINT="${OUTPUT_DIR}/latest_model_${MODEL}.pth"
if [[ -f "$LATEST_CHECKPOINT" ]]; then
  echo "Resuming from ${LATEST_CHECKPOINT}"
  RESUME_FLAGS=(--resume "$LATEST_CHECKPOINT")
fi

srun python -m hybrid_eval.training.train \
  --model "$MODEL" \
  --train-img-dir "${TRAIN_DIR}/img" \
  --train-mask-dir "${TRAIN_DIR}/mask" \
  --val-img-dir "$VAL_IMG_DIR" \
  --val-mask-dir "$VAL_MASK_DIR" \
  --num-classes 19 \
  --image-height 720 \
  --image-width 1440 \
  --crop-size 0 \
  --scale-min 1.0 \
  --scale-max 1.0 \
  --no-photometric-distortion \
  --no-hflip \
  --epochs 35 \
  --warmup-steps 1000 \
  --early-stopping-patience 5 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
  --seed "$SEED" \
  $LR_FLAGS \
  --output-dir "$OUTPUT_DIR" \
  "${RESUME_FLAGS[@]}"
