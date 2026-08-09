# Reproducing the study

Every command that produced the results in `results/shift/`, in order. Run on
hpc3 unless marked otherwise.

Two things about this cluster that are easy to trip over:

- `python3.11` exists only on compute nodes. The venv is built against it, so
  activating it on the head node gives a broken `python`. Anything that needs
  the venv goes through `srun` or `salloc`.
- `/tmp` and `/var/tmp` are 10 MB tmpfs. pip reports a full disk as a network
  error, so `TMPDIR` must point at node-local `/scratch`. The sbatch scripts
  do this themselves; interactive sessions need it set by hand.

## 1. Repository and submodules

Run where you have GitHub access, then rsync across. The cluster blocks
outbound git.

```bash
git clone https://github.com/opometun/ShiftBench.git && cd ShiftBench
git submodule update --init --recursive
```

`--recursive` matters: SADGE needs MASt3R, which needs dust3r, which needs
croco. Three levels.

## 2. Environment

```bash
salloc --partition=workq --ntasks=1 --cpus-per-task=8 --time=01:00:00
srun --pty bash -l
cd <repo> && bash scripts/hpc/setup_venv.sh
source ~/shiftbench-venv/bin/activate
export TMPDIR=/scratch/$USER/tmp && mkdir -p $TMPDIR
pip install -e ".[train,features,image,sadge]"
exit; exit
```

## 3. Data

Place the prepared images at `data/study/streetViewData/` with `train/`,
`validation/` and `test/` subdirectories, each holding `cityscapes/`,
`synscapes/` and `gtaV/` with `img/` and `mask/`. See `data/README.md` for how
the selection and resizing were done.

Then split the study CSVs by split. The metric scripts read every row of
whatever CSV they are given, so without this a mixture would be compared
against itself:

```bash
python scripts/make_shift_manifests.py --check-consistency
```

Writes `data/study/shift_<mixture>_train.csv` (2,000 rows each) and
`data/study/shift_inference.csv` (975 held-out real test images).

## 4. Training

Odd array indices are SegFormer, even are DeepLabV3+, per
`scripts/hpc/study_runs.txt`. `%2` caps concurrent tasks; the gpu partition is
small and usually contended.

```bash
sbatch --array=1,3,5,7,9,11,13,15,17%2 scripts/hpc/train_array.sbatch
sbatch --array=2,4,6,8,10,12,14,16,18%2 scripts/hpc/train_array.sbatch
```

Repeat seeds. Seed 42 writes to `output/<mixture>/<model>/`, anything else to
`output/<mixture>/<model>_seed<N>/`, so these never collide:

```bash
for s in 43 44; do
  sbatch --export=ALL,SEED=$s --array=1,3,5,7,9,11,13,15,17%2 scripts/hpc/train_array.sbatch
  sbatch --export=ALL,SEED=$s --array=2,4,6,8,10,12,14,16,18%2 scripts/hpc/train_array.sbatch
done
```

54 runs total. SegFormer takes roughly 1.75 h each, DeepLabV3+ about 40 min.

## 5. Test-split evaluation

Training reports mIoU on the validation split, which also drove early stopping
and checkpoint selection, so it is optimistically biased. The shift metrics are
computed against the test split, so downstream performance has to come from the
same split or the two axes describe different distributions.

```bash
for m in segformer deeplabv3plus; do for s in 42 43 44; do
  sbatch --export=ALL,MODEL=$m,SEED=$s scripts/hpc/evaluate_test.sbatch
done; done
```

Roughly 10 minutes per job. Writes `test_eval/summary.json` with mIoU, ECE,
entropy and per-class IoU, plus the predicted masks.

## 6. Shift metrics

Split in two on purpose. Only the encoder passes need a GPU, so running the
histogram half on `workq` frees a contended A100 and starts sooner.

```bash
sbatch scripts/hpc/shift_metrics_cpu.sbatch     # colour, texture, 3 label metrics
```

Wait for it to finish, then:

```bash
sbatch scripts/hpc/shift_metrics.sbatch         # DINOv2 and StreetCLIP
```

The driver skips artifacts that already exist, so the second job does only the
embedding work. Writes `results/shift/distances.json`.

## 7. SADGE

DINOv3 is gated on HuggingFace. Accept the licence at
`huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m`, then:

```bash
hf auth login
```

Verify the environment before queuing nine jobs. This checks the submodules,
installs the MASt3R runtime dependencies that `.[sadge]` does not cover, tests
the import chain, and caches both models:

```bash
salloc --partition=gpu --gres=gpu:1 --constraint="A100|H100.80gb" --ntasks=1 --cpus-per-task=8 --time=01:00:00
srun --pty bash -l
cd <repo> && bash scripts/hpc/setup_sadge.sh
exit; exit
```

Then:

```bash
sbatch scripts/hpc/sadge.sbatch
python3 scripts/merge_sadge.py
```

One mixture per array task, about 50 minutes each. `merge_sadge.py` folds the
scores into `distances.json` and warns about any mixture whose fusion sign is
inverted.

## 8. Analysis

```bash
srun --partition=workq --time=00:10:00 --cpus-per-task=2 bash -c \
  "cd <repo> && source ~/shiftbench-venv/bin/activate && \
   for m in segformer deeplabv3plus; do for t in miou ece; do \
     python scripts/analyze_correlation.py --model \$m --target \$t --seeds 42 43 44; \
   done; done"
```

Writes four files: `correlation_<model>.json` and
`correlation_<model>_ece.json`. Each reports Spearman and Pearson against the
downstream target, plus the within-ratio test that holds synthetic fraction
fixed and decides matched pairs with a Welch t-test on the per-seed values.

With `--seeds`, the tie threshold is measured from the observed spread rather
than assumed. With one seed it falls back to a fixed constant and says so.

## Monitoring

```bash
squeue --me -o "%.16i %.9P %.20j %.2t %.10M %.10L %R"
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed | grep -v "\.extern"
tail -5 logs/<job>_<id>.out
```

A stale `SLURM_JOB_ID` left in the shell by an exited `salloc` makes `srun`
fail with "Invalid job id specified". Clear it with
`unset SLURM_JOB_ID SLURM_JOBID`.

## Retrieving results

```bash
rsync -av <user>@hpc3.rz.uos.de:<repo>/results/ results/
rsync -av <user>@hpc3.rz.uos.de:<repo>/output/  output/
```

Checkpoints, predicted masks and encoder embeddings are too large for git and
live on the shared drive; see `FILESHARE.md`.
