# Large artifacts on Google Drive

The repository holds the code and the result JSONs. Study artifacts and
anything too large for GitHub lives on an external storage, laid out to mirror 
the repository so it can be copied straight back over a fresh clone.

## What is there

```
shift_manifests/<dataset>.csv                                    2.3 MB  10 files
checkpoints/<mixture>/<model>[_seed<N>]/best_model_<model>.pth   8.8 GB, 54 files
predictions/<mixture>/<model>[_seed<N>]/test_eval/masks/*.png    6.5 GB, 52,650 files
run_json/<mixture>/<model>[_seed<N>]/...                          29 MB, 108 files
shift_json/correlation_*.json, sadge/*.json                      1.9 MB, 13 files
features/<dataset>_<encoder>.npy                                 115 MB, 20 files
summaries/<dataset>_<summary>.npz                                 99 MB, 70 files
logs/                                                             18 MB  173 files
```

`shift_manifests/` are the split-separated dataset manifests created for the 
study's shift quantification. They are only required for exact study replication. 

`checkpoints/` are the trained models, one per mixture, architecture and seed.
Nine mixtures times two architectures times three seeds. These are the only
artifacts that cannot be reproduced without re-running the study, which took
about 40 GPU hours.

`predictions/` are the per-pixel outputs each model produced on the 975
held-out real test images, 975 PNGs per run. Included so qualitative analysis
needs neither a GPU nor the 16 GB of source images: where one mixture fails and
another does not, which classes get confused, per-image error breakdowns.

`run_json/` holds each run's `training_history_<model>.json` and its
`test_eval/summary.json`, the latter carrying mIoU, ECE, prediction entropy and
per-class IoU. `shift_json/` holds the four correlation outputs and the nine
per-mixture SADGE results.

Only `results/shift/distances.json` stays in the repository, since it is the
one artifact someone evaluating a new shift metric would compare against.

`features/` are the frozen DINOv2 and StreetCLIP embeddings behind the FID,
FCD and centroid metrics. Reproducible, but the encoder passes take a few GPU
hours.

`summaries/` are the per-dataset `.npz` artifacts the histogram metrics
compare. About an hour of CPU to rebuild.

`logs/` are the SLURM job logs. Not needed to reproduce anything, kept as a
record of what ran when.

## What is deliberately not there

**`latest_model_*.pth` (28 GB, 54 files).** These hold the last epoch's weights
plus AdamW moment buffers and scheduler state, which is why each is roughly
three times the size of the matching `best_model`. The optimizer state only
matters for resuming a run, and every run finished. The weights themselves are
from the final epoch rather than the best one, so they are the more overfit of
the two; `training_history_<model>.json` already records per-epoch validation
metrics if that is what you want to look at. `inference.py` loads
`best_model_*.pth`.

**The source images (16 GB), `data/study/streetViewData/`.** On a private 
share drive since dataset licenses prohibit dataset replication.

## Rebuilding a working copy

```bash
git clone <repo> && cd ShiftBench
git submodule update --init --recursive   # only needed for SADGE
rsync -av <share>/shift_manifests/*.csv   data/study/
rsync -av <share>/checkpoints/            output/
rsync -av <share>/predictions/            output/
rsync -av <share>/run_json/               output/
rsync -av <share>/shift_json/             results/shift/
rsync -av <share>/features/               results/shift/features/
rsync -av <share>/summaries/              results/shift/summaries/
```

`checkpoints/`, `predictions/` and `run_json/` all merge into `output/`, since
they follow the same `<mixture>/<model>/` layout. `shift_json/` merges into
`results/shift/` alongside the tracked `distances.json`.

The source images are only needed to retrain or to recompute the shift metrics
from scratch. If needed, you must download them from their official providers
and prepare them as described in [`data/README.md`](data/README.md). 
Without them, `run_json/` and `shift_json/` are enough to re-run 
`scripts/analyze_correlation.py` and reproduce every number in the paper, 
and `predictions/` is enough to inspect what the models actually output.
