# ShiftBench

ShiftBench is an open research benchmark for understanding how the composition of real and synthetic training data affects model performance. It evaluates whether distribution-shift metrics computed before training can identify promising dataset mixtures and reduce the need for expensive trial-and-error experiments.

## Current Scope

- **Distribution-shift metrics** across three families (below), addressable by
  name through one registry.
- **Feature extraction** with frozen encoders (DINOv2, DINOv3, StreetCLIP).
- **A reproducible experiment scaffold**: config-driven runs that validate a
  dataset, compute the configured distances, and write structured artifacts.

Model training, paper-scale configs, artifact versioning, and figure generation
are still to come. See [Known Gaps](#known-gaps).

## The Three Metric Families

Each family is a different lens on how far two datasets are apart:

| family | metrics | reads | measures |
| --- | --- | --- | --- |
| representation-based | `centroid`, `frechet` | embeddings | shift in what a pretrained network sees |
| image-based | `color_js`, `texture_js` | images | shift in raw appearance (HSV, LBP histograms) |
| label-based | `class_frequency_js`, `class_presence_js`, `scene_complexity_js` | masks | shift in the annotations |
| pairwise | `sadge` | image directories | fused geometric + appearance similarity |

The first three follow one pattern — *summarize each dataset, then compare the
two summaries* — with a Gaussian or an averaged histogram as the summary.
SADGE is the exception: it matches images across the two datasets, cannot
summarize one dataset alone, and (unlike every distance above) is a
**similarity — higher means more alike**, which is recorded next to its score.

## Environment

The project targets Python `3.11.9`.

```bash
git clone --recursive <repo>   # --recursive is needed only for SADGE (MASt3R)
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

`numpy` and `scipy` are the only required dependencies. Extras per family:

| extra | installs | needed for |
| --- | --- | --- |
| `.[features]` | torch, transformers, pillow | feature extraction |
| `.[image]` | opencv-python, scikit-image, pillow | image metrics, decoding files |
| `.[sadge]` | opencv-python + the torch stack | SADGE (see below) |

SADGE additionally needs the MASt3R **git submodule** (`git submodule update
--init --recursive` if you didn't clone recursively) and MASt3R's own runtime
requirements; model weights download from Hugging Face on first use.

The scripts insert `src/` on `sys.path` themselves, so they also run from a
clone without installing the package.

## Measuring Distribution Shift

Three steps: get each dataset's raw data into arrays, summarize each dataset,
compare two summaries.

### 1. Extract features (representation metrics only)

```bash
python3 scripts/extract_features.py real.npy --config configs/real_images.toml
python3 scripts/extract_features.py synth.npy --manifest data/synthetic.csv --encoder streetclip
```

`--encoder` is `dinov2` (default), `dinov3`, or `streetclip`; `--model`
overrides the checkpoint. Alongside `real.npy` this writes `real.npy.json`
recording the encoder and checkpoint — keep the two files together.

### 2. Summarize each dataset

```bash
python3 scripts/compute_feature_stats.py real.npy real.npz                       # gaussian (default)
python3 scripts/compute_feature_stats.py real.csv real_color.npz --summary color --image-column image_path
python3 scripts/compute_feature_stats.py real.csv real_cls.npz --summary class_frequency \
    --mask-column seg_path --num-classes 19
```

Gaussian summaries read an embeddings `.npy`; histogram summaries read a CSV
manifest and decode the images or masks it lists (masks may be `.png` or
`.npy`). Every artifact records its summary kind, hyperparameters, and — for
embeddings — the encoder.

### 3. Compare two datasets

```bash
python3 scripts/compute_distance.py real.npz synth.npz --metric frechet
python3 scripts/compute_distance.py real_cls.npz synth_cls.npz --metric class_frequency_js -o d.txt
```

`--metric` is required: the output is a bare number that does not say which
distance it is. Mismatched artifacts **refuse to compare** — different
encoders, different checkpoints, different summary kinds, or different
hyperparameters (a 32-bin color histogram is not comparable to a 16-bin one)
all fail with an error instead of producing a plausible wrong number.

SADGE has no summary artifact and runs only through an experiment (below).

## Experiments

```bash
PYTHONPATH=src python3 -m shiftbench.experiments.run --config configs/smoke.toml
```

Each run creates a directory under `runs/` containing `config.toml`,
`config.normalized.json`, `validation.json`, `metrics.json`, and logs
(including the git commit).

### Recording distances in a run

The optional `[shift]` table selects metrics by name and supplies whatever
inputs those metrics declare — validated at config load, so a bad setup fails
before any work starts:

```toml
[shift]
metrics = ["frechet", "color_js", "class_frequency_js", "sadge"]
stats_a = "../features/real.npz"        # embeddings metrics
stats_b = "../features/synthetic.npz"
manifest_a = "../data/real.csv"         # image/mask metrics
manifest_b = "../data/synthetic.csv"
image_column = "image_path"
mask_column = "seg_path"
num_classes = 19
image_dir_a = "../data/real_images"     # pairwise metrics (SADGE)
image_dir_b = "../data/synth_images"
```

`metrics.json` then gains a `shift` block with every distance next to the
provenance and hyperparameters that produced it:

```json
"shift": {
  "distances": {"frechet": 8.59, "class_frequency_js": 0.59},
  "encoder": "dinov2", "model": "facebook/dinov2-base",
  "params": {"class_frequency_js": {"mask_column": "seg_path", "num_classes": 19}}
}
```

## Dataset Configs

A dataset declares its columns in TOML. `text_column` and `image_column` are
both optional but at least one must be set; `mask_column` is supplementary and
requires `num_classes`, so every label metric records the class count it
assumed. Validation checks that declared image and mask files actually exist.
[`configs/smoke.toml`](configs/smoke.toml) is a text-only example.

## Repository Layout

```text
configs/                     Experiment configurations
data/sample/                 Tiny tracked data for tests and smoke runs
runs/, results/              Local artifacts, ignored by Git
scripts/                     Command-line entry points (parsing + I/O only)
src/shiftbench/
  config.py                  TOML config loading and validation
  provenance.py              What produced which artifact, and comparability
  metrics/                   Metric + Summary registries; gaussian, JS,
                             histogram, and pairwise implementations
  datasets/                  Schema validation, manifest reading, file decoding
  features/                  Frozen encoders (dinov2, dinov3, streetclip) and
                             their registry
  experiments/               Reproducible run scaffold
  shift_quantification_metrics/
                             Metric family implementations (histograms, SADGE;
                             SADGE vendors MASt3R as a git submodule)
tests/                       Unit, characterization, and smoke tests
docs/integration-plan.md     How the metric families were integrated, and the
                             decisions taken along the way
```

## Tests

```bash
python3 -m unittest discover -s tests
```

102 tests requiring only `numpy` and `scipy`; 7 skip without the optional
extras. They cover config validation (including per-metric `[shift]` rules),
schema and manifest handling, the metric registries, characterization pins of
every histogram metric's numbers, equivalence of the summary-artifact path
with the original implementations, and run artifacts including refusal paths.
Encoder backends are wiring-tested against stubs; real forward passes need
`.[features]` and a checkpoint download.

## Known Gaps

- **No real model forward pass has been executed in development.** All three
  encoders and SADGE are verified at the wiring level only; the first real
  extraction/SADGE run is still ahead. SADGE's MASt3R runtime requirements
  (einops etc.) are not covered by the `.[sadge]` extra.
- `provenance.py` is untested, and unknown provenance warns rather than fails,
  so one unlabeled artifact can still slip through a comparison.
- Feature provenance is a sidecar file. Move a `.npy` without its `.json` and
  the record is lost.
- `scripts/extract_features.py --help` requires `torch`, since the import
  happens at module load.
- No committed config demonstrates `image_column` or `mask_column`.

## Data Policy

Only small sample data should be committed to Git. Large source datasets,
generated datasets, model outputs, and paper artifacts belong in ignored local
directories until the project chooses a versioning mechanism such as DVC, Git
LFS, Hugging Face Datasets, or an archived release with checksums.

Paper-critical results should always be reproducible from committed configs and
documented commands.
