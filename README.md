# ShiftBench

ShiftBench is an open research benchmark for understanding how the composition of real and synthetic training data affects model performance. It evaluates whether distribution-shift metrics computed before training can identify promising dataset mixtures and reduce the need for expensive trial-and-error experiments.

## Current Scope

The repository contains two pieces:

1. **A distribution-shift measurement pipeline** (`scripts/` + `src/shiftbench/`):
   frozen-encoder feature extraction, Gaussian summary statistics, and the
   centroid and Fréchet distances between two datasets.
2. **A reproducible experiment scaffold** (`src/shiftbench/experiments/run.py`):
   config-driven runs that validate a dataset, compute the configured
   distances, and write structured artifacts.

Feature extraction stays a separate manual step so that running an experiment
does not require `torch`; a run consumes the statistics it produced.

Model training, paper-scale configs, artifact versioning, and figure generation
are still to come. See [Known Gaps](#known-gaps) for what is deliberately
unfinished.

## Environment

The project targets Python `3.11.9`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

`numpy` and `scipy` are required. Feature extraction additionally needs
`torch`, `transformers`, and `pillow`, kept behind an extra so that comparing
already-extracted statistics does not pull in a deep learning stack:

```bash
python3 -m pip install -e '.[features]'
```

The scripts insert `src/` on `sys.path` themselves, so they also run from a
clone without installing the package.

## Measuring Distribution Shift

Three steps: embed each dataset, summarize each as a Gaussian, then compare two
summaries.

### 1. Extract features

```bash
python3 scripts/extract_features.py real.npy --config configs/real_images.toml
python3 scripts/extract_features.py synth.npy --manifest data/synthetic.csv
```

The manifest source is either `--config` (an experiment TOML, which supplies
both the manifest path and its `image_column`) or `--manifest` (a raw CSV,
whose image column is guessed). Prefer `--config`.

| flag | default | purpose |
| --- | --- | --- |
| `--encoder` | `dinov2` | `dinov2` or `streetclip` |
| `--model` | the encoder's own default | Hugging Face checkpoint id |
| `--image-column` | from config, else guessed | manifest column holding image paths |

Alongside `real.npy` this writes `real.npy.json` recording which encoder and
checkpoint produced it. Keep the two files together.

### 2. Summarize each dataset

```bash
python3 scripts/compute_feature_stats.py real.npy real.npz
```

Writes a mean vector and covariance matrix, carrying the encoder provenance
across from the sidecar. The `.npz` is a fixed small size no matter how many
images went into it, so it is the artifact worth keeping.

### 3. Compare two datasets

```bash
python3 scripts/compute_distance.py real.npz synth.npz --metric frechet
python3 scripts/compute_distance.py real.npz synth.npz --metric centroid -o d.txt
```

- **`centroid`** — distance between the two means. Cheap, but blind to spread:
  a generator that collapsed to near-identical outputs can still score ~0.
- **`frechet`** — the 2-Wasserstein distance between the two Gaussians, so it
  sees both position and spread. This is the FID formula over the chosen
  encoder's features rather than Inception's.

`--metric` is required: the output is a bare number that does not say which
distance it is.

### Encoders are not interchangeable

A distance is only meaningful between features from the *same* encoder and
checkpoint. DINOv2 uses the CLS token unnormalized; StreetCLIP uses the
projection head and L2-normalizes, putting every point on a unit sphere. The
numbers are on different scales and are not comparable. `compute_distance.py`
refuses to compare stats whose recorded provenance disagrees.

## Smoke Experiment

Run the tiny config-driven experiment:

```bash
PYTHONPATH=src python3 -m shiftbench.experiments.run --config configs/smoke.toml
```

or:

```bash
bash scripts/run_smoke_experiment.sh
```

Each run creates a directory under `runs/` containing:

- `config.toml`: exact config used for the run
- `config.normalized.json`: resolved config paths
- `validation.json`: dataset schema validation summary
- `metrics.json`: dataset counts, plus any configured distances
- `logs.json` and `logs.txt`: run metadata, including Python version and Git commit

### Recording distances in a run

Add an optional `[shift]` table naming two `.npz` stats files from step 2:

```toml
[shift]
stats_a = "../features/real.npz"
stats_b = "../features/synthetic.npz"
metrics = ["centroid", "frechet"]
```

`metrics.json` then gains a `shift` block holding the distances together with
the encoder and checkpoint they were computed under:

```json
"shift": {
  "encoder": "dinov2",
  "model": "facebook/dinov2-base",
  "stats_a": "/abs/path/real.npz",
  "stats_b": "/abs/path/synthetic.npz",
  "distances": {"centroid": 8.51, "frechet": 8.59}
}
```

Unknown metric names are rejected when the config loads. Comparing stats from
different encoders fails the run and is recorded in `logs.json`, rather than
producing a number that looks fine.

## Dataset Configs

A dataset declares its columns in TOML. `text_column` and `image_column` are
both optional, but at least one must be set, so an image dataset does not have
to invent a text column. Any column named this way must also appear in
`required_columns`. When `image_column` is set, validation checks that every
referenced image actually exists, resolved relative to the manifest.

[`configs/smoke.toml`](configs/smoke.toml) is a text-only example.

## Repository Layout

```text
configs/                 Experiment configurations
data/
  sample/                Tiny tracked data for tests and smoke runs
  raw/                   Local raw data, ignored by Git
  processed/             Local derived data, ignored by Git
  external/              Local third-party artifacts, ignored by Git
runs/                    Local run artifacts, ignored by Git
results/                 Local aggregated results, ignored by Git
scripts/                 Command-line entry points
src/shiftbench/
  config.py              TOML config loading and validation
  metrics.py             Distances, and the registry naming them
  provenance.py          Which encoder produced which artifact
  datasets/              Schema validation and manifest reading
  features/              Frozen encoders and the registry naming them
  experiments/           Reproducible run scaffold
tests/                   Unit and smoke tests
```

Scripts hold argument parsing and file I/O only; everything testable lives in
`src/shiftbench/`.

## Tests

```bash
python3 -m unittest discover -s tests
```

48 tests, requiring only `numpy` and `scipy`:

- config loading, including dataset modality rules and `[shift]` parsing
- dataset schema validation, including missing image files
- manifest reading and image-column resolution
- the distance metrics and their registry
- structured run outputs, including recorded distances and encoder mismatch

The encoder backends are not covered, since testing them needs `torch` and a
model download.

## Known Gaps

- Feature extraction is not part of a run, so a run records which statistics it
  compared but cannot itself reproduce them from images.
- `provenance.py` is untested, and unknown provenance warns rather than fails,
  so one unlabeled artifact can still slip through a comparison.
- Feature provenance is a sidecar file. Move a `.npy` without its `.json` and
  the record is lost.
- `scripts/extract_features.py --help` requires `torch` to be installed, since
  the import happens at module load.
- No committed config demonstrates `image_column`.

## Data Policy

Only small sample data should be committed to Git. Large source datasets,
generated datasets, model outputs, and paper artifacts belong in ignored local
directories until the project chooses a versioning mechanism such as DVC, Git
LFS, Hugging Face Datasets, or an archived release with checksums.

Paper-critical results should always be reproducible from committed configs and
documented commands.
