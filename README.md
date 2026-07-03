# ShiftBench

ShiftBench is an open research benchmark for understanding how the composition of real and synthetic training data affects model performance. It evaluates whether distribution-shift metrics computed before training can identify promising dataset mixtures and reduce the need for expensive trial-and-error experiments.

## Current Scope

This repository currently contains the beginning infrastructure for a
reproducible scientific experiment:

- deterministic Python environment metadata
- config-driven experiment execution
- structured run outputs
- dataset schema validation
- a tiny tracked sample dataset
- minimal smoke and validation tests

Model training, distribution-shift metrics, paper-scale experiment configs,
artifact versioning, and figure/table generation should be added once the first
real experiment pipeline is stable.

## Environment

The project targets Python `3.11.9`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

The package currently has no runtime dependencies. This is intentional for the
initial scaffold: external scientific dependencies should be added and pinned
when the corresponding experiment code is introduced.

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
scripts/                 Convenience commands
src/shiftbench/          ShiftBench Python package
tests/                   Unit and smoke tests
```

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
- `metrics.json`: smoke metrics and dataset counts
- `logs.json` and `logs.txt`: run metadata, including Python version and Git commit

## Tests

Run the current test suite without extra dependencies:

```bash
python3 -m unittest discover -s tests
```

The tests cover:

- config loading and validation
- dataset schema validation
- structured smoke experiment outputs

## Data Policy

Only small sample data should be committed to Git. Large source datasets,
generated datasets, model outputs, and paper artifacts belong in ignored local
directories until the project chooses a versioning mechanism such as DVC, Git
LFS, Hugging Face Datasets, or an archived release with checksums.

Paper-critical results should always be reproducible from committed configs and
documented commands.
