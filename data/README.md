# Data Layout

ShiftBench keeps large or generated data out of Git.

- `sample/`: tiny tracked datasets used for tests and smoke runs.
- `raw/`: local downloaded source data, ignored by Git.
- `processed/`: local derived datasets, ignored by Git.
- `external/`: local third-party artifacts, ignored by Git.

Paper-critical datasets should later be versioned with DVC, Git LFS, Hugging
Face Datasets, or an archived release with checksums.
