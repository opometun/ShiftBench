# ShiftBench

ShiftBench is an open research benchmark for understanding how the composition of real and synthetic training data affects model performance. It evaluates whether distribution-shift metrics computed before training can identify promising dataset mixtures and reduce the need for expensive trial-and-error experiments.

## Current Scope

- **Distribution-shift metrics** across three families (below), addressable by
  name through one registry.
- **Feature extraction** with frozen encoders (DINOv2, DINOv3, StreetCLIP).
- **A reproducible experiment scaffold**: config-driven runs that validate a
  dataset, compute the configured distances, and write structured artifacts.
- **Image segmentation** using pretrained models (DeepLabV3(+), SegFormer), 
  along with argument‑driven training and inference on custom datasets.
- **Paper-scale configs** for all experiments conducted in our study.
- **Correlation analysis** between quantified shift and downstream model performance.

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
python3 -m pip install -e ".[features]" 
```

`numpy` and `scipy` are the only *declared* base dependencies, but a bare
`pip install -e .` is not enough to import the package: `shiftbench.metrics`
imports `torch` at module scope for SADGE, and `shiftbench.config` imports
the metric registry in turn. `.[features]` is the practical minimum for
running the tests or computing any metric. Extras per family:

| extra | installs | needed for |
| --- | --- | --- |
| `.[ds_prep]` | pillow | data preparation |
| `.[train]` | torch, torchvision, transformer, pillow | segmentation model training/inference |
| `.[features]` | torch, transformers, pillow | feature extraction |
| `.[image]` | opencv-python, scikit-image, pillow | image metrics, decoding files |
| `.[sadge]` | opencv-python + the torch stack | SADGE (see below) |
| `.[dev]` | pytest | development workflow |
| `.[plot]` | matplotlib, seaborn | replicating our study's plots |

SADGE additionally needs the MASt3R **git submodule** (`git submodule update
--init --recursive` if you didn't clone recursively) and MASt3R's own runtime
requirements; model weights download from Hugging Face on first use.

The scripts insert `src/` on `sys.path` themselves, so they also run from a
clone without installing the package.

## Measuring Distribution Shift

The pipeline to get the *shift quantification of a **single** metric* between two datasets comprises the following three steps:
1. get both dataset's raw data into arrays
2. summarize both dataset
3. compare the two summaries

### 1. Extract features (representation-based metrics only)

```bash
python3 scripts/extract_features.py "./runs/cityscapes100/frechet_dinov2/train_features.npy" --config "configs/cityscapes100.toml" --encoder "dinov2" --split "train"
python3 scripts/extract_features.py "./runs/cityscapes100/frechet_dinov2/test_features.npy" --config "configs/cityscapes100.toml" --encoder "dinov2" --split "test"
```

`--encoder` is `dinov2` (default), `dinov3`, or `streetclip`; `--model`
overrides the checkpoint. Alongside `real.npy` this writes `real.npy.json`
recording the encoder and checkpoint — keep the two files together.

When passing `--config` as argument, the configuration file is used to access the manifest.
Alternatively, `--manifest` can be passed as argument for direct access (input column is guessed).

By default, feature extraction is applied to all samples in the manifest.
To apply feature extraction to only a certain split, specify the `--split` argument.
Be aware that when providing no config, the split column must be named `split` in the manifest.

Running the script [`extract_features.py`](scripts/extract_features.py) returns a .npy file storing the extracted features 
and a .json file storing the information of the embedding model that was used for feature extraction.


### 2. Summarize each dataset

```bash
python3 scripts/compute_feature_stats.py "./runs/cityscapes100/frechet_dinov2/train_features.npy" "./runs/cityscapes100/frechet_dinov2/train_feature_stats.npz" --summary "gaussian"
python3 scripts/compute_feature_stats.py "./runs/cityscapes100/frechet_dinov2/test_features.npy" "./runs/cityscapes100/frechet_dinov2/test_feature_stats.npz" --summary "gaussian"
```

Gaussian summaries read an embeddings `.npy`; histogram summaries read a CSV
manifest and decode the images or masks it lists (masks may be `.png` or
`.npy`). Every artifact records its summary kind, hyperparameters, and — for
embeddings — the encoder.

By default, all images/masks are loaded in histogram summaries. 
To consider only a certain split, specify the `--split` argument.
Be aware that the split column must be named `split` in the manifest.

Running the script [`compute_feature_stats.py`](scripts/compute_feature_stats.py) returns a .npz file storing the summary (e.g. for `gaussian` it is mu and sigma).

### 3. Compare two datasets

```bash
python3 scripts/compute_distance.py "./runs/cityscapes100/frechet_dinov2/train_feature_stats.npz" "./runs/cityscapes100/frechet_dinov2/test_feature_stats.npz" --metric "frechet" --output-path "./runs/cityscapes100/frechet_dinov2/shift_score.txt"
```

`--metric` is required: the output is a bare number that does not say which
distance it is. Mismatched artifacts **refuse to compare** — different
encoders, different checkpoints, different summary kinds, or different
hyperparameters (a 32-bin color histogram is not comparable to a 16-bin one)
all fail with an error instead of producing a plausible wrong number.

Running the script [`compute_distance.py`](scripts/compute_distance.py) returns a .txt file storing the shift score (typically a float).

SADGE has no summary artifact and runs only through an experiment (below).

## Experiments

For computing the *shift quantification of **multiple selected** metrics* between two datasets, we suggest running an experiment. This avoids needing to manually conduct the scripts [`extract_features.py`](scripts/extract_features.py) (only for representation-based metrics), [`compute_feature_stats.py`](scripts/compute_feature_stats.py), and [`compute_distance.py`](scripts/compute_distance.py) for each metric.

To conduct this experiment, a `ShiftConfig` needs to be added to the experiment configuration file (the `.toml` file defined in `./configs/`).
It selects metrics by name and supplies whatever inputs those metrics declare. It is validated at config load, so a bad setup fails before any work starts:

```toml
[shift]
metrics = ["frechet", "color_js", "class_frequency_js", "sadge"]
stats_a = "../features/real.npz"        # embeddings metrics
stats_b = "../features/synthetic.npz"
manifest_a = "../data/real.csv"         # image/mask/pairwise metrics
manifest_b = "../data/synthetic.csv"
image_column = "image_path"
mask_column = "seg_path"
num_classes = 19
```

Optional arguments `split_a` and `split_b` allow to consider only a specific split of the manifests.

If you want to apply representation-based metrics, then you need to precompute the summaries for both datasets (path to summaries need to be stored in the config's `stats_a` and `stats_b`). To precompute the summaries, [`extract_features.py`](scripts/extract_features.py) (to get the embeddings) and [`compute_feature_stats.py`](scripts/compute_feature_stats.py) (to compute mu and sigma) need to be run for each representation-based metric and dataset involved in requested shift quantification. Precomputation is enforced here because embedding summaries are expensive to compute.

```bash
PYTHONPATH=src python3 -m shiftbench.experiments.run --config "configs/cityscapes100.toml"
```

Each run creates a directory under `runs/` containing `config.toml`,
`config.normalized.json`, `validation.json`, `metrics.json`, logs
(including the git commit), and summaries of non-embedding-based metrics.

The shift scores ("distances") and hyperparameter settings that produced them can be found in the `shift` block in `metrics.json`:

```json
"shift": {
  "distances": {"frechet": 8.59, "class_frequency_js": 0.59},
  "encoder": "dinov2", "model": "facebook/dinov2-base",
  "params": {"class_frequency_js": {"mask_column": "seg_path", "num_classes": 19}}
}
```

  NOTE: Our predefined experiment configs in [`configs/`](configs/) do not include `sadge` in the Shift configuration's `metrics` list, because it demands GPU and special authentification. If you want to run SADGE this way, you must add it to the list and ensure that all requirements are satisfied:
- create a HuggingFace account and an access token with read permission
- request access to `facebook/dinov3-vitl16-pretrain-lvd1689m`
- log into HuggingFace via terminal `hf auth login` and use your access token

## Dataset Configs

A dataset declares its columns in TOML. In our study, `input_column` and `label_column` store the input image and segmentation mask directories respectively. Validation checks that declared image and mask files actually exist. Although our study's experiments focus on image segmentation, where both inputs and labels are images, the dataset configuration also supports other input and label modalities. <br>
[`configs/smoke.toml`](configs/smoke.toml) is a text-only example. <br>
[`configs/`](configs/) contains the configuration files of our study that can be used to replicate our experiments. 

To replicate our study, follow the dataset preparation steps stated in [`data.README.md`](data/README.md).

## Model training/inference

This repository only provides the implementation of a few selected segmentation models.

Model training and inference expect directory paths that contain all and only the images or masks of a single split (train, val, or test). However, our raw dataset `streetViewData` is organized by source (see [`data/README.md`](data/README.md)). <br>
Most experiment training datasets are a hybrid mix of two sources. Because these samples are spread across multiple source folders, there exists no directory that contains exactly the train images for a given configuration. <br>
Validation and test splits do not have this problem, since they consist solely of Cityscapes and already live in single‑source folders. <br>
To prepare the train split, [`materialize_train_split.py`](scripts/materialize_train_split.py) reads the unified manifest, filters rows belonging to the train split, resolves image and mask paths, and materializes them into temporary directories `<output>/img/` and `<output>/mask/`. <br>
These temporary directories contain exactly the train samples for the chosen configuration and can be passed directly to the training script ([`train.py`](src/hybrid_eval/training/train.py)). <br>
At inference ([`inference.py`](src/hybrid_eval/inference.py)), the trained checkpoint is loaded and used to generate predictions on the test split.

## Shift quantification and correlation analyis (study replicability)

Individual manifests were created for each relevant split using [`make_shift_manifests.py`](scripts/make_shift_manifests.py), because at the time of coducting the experiments, shift quantification always considered the entire manifest. With the introduction of `split_a` and `split_b`, this separated manifest creation isn't necessary anymore for shift quantification. <br>
Our study used [`run_shift_metrics.py`](scripts/run_shift_metrics.py) to compute the shift scores of each dataset configuration. However, the shift scores can also be computed in the usual way described above. <br>
The SADGE metric is considered separately by running [`run_sadge.py`](scripts/run_sadge.py) in a controlled per-mixture manner, and then merging the per-mixture SADGE scores using [`merge_sadge.py`](scripts/merge_sadge.py).

Correlation analysis was conducted in [`analyze_correlation.py`](scripts/analyze_correlation.py).

For more detailed insights, see [`REPRODUCE.md`](REPRODUCE.md).

## Repository Layout

```text
configs/                                 Experiment configurations
data/
├── sample/                              Tiny tracked data for tests and smoke runs
└── study/                               Tracked data to replicate our study's experiments
results/                                 Local artifacts, ignored by Git
└── shift/
    └── study/                           Shift scores computed in our study
runs/                                    Local artifacts, ignored by Git
scripts/                                 Command-line entry points, HPC handling
src/
├── hybrid_eval/
│   ├── models/                          Model definitions of SegFormer-B2 and DeepLabV3(+)
│   ├── training/                        Utilities for training 
│   │                                    (data, metrics, checkpoints, training CLI)
│   └── inference.py                     Checkpoint-driven inference CLI
└── shiftbench/
    ├── datasets/                        Schema validation, manifest reading, 
    │                                    dataset preparation, file decoding
    ├── experiments/                     Reproducible run scaffold
    ├── features/                        Frozen encoders (dinov2, dinov3, streetclip) and
    │                                    their registry
    ├── metrics/                         Metric + Summary registries; gaussian, JS,
    │                                    histogram, and pairwise implementations
    ├── shift_quantification_metrics/    Metric family implementations (histograms, SADGE;
    │                                    SADGE vendors MASt3R as a git submodule)
    ├── config.py                        TOML config loading and validation
    └── provenance.py                    What produced which artifact, and comparability
tests/                                   Unit, characterization, and smoke tests
paper/                                   Scientific paper (IEEE format)
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests need the same `.[features]` install as the rest of the project; a
few additionally skip without the other optional extras. They cover config
validation (including per-metric `[shift]` rules), schema and manifest
handling, the metric registries, characterization pins of every histogram
metric's numbers, equivalence of the summary-artifact path with the original
implementations, and run artifacts including refusal paths. Encoder backends
are wiring-tested against stubs; real forward passes also need a checkpoint
download.

## Known Gaps

- no unit tests for [`prepare.py`](src/shiftbench/datasets/prepare.py) 
- [`provenance.py`](src/shiftbench/provenance.py) is untested, and unknown provenance warns rather than fails,
  so one unlabeled artifact can still slip through a comparison.
- Feature provenance is a sidecar file. Move a `.npy` without its `.json` and
  the record is lost.
- currently only argument-driven model training and inference (might want to switch to config-driven in future)
- figure generation still missing
- we did not test running SADGE via config yet 
- `load_masks` (see [`loaders.py`](src/shiftbench/datasets/loaders.py)) has Cityscapes LUT as default transform. Hence, every component that relies on `load_masks`, including [`compute_feature_stats.py`](scripts/compute_feature_stats.py), [`run.py`](src/shiftbench/experiments/run.py), and some tests, implicilty uses that mask transform. Swapping in a different LUT (or None at all) therefore requires touching several places in the codebase rather than adjusting a single configuration entry.

## Data Policy

Always review the LICENSE terms of any dataset you use. The datasets employed in our study (Cityscapes, Synscapes, and GTA-V) cannot be redistributed under the terms set by their original creators. Hence, for replicating our experiments, please download the data from their original sources and prepare it as described in [`data/README.md`](data/README.md). 

## Study artifacts

`results/shift/distances.json` holds the shift scores for all nine mixtures
across twelve metrics, and is the file to compare a new metric against.

Everything else the study produced is on Google Drive: 
the split-seperated dataset manifests, the 108 trained checkpoints, 
their predicted masks on the test split, per-run training
histories and test summaries, encoder embeddings, and the correlation outputs.
[`FILESHARE.md`](FILESHARE.md) describes the layout and how to merge it back over a clone.
Contact the maintainers for access.

Evaluating a new shift metric does not require retraining. The per-mixture mIoU
and ECE our models achieved are in `shift_json/correlation_*.json` on the
Drive, so a new metric can be correlated against the same downstream numbers we
used. `shift_json/seeds6/` holds the same analysis over all six seeds, which is
the version the paper reports; the files directly under `shift_json/` are the
earlier three-seed run and are kept only so the older figures remain traceable.

[`REPRODUCE.md`](REPRODUCE.md) lists every command that produced these results, in order.
