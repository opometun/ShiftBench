# Data Layout
ShiftBench keeps large or generated data out of Git.

- `sample/`: tiny tracked datasets used for tests and smoke runs.
- `study/`: tracked datasets used for experiments in the street view image segementation study.
- `study/streetViewData/`: local downloaded source data, ignored by Git.


# Preparation
To replicate our study, you need to prepare the data.

### Step 1: Download raw data
#### Cityscapes
The Cityscapes data is publically available at https://www.cityscapes-dataset.com/ after registration. <br>

To replicate our dataset, download the following two packages: 
- `leftImg8bit_trainvaltest.zip` (11GB) for the input images
- `gtFine_trainvaltest.zip` (241MB) for the image annotations

We used all data samples where fine annotations were publically available (the annotations of the 1525 test samples are not available), because the annotations are required for measuring the label-based shift. <br>

This resulted in the following split (city-based to avoid information leakage): <br>
| Split | Total number of samples | Cities and their number of samples |
| ----- | ----------------------- | -----------------------------------|
| Train | 2,000                   | <ul><li> Bochum (96 samples) </li><li> Cologne (154 samples) </li><li> Darmstadt (85 samples) </li><li> Erfurt (109 samples) </li><li> Hamburg (248 samples) </li><li> Hanover (196 samples) </li><li> Jena (119 samples) </li><li> Krefeld (99 samples) </li><li> Monchengladbach (94 samples) </li><li> Strasbourg (365 samples) </li><li> Stuttgart (196 samples) </li><li> Tubingen (144 samples) </li><li> Ulm (95 samples)</li></ul> |
| Validation | 500                | <ul><li> Frankfurt (267 samples) </li><li> Lindau (59 samples) </li><li> Munster (174 samples) </li></ul> |
| Test | 975                      | <ul><li> Aachen (174 samples) </li><li> Bremen (316 samples) </li><li> Dusseldorf (221 samples) </li><li> Weimar (142 samples) </li><li> Zurich (122 samples) </li></ul> |

For the exact overview, please check out the `./study/cityscapes100.csv` file, which includes all file names of the data samples used. <br>

We used the label ID annotation masks as label in our study. You need to remove the filename ending `_gtFine_labelIds`, because the code expects each (img, label)-pair to share the exact same filename.

#### Synscapes
The Synscapes data is publically available at https://synscapes.on.liu.se/index.html. <br>
You need to download the full dataset (337GB) as described in https://synscapes.on.liu.se/download.html. This will create a `Synscapes/`folder.

To replicate our dataset, select the first 2,000 samples. <br>
We used the input images in original resolution (1440 x 720) rather than their upscaled version (2024 x 1012). The input images can be found in `Synscapes/img/rgb/`. <br>
The segmentation masks can be found in `Synscapes/img/class/`. <br>

For the exact overview, please check out the `./study/synscapes100.csv` file, which includes all file names of the data samples used. Note that we used Synscapes data only for the train split.

#### GTA-V (TU Darmstadt)
The GTA-V data is publically available at https://download.visinf.tu-darmstadt.de/data/from_games/. <br>

To replicate our dataset, download the first two image (5.71GB and 5.73GB) and label (69.2MB and 71.8MB) packages. <br>
We selected the first 2,526 data samples, but skipped 526 images that were taken in poor natural light condition (night, sunset/sunrise) as well as in a rather rural environment (i.e. no pedestrian walkway, street signs/lights, or buildings). This selection was made due to the original Cityscapes task restriction of **daytime urban** street scene segmentation.<br>

For the exact overview, please check out the `./study/gta100.csv` file, which includes all 2,000 file names of the data samples used. Note that we used GTA-V data only for the train split.

### Step 2: Store data samples in required folder structure
We used a specific folder structure in our study. Please structure your data accordingly if you want to replicate our study, and put the `streetViewData` folder under `data/study/` in your local repository.

```
streetViewData/
    train/
        cityscapes/
	        img/           # the 2,000 Cityscapes train input samples
                mask/          # the 2,000 Cityscapes train segmentation mask
        synscapes/
	        img/           # the 2,000 Synscapes train input samples
                mask/          # the 2,000 Synscapes train segmentation mask
        gtaV/
	        img/           # the 2,000 GTA-V train input samples
                mask/          # the 2,000 GTA-V train segmentation mask
    validation/
        cityscapes/
	        img/           # the 500 Cityscapes validation input samples
                mask/          # the 500 Cityscapes validation segmentation masks
    test/
        cityscapes/
	        img/           # the 975 Cityscapes test input samples
                mask/          # the 975 Cityscapes test segmentation masks
```

### Step 3 (optional): Create the dataset CSVs
We provided our CSV files for each experiment's dataset in `./study/`. In case you want to recreate them or vary the randomly selected samples in each hybrid configuration, you can run the following commands from the `data/study/` folder:
- **100% Cityscapes training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/cityscapes100.toml" --ds-root-path "./streetViewData"`
- **75% Cityscapes + 25% Synscapes training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/cityscapes75_synscapes25.toml" --ds-root-path "./streetViewData"`
- **50% Cityscapes + 50% Synscapes training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/cityscapes50_synscapes50.toml" --ds-root-path "./streetViewData"`
- **25% Cityscapes + 75% Synscapes training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/cityscapes25_synscapes75.toml" --ds-root-path "./streetViewData"`
- **100% Synscapes training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/synscapes100.toml" --ds-root-path "./streetViewData"`
- **75% Cityscapes + 25% GTA-V training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/cityscapes75_gta25.toml" --ds-root-path "./streetViewData"`
- **50% Cityscapes + 50% GTA-V training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/cityscapes50_gta50.toml" --ds-root-path "./streetViewData"`
- **25% Cityscapes + 75% GTA-V training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/cityscapes25_gta75.toml" --ds-root-path "./streetViewData"`
- **100% GTA-V training dataset**: `python ..\..\scripts\rebuild_streetview_dataset_csv.py --experiment-path "../../configs/gta100.toml" --ds-root-path "./streetViewData"`

### Step 4: Resize data samples to uniform resolution
We used a uniform resolution of (1440 x 720) to cancel out the potential impact of resolution on the shift and model performance. Hence, we downscaled Cityscapes images, and downscaled + center cropped GTA-V images. <br>
If you want to replicate our study, run the following commands from the `data/` folder:
- **Resize Cityscapes data**: `python ..\scripts\prepare_data.py --experiment-path "../configs/cityscapes100.toml"`
- **Resize GTA-V data**: `python ..\scripts\prepare_data.py --experiment-path "../configs/gta100.toml"`