from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plotting" / "figure_dataset_examples.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("figure_dataset_examples", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_find_dataset_examples_chooses_first_image_per_dataset(tmp_path):
    module = _load_module()

    train_root = tmp_path / "train"
    dataset_specs = {
        "cityscapes": [(10, 10, (255, 0, 0))],
        "synscapes": [(10, 10, (0, 255, 0))],
        "gtaV": [(10, 10, (0, 0, 255))],
    }

    for dataset, samples in dataset_specs.items():
        image_dir = train_root / dataset / "img"
        image_dir.mkdir(parents=True)
        for index, (_, _, color) in enumerate(samples):
            image = Image.new("RGB", (10, 10), color=color)
            image.save(image_dir / f"sample_{index}.png")

    examples = module.find_dataset_examples(train_root)

    assert set(examples) == {"cityscapes", "synscapes", "gtaV"}
    assert all(path.is_file() for path in examples.values())
    assert [path.name for path in examples.values()] == [
        "sample_0.png",
        "sample_0.png",
        "sample_0.png",
    ]
