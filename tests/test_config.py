from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shiftbench.config import load_experiment_config


class ExperimentConfigTest(unittest.TestCase):
    def test_loads_smoke_config_with_resolved_paths(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.toml")

        self.assertEqual(config.name, "smoke")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.dataset.path.name, "tiny_shiftbench.csv")
        self.assertTrue(config.dataset.path.is_absolute())

    def test_rejects_missing_schema_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "bad.toml"
            config_path.write_text(
                """
[experiment]
name = "bad"
seed = 1
output_root = "runs"
train_selection = {real = 2000}

[dataset]
path = "data.csv"
required_columns = ["sample_id"]
id_column = "sample_id"
split_column = "split"
source_column = "source"
label_column = "label"
input_column = "input"
allowed_splits = ["train"]
allowed_sources = ["real"]
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "required_columns"):
                load_experiment_config(config_path)


class DatasetConfigTest(unittest.TestCase):
    """We always need to provide input_column and label_column."""

    def _write_config(self, directory: str, dataset_body: str) -> Path:
        config_path = Path(directory) / "modality.toml"
        config_path.write_text(
            f"""
[experiment]
name = "modality"
seed = 1
output_root = "runs"
train_selection = {{ real = 2000 }}

[dataset]
path = "data.csv"
id_column = "sample_id"
split_column = "split"
source_column = "source"
allowed_splits = ["train"]
allowed_sources = ["real"]
{dataset_body}
""".strip(),
            encoding="utf-8",
        )
        return config_path

    def test_input_column_must_be_required(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label"]\n'
                'input_column = "image_path"\n'
                'label_column = "label"',
            )

            with self.assertRaisesRegex(ValueError, "image_path"):
                load_experiment_config(config_path)

    def test_label_column_must_be_required(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "image_path"]\n'
                'input_column = "image_path"\n'
                'label_column = "label"',
            )

            with self.assertRaisesRegex(ValueError, "label"):
                load_experiment_config(config_path)

    def test_input_column_must_be_provided(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label"]\n'
                'label_column = "label"',
            )

            with self.assertRaisesRegex(ValueError, "input_column"):
                load_experiment_config(config_path)

    def test_label_column_must_be_provided(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "image_path"]\n'
                'input_column = "image_path"',
            )

            with self.assertRaisesRegex(ValueError, "label_column"):
                load_experiment_config(config_path)


class MaskConfigTest(unittest.TestCase):
    """Class count num_classes requirements for the config."""

    def _write_config(self, directory: str, dataset_body: str) -> Path:
        config_path = Path(directory) / "mask.toml"
        config_path.write_text(
            f"""
[experiment]
name = "mask"
seed = 1
output_root = "runs"
train_selection = {{ real = 2000 }}

[dataset]
path = "data.csv"
id_column = "sample_id"
split_column = "split"
source_column = "source"
label_column = "label"
input_column = "image_path"
allowed_splits = ["train"]
allowed_sources = ["real"]
{dataset_body}
""".strip(),
            encoding="utf-8",
        )
        return config_path

    def test_rejects_non_positive_num_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label", '
                '"image_path"]\n'
                "num_classes = 0",
            )

            with self.assertRaisesRegex(ValueError, "positive integer"):
                load_experiment_config(config_path)


class ShiftValidationTest(unittest.TestCase):
    """A [shift] table must fail at config load when a requested metric's
    inputs are missing, not mid-run."""

    def _write_config(self, directory: str, shift_body: str) -> Path:
        config_path = Path(directory) / "shift.toml"
        config_path.write_text(
            f"""
[experiment]
name = "shift"
seed = 1
output_root = "runs"
train_selection = {{ real = 2000 }}

[dataset]
path = "data.csv"
required_columns = ["sample_id", "split", "source", "label", "text"]
id_column = "sample_id"
split_column = "split"
source_column = "source"
label_column = "label"
input_column = "text"
allowed_splits = ["train"]
allowed_sources = ["real"]

[shift]
{shift_body}
""".strip(),
            encoding="utf-8",
        )
        return config_path

    def test_accepts_mixed_inputs_when_all_sources_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'metrics = ["frechet", "class_frequency_js"]\n'
                'stats_a = "a.npz"\nstats_b = "b.npz"\n'
                'manifest_a = "a.csv"\nmanifest_b = "b.csv"\n'
                'mask_column = "seg_path"\nnum_classes = 19',
            )

            shift = load_experiment_config(config_path).shift

            self.assertEqual(
                shift.metrics, ("frechet", "class_frequency_js")
            )
            self.assertEqual(shift.num_classes, 19)

    def test_rejects_embeddings_metric_without_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'metrics = ["frechet"]\n'
                'manifest_a = "a.csv"\nmanifest_b = "b.csv"',
            )

            with self.assertRaisesRegex(ValueError, "stats_a and stats_b"):
                load_experiment_config(config_path)

    def test_rejects_image_metric_without_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'metrics = ["color_js"]\nstats_a = "a.npz"\nstats_b = "b.npz"',
            )

            with self.assertRaisesRegex(ValueError, "manifest_a and manifest_b"):
                load_experiment_config(config_path)

    def test_rejects_mask_metric_without_num_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'metrics = ["class_frequency_js"]\n'
                'manifest_a = "a.csv"\nmanifest_b = "b.csv"\n'
                'mask_column = "seg_path"',
            )

            with self.assertRaisesRegex(ValueError, "num_classes"):
                load_experiment_config(config_path)

    def test_accepts_pairwise_metric_with_image_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'metrics = ["sadge"]\n'
                'image_dir_a = "real_images"\nimage_dir_b = "synth_images"',
            )

            shift = load_experiment_config(config_path).shift

            self.assertTrue(str(shift.image_dir_a).endswith("real_images"))

    def test_rejects_pairwise_metric_without_image_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'metrics = ["sadge"]\nstats_a = "a.npz"\nstats_b = "b.npz"',
            )

            with self.assertRaisesRegex(ValueError, "image_dir_a and image_dir_b"):
                load_experiment_config(config_path)

    def test_rejects_half_a_stats_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'metrics = ["frechet"]\nstats_a = "a.npz"',
            )

            with self.assertRaisesRegex(ValueError, "set together"):
                load_experiment_config(config_path)


if __name__ == "__main__":
    unittest.main()
