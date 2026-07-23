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

[dataset]
path = "data.csv"
required_columns = ["sample_id"]
id_column = "sample_id"
split_column = "split"
source_column = "source"
label_column = "label"
text_column = "text"
allowed_splits = ["train"]
allowed_sources = ["real"]
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "required_columns"):
                load_experiment_config(config_path)


class DatasetModalityTest(unittest.TestCase):
    """An image dataset must not have to invent a text column."""

    def _write_config(self, directory: str, dataset_body: str) -> Path:
        config_path = Path(directory) / "modality.toml"
        config_path.write_text(
            f"""
[experiment]
name = "modality"
seed = 1
output_root = "runs"

[dataset]
path = "data.csv"
id_column = "sample_id"
split_column = "split"
source_column = "source"
label_column = "label"
allowed_splits = ["train"]
allowed_sources = ["real"]
{dataset_body}
""".strip(),
            encoding="utf-8",
        )
        return config_path

    def test_accepts_image_only_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label", '
                '"image_path"]\nimage_column = "image_path"',
            )

            config = load_experiment_config(config_path)

            self.assertEqual(config.dataset.image_column, "image_path")
            self.assertIsNone(config.dataset.text_column)

    def test_accepts_dataset_with_both_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label", '
                '"text", "image_path"]\ntext_column = "text"\n'
                'image_column = "image_path"',
            )

            config = load_experiment_config(config_path)

            self.assertEqual(config.dataset.text_column, "text")
            self.assertEqual(config.dataset.image_column, "image_path")

    def test_rejects_dataset_with_neither_modality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label"]',
            )

            with self.assertRaisesRegex(ValueError, "text_column, image_column"):
                load_experiment_config(config_path)

    def test_image_column_must_be_declared_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label"]\n'
                'image_column = "image_path"',
            )

            with self.assertRaisesRegex(ValueError, "image_path"):
                load_experiment_config(config_path)


class MaskConfigTest(unittest.TestCase):
    """mask_column and num_classes must travel together, so the class count a
    label metric assumed is always recorded in the config."""

    def _write_config(self, directory: str, dataset_body: str) -> Path:
        config_path = Path(directory) / "mask.toml"
        config_path.write_text(
            f"""
[experiment]
name = "mask"
seed = 1
output_root = "runs"

[dataset]
path = "data.csv"
id_column = "sample_id"
split_column = "split"
source_column = "source"
label_column = "label"
image_column = "image_path"
allowed_splits = ["train"]
allowed_sources = ["real"]
{dataset_body}
""".strip(),
            encoding="utf-8",
        )
        return config_path

    def test_accepts_mask_column_with_num_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label", '
                '"image_path", "seg_path"]\nmask_column = "seg_path"\n'
                "num_classes = 19",
            )

            config = load_experiment_config(config_path)

            self.assertEqual(config.dataset.mask_column, "seg_path")
            self.assertEqual(config.dataset.num_classes, 19)

    def test_rejects_mask_column_without_num_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label", '
                '"image_path", "seg_path"]\nmask_column = "seg_path"',
            )

            with self.assertRaisesRegex(ValueError, "num_classes"):
                load_experiment_config(config_path)

    def test_rejects_num_classes_without_mask_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label", '
                '"image_path"]\nnum_classes = 19',
            )

            with self.assertRaisesRegex(ValueError, "mask_column"):
                load_experiment_config(config_path)

    def test_rejects_non_positive_num_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label", '
                '"image_path", "seg_path"]\nmask_column = "seg_path"\n'
                "num_classes = 0",
            )

            with self.assertRaisesRegex(ValueError, "positive integer"):
                load_experiment_config(config_path)

    def test_mask_column_must_be_declared_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                'required_columns = ["sample_id", "split", "source", "label", '
                '"image_path"]\nmask_column = "seg_path"\nnum_classes = 19',
            )

            with self.assertRaisesRegex(ValueError, "seg_path"):
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

[dataset]
path = "data.csv"
required_columns = ["sample_id", "split", "source", "label", "text"]
id_column = "sample_id"
split_column = "split"
source_column = "source"
label_column = "label"
text_column = "text"
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
