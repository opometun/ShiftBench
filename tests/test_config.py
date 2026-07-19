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


if __name__ == "__main__":
    unittest.main()
