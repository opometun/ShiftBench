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


if __name__ == "__main__":
    unittest.main()
