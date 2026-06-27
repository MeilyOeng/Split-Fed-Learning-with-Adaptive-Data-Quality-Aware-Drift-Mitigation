from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DQ_ROOT = PROJECT_ROOT / "data quality"
if str(DQ_ROOT) not in sys.path:
    sys.path.insert(0, str(DQ_ROOT))

from modules.label_dataset import build_demand_label_dataset


class RuntimeQualityArtifactTests(unittest.TestCase):
    def test_runtime_quality_artifact_contains_window_keys_and_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            current_path = root / "current.csv"
            output_path = root / "quality_scores.csv"

            frame = pd.DataFrame(
                {
                    "Fecha_Hora": pd.date_range("2024-01-01", periods=8, freq="h"),
                    "9901": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                }
            )
            frame.to_csv(clean_path, index=False)
            frame.to_csv(current_path, index=False)

            dataset = build_demand_label_dataset(
                clean_path=clean_path,
                current_path=current_path,
                output_path=output_path,
                score_config={},
                window_size_steps=3,
                stride_steps=1,
                forecast_horizon=1,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                pca_mode="fit",
            )

            self.assertTrue(
                {"client_id", "mode", "window_start_index", "window_end_index", "quality_score"}.issubset(dataset.columns)
            )
            self.assertTrue(
                {
                    "dimension_completeness",
                    "dimension_validity",
                    "dimension_consistency",
                    "dimension_accuracy",
                    "dimension_freshness",
                }.issubset(dataset.columns)
            )

            expected_modes = {
                0: "train",
                1: "train",
                2: "val",
                3: "test",
                4: "test",
            }
            actual_modes = dict(zip(dataset["window_start_index"].astype(int), dataset["mode"]))
            self.assertEqual(actual_modes, expected_modes)
            self.assertTrue(dataset["quality_score"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
