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
from modules.pca_quality import compute_pca_quality_scores


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
                score_config={
                    "validity": {"upper_bound": 100.0},
                    "consistency": {"rate_of_change_threshold": 100.0},
                },
                window_size_steps=3,
                stride_steps=1,
                forecast_horizon=1,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                pca_mode="fit",
                artifact_mode="runtime_scores",
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
            self.assertFalse(any(col.startswith("y_") for col in dataset.columns))


    def test_client_average_final_score_is_dimension_mean(self) -> None:
        scores = pd.DataFrame(
            {
                "client_id": ["9901", "9902"],
                "completeness": [1.0, 0.2],
                "validity": [0.8, 0.4],
                "consistency": [0.6, 0.6],
                "accuracy": [0.4, 0.8],
                "freshness": [0.2, 1.0],
                "sample_count": [10, 10],
            }
        )

        out, model = compute_pca_quality_scores(scores, config={"final_score": {"method": "average"}})

        self.assertIsNone(model)
        self.assertEqual(set(out["score_method"]), {"average"})
        expected = scores[["completeness", "validity", "consistency", "accuracy", "freshness"]].mean(axis=1)
        self.assertTrue((out["quality_score"].round(12) == expected.round(12)).all())

    def test_runtime_average_final_score_is_window_dimension_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            current_path = root / "current.csv"
            output_path = root / "quality_scores.csv"

            clean = pd.DataFrame(
                {
                    "Fecha_Hora": pd.date_range("2024-01-01", periods=8, freq="h"),
                    "9901": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                }
            )
            current = clean.copy()
            current.loc[2, "9901"] = float("nan")
            current.loc[4, "9901"] = 100.0
            clean.to_csv(clean_path, index=False)
            current.to_csv(current_path, index=False)

            dataset = build_demand_label_dataset(
                clean_path=clean_path,
                current_path=current_path,
                output_path=output_path,
                score_config={
                    "final_score": {"method": "average"},
                    "validity": {"upper_bound": 100.0},
                    "consistency": {"rate_of_change_threshold": 100.0},
                },
                window_size_steps=3,
                stride_steps=1,
                forecast_horizon=1,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                pca_mode="fit",
                artifact_mode="runtime_scores",
            )

            dims = dataset[
                [
                    "dimension_completeness",
                    "dimension_validity",
                    "dimension_consistency",
                    "dimension_accuracy",
                    "dimension_freshness",
                ]
            ]
            expected = dims.mean(axis=1)
            self.assertEqual(set(dataset["score_method"]), {"average"})
            self.assertTrue((dataset["quality_score"].round(12) == expected.round(12)).all())
            self.assertFalse(any(col.startswith("y_") for col in dataset.columns))


if __name__ == "__main__":
    unittest.main()
