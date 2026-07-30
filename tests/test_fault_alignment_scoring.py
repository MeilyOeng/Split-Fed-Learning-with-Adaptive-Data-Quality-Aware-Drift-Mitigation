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

from modules.fault_injection import WideDemandFaultInjector
from modules.io import load_wide_demand_csv
from modules.scoring import WideDemandQualityScorer


class FaultAlignmentScoringTests(unittest.TestCase):
    def test_fault_injector_restores_expected_grid_and_marks_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            pd.DataFrame(
                {
                    "Fecha_Hora": pd.date_range("2024-01-01", periods=12, freq="h"),
                    "9901": [float(i + 1) for i in range(12)],
                    "9902": [float((i + 1) * 2) for i in range(12)],
                }
            ).to_csv(clean_path, index=False)

            dataset = load_wide_demand_csv(clean_path, timestamp_col="Fecha_Hora")
            injector = WideDemandFaultInjector(
                config={
                    "faults": {
                        "completeness": {
                            "dropped_timestamp_rows_fraction": 0.2,
                        },
                        "consistency": {
                            "duplicated_timestamp_fraction": 0.2,
                        },
                    },
                    "restore_expected_timestamps_after_injection": True,
                }
            )
            faulty_df, _ = injector.inject(dataset, seed=42)

            self.assertEqual(len(faulty_df), len(dataset.frame))
            self.assertIn("__dq_missing_timestamp__", faulty_df.columns)
            self.assertIn("__dq_duplicate_timestamp_count__", faulty_df.columns)
            self.assertGreater(int(faulty_df["__dq_missing_timestamp__"].sum()), 0)
            self.assertGreater(int(faulty_df["__dq_duplicate_timestamp_count__"].sum()), 0)

    def test_scoring_penalizes_restored_missing_and_duplicate_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            current_path = root / "current.csv"

            clean = pd.DataFrame(
                {
                    "Fecha_Hora": pd.date_range("2024-01-01", periods=6, freq="h"),
                    "9901": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                }
            )
            current = clean.copy()
            current.loc[2, "9901"] = float("nan")
            current["__dq_missing_timestamp__"] = [0, 0, 1, 0, 0, 0]
            current["__dq_duplicate_timestamp_count__"] = [0, 1, 0, 0, 0, 0]
            clean.to_csv(clean_path, index=False)
            current.to_csv(current_path, index=False)

            clean_dataset = load_wide_demand_csv(clean_path, timestamp_col="Fecha_Hora")
            current_dataset = load_wide_demand_csv(current_path, timestamp_col="Fecha_Hora")
            scorer = WideDemandQualityScorer(
                config={
                    "validity": {"upper_bound": 100.0},
                    "consistency": {"rate_of_change_threshold": 100.0},
                }
            )
            scores = scorer.score(current_dataset, clean_dataset)
            row = scores.loc[scores["client_id"] == "9901"].iloc[0]

            self.assertLess(float(row["C3"]), 1.0)
            self.assertLess(float(row["CS1"]), 1.0)
            self.assertLess(float(row["CS2"]), 1.0)
            self.assertLess(float(row["CS3"]), 1.0)


if __name__ == "__main__":
    unittest.main()
