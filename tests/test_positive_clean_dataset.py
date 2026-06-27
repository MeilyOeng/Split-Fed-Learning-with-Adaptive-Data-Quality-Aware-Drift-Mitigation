from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.create_positive_clean_dataset import create_positive_clean_dataset


class CreatePositiveCleanDatasetTests(unittest.TestCase):
    def test_create_positive_clean_dataset_flips_demand_columns_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "raw_negative.csv"
            output_path = root / "dataset_positive.csv"
            report_path = root / "report.json"

            pd.DataFrame(
                {
                    "Fecha_Hora": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
                    "9901": [-2.0, -3.5],
                    "9902": [-1.0, 0.0],
                }
            ).to_csv(input_path, index=False)

            report = create_positive_clean_dataset(
                input_path=input_path,
                output_path=output_path,
                report_path=report_path,
                timestamp_col="Fecha_Hora",
            )

            output = pd.read_csv(output_path)
            written_report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(output["9901"].tolist(), [2.0, 3.5])
            self.assertEqual(output["9902"].tolist(), [1.0, -0.0])
            self.assertEqual(report["before"]["negative_count"], 3)
            self.assertEqual(report["after"]["negative_count"], 0)
            self.assertEqual(written_report["after"]["min"], 0.0)


if __name__ == "__main__":
    unittest.main()
