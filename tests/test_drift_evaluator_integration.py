from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from drift.artifacts import write_csv
from drift.contracts import DetectorEvent, InjectionRecord
from drift.evaluator import evaluate_run_artifacts
from drift.injector import inject_series
from tests.test_drift_standalone import config, event


class ArtifactEvaluatorTests(unittest.TestCase):
    def test_run_evaluator_reports_drifted_and_clean_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drift_dir = root / "cluster_1" / "drift"
            records = inject_series(np.ones(12), config(start_index=5, mode="test")).records
            write_csv(drift_dir / "injection_manifest.csv", records, InjectionRecord)
            events = [event(i, alarm=i == 8) for i in range(12)]
            events = [DetectorEvent(**{**asdict(item), "mode": "test"}) for item in events]
            clean = [DetectorEvent(**{**asdict(event(i, alarm=i == 3, client="11")), "mode": "test"}) for i in range(12)]
            write_csv(drift_dir / "detector_events.csv", events + clean, DetectorEvent)
            prediction = root / "cluster_1" / "client_10_test_predictions.csv"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            with prediction.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "window_start_index", "window_end_index", "y_true", "y_pred"])
                writer.writeheader()
                for index in range(8):
                    writer.writerow({"timestamp": index, "window_start_index": index, "window_end_index": index + 3, "y_true": 1, "y_pred": 2})
            summary = evaluate_run_artifacts(root, root / "results" / "drift")
            self.assertEqual(summary["drifted_detector_client_rows"], 1)
            self.assertEqual(summary["clean_detector_client_rows"], 1)
            self.assertEqual(summary["clean_client_false_positive_rate"], 1.0)
            detector = summary["per_detector"]["page_hinkley"]
            self.assertEqual(detector["confusion_matrix"], {
                "true_positive": 1, "false_negative": 0, "false_positive": 1, "true_negative": 0,
            })
            self.assertEqual(detector["accuracy"], 0.5)
            self.assertEqual(detector["false_positive_rate"], 1.0)
            self.assertEqual(detector["recall_post_drift_detection_rate"], 1.0)
            self.assertEqual(detector["detection_delay"]["mean"], 3.0)
            self.assertTrue((root / "results" / "drift" / "forecast_segment_summary.csv").exists())
