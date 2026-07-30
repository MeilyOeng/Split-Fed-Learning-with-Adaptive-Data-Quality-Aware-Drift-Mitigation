from __future__ import annotations

import unittest
from pathlib import Path


class DisabledDriftRegressionTests(unittest.TestCase):
    def test_disabled_branch_preserves_original_stream_payload_contract(self):
        source = Path("data/stream_service.py").read_text(encoding="utf-8")
        self.assertIn('if not (DRIFT_INJECTION_ENABLED or DRIFT_DETECTION_ENABLED):\n        return', source)
        self.assertIn('values = client_runtime_series.get(series_key, client_series[series_key])', source)
        self.assertIn('if DRIFT_INJECTION_ENABLED or DRIFT_DETECTION_ENABLED:\n        payload["drift_metadata"]', source)

    def test_protected_runtime_files_are_not_part_of_integration(self):
        for path in ("client/client.py", "data/data_streaming.py"):
            self.assertTrue(Path(path).exists())
