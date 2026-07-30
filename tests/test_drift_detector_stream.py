from __future__ import annotations

import unittest

from drift.contracts import DetectorKey, DetectorParameters
from drift.detectors import DetectorRegistry


class DetectorEndpointTests(unittest.TestCase):
    def test_overlapping_windows_update_each_unique_endpoint_once(self):
        registry = DetectorRegistry(DetectorParameters(ph_min_instances=2, ph_threshold=3))
        key = DetectorKey("run", "1", "10", "test", "standardized_window_endpoint", "page_hinkley")
        window_endpoints_from_batches = ([23, 24, 25], [25, 26, 27])
        for batch in window_endpoints_from_batches:
            for endpoint in batch:
                registry.update(key, observation_index=endpoint, value=float(endpoint))
        self.assertEqual([event.observation_index for event in registry.events(key)], [23, 24, 25, 26, 27])

    def test_explicit_reset_is_separate_from_stream_offset(self):
        registry = DetectorRegistry()
        key = DetectorKey("run", "1", "10", "test", "x", "adwin")
        stream_offset = 17
        registry.update(key, observation_index=23, value=0.0)
        registry.reset()
        self.assertEqual(stream_offset, 17)
        self.assertEqual(registry.keys(), ())
