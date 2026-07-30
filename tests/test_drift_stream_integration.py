from __future__ import annotations

import unittest

import numpy as np

from drift.contracts import DriftConfig, DriftSource, DriftType
from drift.injector import inject_series


class DriftStreamIntegrationTests(unittest.TestCase):
    def test_runtime_copy_changes_only_detector_visible_test_observations(self):
        base = np.arange(30, dtype=float)
        visible_start, visible_end = 20, 29
        cfg = DriftConfig(
            run_id="r", configuration_id="c", cluster_id="1", client_id="10",
            mode="test", scope="test", drift_type=DriftType.SUDDEN,
            drift_source=DriftSource.INPUT, start_index=23, magnitude=1.0,
            segment_start_index=visible_start, segment_end_index_exclusive=visible_end,
        )
        injected = inject_series(base, cfg)
        runtime = base.copy()
        runtime[visible_start:visible_end] = injected.runtime_values[visible_start:visible_end]
        np.testing.assert_array_equal(runtime[:visible_start], base[:visible_start])
        np.testing.assert_array_equal(runtime[visible_end:], base[visible_end:])
        self.assertTrue(np.array_equal(runtime[visible_start:23], base[visible_start:23]))
        self.assertFalse(np.array_equal(runtime[23:visible_end], base[23:visible_end]))

    def test_train_validation_normalization_uses_clean_base(self):
        base = np.arange(100, dtype=float)
        clean_train = base[:70].copy()
        clean_mean, clean_std = clean_train.mean(), clean_train.std()
        runtime = base.copy()
        runtime[85:] *= 10
        self.assertEqual(clean_mean, base[:70].mean())
        self.assertEqual(clean_std, base[:70].std())
        self.assertNotEqual(runtime.mean(), base.mean())
