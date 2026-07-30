from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from drift.contracts import (
    DetectorEvent,
    DetectorKey,
    DetectorName,
    DetectorParameters,
    DriftConfig,
    DriftSource,
    DriftType,
    TransitionFunction,
)
from drift.detectors import DetectorRegistry
from drift.evaluator import classify_forecast_segment, evaluate_detection, evaluate_forecasts_by_segment
from drift.framework import transition_progress
from drift.injector import derive_client_seed, inject_series


def config(**changes):
    base = dict(
        run_id="run", configuration_id="cfg", cluster_id="1", client_id="10", mode="inference",
        drift_type=DriftType.SUDDEN, drift_source=DriftSource.INPUT, start_index=5, magnitude=1.0,
    )
    base.update(changes)
    return DriftConfig(**base)


def event(index, alarm=False, detector="page_hinkley", client="10"):
    return DetectorEvent(
        run_id="run", cluster_id="1", client_id=client, mode="inference",
        stream_name="standardized_window_end", detector_name=detector,
        event_index=index, observation_index=index, timestamp=f"2026-01-01T{index:02d}:00:00",
        detector_value=float(index), alarm=alarm,
    )


class DriftFrameworkTests(unittest.TestCase):
    def test_sudden_exact_index(self):
        cfg = config(start_index=3)
        self.assertEqual([transition_progress(i, cfg) for i in range(5)], [0, 0, 0, 1, 1])

    def test_gradual_linear_and_sigmoid_boundaries(self):
        linear = config(drift_type="gradual", start_index=2, end_index=6, transition="linear")
        self.assertEqual([transition_progress(i, linear) for i in range(7)], [0, 0, 0, .25, .5, .75, 1])
        sigmoid = replace(linear, transition=TransitionFunction.SIGMOID)
        self.assertEqual(transition_progress(2, sigmoid), 0)
        self.assertEqual(transition_progress(6, sigmoid), 1)

    def test_incremental_is_deterministic(self):
        cfg = config(drift_type="incremental", start_index=1, end_index=5, magnitude=.5)
        first = inject_series(np.ones(7), cfg)
        second = inject_series(np.ones(7), replace(cfg, random_seed=999))
        np.testing.assert_array_equal(first.runtime_values, second.runtime_values)

    def test_recurring_respects_period(self):
        cfg = config(drift_type="recurring", start_index=2, period=4)
        self.assertEqual([transition_progress(i, cfg) for i in range(10)], [0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    def test_strict_validation(self):
        with self.assertRaises(ValueError):
            config(drift_type="mystery")
        with self.assertRaises(ValueError):
            config(drift_type="gradual", start_index=5, end_index=5)
        with self.assertRaises(ValueError):
            config(drift_type="recurring", period=0)
        with self.assertRaises(ValueError):
            config(transition="unknown")

    def test_zero_effect_reports_no_transform(self):
        cfg = config(magnitude=0, offset_delta=0, noise_sigma=0)
        result = inject_series(np.ones(10), cfg)
        self.assertFalse(any(record.transform_applied for record in result.records))
        np.testing.assert_array_equal(result.base_values, result.runtime_values)

    def test_local_rng_reproducibility_and_difference(self):
        cfg = config(drift_type="gradual", start_index=1, end_index=30, noise_sigma=.5, random_seed=7)
        a = inject_series(np.ones(40), cfg)
        b = inject_series(np.ones(40), cfg)
        c = inject_series(np.ones(40), replace(cfg, random_seed=8))
        np.testing.assert_array_equal(a.runtime_values, b.runtime_values)
        self.assertFalse(np.array_equal(a.runtime_values, c.runtime_values))

    def test_base_is_immutable_and_input_unchanged(self):
        source = np.ones(10)
        result = inject_series(source, config())
        source[0] = 99
        self.assertEqual(result.base_values[0], 1)
        with self.assertRaises(ValueError):
            result.base_values[0] = 4

    def test_index_and_timestamp_alignment(self):
        times = [f"2026-01-01T{i:02d}:00:00" for i in range(10)]
        result = inject_series(np.ones(10), config(start_index=4), timestamps=times)
        self.assertEqual([r.observation_index for r in result.records], list(range(10)))
        self.assertEqual([r.timestamp for r in result.records], times)
        self.assertFalse(result.records[3].regime_active)
        self.assertTrue(result.records[4].regime_active)


class DetectorRegistryTests(unittest.TestCase):
    def setUp(self):
        self.params = DetectorParameters(
            ph_min_instances=5, ph_delta=.001, ph_threshold=2,
            adwin_delta=.1, adwin_clock=1, adwin_min_window_length=3, adwin_grace_period=5,
        )

    def test_independent_states_and_duplicate_prevention(self):
        registry = DetectorRegistry(self.params)
        a = DetectorKey("run", "1", "10", "inference", "x", DetectorName.PAGE_HINKLEY)
        b = DetectorKey("run", "1", "11", "inference", "x", DetectorName.PAGE_HINKLEY)
        self.assertIsNotNone(registry.update(a, observation_index=0, value=0))
        self.assertIsNone(registry.update(a, observation_index=0, value=0))
        self.assertIsNotNone(registry.update(b, observation_index=0, value=0))
        self.assertEqual(len(registry.events(a)), 1)
        self.assertEqual(len(registry.events(b)), 1)
        with self.assertRaises(ValueError):
            registry.update(a, observation_index=-1, value=0)

    def test_decreasing_coordinate_and_reset(self):
        registry = DetectorRegistry(self.params)
        key = DetectorKey("run", "1", "10", "inference", "x", "adwin")
        registry.update(key, observation_index=2, value=0)
        with self.assertRaises(ValueError):
            registry.update(key, observation_index=1, value=0)
        registry.reset(key)
        self.assertIsNotNone(registry.update(key, observation_index=1, value=0))


class EvaluationTests(unittest.TestCase):
    def test_pre_alarm_not_success_and_later_alarm_retained(self):
        records = inject_series(np.ones(12), config(start_index=5)).records
        events = [event(i, alarm=i in {2, 8}) for i in range(12)]
        result = evaluate_detection(events, records)
        self.assertTrue(result.false_alarm_present)
        self.assertTrue(result.primary.successful_detection)
        self.assertEqual(result.primary.first_valid_detection, 8)
        self.assertEqual(result.primary.detection_delay, 3)
        self.assertEqual(result.all_alarms, (2, 8))

    def test_pre_alarm_only_is_missed_detection(self):
        records = inject_series(np.ones(10), config(start_index=5)).records
        result = evaluate_detection([event(i, alarm=i == 2) for i in range(10)], records)
        self.assertFalse(result.primary.successful_detection)
        self.assertIsNone(result.primary.detection_delay)

    def test_clean_alarm_is_false_positive(self):
        result = evaluate_detection([event(i, alarm=i == 3) for i in range(8)])
        self.assertTrue(result.clean_client_false_positive)
        self.assertEqual(result.false_alarm_count, 1)

    def test_gradual_has_transition_and_effective_onsets(self):
        cfg = config(drift_type="incremental", start_index=2, end_index=8)
        records = inject_series(np.ones(12), cfg).records
        result = evaluate_detection([event(i, alarm=i == 7) for i in range(12)], records)
        onsets = {item.label: item for item in result.onset_evaluations}
        self.assertEqual(onsets["transition"].onset_index, 2)
        self.assertEqual(onsets["effective_0.5"].onset_index, 5)
        self.assertEqual(onsets["transition"].detection_delay, 5)
        self.assertEqual(onsets["effective_0.5"].detection_delay, 2)

    def test_recurring_intervals_are_separate(self):
        cfg = config(drift_type="recurring", start_index=2, period=4)
        records = inject_series(np.ones(12), cfg).records
        events = [event(i, alarm=i in {3, 5, 7}) for i in range(12)]
        result = evaluate_detection(events, records)
        self.assertEqual([(x.start_index, x.end_index_exclusive) for x in result.interval_evaluations], [(2, 4), (6, 8), (10, 12)])
        self.assertEqual(result.interval_evaluations[0].detection_delay, 1)
        self.assertEqual(result.interval_evaluations[1].detection_delay, 1)
        self.assertFalse(result.interval_evaluations[2].successful_detection)
        self.assertEqual(result.inactive_interval_alarms, (5,))


class SyntheticEndToEndTests(unittest.TestCase):
    def test_inject_detect_log_evaluate(self):
        base = np.ones(240)
        injected = inject_series(base, config(start_index=120, magnitude=9))
        params = DetectorParameters(
            ph_min_instances=5, ph_delta=.001, ph_threshold=2,
            adwin_delta=.1, adwin_clock=1, adwin_min_window_length=3, adwin_grace_period=5,
        )
        registry = DetectorRegistry(params)
        summaries = []
        for detector in (DetectorName.PAGE_HINKLEY, DetectorName.ADWIN):
            key = DetectorKey("run", "1", "10", "inference", "standardized_window_end", detector)
            for record, value in zip(injected.records, injected.runtime_values):
                registry.update(
                    key, observation_index=record.observation_index, value=float(value),
                    timestamp=record.timestamp, regime_active=record.regime_active,
                    transform_applied=record.transform_applied,
                )
            summary = evaluate_detection(registry.events(key), injected.records)
            summaries.append(summary)
            self.assertEqual(summary.observations_seen, 240)
            self.assertTrue(summary.primary.successful_detection)
            self.assertGreaterEqual(summary.primary.first_valid_detection, 120)
        self.assertEqual({s.detector_name for s in summaries}, {"page_hinkley", "adwin"})



class StandaloneExtensionTests(unittest.TestCase):
    def test_stable_client_seed_is_local_and_client_specific(self):
        np.random.seed(123)
        before = np.random.get_state()
        seed_a = derive_client_seed(42, run_id="r", cluster_id="1", client_id="10")
        seed_b = derive_client_seed(42, run_id="r", cluster_id="1", client_id="11")
        after = np.random.get_state()
        self.assertEqual(seed_a, derive_client_seed(42, run_id="r", cluster_id="1", client_id="10"))
        self.assertNotEqual(seed_a, seed_b)
        self.assertTrue(np.array_equal(before[1], after[1]))

    def test_metadata_contains_base_runtime_and_segment_contract(self):
        cfg = config(start_index=5, requested_start_ratio=.25, segment_start_index=2, segment_end_index_exclusive=14)
        result = inject_series(np.ones(14), cfg)
        record = result.records[5]
        self.assertEqual(record.base_value, 1.0)
        self.assertEqual(record.runtime_value, 2.0)
        self.assertEqual(record.scope, "test")
        self.assertEqual((record.segment_start_index, record.segment_end_index_exclusive), (2, 14))

    def test_bounded_event_buffer_and_monotonic_event_ids(self):
        registry = DetectorRegistry(DetectorParameters(ph_min_instances=2, ph_threshold=2), max_buffered_events=3)
        key = DetectorKey("run", "1", "10", "test", "x", "page_hinkley")
        for index in range(5):
            registry.update(key, observation_index=index, value=float(index))
        self.assertEqual([item.event_index for item in registry.events(key)], [2, 3, 4])
        self.assertEqual(len(registry.drain_events(key)), 3)
        self.assertEqual(registry.events(key), ())
        event_after_drain = registry.update(key, observation_index=5, value=5.0)
        self.assertEqual(event_after_drain.event_index, 5)

    def test_page_hinkley_warmup_is_not_applied_to_adwin(self):
        records = inject_series(np.ones(10), config(start_index=2)).records
        ph_events = [replace(event(i, alarm=i in {3, 6}), detector_min_instances=5) for i in range(10)]
        ph = evaluate_detection(ph_events, records)
        self.assertEqual(ph.warmup_end_index, 4)
        self.assertEqual(ph.primary.first_valid_detection, 6)
        self.assertEqual(ph.primary.minimum_achievable_delay, 2)
        adwin_events = [replace(event(i, alarm=i == 3, detector="adwin"), detector_min_instances=None) for i in range(10)]
        adwin = evaluate_detection(adwin_events, records)
        self.assertIsNone(adwin.warmup_end_index)
        self.assertEqual(adwin.primary.first_valid_detection, 3)

    def test_recurring_inactive_alarm_is_not_valid_primary_detection(self):
        records = inject_series(np.ones(10), config(drift_type="recurring", start_index=2, period=4)).records
        result = evaluate_detection([event(i, alarm=i == 5) for i in range(10)], records)
        self.assertFalse(result.primary.successful_detection)
        self.assertEqual(result.inactive_interval_alarms, (5,))

    def test_forecast_segmentation_and_metrics(self):
        labels = [classify_forecast_segment(start, start + 3, 5).segment for start in (0, 3, 5)]
        self.assertEqual(labels, ["pre_drift", "mixed_window", "fully_drifted"])
        metrics = evaluate_forecasts_by_segment([0, 3, 5], [3, 6, 8], [1, 2, 3], [2, 2, 1], drift_start_index=5)
        by_name = {item.segment: item for item in metrics}
        self.assertEqual(by_name["mixed_window"].mae, 0.0)
        self.assertEqual(by_name["fully_drifted"].rmse, 2.0)

if __name__ == "__main__":
    unittest.main()
