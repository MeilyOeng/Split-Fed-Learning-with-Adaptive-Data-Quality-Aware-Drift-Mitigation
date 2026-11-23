"""Tests for drift.mitigation.prepared.faults (window-level fault injection,
raw-scale, applied after windowing and before normalization).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from drift.mitigation.prepared import faults

REAL_FAULT_CONFIG = Path(__file__).resolve().parents[1] / "data quality" / "configs" / "fault_injection_controlled_window.yaml"
WINDOW_SIZE = 24


def _small_fault_config(tmp: Path, *, total_faulty_window_fraction: float = 0.5, dimension_window_fractions: dict | None = None) -> Path:
    config = {
        "fault_mode": "controlled_window",
        "seed": 42,
        "window_faulting": {
            "window_size": 4, "stride": 1,  # deliberately WRONG on purpose -- apply_window_faults must override these
            "sampling_scope": "per_cluster_per_client",
            "total_faulty_window_fraction": total_faulty_window_fraction,
            "dimension_window_fractions": dimension_window_fractions or {
                "completeness": 0.1, "consistency": 0.1, "accuracy": 0.1, "validity": 0.1, "freshness": 0.0, "mixed": 0.1,
            },
            "min_faults_per_faulty_window": 1,
            "max_faults_per_faulty_window": 1,
            "multi_fault_probability": 0.0,
        },
        "clients": {"selection_mode": "per_cluster_random", "selected": [], "random_count": 1},
        "cluster_assignment": {"client_id_column": "ID", "cluster_id_column": "Cluster"},
        "faults": {
            "completeness": {"random_missing_rate": 0.0, "block_missing": {"min_length": 1, "max_length": 2}},
            "consistency": {"lag_choices": [1], "jump_multiplier": 6.0},
            "accuracy": {"spike_multiplier": 4.0, "noise_std_fraction": 0.05},
            "validity": {"high_multiplier": 3.0},
            "freshness": {"delay_steps": 1},
        },
        "restore_expected_timestamps_after_injection": True,
    }
    path = tmp / "small_fault_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _synthetic_population(n_clusters: int = 2, n_clients_per_cluster: int = 3, n_pre_drift: int = 4, n_eligible: int = 8, seed: int = 0):
    """n_pre_drift + n_eligible deployment windows per client, each window a
    DISTINCT constant value so leakage between windows is immediately visible.
    """
    rng = np.random.default_rng(seed)
    n_windows = n_pre_drift + n_eligible
    X_drift: dict[tuple[str, str], np.ndarray] = {}
    assignment_rows: list[dict[str, str]] = []
    segment_by_window = np.array(["PRE_DRIFT"] * n_pre_drift + ["ADAPTATION"] * (n_eligible // 2) + ["FINAL_HOLDOUT"] * (n_eligible - n_eligible // 2))
    target_index_by_window = np.arange(n_windows) * 100 + 1000  # arbitrary but distinct raw indices
    timestamps = pd.date_range("2021-01-01", periods=200000, freq="h").to_numpy()  # long enough to index target_index_by_window safely

    client_counter = 2000
    for cluster_num in range(1, n_clusters + 1):
        cluster_id = str(cluster_num)
        for _ in range(n_clients_per_cluster):
            client_counter += 1
            client_id = str(client_counter)
            base = rng.uniform(10, 20)
            X = np.array([np.full((WINDOW_SIZE, 1), base * 1000 + w) for w in range(n_windows)])
            X_drift[(cluster_id, client_id)] = X
            assignment_rows.append({"cluster_id": cluster_id, "client_id": client_id})
    return X_drift, segment_by_window, target_index_by_window, timestamps, assignment_rows


class MakeModelConsumableTests(unittest.TestCase):
    def test_a_single_missing_point(self):
        X = np.array([[1.0, 2.0, np.nan, 4.0]]).reshape(1, 4, 1)
        result, meta = faults.make_model_consumable(X, clean_training_mean_raw=0.0)
        np.testing.assert_array_equal(result.reshape(-1), [1.0, 2.0, 2.0, 4.0])
        self.assertEqual(meta["missing_values_before_imputation"], 1)
        self.assertEqual(meta["forward_filled_count"], 1)
        self.assertEqual(meta["leading_fallback_count"], 0)
        self.assertEqual(meta["missing_values_after_imputation"], 0)

    def test_b_consecutive_missing_block(self):
        X = np.array([[1.0, np.nan, np.nan, 4.0]]).reshape(1, 4, 1)
        result, meta = faults.make_model_consumable(X, clean_training_mean_raw=0.0)
        np.testing.assert_array_equal(result.reshape(-1), [1.0, 1.0, 1.0, 4.0])
        self.assertEqual(meta["forward_filled_count"], 2)

    def test_c_leading_nans_use_clean_training_mean(self):
        X = np.array([[np.nan, np.nan, 3.0, 4.0]]).reshape(1, 4, 1)
        result, meta = faults.make_model_consumable(X, clean_training_mean_raw=0.0)
        np.testing.assert_array_equal(result.reshape(-1), [0.0, 0.0, 3.0, 4.0])
        self.assertEqual(meta["leading_fallback_count"], 2)
        self.assertEqual(meta["forward_filled_count"], 0)

    def test_c_leading_nans_use_nonzero_mean(self):
        X = np.array([[np.nan, np.nan, 3.0, 4.0]]).reshape(1, 4, 1)
        result, meta = faults.make_model_consumable(X, clean_training_mean_raw=7.5)
        np.testing.assert_array_equal(result.reshape(-1), [7.5, 7.5, 3.0, 4.0])

    def test_d_entire_window_missing(self):
        X = np.full((1, 4, 1), np.nan)
        result, meta = faults.make_model_consumable(X, clean_training_mean_raw=2.0)
        np.testing.assert_array_equal(result.reshape(-1), [2.0, 2.0, 2.0, 2.0])
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(meta["leading_fallback_count"], 4)

    def test_e_input_never_mutated(self):
        X = np.array([[1.0, np.nan, 3.0, 4.0]]).reshape(1, 4, 1)
        original = X.copy()
        faults.make_model_consumable(X, clean_training_mean_raw=0.0)
        np.testing.assert_array_equal(X, original)
        self.assertTrue(np.isnan(X[0, 1, 0]))

    def test_no_op_on_already_finite_input(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0]]).reshape(1, 4, 1)
        result, meta = faults.make_model_consumable(X, clean_training_mean_raw=0.0)
        np.testing.assert_array_equal(result.reshape(-1), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(meta["missing_values_before_imputation"], 0)
        self.assertEqual(meta["missing_values_after_imputation"], 0)

    def test_multiple_windows_independent(self):
        X = np.array([
            [1.0, np.nan, 3.0, 4.0],
            [np.nan, np.nan, np.nan, np.nan],
            [5.0, 6.0, 7.0, 8.0],
        ]).reshape(3, 4, 1)
        result, meta = faults.make_model_consumable(X, clean_training_mean_raw=0.0)
        np.testing.assert_array_equal(result[0].reshape(-1), [1.0, 1.0, 3.0, 4.0])
        np.testing.assert_array_equal(result[1].reshape(-1), [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result[2].reshape(-1), [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(meta["missing_values_before_imputation"], 5)


class ApplyWindowFaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="faults_window_test_"))
        self.fault_config_path = _small_fault_config(self.tmp)
        self.X_drift, self.segment_by_window, self.target_index_by_window, self.timestamps, self.assignment_rows = _synthetic_population()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, **overrides):
        kwargs = dict(
            X_drift=self.X_drift, segment_by_window=self.segment_by_window,
            target_index_by_window=self.target_index_by_window, timestamps=self.timestamps,
            assignment_rows=self.assignment_rows, fault_config_path=self.fault_config_path,
            fault_selection_seed=42, window_size=WINDOW_SIZE, manifests_dir=self.tmp / "manifests",
        )
        kwargs.update(overrides)
        return faults.apply_window_faults(**kwargs)

    def test_pre_drift_windows_are_byte_identical(self):
        result = self._run()
        n_pre_drift = int(np.sum(self.segment_by_window == "PRE_DRIFT"))
        for key, X_observed in result["X_observed"].items():
            np.testing.assert_array_equal(X_observed[:n_pre_drift], self.X_drift[key][:n_pre_drift])

    def test_x_observed_shape_matches_x_drift(self):
        result = self._run()
        for key, X_observed in result["X_observed"].items():
            self.assertEqual(X_observed.shape, self.X_drift[key].shape)

    def test_fault_selected_is_a_strict_nonempty_subset(self):
        result = self._run(manifests_dir=self.tmp / "manifests_subset")
        all_clients = {(row["cluster_id"], row["client_id"]) for row in self.assignment_rows}
        self.assertGreater(len(result["fault_selected"]), 0)
        self.assertLess(len(result["fault_selected"]), len(all_clients))
        self.assertTrue(result["fault_selected"] <= all_clients)

    def test_unselected_clients_have_zero_x_diff(self):
        result = self._run(manifests_dir=self.tmp / "manifests_unselected")
        for key, X_observed in result["X_observed"].items():
            if key not in result["fault_selected"]:
                np.testing.assert_array_equal(X_observed, self.X_drift[key])

    def test_realization_frame_has_no_synthetic_timestamps(self):
        result = self._run(manifests_dir=self.tmp / "manifests_coords")
        frame = result["realizations_frame"]
        for forbidden in ("start_time", "end_time", "affected_timestamps", "window_start", "window_end", "_synthetic_position"):
            self.assertNotIn(forbidden, frame.columns)
        expected_columns = {
            "cluster_id", "client_id", "deployment_window_index", "segment",
            "forecast_target_index_raw", "forecast_target_timestamp",
            "fault_position_within_window", "fault_dimension", "fault_type",
        }
        if not frame.empty:
            self.assertEqual(set(frame.columns), expected_columns)
            self.assertTrue((frame["segment"] != "PRE_DRIFT").all())

    def test_realization_deployment_window_index_matches_real_coordinates(self):
        result = self._run(manifests_dir=self.tmp / "manifests_realcoord")
        frame = result["realizations_frame"]
        n_pre_drift = int(np.sum(self.segment_by_window == "PRE_DRIFT"))
        for _, row in frame.iterrows():
            idx = int(row["deployment_window_index"])
            self.assertGreaterEqual(idx, n_pre_drift)
            self.assertEqual(row["segment"], self.segment_by_window[idx])
            self.assertEqual(int(row["forecast_target_index_raw"]), int(self.target_index_by_window[idx]))

    def test_reproducible_given_same_seed(self):
        result1 = self._run(manifests_dir=self.tmp / "manifests_a")
        result2 = self._run(manifests_dir=self.tmp / "manifests_b")
        self.assertEqual(result1["fault_selected"], result2["fault_selected"])
        for key in result1["X_observed"]:
            np.testing.assert_array_equal(result1["X_observed"][key], result2["X_observed"][key])

    def test_mismatched_assignment_rows_fails_closed(self):
        with self.assertRaises(ValueError):
            self._run(assignment_rows=self.assignment_rows[:-1], manifests_dir=self.tmp / "manifests_bad")

    def test_mixed_fault_confined_to_one_window_never_crosses_chunk_boundary(self):
        # Force ALL allocation onto "mixed" specifically, high fraction, to
        # exercise mixed-dimension selection directly (not just the other
        # four dimensions already covered by the default small config).
        config_path = _small_fault_config(
            self.tmp, total_faulty_window_fraction=0.75,
            dimension_window_fractions={"completeness": 0.0, "consistency": 0.0, "accuracy": 0.0, "validity": 0.0, "freshness": 0.0, "mixed": 0.75},
        )
        result = self._run(fault_config_path=config_path, manifests_dir=self.tmp / "manifests_mixed")
        frame = result["realizations_frame"]
        self.assertFalse(frame.empty, "expected at least one realization from mixed-dimension allocation")
        # "mixed" is a meta-dimension: the injector allocates a WINDOW to
        # "mixed", then recursively applies 2 of the 5 real dimensions to
        # it -- so individual FaultEvent rows are labeled with their actual
        # sub-dimension (e.g. "validity"), never literally "mixed". Allocation
        # itself is verified via the per-client window summary instead.
        summary = result["window_summary_frame"]
        self.assertIn("windows_allocated_mixed", summary.columns)
        self.assertTrue((summary["windows_allocated_mixed"] > 0).any())
        self.assertTrue(frame["fault_dimension"].isin(["completeness", "consistency", "accuracy", "validity", "freshness"]).all())
        # every reported fault position must fall within [0, window_size)
        # relative to its OWN window -- never spilling into an adjacent
        # window's 24-step span.
        for _, row in frame.iterrows():
            position = row["fault_position_within_window"]
            if position is None or (isinstance(position, float) and np.isnan(position)):
                continue
            self.assertGreaterEqual(int(position), 0)
            self.assertLess(int(position), WINDOW_SIZE)
        # and directly on the arrays: a mixed-faulted window's corruption
        # must never appear in a NEIGHBOR that was not itself independently
        # selected -- with 6 of 8 windows faulted in this config, some
        # neighbors legitimately carry their own independent fault, so only
        # neighbors absent from faulted_window_keys are checked.
        faulted_window_keys = {
            (str(r["cluster_id"]), str(r["client_id"]), int(r["deployment_window_index"])) for _, r in frame.iterrows()
        }
        for _, row in frame.iterrows():
            key = (str(row["cluster_id"]), str(row["client_id"]))
            idx = int(row["deployment_window_index"])
            observed = result["X_observed"][key]
            drift = self.X_drift[key]
            neighbor_indices = [i for i in (idx - 1, idx + 1) if 0 <= i < len(drift)]
            for neighbor in neighbor_indices:
                if (key[0], key[1], neighbor) in faulted_window_keys:
                    continue  # neighbor has its own independent fault -- not a leak
                np.testing.assert_array_equal(
                    observed[neighbor], drift[neighbor],
                    err_msg=f"mixed fault at window {idx} leaked into neighboring window {neighbor}",
                )

    def test_overlap_independence_regression(self):
        # The fundamental property the old raw-timeline implementation
        # violated: two windows that WOULD overlap in a real raw timeline
        # (adjacent deployment_window_index values, sharing 23 of 24 raw
        # positions) must be corrupted completely independently. Run with a
        # fraction guaranteed to leave some windows untouched, then assert:
        # for any two ADJACENT eligible windows i, i+1 where i was faulted
        # and i+1 was not independently selected, X_observed[i+1] ==
        # X_drift[i+1] exactly.
        result = self._run(manifests_dir=self.tmp / "manifests_overlap")
        frame = result["realizations_frame"]
        faulted_window_keys = {(str(row["cluster_id"]), str(row["client_id"]), int(row["deployment_window_index"])) for _, row in frame.iterrows()}
        checked_any = False
        for key, X_observed in result["X_observed"].items():
            drift = self.X_drift[key]
            for idx in range(len(drift) - 1):
                this_faulted = (key[0], key[1], idx) in faulted_window_keys
                next_faulted = (key[0], key[1], idx + 1) in faulted_window_keys
                if this_faulted and not next_faulted:
                    checked_any = True
                    np.testing.assert_array_equal(
                        X_observed[idx + 1], drift[idx + 1],
                        err_msg=f"window {idx}'s fault leaked into adjacent (overlapping-in-raw-terms) window {idx + 1} for client {key}",
                    )
        self.assertTrue(checked_any, "test setup did not produce the faulted/unfaulted-neighbor pattern needed to exercise this check")


if __name__ == "__main__":
    unittest.main()
