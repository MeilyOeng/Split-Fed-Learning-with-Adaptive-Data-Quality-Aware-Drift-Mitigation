from __future__ import annotations

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
import run_split
from torch import nn

from utils.client_optimization import apply_client_optimizer_step
from drift.configuration import load_drift_config
from drift.contracts import SchedulePoint
from drift.mitigation.legacy.checkpoint_integration import run_tiny_checkpoint_H_fixture
from drift.mitigation.checkpoint_registry import CheckpointHRegistry, hash_checkpoint_payload, register_checkpoint_h, stage_method_checkpoint
from drift.mitigation.legacy.production_runner import build_method_environment
from data.deployment_stream import resolve_deployment_slice
from drift.mitigation.core import (
    Method,
    ProcessSpec,
    Segment,
    SegmentSizes,
    UpdatePolicy,
    assert_exact_prediction_keys,
    assert_materialized_timeline_reuse,
    build_segment_rows,
    counter_state_hash,
    guard_output_directory,
    load_suite_yaml,
    model_state_hashes,
    optimizer_state_hash,
    prediction_key_hash,
    register_method_policy,
    registered_method_names,
    resolve_method_policy,
    stable_select_per_cluster,
    canonical_hash, file_sha256, write_csv_rows,
)
from drift.mitigation.legacy.service_adapter import (
    run_full_adapter_dry_run,
    run_prediction_only_final_holdout,
    run_two_cluster_fedavg_fixture,
    validate_full_population,
)
from drift.mitigation.legacy.smoke import (
    _normal_round_update,
    _state_snapshot,
    create_checkpoint_fixture,
    load_runtime,
    prepare_smoke_data,
    run_method,
    run_smoke_suite,
)
from drift.process import generate_process_timeline
from drift.windowing import build_forecasting_windows, fit_clean_training_scalers, normalize_windows
from models.split_lstm import LSTMConfig, build_split_lstm
from run_split import chunk_round_clients, order_round_clients, run_m2_oracle_client_only
from server.frozen_gradient import frozen_server_activation_step


def schedule(length: int, onset: int) -> list[SchedulePoint]:
    rows = []
    for index in range(length):
        active = index >= onset
        rows.append(SchedulePoint(
            absolute_index=index,
            timestamp=str(pd.Timestamp("2025-01-01") + pd.Timedelta(hours=index)),
            configured_regime="new" if active else "old",
            old_concept_id="old",
            new_concept_id="new",
            active_concept_id="new" if active else "old",
            drift_weight=1.0 if active else 0.0,
            regime_active=active,
            change_point_event=index == onset,
            new_regime_selected=active,
        ))
    return rows


def clean_ar(length: int, innovations: np.ndarray | None = None) -> np.ndarray:
    innovations = np.zeros(length) if innovations is None else innovations
    values = np.empty(length, dtype=float)
    values[0] = 2.0
    for index in range(1, length):
        values[index] = 1.0 + 0.50 * values[index - 1] + innovations[index]
    return values


def process_parameters(*, intercept_delta: float = 0.40, coefficient_delta: float = 0.10, source: str = "observed_process_residual"):
    return {
        "old": {"intercept": 1.0, "ar_coefficients": [0.50], "innovation_sigma": 1.0},
        "changes": {"intercept_delta": intercept_delta, "ar_coefficient_delta": [coefficient_delta], "innovation_scale": 1.0},
        "innovation_source": source,
    }


class ProcessDriftTests(unittest.TestCase):
    def test_exact_old_and_new_equations_and_onset_recursion(self):
        innovations = np.linspace(-0.2, 0.2, 20)
        clean = clean_ar(20, innovations)
        result = generate_process_timeline(clean, schedule(20, 8), process_parameters(), target_feature_index=0, client_seed=42, component_id="process")
        for index in range(1, 8):
            expected = 1.0 + 0.50 * result.values[index - 1, 0] + innovations[index]
            self.assertAlmostEqual(result.values[index, 0], expected, places=12)
        for index in range(8, 20):
            expected = 1.4 + 0.60 * result.values[index - 1, 0] + innovations[index]
            self.assertAlmostEqual(result.values[index, 0], expected, places=12)
        self.assertFalse(result.changed[:8].any())
        self.assertTrue(result.changed[8:].any())

    def test_arbitrary_source_is_literal_before_onset_and_observed_residual_after(self):
        clean = np.asarray([11.0, 8.0, 13.5, 7.25, 9.0, 4.0, 12.0, 3.0, 5.0, 6.0])
        onset = 5
        result = generate_process_timeline(
            clean, schedule(len(clean), onset), process_parameters(),
            target_feature_index=0, client_seed=42, component_id="observed-residual",
        )
        np.testing.assert_array_equal(result.values[:onset, 0], clean[:onset])
        expected_residuals = np.zeros_like(clean)
        expected_residuals[1:] = clean[1:] - (1.0 + 0.5 * clean[:-1])
        np.testing.assert_array_equal(result.innovations, expected_residuals)
        self.assertEqual(result.innovation_source, "observed_process_residual")
        for index in range(onset, len(clean)):
            self.assertAlmostEqual(
                result.values[index, 0],
                1.4 + 0.6 * result.values[index - 1, 0] + expected_residuals[index],
                places=12,
            )

    def test_observed_residual_path_does_not_depend_on_random_seed(self):
        clean = np.asarray([2.0, 2.2, 1.8, 3.1, 2.7, 4.2, 2.4, 3.0])
        first = generate_process_timeline(clean, schedule(len(clean), 4), process_parameters(), target_feature_index=0, client_seed=1, component_id="residual")
        second = generate_process_timeline(clean, schedule(len(clean), 4), process_parameters(), target_feature_index=0, client_seed=999, component_id="residual")
        np.testing.assert_array_equal(first.innovations, second.innovations)
        np.testing.assert_array_equal(first.values, second.values)

    def test_resolved_parameters_are_1p4_and_0p6_with_scale_one(self):
        spec = ProcessSpec()
        spec.validate()
        self.assertAlmostEqual(spec.new_intercept, 1.4)
        self.assertEqual(spec.new_ar_coefficients, (0.6,))
        self.assertAlmostEqual(spec.innovation_scale, 1.0)
        self.assertEqual(spec.ar_coefficient_delta, (0.10,))
        manifest = spec.equation_manifest()
        self.assertEqual(
            manifest["old_and_new_zero_innovation_fixed_points"],
            {"old": 2.0, "new": 3.5},
        )
        self.assertNotIn("old_and_new_stationary_means", manifest)
        self.assertIn(
            "not empirical client means", manifest["fixed_point_interpretation"]
        )

    def test_process_validate_accepts_a_non_golden_stationary_scenario(self):
        # Extensibility: a scenario config other than the one approved
        # production suite must still be accepted so long as it is a
        # genuine, well-posed, stationary AR(1) sudden-drift scenario.
        spec = ProcessSpec(old_intercept=0.0, old_ar_coefficients=(0.2,), intercept_delta=1.0, ar_coefficient_delta=(0.3,), innovation_sigma=0.5, innovation_scale=2.0)
        spec.validate()
        self.assertAlmostEqual(spec.new_intercept, 1.0)
        self.assertAlmostEqual(spec.new_ar_coefficients[0], 0.5)

    def test_process_validate_rejects_non_stationary_ar_coefficient(self):
        with self.assertRaisesRegex(ValueError, "stationary"):
            ProcessSpec(old_ar_coefficients=(0.95,), ar_coefficient_delta=(0.10,)).validate()

    def test_process_validate_rejects_a_no_op_drift(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            ProcessSpec(intercept_delta=0.0, ar_coefficient_delta=(0.0,)).validate()

    def test_process_validate_rejects_non_positive_innovation_parameters(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            ProcessSpec(innovation_sigma=0.0).validate()
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            ProcessSpec(innovation_scale=-1.0).validate()

    def test_zero_delta_is_identity(self):
        innovations = np.linspace(-0.1, 0.1, 24)
        clean = clean_ar(24, innovations)
        result = generate_process_timeline(clean, schedule(24, 9), process_parameters(intercept_delta=0.0, coefficient_delta=0.0), target_feature_index=0, client_seed=42, component_id="identity")
        np.testing.assert_allclose(result.values[:, 0], clean, rtol=0.0, atol=1e-12)
        self.assertFalse(result.changed.any())

    def test_random_innovations_are_deterministic_under_seed(self):
        clean = clean_ar(30)
        first = generate_process_timeline(clean, schedule(30, 10), process_parameters(source="sampled_gaussian"), target_feature_index=0, client_seed=42, component_id="normal")
        second = generate_process_timeline(clean, schedule(30, 10), process_parameters(source="sampled_gaussian"), target_feature_index=0, client_seed=42, component_id="normal")
        np.testing.assert_array_equal(first.innovations, second.innovations)
        np.testing.assert_array_equal(first.values, second.values)

    def test_legacy_magnitude_is_rejected_and_absent(self):
        config = ProcessSpec().drift_config(client_ids=["1"], onset_index=10, window_size=4, horizon=1, seed=42)
        self.assertNotIn("magnitude", json.dumps(config).lower())
        loaded = load_drift_config(config)
        self.assertEqual(loaded.drift_source.value, "process")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            text = Path("drift/configs/mitigation_sudden_process_seed42.yaml").read_text(encoding="utf-8")
            path.write_text(text + "magnitude: 0.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "magnitude"):
                load_suite_yaml(path)

    def test_load_suite_yaml_resolves_full_seed_contract_with_defaults(self):
        config = load_suite_yaml("drift/configs/mitigation_sudden_process_seed42.yaml")
        self.assertEqual(
            set(config["seeds"]),
            {
                "experiment_seed", "client_selection_seed", "data_order_seed", "drift_seed",
                "h_training_seed", "adaptation_seed", "fault_selection_seed", "fault_generation_seed",
            },
        )
        # Declared in the golden config.
        self.assertEqual(config["seeds"]["h_training_seed"], 42)
        # Reserved for Stage E and absent from the golden config; must still
        # resolve to the documented default rather than a missing key.
        self.assertEqual(config["seeds"]["fault_selection_seed"], 42)

    def test_load_suite_yaml_accepts_a_differently_named_scenario_suite_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.yaml"
            text = Path("drift/configs/mitigation_sudden_process_seed42.yaml").read_text(encoding="utf-8")
            text = text.replace(
                "suite_id: sudden_process_ar_int0p40_phi0p10_innov1p0_seed42_pre30_adapt40_holdout30",
                "suite_id: sudden_process_ar_int0p40_phi0p10_innov1p0_seed7_pre30_adapt40_holdout30",
            )
            path.write_text(text, encoding="utf-8")
            config = load_suite_yaml(path)
        self.assertEqual(config["suite_id"], "sudden_process_ar_int0p40_phi0p10_innov1p0_seed7_pre30_adapt40_holdout30")

    def test_load_suite_yaml_rejects_a_malformed_suite_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_suite_id.yaml"
            text = Path("drift/configs/mitigation_sudden_process_seed42.yaml").read_text(encoding="utf-8")
            text = text.replace(
                "suite_id: sudden_process_ar_int0p40_phi0p10_innov1p0_seed42_pre30_adapt40_holdout30",
                "suite_id: 'has spaces/and slashes'",
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "suite_id must be"):
                load_suite_yaml(path)

    def test_timeline_first_coherent_X_y_and_double_injection_guard(self):
        prepared, _ = prepare_smoke_data()
        selected = next(item for item in prepared.values() if item.selected)
        for position, target_index in enumerate(selected.drifted_windows.target_indices):
            self.assertAlmostEqual(
                selected.drifted_windows.y[position],
                (selected.drifted_timeline[target_index] - selected.scaler.y_location) / selected.scaler.y_scale,
                places=12,
            )
        with self.assertRaisesRegex(ValueError, "already materialized"):
            assert_materialized_timeline_reuse(materialized=True, runtime_injection_requested=True)


class SegmentationSelectionScalerTests(unittest.TestCase):
    def test_exact_full_segmentation_and_boundary_flags(self):
        sizes = SegmentSizes()
        sizes.validate()
        starts = np.arange(6988, 6988 + sizes.total)
        timestamps = pd.date_range("2014-01-01", periods=8760, freq="h")
        rows = build_segment_rows(window_starts=starts, window_size=24, horizon=1, timestamps=timestamps, sizes=sizes, onset_index=7536, selected=True)
        self.assertEqual(rows[523]["segment"], Segment.PRE_DRIFT.value)
        boundary = rows[524]
        self.assertEqual(boundary["segment"], Segment.ADAPTATION.value)
        self.assertEqual((boundary["window_start_index"], boundary["window_end_index"], boundary["target_index"]), (7512, 7535, 7536))
        self.assertEqual(boundary["process_regime_at_window_start"], "OLD")
        self.assertEqual(boundary["process_regime_at_window_end"], "OLD")
        self.assertEqual(boundary["process_regime_at_target"], "NEW")
        self.assertFalse(boundary["window_contains_any_drifted_input"])
        self.assertFalse(boundary["window_input_fully_drifted"])
        self.assertTrue(boundary["target_drifted"])
        self.assertEqual(rows[1224]["segment"], Segment.FINAL_HOLDOUT.value)

    def test_deterministic_eight_per_cluster_selection(self):
        clusters = {str(index): [f"c{index}_{item:02d}" for item in range(21)] for index in range(1, 5)}
        first = stable_select_per_cluster(clusters, per_cluster=8, seed=42)
        second = stable_select_per_cluster(clusters, per_cluster=8, seed=42)
        self.assertEqual(first, second)
        for cluster in clusters:
            selected = [row for row in first if row["cluster_id"] == cluster and row["selected_for_drift"]]
            self.assertEqual(len(selected), 8)

    def test_shared_scaler_identity_and_training_only_fit(self):
        clean = np.linspace(1.0, 20.0, 30)
        mutated = clean.copy()
        mutated[15:] += 1000.0
        first = fit_clean_training_scalers(clean, clean, training_window_starts=range(5), window_size=4, forecast_horizon=1, target_feature_index=0, share_target_scaler=True)
        second = fit_clean_training_scalers(mutated, mutated, training_window_starts=range(5), window_size=4, forecast_horizon=1, target_feature_index=0, share_target_scaler=True)
        self.assertEqual(first.target_scaling_strategy, "shared")
        self.assertEqual(first.x_location[0], first.y_location)
        self.assertEqual(first.x_scale[0], first.y_scale)
        self.assertEqual(first.shared_target_union_hash, second.shared_target_union_hash)
        self.assertEqual(first.x_location[0], second.x_location[0])

    def test_cross_method_pre_drift_data_equality(self):
        prepared, metadata = prepare_smoke_data()
        self.assertEqual(metadata["innovation_source"], "observed_process_residual")
        self.assertTrue(metadata["source_dataset_path"].endswith("dataset/output_clean/cluster_1.pkl"))
        for item in prepared.values():
            n = metadata["sizes"]["pre_drift"]
            np.testing.assert_array_equal(item.clean_timeline[:metadata["onset_index"]], item.drifted_timeline[:metadata["onset_index"]])
            np.testing.assert_array_equal(item.clean_windows.X[:n], item.drifted_windows.X[:n])
            np.testing.assert_array_equal(item.clean_windows.y[:n], item.drifted_windows.y[:n])


class MethodPolicyRegistryTests(unittest.TestCase):
    def test_built_in_methods_are_pre_registered_in_declaration_order(self):
        names = registered_method_names()
        self.assertEqual(names[:4], ["R0_clean_frozen", "M0_drift_frozen", "M1_continued_splitfed", "M2_oracle_client_only"])

    def test_resolve_method_policy_accepts_enum_member_or_string(self):
        by_enum = resolve_method_policy(Method.M1)
        by_string = resolve_method_policy("M1_continued_splitfed")
        self.assertEqual(by_enum, by_string)
        self.assertTrue(by_enum.selected_clients_update)
        self.assertEqual(by_enum.method_role, "mitigation_candidate")

    def test_exactly_one_clean_and_one_frozen_baseline_among_built_ins(self):
        roles = {name: resolve_method_policy(name).method_role for name in registered_method_names()}
        self.assertEqual(sum(1 for role in roles.values() if role == "clean_baseline"), 1)
        self.assertEqual(sum(1 for role in roles.values() if role == "frozen_baseline"), 1)

    def test_resolve_method_policy_rejects_an_unregistered_method(self):
        with self.assertRaisesRegex(ValueError, "no policy registered"):
            resolve_method_policy("M99_not_registered")

    def test_register_method_policy_opens_a_new_method_without_editing_the_enum(self):
        register_method_policy(
            "M3_dq_aware_test_only",
            UpdatePolicy(True, True, False, False, False, False, method_role="mitigation_candidate"),
        )
        try:
            self.assertIn("M3_dq_aware_test_only", registered_method_names())
            self.assertTrue(resolve_method_policy("M3_dq_aware_test_only").selected_clients_update)
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_method_policy("M3_dq_aware_test_only", resolve_method_policy(Method.M1))
        finally:
            from drift.mitigation import core as _core_module
            _core_module._METHOD_POLICY_REGISTRY.pop("M3_dq_aware_test_only", None)


class ProductionAdapterIntegrationTests(unittest.TestCase):
    def test_full_population_shape_is_strict(self):
        valid = {str(cluster): [f"{cluster}_{index}" for index in range(21)] for cluster in range(1, 5)}
        validate_full_population(valid)
        with self.assertRaisesRegex(ValueError, "four clusters"):
            validate_full_population({"1": ["a"]})

    def test_production_scheduler_adapter_dry_run(self):
        report = run_full_adapter_dry_run()
        self.assertTrue(report["accepted"])
        self.assertEqual(sum(row["event"] == "train_batch" for row in report["events"]), 2)
        self.assertTrue(all(not row["scheduler_invoked"] for row in report["non_M1_policies"].values()))

    def test_prediction_only_adapter_starts_at_final_holdout_offset(self):
        with patch("drift.mitigation.legacy.service_adapter.run_split.run_test_phase", return_value={}) as call:
            run_prediction_only_final_holdout(
                ["fixture"],
                {"fixture": ["a", "b"]},
                {"RUN_CLUSTERS_IN_PARALLEL": "false"},
                final_holdout_window_offset=7,
                require_full_population=False,
            )
        self.assertEqual(call.call_args.kwargs["window_offset"], 7)

    def test_two_cluster_real_fedavg_event_and_parity(self):
        report = run_two_cluster_fedavg_fixture()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["fedavg_event_count"], 1)
        self.assertEqual(report["actual_weight"], [4.0, 6.0])

    def test_tiny_real_run_split_checkpoint_H_loads_into_all_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_tiny_checkpoint_H_fixture(Path(tmp) / "tiny_H")
        self.assertTrue(report["accepted"])
        self.assertTrue(report["all_methods_loaded_identical_H"])
        self.assertTrue(report["normalization_contract_ok"])
        self.assertEqual(report["normalization_fit_indices"], list(range(14)))
        self.assertTrue(report["checkpoint_H_one_optimizer_step_per_client_batch"])
        self.assertTrue(all(steps == [1] for steps in report["checkpoint_H_client_adam_steps"].values()))


class TrainingPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared, cls.metadata = prepare_smoke_data()
        cls.client_states, cls.server_state, cls.cfg = create_checkpoint_fixture(cls.metadata["clients"])

    def test_no_updates_before_adaptation_or_during_holdout(self):
        # This generation-1 pipeline predates R1/M3 (Method now has more
        # members than it does bespoke dispatch for); exercise exactly the
        # four methods this suite was written against.
        for method in (Method.R0, Method.M0, Method.M1, Method.M2):
            result = run_method(method, self.prepared, self.metadata, self.client_states, self.server_state, self.cfg)
            self.assertTrue(all(event.get("segment") == Segment.ADAPTATION.value for event in result["adaptation_events"]))
            self.assertFalse(any(event.get("segment") == Segment.FINAL_HOLDOUT.value for event in result["adaptation_events"]))

    def test_persistent_optimizer_lifecycle(self):
        state = load_runtime(self.client_states, self.server_state, self.cfg)
        ids_before = {client_id: id(opt) for client_id, opt in state.client_optimizers.items()}
        server_id = id(state.server_optimizer)
        _normal_round_update(state, self.prepared, sorted(self.prepared), [4, 5])
        self.assertEqual(ids_before, {client_id: id(opt) for client_id, opt in state.client_optimizers.items()})
        self.assertEqual(server_id, id(state.server_optimizer))

    def test_client_optimizer_permission_call_count_and_adam_state(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        activations = model(torch.tensor([[1.0, -2.0]]))
        gradient = torch.ones_like(activations)
        original_step = optimizer.step
        optimizer.step = Mock(wraps=original_step)
        optimizer.zero_grad(set_to_none=True)
        activations.backward(gradient=gradient)
        self.assertTrue(
            apply_client_optimizer_step(optimizer, update_allowed=True)
        )
        self.assertEqual(optimizer.step.call_count, 1)
        adam_steps = {
            int(value["step"].item())
            for value in optimizer.state.values()
            if "step" in value
        }
        self.assertEqual(adam_steps, {1})
        denied_model = nn.Linear(2, 1)
        denied_optimizer = torch.optim.Adam(denied_model.parameters(), lr=0.01)
        parameters_before = [parameter.detach().clone() for parameter in denied_model.parameters()]
        denied_activations = denied_model(torch.tensor([[1.0, -2.0]]))
        denied_original_step = denied_optimizer.step
        denied_optimizer.step = Mock(wraps=denied_original_step)
        self.assertFalse(
            apply_client_optimizer_step(denied_optimizer, update_allowed=False)
        )
        self.assertEqual(denied_optimizer.step.call_count, 0)
        self.assertEqual(len(denied_optimizer.state), 0)
        for before, after in zip(parameters_before, denied_model.parameters()):
            torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)

    def test_m2_activation_gradient_and_all_server_state_frozen(self):
        state = load_runtime(self.client_states, self.server_state, self.cfg)
        before = _state_snapshot(state)["server"]
        client_id = next(item for item in self.prepared if self.prepared[item].selected)
        X = torch.as_tensor(self.prepared[client_id].drifted_windows.X[[4, 5]], dtype=torch.float32)
        y = torch.as_tensor(self.prepared[client_id].drifted_windows.y[[4, 5]], dtype=torch.float32)
        activation = state.clients[client_id](X)[:, -1, :]
        response = frozen_server_activation_step(state.server, nn.MSELoss(), activation, y, device="cpu")
        self.assertEqual(response["grad_activations"].shape, activation.shape)
        self.assertGreater(float(response["grad_activations"].abs().sum()), 0.0)
        activation.backward(response["grad_activations"])
        after = _state_snapshot(state)["server"]
        self.assertEqual(before, after)
        self.assertFalse(state.server.training)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.backends.cudnn.is_available(),
        "CUDA with cuDNN is required for the production M2 RNN regression",
    )
    def test_m2_cuda_cudnn_lstm_activation_gradient_keeps_server_frozen(self):
        cfg = LSTMConfig(
            input_size=1,
            hidden_size=8,
            num_layers=3,
            dropout=0.4,
            output_size=1,
        )
        _, server = build_split_lstm(cfg, split_layer=2)
        server = server.to("cuda")
        server.train()
        before = model_state_hashes(server)

        response = frozen_server_activation_step(
            server,
            nn.MSELoss(),
            torch.randn(3, 8),
            torch.randn(3, 1),
            device="cuda",
        )

        self.assertTrue(response["cudnn_rnn_disabled_for_input_gradient"])
        self.assertEqual(response["grad_activations"].shape, (3, 8))
        self.assertGreater(float(response["grad_activations"].abs().sum()), 0.0)
        self.assertEqual(before, model_state_hashes(server))
        self.assertTrue(all(parameter.grad is None for parameter in server.parameters()))
        self.assertFalse(server.training)

    def test_m2_controls_invariant_and_selected_change(self):
        result = run_method(Method.M2, self.prepared, self.metadata, self.client_states, self.server_state, self.cfg)
        for client_id, item in self.prepared.items():
            changed = result["before"]["clients"][client_id]["parameters"] != result["after"]["clients"][client_id]["parameters"]
            self.assertEqual(changed, item.selected)
        self.assertEqual(result["before"]["server"], result["after"]["server"])
        self.assertEqual(len(result["oracle_events"]), 2)
        for client_id, item in self.prepared.items():
            expected_steps = 3 if item.selected else 0
            self.assertEqual(result["state"].client_counters[client_id]["optimizer_step_count"], expected_steps)
        self.assertEqual(result["state"].server_counters["optimizer_step_count"], 0)

    def test_m1_one_client_optimizer_step_per_permitted_batch(self):
        result = run_method(Method.M1, self.prepared, self.metadata, self.client_states, self.server_state, self.cfg)
        for client_id in self.prepared:
            self.assertEqual(result["state"].client_counters[client_id]["optimizer_step_count"], 3)
            adam_steps = {
                int(value["step"].item())
                for value in result["state"].client_optimizers[client_id].state.values()
                if "step" in value
            }
            self.assertEqual(adam_steps, {3})

    def test_prequential_onset_prediction_reveal_oracle_then_update(self):
        result = run_method(Method.M2, self.prepared, self.metadata, self.client_states, self.server_state, self.cfg)
        for client_id in self.metadata["selected"]:
            events = sorted(
                (item for item in result["prequential_onset_events"] if item["client_id"] == client_id),
                key=lambda item: item["sequence"],
            )
            self.assertEqual(
                [item["event"] for item in events],
                ["prediction_recorded", "target_revealed", "oracle_onset", "client_optimizer_step"],
            )
            self.assertTrue(all(item["target_index"] == self.metadata["onset_index"] for item in events))
            self.assertEqual(events[0]["prediction_phase"], "pre_update")
            self.assertEqual(events[2]["latency_windows"], 0)
            self.assertEqual(events[3]["optimizer_step_count"], 1)
        full_boundary = build_segment_rows(
            window_starts=np.arange(6988, 6988 + SegmentSizes().total),
            window_size=24,
            horizon=1,
            timestamps=pd.date_range("2014-01-01", periods=8760, freq="h"),
            sizes=SegmentSizes(),
            onset_index=7536,
            selected=True,
        )[524]
        self.assertEqual(full_boundary["target_index"], 7536)
        self.assertTrue(full_boundary["target_drifted"])

    def test_existing_scheduler_order_and_partial_chunk_behavior(self):
        import random
        rng1, rng2 = random.Random(42), random.Random(42)
        clients = ["a", "b", "c", "d", "e"]
        self.assertEqual(order_round_clients(clients, "random_per_round", rng1, 1, 1), order_round_clients(clients, "random_per_round", rng2, 1, 1))
        self.assertEqual(chunk_round_clients(clients, 2), [["a", "b"], ["c", "d"], ["e"]])

    def test_m1_parity_with_existing_server_train_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "OUTPUT_DIR": tmp,
                "CLUSTER_ID": "parity",
                "GPU_ENABLED": "false",
                "SERVER_LR": str(self.cfg.learning_rate),
                "ACCUM_STEPS_SERVER": "4",
                "CHECKPOINT_EVERY_ROUNDS": "0",
                "MODEL_INPUT_SIZE": "1",
                "MODEL_HIDDEN_SIZE": "8",
                "MODEL_NUM_LAYERS": "3",
                "MODEL_SPLIT_LAYER": "2",
                "MODEL_OUTPUT_SIZE": "1",
                "MODEL_DROPOUT": "0.1",
                "USE_QUALITY_WEIGHTS": "false",
                "USE_QUALITY_AS_INPUT": "false",
                "USE_QUALITY_AGGREGATION_WEIGHTS": "false",
                "USE_QUALITY_ANCHOR_AGGREGATION": "false",
            }
            with patch.dict(os.environ, env, clear=False):
                module = importlib.import_module("server.server")
                production = module.SplitLearningServerState()
            production.model.load_state_dict(self.server_state, strict=True)
            production.optimizer = torch.optim.Adam(production.model.parameters(), lr=self.cfg.learning_rate)
            reference = load_runtime(self.client_states, self.server_state, self.cfg)
            smoke = load_runtime(self.client_states, self.server_state, self.cfg)
            order = sorted(self.prepared)
            positions = [4, 5]
            torch.manual_seed(777)
            _normal_round_update(smoke, self.prepared, order, positions)
            production.set_round(order, round_id=1001, effective_accum_steps_target=len(order), global_round=1, logical_round=1, scheduler_epoch=1)
            torch.manual_seed(777)
            for client_id in order:
                model = reference.clients[client_id]
                optimizer = reference.client_optimizers[client_id]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                windows = self.prepared[client_id].drifted_windows
                X = torch.as_tensor(windows.X[positions], dtype=torch.float32)
                y = torch.as_tensor(windows.y[positions], dtype=torch.float32)
                activation = model(X)[:, -1, :]
                response = production.train_step(client_id, activation.detach(), y)
                activation.backward(response["grad_activations"])
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            for client_id in order:
                for name, tensor in smoke.clients[client_id].state_dict().items():
                    torch.testing.assert_close(tensor, reference.clients[client_id].state_dict()[name], rtol=0.0, atol=0.0)
            for name, tensor in smoke.server.state_dict().items():
                torch.testing.assert_close(tensor, production.model.state_dict()[name], rtol=0.0, atol=0.0)
            self.assertEqual(smoke.server_counters["optimizer_step_count"], production.optimizer_step_count)
            self.assertEqual(production.pending_accum_steps, 0)


class HashingOutputAndSmokeTests(unittest.TestCase):
    def test_canonical_parameter_buffer_optimizer_and_counter_hashes(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before_model = model_state_hashes(model)
        before_optimizer = optimizer_state_hash(optimizer)
        before_counter = counter_state_hash({"steps": 0})
        loss = model(torch.ones(3, 2)).sum()
        loss.backward()
        optimizer.step()
        after_model = model_state_hashes(model)
        self.assertNotEqual(before_model["parameters"], after_model["parameters"])
        self.assertNotEqual(before_model["buffers"], after_model["buffers"])
        self.assertNotEqual(before_optimizer, optimizer_state_hash(optimizer))
        self.assertNotEqual(before_counter, counter_state_hash({"steps": 1}))

    def test_output_overwrite_and_resume_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite"
            guard_output_directory(path)
            with self.assertRaises(FileExistsError):
                guard_output_directory(path)
            (path / "resume_hashes.json").write_text(json.dumps({"config": "a"}), encoding="utf-8")
            guard_output_directory(path, resume=True, expected_hashes={"config": "a"})
            with self.assertRaisesRegex(ValueError, "resume hash"):
                guard_output_directory(path, resume=True, expected_hashes={"config": "b"})
            (path / "acceptance_report.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed"):
                guard_output_directory(path, resume=True, expected_hashes={"config": "a"})

    def test_missing_and_duplicate_prediction_key_detection(self):
        rows = [{"client_id": "a", "deployment_window_number": 1, "window_start_index": 0, "window_end_index": 3, "target_index": 4, "segment": "PRE_DRIFT"}]
        self.assertTrue(prediction_key_hash(rows))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prediction_key_hash(rows + rows)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            assert_exact_prediction_keys(rows, [("a", 1, 4, "PRE_DRIFT"), ("a", 2, 5, "PRE_DRIFT")])

    def test_complete_four_method_smoke_artifacts_and_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "smoke"
            result = run_smoke_suite(root)
            self.assertTrue(result["acceptance"]["accepted"])
            required = {
                "resolved_config.yaml", "run_manifest.json", "seeds.json", "checkpoint_hashes_before.json",
                "checkpoint_hashes_after.json", "tensor_state_hashes_before.json", "tensor_state_hashes_after.json",
                "parameter_change_report.csv", "oracle_events.csv", "segment_manifest.csv", "adaptation_schedule.csv",
                "prediction_rows.csv", "client_metrics.csv", "cluster_metrics.csv", "group_metrics.csv",
                "adaptation_events.csv", "optimizer_step_summary.json", "fedavg_event_summary.json",
                "runtime_summary.json", "acceptance_report.json", "adaptation_stream_metrics.csv",
                "onset_order_evidence.json",
            }
            # This generation-1 smoke pipeline predates R1/M3; it stays
            # scoped to the original four methods it was written against.
            for method in (Method.R0, Method.M0, Method.M1, Method.M2):
                run = root / method.value
                self.assertTrue(required.issubset({item.name for item in run.iterdir()}))
                self.assertTrue((run / "logs" / "stdout.log").exists())
                self.assertTrue((run / "logs" / "stderr.log").exists())
                self.assertTrue((run / "logs" / "events.jsonl").exists())
                event_rows = [json.loads(line) for line in (run / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                expected_oracle_events = 2 if method is Method.M2 else 0
                self.assertEqual(sum(row["event"] == "oracle_onset" for row in event_rows), expected_oracle_events)
                acceptance = json.loads((run / "acceptance_report.json").read_text(encoding="utf-8"))
                self.assertTrue(acceptance["accepted"], msg=f"{method}: {acceptance}")

            m2 = root / Method.M2.value
            self.assertTrue((m2 / "counterfactual_clean_holdout" / "prediction_rows.csv").exists())
            m2_acceptance = json.loads((m2 / "acceptance_report.json").read_text(encoding="utf-8"))
            self.assertEqual(m2_acceptance["maximum_absolute_clean_control_prediction_difference"], 0.0)


class ProductionPreflightTests(unittest.TestCase):
    def test_real_checkpoint_H_registry_accepts_complete_population(self):
        root = Path("outputs/drift_mitigation/checkpoint_H_clean_shared_scaler_seed42_epochs130")
        if not root.exists():
            self.skipTest("production checkpoint H is not present")
        registry = register_checkpoint_h(root)
        self.assertEqual(len(registry.clients), 84)
        self.assertEqual(len(registry.servers), 4)
        self.assertEqual(len({row["tensor_state"] for row in registry.servers}), 1)
        self.assertTrue(registry.normalization_hash)

    def test_incomplete_checkpoint_H_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "exactly 84"):
                register_checkpoint_h(tmp)

    def test_staging_preserves_source_and_uses_fresh_deployment_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "H"; method = Path(tmp) / "M0"; (root / "cluster_1").mkdir(parents=True); (root / "global").mkdir()
            model = nn.Linear(2, 1); optimizer = torch.optim.Adam(model.parameters())
            payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": 130}
            client_path = root / "cluster_1" / "client_1_final.pt"; server_path = root / "cluster_1" / "server_final.pt"; global_path = root / "global" / "global_server_latest.pt"
            torch.save(payload, client_path); torch.save(payload, server_path); torch.save(payload, global_path)
            client_hash, server_hash, global_hash = hash_checkpoint_payload(client_path), hash_checkpoint_payload(server_path), hash_checkpoint_payload(global_path)
            registry = CheckpointHRegistry(root, ({"cluster_id": "1", "client_id": "1", "relative_path": "cluster_1/client_1_final.pt", **client_hash},), ({"cluster_id": "1", "relative_path": "cluster_1/server_final.pt", **server_hash},), {"relative_path": "global/global_server_latest.pt", **global_hash}, canonical_hash([client_hash["tensor_state"], server_hash["tensor_state"], global_hash["tensor_state"]]), canonical_hash([client_hash["artifact_file"], server_hash["artifact_file"], global_hash["artifact_file"]]), "scaler")
            source_before = file_sha256(client_path)
            report = stage_method_checkpoint(registry, method)
            self.assertEqual(file_sha256(client_path), source_before)
            self.assertFalse(report["baseline_progress_files_copied"])
            progress = json.loads(next((method / "checkpoints" / "deployment_progress").glob("*.json")).read_text())
            self.assertEqual(progress["client_optimizer_steps"], 0)
            self.assertEqual(progress["adaptation_windows_seen"], 0)

    def test_prepared_deployment_modes_are_exact_and_protected(self):
        adaptation = resolve_deployment_slice("deployment_adaptation", window_offset=524, window_limit=700, materialized=True, runtime_drift_injection=False, for_training=True)
        self.assertEqual((adaptation.source_mode, adaptation.window_offset, adaptation.window_limit), ("test", 524, 700))
        holdout = resolve_deployment_slice("deployment_final_holdout", window_offset=1224, window_limit=524, materialized=True)
        self.assertFalse(holdout.training_allowed)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            resolve_deployment_slice("deployment_final_holdout", materialized=True, for_training=True)
        with self.assertRaisesRegex(ValueError, "runtime drift injection"):
            resolve_deployment_slice("deployment_adaptation", materialized=True, runtime_drift_injection=True)
        with self.assertRaisesRegex(ValueError, "window_limit=700"):
            resolve_deployment_slice("deployment_adaptation", window_limit=699, materialized=True)

    def test_production_M2_round_robin_selected_order(self):
        calls = []
        counts = {}
        def fake_post(url, payload, timeout=600.0):
            port = int(url.split("localhost:")[1].split("/")[0])
            calls.append(port); counts[port] = counts.get(port, 0) + 1
            return {"status": "batch_processed" if counts[port] == 1 else "adaptation_complete"}
        with patch("run_split.post_json", side_effect=fake_post):
            rows = run_m2_oracle_client_only(["1"], {"1": ["a", "b", "control"]}, {"a", "b"})
        self.assertEqual(calls, [run_split.CLIENT_WORKER_BASE_PORT, run_split.CLIENT_WORKER_BASE_PORT + 1, run_split.CLIENT_WORKER_BASE_PORT, run_split.CLIENT_WORKER_BASE_PORT + 1])
        self.assertEqual(len(rows), 4)

    def test_production_M2_server_endpoint_preserves_all_state(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "OUTPUT_DIR": tmp, "CLUSTER_ID": "m2prod", "GPU_ENABLED": "false",
            "MODEL_INPUT_SIZE": "1", "MODEL_HIDDEN_SIZE": "8", "MODEL_NUM_LAYERS": "3",
            "MODEL_SPLIT_LAYER": "2", "MODEL_OUTPUT_SIZE": "1", "MODEL_DROPOUT": "0.4",
            "USE_QUALITY_WEIGHTS": "false", "USE_QUALITY_AS_INPUT": "false",
            "USE_QUALITY_AGGREGATION_WEIGHTS": "false", "USE_QUALITY_ANCHOR_AGGREGATION": "false",
        }, clear=False):
            module = importlib.import_module("server.server")
            server = module.SplitLearningServerState()
            before_model = model_state_hashes(server.model)
            before_optimizer = optimizer_state_hash(server.optimizer)
            before_counters = counter_state_hash(server.round_state())
            result = server.frozen_activation_step("selected", torch.randn(3, 8), torch.randn(3, 1))
            self.assertTrue(result["server_state_unchanged"])
            self.assertEqual(before_model, model_state_hashes(server.model))
            self.assertEqual(before_optimizer, optimizer_state_hash(server.optimizer))
            self.assertEqual(before_counters, counter_state_hash(server.round_state()))
            self.assertEqual(server.optimizer_step_count, 0)
            self.assertFalse(server.model.training)

    def test_method_environment_routes_M1_to_existing_scheduler_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "common" / "drifted_deployment_timeline").mkdir(parents=True); (root / "common" / "clean_deployment_timeline").mkdir()
            rows = [{"cluster_id": str(cluster), "client_id": f"{cluster}{index:02d}", "selected_for_drift": index < 8} for cluster in range(1, 5) for index in range(21)]
            write_csv_rows(root / "selected_drift_clients.csv", list(rows[0]), rows)
            env = build_method_environment(Method.M1, {"root": root})
            self.assertEqual(env["TRAIN_STREAM_MODE"], "deployment_adaptation")
            self.assertEqual((env["NUM_EPOCHS"], env["BATCH_SIZE"]), ("1", "128"))
            self.assertEqual(env["TRAINING_MODE"], "client_batch_random")
            self.assertEqual(env["CLIENT_ORDER_STRATEGY"], "random_per_round")
            self.assertEqual(env["SKIP_TRAINING"], "false")


if __name__ == "__main__":
    unittest.main()
