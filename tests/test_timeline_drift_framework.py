from __future__ import annotations

import copy
from pathlib import Path
import functools
import unittest
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class _Approx:
    def __init__(self, expected: float) -> None:
        self.expected = float(expected)

    def __eq__(self, actual: object) -> bool:
        return bool(np.isclose(float(actual), self.expected, rtol=1e-7, atol=1e-12))

    def __ne__(self, actual: object) -> bool:
        return not self == actual


class _Mark:
    @staticmethod
    def parametrize(names_text: str, values: Sequence[object]):
        names = tuple(item.strip() for item in names_text.split(","))
        def decorator(function):
            @functools.wraps(function)
            def wrapped():
                for value in values:
                    if len(names) == 1:
                        function(value)
                    else:
                        function(*value)
            return wrapped
        return decorator


class _TestCompat:
    mark = _Mark()

    @staticmethod
    def approx(expected: float) -> _Approx:
        return _Approx(expected)

    @staticmethod
    def raises(exception_type, *, match: str):
        return unittest.TestCase().assertRaisesRegex(exception_type, match)


pytest = _TestCompat()

from drift.configuration import legacy_environment_config, load_drift_config
from drift.contracts import DetectorEvent, DriftSource, DriftType, SchedulePoint
from drift.generators import (
    FrozenLinearSupervisedGenerator,
    GeneratorRegistry,
    StrictCovariateGeneration,
    TargetMappingRegistry,
)
from drift.evaluator import evaluate_detection_from_manifest
from drift.pipeline import (
    generate_supervised_strict_covariate_benchmark,
    generate_target_mapping_benchmark,
    inject_timeline,
)
from drift.schedule import build_drift_schedule, resolve_experiment_schedule
from drift.windowing import (
    assert_horizon_one_consistency,
    build_forecasting_windows,
    fit_clean_training_scalers,
    normalize_windows,
)


def _raw_config(
    temporal: Mapping[str, object],
    source: Mapping[str, object],
    *,
    clients: Sequence[str] = ("1",),
    enabled: bool = True,
    seed: int = 73,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "experiment_id": "timeline-test",
        "enabled": enabled,
        "experiment_seed": seed,
        "clients": {"ids": list(clients)},
        "modes": ["test"],
        "temporal": dict(temporal),
        "source": dict(source),
        "windowing": {
            "window_size": 4,
            "forecast_horizon": 1,
            "target_feature_index": 0,
        },
    }


def _measurement_source(*specs: Mapping[str, object]) -> dict[str, object]:
    return {"type": "measurement", "feature_transforms": {0: list(specs)}}


def _coordinates(length: int = 40, clients: Sequence[str] = ("1",)):
    timestamps = list(pd.date_range("2024-01-01", periods=length, freq="h"))
    return {("c", client): (list(range(length)), timestamps) for client in clients}


def _resolve(config, length: int = 40, clients: Sequence[str] = ("1",)):
    return resolve_experiment_schedule(config, _coordinates(length, clients))


def _clean_ar(length: int = 40, intercept: float = 1.0, coefficient: float = 0.5) -> np.ndarray:
    values = np.empty(length, dtype=float)
    values[0] = 2.0
    for index in range(1, length):
        values[index] = intercept + coefficient * values[index - 1]
    return values


def _process_source(**changes: object) -> dict[str, object]:
    return {
        "type": "process",
        "old": {
            "intercept": 1.0,
            "ar_coefficients": [0.5],
            "innovation_sigma": 1.0,
            "seasonal_amplitude": 0.0,
            "seasonal_period": 24,
        },
        "changes": changes,
        "innovation_mode": "clean_residual",
    }


def test_sudden_schedule_switches_at_exact_absolute_onset() -> None:
    config = load_drift_config(_raw_config(
        {"type": "sudden", "start_index": 10},
        _measurement_source({"type": "offset", "delta": 2.0}),
    ))
    schedule = build_drift_schedule(
        config,
        client_id="1",
        absolute_indices=list(range(20)),
        timestamps=list(pd.date_range("2024-01-01", periods=20, freq="h")),
    )
    assert not schedule[9].regime_active
    assert schedule[9].drift_weight == 0.0
    assert schedule[10].regime_active
    assert schedule[10].drift_weight == 1.0
    assert schedule[10].change_point_event
    assert sum(point.change_point_event for point in schedule) == 1


@pytest.mark.parametrize("transition", ["linear", "sigmoid", "exponential"])
def test_incremental_transition_has_exact_endpoint_weights(transition: str) -> None:
    config = load_drift_config(_raw_config(
        {"type": "incremental", "start_index": 10, "end_index": 20, "transition": transition},
        _measurement_source({"type": "scale", "factor": 1.2}),
    ))
    schedule = _resolve(config, length=30).for_client("c", "1")
    assert schedule[10].drift_weight == pytest.approx(0.0)
    if transition in {"linear", "sigmoid"}:
        assert schedule[15].drift_weight == pytest.approx(0.5)
    else:
        assert 0.0 < schedule[15].drift_weight < 1.0
    assert schedule[20].drift_weight == pytest.approx(1.0)
    assert schedule[15].active_concept_id == "blend"


def test_gradual_mask_is_materialized_once_and_seeded_per_client() -> None:
    config = load_drift_config(_raw_config(
        {"type": "gradual", "start_index": 10, "end_index": 180},
        _measurement_source({"type": "offset", "delta": 3.0}),
        clients=("1", "2"),
        seed=90210,
    ))
    first = _resolve(config, length=200, clients=("1", "2"))
    second = _resolve(config, length=200, clients=("1", "2"))
    mask_1 = tuple(point.new_regime_selected for point in first.for_client("c", "1"))
    mask_1_repeat = tuple(point.new_regime_selected for point in second.for_client("c", "1"))
    mask_2 = tuple(point.new_regime_selected for point in first.for_client("c", "2"))
    assert mask_1 == mask_1_repeat
    assert mask_1 != mask_2
    assert first.resolved_schedule_hash == second.resolved_schedule_hash
    seeds = {item.client_id: item.client_seed for item in first.selections}
    assert seeds["1"] != seeds["2"]


def test_recurring_cycle_boundaries_are_explicit() -> None:
    config = load_drift_config(_raw_config(
        {
            "type": "recurring",
            "start_index": 5,
            "cycle_length": 6,
            "new_duration": 2,
            "old_duration": 4,
        },
        _measurement_source({"type": "offset", "delta": 1.0}),
    ))
    schedule = _resolve(config, length=20).for_client("c", "1")
    assert [schedule[index].active_concept_id for index in range(5, 12)] == [
        "new", "new", "old", "old", "old", "old", "new"
    ]
    assert schedule[5].change_point_event
    assert schedule[7].change_point_event
    assert schedule[11].change_point_event


def test_seasonal_change_is_a_process_component_not_a_temporal_type() -> None:
    raw = _raw_config(
        {"type": "sudden", "start_index": 10},
        {
            **_process_source(intercept_delta=0.0),
            "seasonal_change": {"amplitude_delta": 2.0, "phase_delta": 0.0, "period_new": 4},
        },
    )
    raw["source"]["changes"] = {}
    config = load_drift_config(raw)
    resolved = _resolve(config, length=24)
    result = inject_timeline(
        _clean_ar(24),
        timestamps=_coordinates(24)[("c", "1")][1],
        config=config,
        resolved_schedule=resolved,
        cluster_id="c",
        client_id="1",
    )
    rows = [row for row in result.manifest if row.component_type == "process"]
    assert rows[10].drift_type == "sudden"
    process = result.latent_timeline[:, 0]
    assert process[11] != pytest.approx(_clean_ar(24)[11])
    invalid = copy.deepcopy(raw)
    invalid["temporal"]["type"] = "seasonal"
    with pytest.raises(ValueError, match="Unsupported DriftType"):
        load_drift_config(invalid)


def test_measurement_transform_is_reproducible_feature_specific_and_additive() -> None:
    raw = _raw_config(
        {"type": "sudden", "start_index": 5},
        {
            "type": "measurement",
            "feature_transforms": {
                0: [{"type": "scale", "factor": 2.0}, {"type": "offset", "delta": 3.0}],
                1: {"type": "noise", "sigma": 0.1},
            },
        },
    )
    config = load_drift_config(raw)
    resolved = _resolve(config, length=16)
    clean = np.column_stack([np.arange(16, dtype=float), np.full(16, 10.0)])
    timestamps = _coordinates(16)[("c", "1")][1]
    first = inject_timeline(clean, timestamps=timestamps, config=config, resolved_schedule=resolved, cluster_id="c", client_id="1")
    second = inject_timeline(clean, timestamps=timestamps, config=config, resolved_schedule=resolved, cluster_id="c", client_id="1")
    np.testing.assert_array_equal(first.observed_timeline, second.observed_timeline)
    np.testing.assert_allclose(first.observed_timeline[5:, 0], clean[5:, 0] * 2.0 + 3.0)
    assert first.observed_timeline[5, 0] == 13.0
    assert not np.array_equal(first.observed_timeline[5:, 1], clean[5:, 1])
    np.testing.assert_array_equal(first.target_timeline, clean)
    assert not first.autoregressive_consistency_expected
    assert first.exception_reason == "measurement_drift"




def test_incremental_composite_transform_interpolates_identity_to_final_g() -> None:
    config = load_drift_config(_raw_config(
        {"type": "incremental", "start_index": 5, "end_index": 9},
        _measurement_source(
            {"type": "scale", "factor": 2.0},
            {"type": "scale", "factor": 3.0},
        ),
    ))
    clean = np.full(14, 10.0)
    result = inject_timeline(
        clean,
        timestamps=_coordinates(14)[("c", "1")][1],
        config=config,
        resolved_schedule=_resolve(config, length=14),
        cluster_id="c",
        client_id="1",
    )
    assert result.manifest[7].drift_weight == pytest.approx(0.5)
    assert result.observed_timeline[7, 0] == pytest.approx(35.0)

def test_overlapping_windows_reuse_one_transformed_timestamp() -> None:
    config = load_drift_config(_raw_config(
        {"type": "sudden", "start_index": 7},
        _measurement_source({"type": "offset", "delta": 10.0}),
    ))
    clean = np.arange(20, dtype=float)
    result = inject_timeline(
        clean,
        timestamps=_coordinates(20)[("c", "1")][1],
        config=config,
        resolved_schedule=_resolve(config, length=20),
        cluster_id="c",
        client_id="1",
    )
    windows = build_forecasting_windows(
        result.observed_timeline,
        result.target_timeline,
        window_size=4,
        forecast_horizon=1,
        target_feature_index=0,
    )
    assert windows.X[4, 3, 0] == windows.X[5, 2, 0] == windows.X[6, 1, 0] == windows.X[7, 0, 0]
    with pytest.raises(AssertionError, match="chronological consistency"):
        assert_horizon_one_consistency(windows.X, windows.y, target_feature_index=0)


def test_recursive_process_uses_coherent_inputs_and_targets() -> None:
    config = load_drift_config(_raw_config(
        {"type": "sudden", "start_index": 10},
        _process_source(intercept_delta=0.5, ar_coefficient_delta=[0.1]),
    ))
    clean = _clean_ar(30)
    result = inject_timeline(
        clean,
        timestamps=_coordinates(30)[("c", "1")][1],
        config=config,
        resolved_schedule=_resolve(config, length=30),
        cluster_id="c",
        client_id="1",
    )
    np.testing.assert_array_equal(result.latent_timeline[:10, 0], clean[:10])
    windows = build_forecasting_windows(
        result.observed_timeline,
        result.target_timeline,
        window_size=4,
        forecast_horizon=1,
        target_feature_index=0,
    )
    assert_horizon_one_consistency(windows.X, windows.y, target_feature_index=0)
    assert result.autoregressive_consistency_expected
    assert result.latent_timeline[11, 0] != pytest.approx(clean[11])


def test_incremental_metadata_separates_onset_activity_and_actual_transform() -> None:
    config = load_drift_config(_raw_config(
        {"type": "incremental", "start_index": 5, "end_index": 9},
        _measurement_source({"type": "offset", "delta": 4.0}),
    ))
    result = inject_timeline(
        np.arange(15, dtype=float),
        timestamps=_coordinates(15)[("c", "1")][1],
        config=config,
        resolved_schedule=_resolve(config, length=15),
        cluster_id="c",
        client_id="1",
    )
    onset = result.manifest[5]
    assert onset.change_point_event
    assert onset.regime_active
    assert onset.drift_weight == 0.0
    assert not onset.new_regime_selected
    assert not onset.transformation_applied
    assert result.manifest[7].active_concept_id == "blend"
    assert result.manifest[7].transformation_applied


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda raw: raw["source"]["feature_transforms"][0][0].update({"factor": 1.0}), "different from 1"),
        (lambda raw: raw["temporal"].update({"type": "unknown"}), "Unsupported DriftType"),
    ],
)
def test_invalid_and_noop_configuration_is_rejected(mutator, message: str) -> None:
    raw = _raw_config(
        {"type": "sudden", "start_index": 5},
        _measurement_source({"type": "scale", "factor": 1.2}),
    )
    mutator(raw)
    with pytest.raises(ValueError, match=message):
        load_drift_config(raw)


def test_process_noop_and_bad_recurring_period_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero process change"):
        load_drift_config(_raw_config(
            {"type": "sudden", "start_index": 5},
            _process_source(),
        ))
    inactive_season = _process_source()
    inactive_season["changes"] = {}
    inactive_season["seasonal_change"] = {"phase_delta": 1.0, "period_new": 12}
    with pytest.raises(ValueError, match="non-zero process change"):
        load_drift_config(_raw_config(
            {"type": "sudden", "start_index": 5},
            inactive_season,
        ))

    with pytest.raises(ValueError, match="must equal cycle_length"):
        load_drift_config(_raw_config(
            {"type": "recurring", "start_index": 5, "cycle_length": 8, "new_duration": 2, "old_duration": 3},
            _measurement_source({"type": "offset", "delta": 1.0}),
        ))




def test_schema_rejects_unknown_and_incompatible_fields() -> None:
    bad_process = _raw_config(
        {"type": "sudden", "start_index": 5},
        _process_source(intercept_delta=0.5),
    )
    bad_process["source"]["changes"]["interept_delta"] = 0.5
    with pytest.raises(ValueError, match="unknown process.changes field"):
        load_drift_config(bad_process)

    bad_transition = _raw_config(
        {"type": "sudden", "start_index": 5, "transition": "linear"},
        _measurement_source({"type": "offset", "delta": 1.0}),
    )
    with pytest.raises(ValueError, match="only compatible with incremental or gradual"):
        load_drift_config(bad_transition)

    bad_feature = _raw_config(
        {"type": "sudden", "start_index": 5},
        {"type": "measurement", "feature_transforms": {"not-an-index": {"type": "offset", "delta": 1.0}}},
    )
    with pytest.raises(ValueError, match="is not an integer"):
        load_drift_config(bad_feature)

    bad_boolean = _raw_config(
        {"type": "sudden", "start_index": 5},
        _measurement_source({"type": "offset", "delta": 1.0}),
    )
    bad_boolean["enabled"] = "false"
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        load_drift_config(bad_boolean)


def test_legacy_source_names_canonicalize_to_corrected_terms() -> None:
    assert DriftSource.parse("input") is DriftSource.MEASUREMENT
    assert DriftSource.INPUT is DriftSource.MEASUREMENT
    assert DriftSource.parse("target") is DriftSource.TARGET_MAPPING
    assert DriftSource.TARGET is DriftSource.TARGET_MAPPING
    legacy_config = _raw_config(
        {"type": "sudden", "start_index": 5},
        _measurement_source({"type": "offset", "delta": 1.0}),
    )
    legacy_config["source"].pop("type")
    legacy_config["source"]["source"] = "input"
    assert load_drift_config(legacy_config).drift_source is DriftSource.MEASUREMENT

    conflicting = copy.deepcopy(legacy_config)
    conflicting["source"]["type"] = "process"
    with pytest.raises(ValueError, match="fields disagree"):
        load_drift_config(conflicting)


def test_config_parse_does_not_mutate_input_and_hash_is_canonical() -> None:
    raw = _raw_config(
        {"type": "sudden", "start_index": 5},
        _measurement_source({"type": "offset", "delta": 1.0}),
    )
    original = copy.deepcopy(raw)
    first = load_drift_config(raw)
    reordered = {key: raw[key] for key in reversed(list(raw))}
    second = load_drift_config(reordered)
    assert raw == original
    assert first.config_hash == second.config_hash
    assert first.canonical_config == second.canonical_config


class _ChronologicalPersistenceGenerator:
    generator_id = "persistence_residual_permutation_v1"
    generator_version = "1"
    residual_distribution_id = "fixed_empirical_residuals_v1"
    benchmark_kind = "chronological"

    def conditional_mean(self, inputs: np.ndarray) -> np.ndarray:
        return np.asarray(inputs, dtype=float)[:, -1, 0]

    def generate(
        self,
        clean_timeline: np.ndarray,
        schedule: Sequence[SchedulePoint],
        *,
        target_feature_index: int,
        window_size: int,
        forecast_horizon: int,
        rng: np.random.Generator,
        parameters: Mapping[str, object],
    ) -> StrictCovariateGeneration:
        from drift.windowing import build_forecasting_windows

        clean = np.asarray(clean_timeline, dtype=float).reshape(-1, 1)
        runtime = clean.copy()
        onset = next(index for index, point in enumerate(schedule) if point.regime_active)
        clean_residuals = np.diff(clean[:, 0])
        replacement = clean_residuals[onset - 1:][::-1]
        for position, residual in zip(range(onset, len(runtime)), replacement):
            runtime[position, 0] = runtime[position - 1, 0] + residual
        reference = build_forecasting_windows(
            clean, clean, window_size=window_size, forecast_horizon=forecast_horizon,
            target_feature_index=target_feature_index,
        )
        generated = build_forecasting_windows(
            runtime, runtime, window_size=window_size, forecast_horizon=forecast_horizon,
            target_feature_index=target_feature_index,
        )
        residuals = generated.y - self.conditional_mean(generated.X)
        return StrictCovariateGeneration(
            benchmark_kind="chronological",
            generator_id_before=self.generator_id,
            generator_id_after=self.generator_id,
            generator_version_before=self.generator_version,
            generator_version_after=self.generator_version,
            residual_distribution_id_before=self.residual_distribution_id,
            residual_distribution_id_after=self.residual_distribution_id,
            reference_inputs=reference.X,
            generated_inputs=generated.X,
            generated_targets=generated.y,
            residuals=residuals,
            chronological_timeline=runtime,
        )


def test_strict_covariate_requires_registered_generator_and_coherent_timeline() -> None:
    source = {
        "type": "strict_covariate",
        "generator_id": "persistence_residual_permutation_v1",
        "benchmark_kind": "chronological",
        "guarantees_autoregressive_consistency": True,
    }
    config = load_drift_config(_raw_config({"type": "sudden", "start_index": 8}, source))
    clean = np.empty(30, dtype=float)
    clean[0] = 10.0
    residual_pattern = (-1.0, 0.0, 2.0)
    for index in range(1, len(clean)):
        clean[index] = clean[index - 1] + residual_pattern[(index - 1) % len(residual_pattern)]
    kwargs = dict(
        timestamps=_coordinates(30)[("c", "1")][1], config=config,
        resolved_schedule=_resolve(config, length=30), cluster_id="c", client_id="1",
    )
    with pytest.raises(ValueError, match="generator registry"):
        inject_timeline(clean, **kwargs)
    registry = GeneratorRegistry()
    registry.register(_ChronologicalPersistenceGenerator())
    result = inject_timeline(clean, generator_registry=registry, **kwargs)
    np.testing.assert_array_equal(result.latent_timeline[:8, 0], clean[:8])
    windows = build_forecasting_windows(
        result.latent_timeline, result.target_timeline,
        window_size=4, forecast_horizon=1, target_feature_index=0,
    )
    assert_horizon_one_consistency(windows.X, windows.y, target_feature_index=0)


def test_strict_covariate_configuration_without_frozen_generator_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires generator_id"):
        load_drift_config(_raw_config(
            {"type": "sudden", "start_index": 8},
            {"type": "strict_covariate", "benchmark_kind": "supervised"},
        ))


def test_supervised_strict_covariate_uses_one_frozen_conditional() -> None:
    source = {
        "type": "strict_covariate",
        "generator_id": "frozen_linear_supervised_v1",
        "benchmark_kind": "supervised",
        "reference_inputs": [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        "input_shift": [2.0, -1.0],
        "residual_sigma": 0.05,
    }
    config = load_drift_config(_raw_config({"type": "sudden", "start_index": 8}, source))
    registry = GeneratorRegistry()
    generator = FrozenLinearSupervisedGenerator([0.5, -0.25], intercept=1.0)
    registry.register(generator)
    generation = generate_supervised_strict_covariate_benchmark(
        np.arange(20, dtype=float), config=config, resolved_schedule=_resolve(config, length=20),
        cluster_id="c", client_id="1", generator_registry=registry,
    )
    np.testing.assert_allclose(
        generation.generated_targets,
        generator.conditional_mean(generation.generated_inputs) + generation.residuals,
    )
    assert generation.generator_id_before == generation.generator_id_after
    assert generation.residual_distribution_id_before == generation.residual_distribution_id_after


def test_target_mapping_is_explicitly_supervised_and_not_autoregressive() -> None:
    source = {
        "type": "target_mapping",
        "old_mapping_id": "last",
        "new_mapping_id": "last_plus_10",
        "target_fed_back": True,
        "supervised_only": True,
    }
    config = load_drift_config(_raw_config({"type": "sudden", "start_index": 10}, source))
    registry = TargetMappingRegistry()
    registry.register("last", lambda X: X[:, -1, 0])
    registry.register("last_plus_10", lambda X: X[:, -1, 0] + 10.0)
    result = generate_target_mapping_benchmark(
        np.arange(24, dtype=float), timestamps=_coordinates(24)[("c", "1")][1],
        config=config, resolved_schedule=_resolve(config, length=24), cluster_id="c", client_id="1",
        mapping_registry=registry,
    )
    before = result.windows.target_indices < 10
    np.testing.assert_allclose(result.windows.y[before], result.old_targets[before])
    np.testing.assert_allclose(result.windows.y[~before], result.new_targets[~before])
    assert not result.autoregressive_consistency_expected
    with pytest.raises(AssertionError, match="chronological consistency"):
        assert_horizon_one_consistency(result.windows.X, result.windows.y, target_feature_index=0)


def test_mixed_manifest_has_one_row_per_component_timestamp_and_all_client_selections() -> None:
    source = {
        "type": "mixed",
        "components": [
            {"id": "process", **_process_source(intercept_delta=0.25)},
            {
                "id": "sensor",
                "type": "measurement",
                "feature_transforms": {1: {"type": "offset", "delta": 5.0}},
            },
        ],
    }
    config = load_drift_config(_raw_config(
        {"type": "sudden", "start_index": 10}, source, clients=("1",),
    ))
    resolved = _resolve(config, length=25, clients=("1", "2"))
    target = _clean_ar(25)
    clean = np.column_stack([target, np.linspace(0.0, 1.0, 25)])
    result = inject_timeline(
        clean, timestamps=_coordinates(25)[("c", "1")][1], config=config,
        resolved_schedule=resolved, cluster_id="c", client_id="1",
    )
    assert len(result.manifest) == 50
    assert {row.component_id for row in result.manifest} == {"process", "sensor"}
    measurement_rows = [row for row in result.manifest if row.component_id == "sensor"]
    assert all(not row.autoregressive_consistency_expected for row in measurement_rows)
    assert all(row.exception_reason == "measurement_drift" for row in measurement_rows)
    assert len(resolved.selections) == 2
    assert {row.client_id: row.affected for row in resolved.selections} == {"1": True, "2": False}
    assert len(config.config_hash) == len(resolved.resolved_schedule_hash) == 64


def test_clean_disabled_timeline_is_unchanged() -> None:
    raw = _raw_config(
        {"type": "none"}, {"type": "measurement"}, enabled=False, clients=(),
    )
    config = load_drift_config(raw)
    resolved = _resolve(config, length=20)
    clean = np.arange(20, dtype=np.float64)
    result = inject_timeline(
        clean, timestamps=_coordinates(20)[("c", "1")][1], config=config,
        resolved_schedule=resolved, cluster_id="c", client_id="1",
    )
    np.testing.assert_array_equal(result.clean_timeline[:, 0], clean)
    np.testing.assert_array_equal(result.observed_timeline[:, 0], clean)
    np.testing.assert_array_equal(result.target_timeline[:, 0], clean)
    assert not result.manifest


def test_clean_scalers_use_distinct_exact_x_and_y_training_unions() -> None:
    clean = np.arange(30, dtype=float)
    scaler = fit_clean_training_scalers(
        clean, clean, training_window_starts=range(0, 10), window_size=4,
        forecast_horizon=1, target_feature_index=0, method="zscore",
    )
    assert scaler.x_observation_indices == tuple(range(13))
    assert scaler.y_target_indices == tuple(range(4, 14))
    assert scaler.x_union_hash != scaler.y_union_hash
    assert scaler.x_location[0] == pytest.approx(np.mean(clean[:13]))
    assert scaler.y_location == pytest.approx(np.mean(clean[4:14]))

def test_shared_clean_scaler_uses_exact_x_y_coordinate_union() -> None:
    clean = np.arange(30, dtype=float)
    scaler = fit_clean_training_scalers(
        clean, clean, training_window_starts=range(0, 10), window_size=4,
        forecast_horizon=1, target_feature_index=0, method="zscore",
        share_target_scaler=True,
    )
    assert scaler.target_scaling_strategy == "shared"
    assert scaler.x_observation_indices == tuple(range(13))
    assert scaler.y_target_indices == tuple(range(4, 14))
    assert scaler.shared_target_indices == tuple(range(14))
    assert scaler.shared_target_union_hash is not None
    assert scaler.x_location[0] == pytest.approx(np.mean(clean[:14]))
    assert scaler.y_location == pytest.approx(scaler.x_location[0])
    assert scaler.y_scale == pytest.approx(scaler.x_scale[0])

    windows = build_forecasting_windows(
        clean, clean, window_size=4, forecast_horizon=1, target_feature_index=0,
    )
    normalized = normalize_windows(windows, scaler)
    assert_horizon_one_consistency(normalized.X, normalized.y, target_feature_index=0)

def test_shared_clean_scaler_rejects_different_overlapping_target_signal() -> None:
    clean = np.arange(30, dtype=float)
    different_target = clean.copy()
    different_target[5] += 1.0
    with pytest.raises(ValueError, match="same clean physical variable"):
        fit_clean_training_scalers(
            clean, different_target, training_window_starts=range(0, 10), window_size=4,
            forecast_horizon=1, target_feature_index=0, method="zscore",
            share_target_scaler=True,
        )


def test_drift_is_applied_in_raw_units_before_windowing_and_clean_normalization() -> None:
    clean = np.arange(30, dtype=float)
    scaler = fit_clean_training_scalers(
        clean, clean, training_window_starts=range(0, 8), window_size=4,
        forecast_horizon=1, target_feature_index=0, method="zscore",
    )
    config = load_drift_config(_raw_config(
        {"type": "sudden", "start_index": 15},
        _measurement_source({"type": "offset", "delta": 5.0}),
    ))
    result = inject_timeline(
        clean, timestamps=_coordinates(30)[("c", "1")][1], config=config,
        resolved_schedule=_resolve(config, length=30), cluster_id="c", client_id="1",
    )
    assert result.observed_timeline[15, 0] - clean[15] == 5.0
    raw_windows = build_forecasting_windows(
        result.observed_timeline, result.target_timeline,
        window_size=4, forecast_horizon=1, target_feature_index=0,
    )
    normalized = normalize_windows(raw_windows, scaler)
    expected = (result.observed_timeline[15, 0] - scaler.x_location[0]) / scaler.x_scale[0]
    assert normalized.X[12, 3, 0] == pytest.approx(expected)
    assert scaler.x_location[0] == pytest.approx(np.mean(clean[:11]))


def test_legacy_seasonal_alias_migrates_to_recurring_measurement_without_mutation() -> None:
    env = {
        "DRIFT_INJECTION_ENABLED": "true",
        "DRIFT_TYPE": "seasonal",
        "DRIFT_PERIOD": "8",
        "DRIFT_MAGNITUDE": "0.5",
        "DRIFT_CLIENT_FRACTION": "1.0",
    }
    config = legacy_environment_config(env, series_length=40)
    assert config.temporal.drift_type is DriftType.RECURRING
    assert config.drift_source is DriftSource.MEASUREMENT
    assert config.components[0].feature_transforms[0].parameters["factor"] == 1.5


def test_splitfed_training_math_files_are_not_imported_by_timeline_engine() -> None:
    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "drift/configuration.py", "drift/schedule.py", "drift/transforms.py",
            "drift/process.py", "drift/generators.py", "drift/pipeline.py", "drift/windowing.py",
        )
    )
    assert "fed_server" not in source
    assert "models." not in source
    assert "optimizer" not in source.lower()



def test_schema_v2_manifest_evaluator_uses_offline_selection_truth() -> None:
    config = load_drift_config(_raw_config(
        {"type": "sudden", "start_index": 10},
        _measurement_source({"type": "offset", "delta": 2.0}),
    ))
    result = inject_timeline(
        np.arange(20, dtype=float), timestamps=_coordinates(20)[("c", "1")][1],
        config=config, resolved_schedule=_resolve(config, length=20),
        cluster_id="c", client_id="1",
    )
    events = [
        DetectorEvent(
            run_id=config.experiment_id,
            cluster_id="c",
            client_id="1",
            mode="test",
            stream_name="endpoint",
            detector_name="page_hinkley",
            event_index=event_index,
            observation_index=absolute_index,
            timestamp=None,
            detector_value=float(absolute_index),
            alarm=absolute_index == 12,
        )
        for event_index, absolute_index in enumerate(range(5, 20))
    ]
    evaluated = evaluate_detection_from_manifest(events, result.manifest, affected=True)
    assert evaluated.primary is not None
    assert evaluated.primary.onset_index == 10
    assert evaluated.primary.first_valid_detection == 12
    assert evaluated.primary.detection_delay == 2
    clean = evaluate_detection_from_manifest(events, (), affected=False)
    assert clean.clean_client_false_positive
def load_tests(loader, standard_tests, pattern):
    del loader, standard_tests, pattern
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            suite.addTest(unittest.FunctionTestCase(function, description=name))
    return suite


if __name__ == "__main__":
    unittest.main()
