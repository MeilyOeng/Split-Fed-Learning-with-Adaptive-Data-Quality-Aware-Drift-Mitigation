"""Source-aware timeline drift orchestration and component-level manifests."""

from __future__ import annotations
from dataclasses import dataclass

import json
from typing import Optional, Sequence

import numpy as np

from .configuration import canonical_json
from .contracts import (
    DriftExperimentConfig,
    DriftManifestRow,
    DriftSource,
    DriftType,
    TimelineInjectionResult,
)
from .generators import (
    GeneratorRegistry,
    StrictCovariateGeneration,
    TargetMappingRegistry,
    validate_strict_covariate_generation,
)
from .process import generate_process_timeline
from .schedule import ResolvedExperimentSchedule, derive_stable_seed
from .transforms import apply_measurement_transforms, as_2d_timeline
from .windowing import ForecastWindows, build_forecasting_windows


@dataclass(frozen=True)
class TargetMappingBenchmarkResult:
    windows: ForecastWindows
    old_targets: np.ndarray
    new_targets: np.ndarray
    manifest: tuple[DriftManifestRow, ...]
    autoregressive_consistency_expected: bool = False
    exception_reason: str = "supervised_target_mapping_not_autoregressive"



def _json_values(row: np.ndarray) -> str:
    values = [None if not np.isfinite(item) else float(item) for item in np.asarray(row).reshape(-1)]
    return json.dumps(values, separators=(",", ":"))


def _manifest_rows(
    *,
    config: DriftExperimentConfig,
    resolved: ResolvedExperimentSchedule,
    cluster_id: str,
    client_id: str,
    component_id: str,
    component_type: str,
    base: np.ndarray,
    runtime: np.ndarray,
    changed: np.ndarray,
    affected_features: Sequence[int],
    parameters: dict,
    autoregressive_consistency_expected: bool,
    exception_reason: Optional[str],
) -> list[DriftManifestRow]:
    schedule = resolved.for_client(cluster_id, client_id)
    client_record = next(
        item
        for item in resolved.selections
        if item.cluster_id == str(cluster_id) and item.client_id == str(client_id)
    )
    return [
        DriftManifestRow(
            experiment_id=config.experiment_id,
            configuration_id=config.config_hash[:16],
            cluster_id=str(cluster_id),
            client_id=str(client_id),
            absolute_index=point.absolute_index,
            timestamp=point.timestamp,
            component_id=component_id,
            component_type=component_type,
            drift_type=config.temporal.drift_type.value,
            drift_source=component_type,
            configured_regime=point.configured_regime,
            old_concept_id=point.old_concept_id,
            new_concept_id=point.new_concept_id,
            active_concept_id=point.active_concept_id,
            drift_weight=point.drift_weight,
            regime_active=point.regime_active,
            change_point_event=point.change_point_event,
            new_regime_selected=point.new_regime_selected,
            transformation_applied=bool(changed[position]),
            affected_features=canonical_json(list(affected_features)),
            transformation_parameters=canonical_json(parameters),
            experiment_seed=config.experiment_seed,
            client_seed=client_record.client_seed,
            config_hash=config.config_hash,
            resolved_schedule_hash=resolved.resolved_schedule_hash,
            autoregressive_consistency_expected=autoregressive_consistency_expected,
            exception_reason=exception_reason,
            base_values=_json_values(base[position]),
            runtime_values=_json_values(runtime[position]),
        )
        for position, point in enumerate(schedule)
    ]


def inject_timeline(
    clean_values: Sequence[float] | np.ndarray,
    *,
    timestamps: Sequence[object],
    config: DriftExperimentConfig,
    resolved_schedule: ResolvedExperimentSchedule,
    cluster_id: str,
    client_id: str,
    generator_registry: Optional[GeneratorRegistry] = None,
) -> TimelineInjectionResult:
    """Apply process-like components before measurement components on one raw timeline."""
    clean, _ = as_2d_timeline(clean_values)
    schedule = resolved_schedule.for_client(cluster_id, client_id)
    if not schedule:
        immutable = np.array(clean, copy=True)
        immutable.setflags(write=False)
        return TimelineInjectionResult(
            clean_timeline=immutable,
            latent_timeline=np.array(clean, copy=True),
            observed_timeline=np.array(clean, copy=True),
            target_timeline=np.array(clean, copy=True),
            manifest=(),
            autoregressive_consistency_expected=True,
            exception_reason=None,
        )
    if len(schedule) != clean.shape[0] or len(timestamps) != clean.shape[0]:
        raise ValueError("resolved schedule and timestamps must align with the raw timeline")
    selection = next(
        item for item in resolved_schedule.selections
        if item.cluster_id == str(cluster_id) and item.client_id == str(client_id)
    )

    latent = np.array(clean, copy=True)
    observed = np.array(clean, copy=True)
    innovation_source: Optional[str] = None
    innovations: Optional[np.ndarray] = None
    manifest: list[DriftManifestRow] = []
    process_like = [
        component for component in config.components
        if component.source in {DriftSource.PROCESS, DriftSource.STRICT_COVARIATE}
    ]
    measurement = [
        component for component in config.components
        if component.source is DriftSource.MEASUREMENT
    ]
    unsupported = [
        component for component in config.components
        if component.source is DriftSource.TARGET_MAPPING
    ]
    if unsupported:
        raise ValueError(
            "TARGET_MAPPING is a supervised benchmark and must use generate_target_mapping_benchmark"
        )
    if len(process_like) > 1:
        raise ValueError("a timeline may contain only one process or strict-covariate component")

    for component in process_like:
        before = np.array(latent, copy=True)
        if component.source is DriftSource.PROCESS:
            generated = generate_process_timeline(
                latent,
                schedule,
                component.parameters,
                target_feature_index=config.target_feature_index,
                client_seed=selection.client_seed,
                component_id=component.component_id,
            )
            latent = generated.values
            changed = generated.changed
            innovation_source = generated.innovation_source
            innovations = np.array(generated.innovations, copy=True)
            component_type = DriftSource.PROCESS.value
            affected_features = (config.target_feature_index,)
        else:
            if generator_registry is None:
                raise ValueError("strict_covariate timeline injection requires a generator registry")
            generator_id = str(component.parameters["generator_id"])
            generator = generator_registry.require(generator_id)
            if str(component.parameters["benchmark_kind"]).lower() != "chronological":
                raise ValueError(
                    "non-autoregressive supervised strict-covariate generation cannot be used in overlapping streaming"
                )
            generation = generator.generate(
                latent,
                schedule,
                target_feature_index=config.target_feature_index,
                window_size=config.window_size,
                forecast_horizon=config.forecast_horizon,
                rng=np.random.default_rng(
                    derive_stable_seed(selection.client_seed, component.component_id, "strict-covariate")
                ),
                parameters=component.parameters,
            )
            validate_strict_covariate_generation(
                generation,
                generator,
                require_chronological=True,
                target_feature_index=config.target_feature_index,
                window_size=config.window_size,
                forecast_horizon=config.forecast_horizon,
            )
            latent, _ = as_2d_timeline(generation.chronological_timeline)
            forbidden_changes = [
                position
                for position, point in enumerate(schedule)
                if (not point.regime_active or (point.change_point_event and point.drift_weight == 0.0))
                and not np.allclose(latent[position], before[position], rtol=0.0, atol=1e-12)
            ]
            if forbidden_changes:
                raise ValueError(
                    "chronological strict-covariate generator changed values before schedule activation"
                )
            changed = np.any(~np.isclose(latent, before, rtol=0.0, atol=1e-12), axis=1)
            feature_changes = ~np.isclose(latent, before, rtol=0.0, atol=1e-12)
            affected_features = tuple(
                int(index) for index in np.flatnonzero(np.any(feature_changes, axis=0))
            )
            component_type = DriftSource.STRICT_COVARIATE.value
        manifest.extend(
            _manifest_rows(
                config=config,
                resolved=resolved_schedule,
                cluster_id=cluster_id,
                client_id=client_id,
                component_id=component.component_id,
                component_type=component_type,
                base=before,
                runtime=latent,
                changed=changed,
                affected_features=affected_features,
                parameters=dict(component.parameters),
                autoregressive_consistency_expected=True,
                exception_reason=None,
            )
        )

    observed = np.array(latent, copy=True)
    for component in measurement:
        before = np.array(observed, copy=True)
        transformed = apply_measurement_transforms(
            observed,
            schedule,
            component.feature_transforms,
            client_seed=selection.client_seed,
            component_id=component.component_id,
        )
        observed = transformed.values
        manifest.extend(
            _manifest_rows(
                config=config,
                resolved=resolved_schedule,
                cluster_id=cluster_id,
                client_id=client_id,
                component_id=component.component_id,
                component_type=DriftSource.MEASUREMENT.value,
                base=before,
                runtime=observed,
                changed=transformed.changed,
                affected_features=transformed.affected_features,
                parameters={
                    "feature_transforms": [
                        {
                            "feature_index": item.feature_index,
                            "type": item.transform_type.value,
                            **dict(item.parameters),
                        }
                        for item in component.feature_transforms
                    ]
                },
                autoregressive_consistency_expected=False,
                exception_reason="measurement_drift",
            )
        )

    has_measurement = bool(measurement)
    immutable_clean = np.array(clean, copy=True)
    immutable_clean.setflags(write=False)
    return TimelineInjectionResult(
        clean_timeline=immutable_clean,
        latent_timeline=latent,
        observed_timeline=observed,
        target_timeline=np.array(latent, copy=True),
        manifest=tuple(manifest),
        autoregressive_consistency_expected=not has_measurement,
        exception_reason="measurement_drift" if has_measurement else None,
        innovation_source=innovation_source,
        innovations=innovations,
        equation_units="raw_source_units",
    )


def generate_supervised_strict_covariate_benchmark(
    clean_values: Sequence[float] | np.ndarray,
    *,
    config: DriftExperimentConfig,
    resolved_schedule: ResolvedExperimentSchedule,
    cluster_id: str,
    client_id: str,
    generator_registry: GeneratorRegistry,
) -> StrictCovariateGeneration:
    components = [
        item for item in config.components if item.source is DriftSource.STRICT_COVARIATE
    ]
    if len(components) != 1:
        raise ValueError("supervised strict-covariate benchmark requires exactly one component")
    component = components[0]
    if str(component.parameters.get("benchmark_kind", "")).lower() != "supervised":
        raise ValueError("component is not configured as a supervised benchmark")
    selection = next(
        item for item in resolved_schedule.selections
        if item.cluster_id == str(cluster_id) and item.client_id == str(client_id)
    )
    generator = generator_registry.require(str(component.parameters["generator_id"]))
    generation = generator.generate(
        np.asarray(clean_values, dtype=float),
        resolved_schedule.for_client(cluster_id, client_id),
        target_feature_index=config.target_feature_index,
        window_size=config.window_size,
        forecast_horizon=config.forecast_horizon,
        rng=np.random.default_rng(
            derive_stable_seed(selection.client_seed, component.component_id, "strict-covariate")
        ),
        parameters=component.parameters,
    )
    validate_strict_covariate_generation(
        generation,
        generator,
        require_chronological=False,
        target_feature_index=config.target_feature_index,
        window_size=config.window_size,
        forecast_horizon=config.forecast_horizon,
    )
    return generation


def generate_target_mapping_benchmark(
    clean_values: Sequence[float] | np.ndarray,
    *,
    timestamps: Sequence[object],
    config: DriftExperimentConfig,
    resolved_schedule: ResolvedExperimentSchedule,
    cluster_id: str,
    client_id: str,
    mapping_registry: TargetMappingRegistry,
) -> TargetMappingBenchmarkResult:
    """Generate an explicitly non-autoregressive supervised mapping benchmark."""
    components = [
        item for item in config.components if item.source is DriftSource.TARGET_MAPPING
    ]
    if len(components) != 1:
        raise ValueError("target-mapping benchmark requires exactly one target-mapping component")
    component = components[0]
    if not bool(component.parameters.get("supervised_only", False)):
        raise ValueError("target-mapping benchmark must set supervised_only=true")

    clean, _ = as_2d_timeline(clean_values)
    schedule = resolved_schedule.for_client(cluster_id, client_id)
    if len(schedule) != clean.shape[0] or len(timestamps) != clean.shape[0]:
        raise ValueError("resolved schedule and timestamps must align with the raw timeline")
    base_windows = build_forecasting_windows(
        clean,
        clean,
        window_size=config.window_size,
        forecast_horizon=config.forecast_horizon,
        target_feature_index=config.target_feature_index,
    )
    old_mapping = mapping_registry.require(str(component.parameters["old_mapping_id"]))
    new_mapping = mapping_registry.require(str(component.parameters["new_mapping_id"]))
    old_targets = np.asarray(old_mapping(base_windows.X), dtype=float).reshape(-1)
    new_targets = np.asarray(new_mapping(base_windows.X), dtype=float).reshape(-1)
    if old_targets.size != base_windows.y.size or new_targets.size != base_windows.y.size:
        raise ValueError("target mappings must return one target per input window")
    if not np.isfinite(old_targets).all() or not np.isfinite(new_targets).all():
        raise ValueError("target mappings returned non-finite values")
    if np.allclose(old_targets, new_targets, rtol=0.0, atol=1e-15):
        raise ValueError("old and new target mappings are numerically identical")

    runtime_targets = np.array(old_targets, copy=True)
    for window_position, target_position in enumerate(base_windows.target_indices):
        point = schedule[int(target_position)]
        if config.temporal.drift_type is DriftType.INCREMENTAL:
            runtime_targets[window_position] = (
                (1.0 - point.drift_weight) * old_targets[window_position]
                + point.drift_weight * new_targets[window_position]
            )
        elif point.new_regime_selected:
            runtime_targets[window_position] = new_targets[window_position]

    base_by_timestamp = np.full((clean.shape[0], 1), np.nan, dtype=float)
    runtime_by_timestamp = np.full((clean.shape[0], 1), np.nan, dtype=float)
    changed = np.zeros(clean.shape[0], dtype=bool)
    for window_position, target_position in enumerate(base_windows.target_indices):
        position = int(target_position)
        base_by_timestamp[position, 0] = old_targets[window_position]
        runtime_by_timestamp[position, 0] = runtime_targets[window_position]
        changed[position] = not np.isclose(
            old_targets[window_position], runtime_targets[window_position],
            rtol=0.0, atol=1e-12,
        )
    manifest = _manifest_rows(
        config=config,
        resolved=resolved_schedule,
        cluster_id=cluster_id,
        client_id=client_id,
        component_id=component.component_id,
        component_type=DriftSource.TARGET_MAPPING.value,
        base=base_by_timestamp,
        runtime=runtime_by_timestamp,
        changed=changed,
        affected_features=[config.target_feature_index],
        parameters=dict(component.parameters),
        autoregressive_consistency_expected=False,
        exception_reason="supervised_target_mapping_not_autoregressive",
    )
    return TargetMappingBenchmarkResult(
        windows=ForecastWindows(
            base_windows.X,
            runtime_targets,
            base_windows.window_start_indices,
            base_windows.window_end_indices,
            base_windows.target_indices,
        ),
        old_targets=old_targets,
        new_targets=new_targets,
        manifest=tuple(manifest),
    )
