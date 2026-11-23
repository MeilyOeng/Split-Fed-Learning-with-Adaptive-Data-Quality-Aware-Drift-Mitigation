"""Recursive coherent autoregressive process drift."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .contracts import SchedulePoint
from .schedule import derive_stable_seed
from .transforms import as_2d_timeline


@dataclass(frozen=True)
class ARParameters:
    intercept: float
    coefficients: tuple[float, ...]
    seasonal_amplitude: float = 0.0
    seasonal_phase: float = 0.0
    seasonal_period: float = 24.0
    innovation_sigma: float = 0.0


@dataclass(frozen=True)
class ProcessResult:
    values: np.ndarray
    changed: np.ndarray
    old_candidates: np.ndarray
    new_candidates: np.ndarray
    innovations: np.ndarray
    innovation_source: str


def _finite(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _parse_old(parameters: Mapping[str, object]) -> ARParameters:
    old = dict(parameters.get("old", {}) or {})
    coefficients = tuple(_finite("ar coefficient", item) for item in old.get("ar_coefficients", []))
    if not coefficients:
        raise ValueError("process.old.ar_coefficients must be non-empty")
    period = _finite("seasonal_period", old.get("seasonal_period", 24.0))
    if period <= 0:
        raise ValueError("seasonal_period must be > 0")
    sigma = _finite("innovation_sigma", old.get("innovation_sigma", 0.0))
    if sigma < 0:
        raise ValueError("innovation_sigma must be >= 0")
    return ARParameters(
        intercept=_finite("intercept", old.get("intercept", 0.0)),
        coefficients=coefficients,
        seasonal_amplitude=_finite("seasonal_amplitude", old.get("seasonal_amplitude", 0.0)),
        seasonal_phase=_finite("seasonal_phase", old.get("seasonal_phase", 0.0)),
        seasonal_period=period,
        innovation_sigma=sigma,
    )


def _new_parameters(old: ARParameters, parameters: Mapping[str, object]) -> tuple[ARParameters, float, float]:
    changes = dict(parameters.get("changes", {}) or {})
    deltas = changes.get("ar_coefficient_delta", [0.0] * len(old.coefficients))
    if len(deltas) != len(old.coefficients):
        raise ValueError("ar_coefficient_delta length must match old ar_coefficients")
    seasonal = dict(parameters.get("seasonal_change", {}) or {})
    period = float(seasonal.get("period_new", old.seasonal_period))
    if period <= 0:
        raise ValueError("new seasonal period must be > 0")
    innovation_scale = _finite("innovation_scale", changes.get("innovation_scale", 1.0))
    if innovation_scale < 0:
        raise ValueError("innovation_scale must be >= 0")
    result = ARParameters(
        intercept=old.intercept + _finite("intercept_delta", changes.get("intercept_delta", 0.0)),
        coefficients=tuple(
            coefficient + _finite("ar coefficient delta", delta)
            for coefficient, delta in zip(old.coefficients, deltas)
        ),
        seasonal_amplitude=old.seasonal_amplitude
        + _finite("seasonal_amplitude_delta", seasonal.get("amplitude_delta", changes.get("seasonal_amplitude_delta", 0.0))),
        seasonal_phase=old.seasonal_phase
        + _finite("seasonal_phase_delta", seasonal.get("phase_delta", changes.get("seasonal_phase_delta", 0.0))),
        seasonal_period=period,
        innovation_sigma=old.innovation_sigma * innovation_scale,
    )
    return (
        result,
        _finite("scale_factor", changes.get("scale_factor", 1.0)),
        _finite("level_offset", changes.get("level_offset", 0.0)),
    )


def _is_stable(coefficients: Sequence[float]) -> bool:
    if len(coefficients) == 1:
        return abs(coefficients[0]) < 1.0
    polynomial = [-float(item) for item in reversed(coefficients)] + [1.0]
    roots = np.roots(polynomial)
    return bool(np.all(np.abs(roots) > 1.0 + 1e-10))


def _seasonal(spec: ARParameters, absolute_index: int) -> float:
    return spec.seasonal_amplitude * math.sin(
        (2.0 * math.pi * absolute_index / spec.seasonal_period) + spec.seasonal_phase
    )


def _conditional_mean(spec: ARParameters, history: np.ndarray, absolute_index: int) -> float:
    order = len(spec.coefficients)
    if history.size < order:
        raise ValueError(f"process requires at least {order} realized lag values")
    lags = history[-order:][::-1]
    return float(spec.intercept + np.dot(np.asarray(spec.coefficients), lags) + _seasonal(spec, absolute_index))


def generate_process_timeline(
    clean_values: Sequence[float] | np.ndarray,
    schedule: Sequence[SchedulePoint],
    parameters: Mapping[str, object],
    *,
    target_feature_index: int,
    client_seed: int,
    component_id: str,
) -> ProcessResult:
    """Generate one path recursively from common history and innovations."""
    clean, _ = as_2d_timeline(clean_values)
    if target_feature_index < 0 or target_feature_index >= clean.shape[1]:
        raise ValueError("target_feature_index is outside the timeline feature count")
    if not np.isfinite(clean[:, target_feature_index]).all():
        raise ValueError(
            "recursive process drift requires a finite latent target timeline"
        )
    if len(schedule) != clean.shape[0]:
        raise ValueError("schedule length must equal timeline length")
    old = _parse_old(parameters)
    new, scale_factor, level_offset = _new_parameters(old, parameters)
    allow_unstable = bool(parameters.get("allow_unstable_ar", False))
    if not allow_unstable and (not _is_stable(old.coefficients) or not _is_stable(new.coefficients)):
        raise ValueError("old and new AR coefficients must be stable unless allow_unstable_ar=true")

    runtime = np.array(clean, copy=True)
    target_clean = clean[:, target_feature_index]
    length = clean.shape[0]
    old_candidates = np.full(length, np.nan, dtype=float)
    new_candidates = np.full(length, np.nan, dtype=float)
    innovations = np.zeros(length, dtype=float)
    changed = np.zeros(length, dtype=bool)
    legacy_mode = parameters.get("innovation_mode")
    innovation_source = parameters.get("innovation_source")
    if legacy_mode is not None and innovation_source is not None:
        raise ValueError("configure innovation_source, not both innovation_source and innovation_mode")
    if innovation_source is None:
        innovation_source = {
            "clean_residual": "observed_process_residual",
            "normal": "sampled_gaussian",
        }.get(str(legacy_mode or "clean_residual").lower())
    innovation_source = str(innovation_source).lower()
    if innovation_source not in {"observed_process_residual", "sampled_gaussian"}:
        raise ValueError("innovation_source must be observed_process_residual or sampled_gaussian")
    rng = np.random.default_rng(derive_stable_seed(client_seed, component_id, "process-innovations"))

    order = len(old.coefficients)
    for position, point in enumerate(schedule):
        if position < order:
            if point.regime_active:
                raise ValueError("process onset occurs before sufficient AR history is available")
            # The first `order` values are common realized source history.
            continue
        realized_history = runtime[:position, target_feature_index]
        if innovation_source == "observed_process_residual":
            clean_history = target_clean[:position]
            epsilon = target_clean[position] - _conditional_mean(
                old, clean_history, point.absolute_index
            )
        else:
            epsilon = float(rng.normal(0.0, old.innovation_sigma))
        innovations[position] = epsilon
        old_candidate = _conditional_mean(old, realized_history, point.absolute_index) + epsilon
        new_structural = _conditional_mean(new, realized_history, point.absolute_index)
        new_candidate = level_offset + scale_factor * new_structural + epsilon * (
            new.innovation_sigma / old.innovation_sigma
            if old.innovation_sigma > 0
            else float(dict(parameters.get("changes", {}) or {}).get("innovation_scale", 1.0))
        )
        old_candidates[position] = old_candidate
        new_candidates[position] = new_candidate

        if not point.regime_active:
            # Preserve the actual clean source timeline literally. The old
            # transition is used only to extract/record its observed residual.
            value = target_clean[position]
        elif point.active_concept_id == "blend":
            value = old_candidate + point.drift_weight * (new_candidate - old_candidate)
        elif point.new_regime_selected:
            value = new_candidate
        else:
            value = old_candidate
        runtime[position, target_feature_index] = value
        changed[position] = not np.isclose(
            value, target_clean[position], rtol=0.0, atol=1e-12
        )

    return ProcessResult(runtime, changed, old_candidates, new_candidates, innovations, innovation_source)
