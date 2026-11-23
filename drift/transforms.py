"""Feature-specific transformations on chronological physical-unit timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import FeatureTransform, SchedulePoint, TransformType
from .schedule import derive_stable_seed


@dataclass(frozen=True)
class MeasurementResult:
    values: np.ndarray
    changed: np.ndarray
    affected_features: tuple[int, ...]


def as_2d_timeline(values: Sequence[float] | np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    was_1d = array.ndim == 1
    if was_1d:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"timeline must have shape [T] or [T,F], got {array.shape}")
    return np.array(array, copy=True), was_1d


def _nonlinear(values: np.ndarray, parameters: dict) -> np.ndarray:
    kind = str(parameters.get("kind", "")).lower()
    if kind == "square":
        return np.square(values)
    if kind == "log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    if kind == "clip":
        lower = float(parameters.get("lower", -np.inf))
        upper = float(parameters.get("upper", np.inf))
        if not lower < upper:
            raise ValueError("nonlinear clip requires lower < upper")
        return np.clip(values, lower, upper)
    raise ValueError(f"unsupported nonlinear transform kind={kind!r}")


def apply_measurement_transforms(
    values: Sequence[float] | np.ndarray,
    schedule: Sequence[SchedulePoint],
    transforms: Sequence[FeatureTransform],
    *,
    client_seed: int,
    component_id: str,
) -> MeasurementResult:
    """Apply each final transform once per timestamp, interpolating from identity."""
    base, _ = as_2d_timeline(values)
    if len(schedule) != base.shape[0]:
        raise ValueError("schedule length must equal timeline length")
    if not transforms:
        raise ValueError("measurement component requires at least one transform")
    feature_count = base.shape[1]
    for transform in transforms:
        if transform.feature_index >= feature_count:
            raise ValueError(
                f"feature index {transform.feature_index} is outside timeline feature count {feature_count}"
            )

    final = np.array(base, copy=True)
    affected = tuple(sorted({item.feature_index for item in transforms}))

    for transform_number, transform in enumerate(transforms):
        feature = transform.feature_index
        original = np.array(final[:, feature], copy=True)
        transformed = np.array(original, copy=True)
        finite = np.isfinite(original)
        parameters = dict(transform.parameters)
        if transform.transform_type is TransformType.SCALE:
            transformed[finite] = original[finite] * float(parameters["factor"])
        elif transform.transform_type is TransformType.OFFSET:
            transformed[finite] = original[finite] + float(parameters["delta"])
        elif transform.transform_type is TransformType.NOISE:
            rng = np.random.default_rng(
                derive_stable_seed(client_seed, component_id, feature, transform_number, "noise")
            )
            noise = rng.normal(0.0, float(parameters["sigma"]), size=base.shape[0])
            transformed[finite] = original[finite] + noise[finite]
        else:
            transformed[finite] = _nonlinear(original[finite], parameters)
        final[:, feature] = transformed

    runtime = np.array(base, copy=True)
    changed = np.zeros(base.shape[0], dtype=bool)
    for position, point in enumerate(schedule):
        if point.active_concept_id == "blend":
            applied_weight = point.drift_weight
        else:
            applied_weight = 1.0 if point.new_regime_selected else 0.0
        for feature in affected:
            original = base[position, feature]
            if not np.isfinite(original):
                continue
            updated = original + applied_weight * (final[position, feature] - original)
            runtime[position, feature] = updated
            if not np.isclose(updated, original, rtol=0.0, atol=1e-12):
                changed[position] = True

    return MeasurementResult(runtime, changed, affected)
