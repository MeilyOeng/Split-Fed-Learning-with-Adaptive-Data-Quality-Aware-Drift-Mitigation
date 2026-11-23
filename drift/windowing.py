"""Forecast window construction and exact clean-training normalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

from .transforms import as_2d_timeline


@dataclass(frozen=True)
class ForecastWindows:
    X: np.ndarray
    y: np.ndarray
    window_start_indices: np.ndarray
    window_end_indices: np.ndarray
    target_indices: np.ndarray


@dataclass(frozen=True)
class CleanTrainingScalers:
    method: str
    x_location: np.ndarray
    x_scale: np.ndarray
    y_location: float
    y_scale: float
    target_scaling_strategy: str
    x_observation_indices: tuple[int, ...]
    y_target_indices: tuple[int, ...]
    shared_target_indices: tuple[int, ...]
    x_union_hash: str
    y_union_hash: str
    shared_target_union_hash: str | None


def _target_vector(values: Sequence[float] | np.ndarray, feature_index: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if feature_index != 0:
            raise ValueError("one-dimensional target timeline only supports target_feature_index=0")
        return array
    if array.ndim != 2 or feature_index >= array.shape[1]:
        raise ValueError("target timeline feature index is invalid")
    return array[:, feature_index]


def build_forecasting_windows(
    observed_timeline: Sequence[float] | np.ndarray,
    target_timeline: Sequence[float] | np.ndarray,
    *,
    window_size: int,
    forecast_horizon: int,
    target_feature_index: int,
    window_start_indices: Sequence[int] | None = None,
) -> ForecastWindows:
    observed, _ = as_2d_timeline(observed_timeline)
    targets = _target_vector(target_timeline, target_feature_index)
    if observed.shape[0] != targets.size:
        raise ValueError("observed and target timelines must have equal lengths")
    if window_size <= 0 or forecast_horizon <= 0:
        raise ValueError("window_size and forecast_horizon must be > 0")
    total = observed.shape[0] - window_size - forecast_horizon + 1
    if total <= 0:
        raise ValueError("timeline is too short for the requested window geometry")
    starts = np.arange(total, dtype=int) if window_start_indices is None else np.asarray(window_start_indices, dtype=int)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("window_start_indices must be a non-empty vector")
    ends = starts + window_size - 1
    target_indices = ends + forecast_horizon
    if starts.min() < 0 or target_indices.max() >= observed.shape[0]:
        raise ValueError("window coordinates fall outside the timeline")
    X = np.stack([observed[start : start + window_size] for start in starts])
    y = targets[target_indices]
    return ForecastWindows(X, y, starts, ends, target_indices)


def assert_horizon_one_consistency(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_feature_index: int,
    atol: float = 1e-12,
) -> None:
    windows = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float).reshape(-1)
    if windows.ndim != 3 or windows.shape[0] != targets.size:
        raise ValueError("X must have shape [N,L,F] and align with y")
    if target_feature_index < 0 or target_feature_index >= windows.shape[2]:
        raise ValueError("target_feature_index is outside X")
    if targets.size > 1 and not np.allclose(
        targets[:-1],
        windows[1:, -1, target_feature_index],
        rtol=0.0,
        atol=atol,
        equal_nan=True,
    ):
        raise AssertionError("horizon-one chronological consistency failed: y_t != X_(t+1)[-1]")


def _array_hash(indices: np.ndarray, values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(indices, dtype="<i8").tobytes())
    digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def fit_clean_training_scalers(
    clean_input_timeline: Sequence[float] | np.ndarray,
    clean_target_timeline: Sequence[float] | np.ndarray,
    *,
    training_window_starts: Sequence[int],
    window_size: int,
    forecast_horizon: int,
    target_feature_index: int,
    method: str = "zscore",
    eps: float = 1e-8,
    share_target_scaler: bool = False,
) -> CleanTrainingScalers:
    """Fit training-only scalers, optionally sharing target-feature and y parameters."""
    inputs, _ = as_2d_timeline(clean_input_timeline)
    targets = _target_vector(clean_target_timeline, target_feature_index)
    if target_feature_index < 0 or target_feature_index >= inputs.shape[1]:
        raise ValueError("target_feature_index is outside the clean input timeline")
    starts = np.asarray(training_window_starts, dtype=int)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("training_window_starts must be non-empty")
    x_indices = np.unique(
        np.concatenate([np.arange(start, start + window_size, dtype=int) for start in starts])
    )
    y_indices = np.unique(starts + window_size - 1 + forecast_horizon)
    if x_indices.min() < 0 or y_indices.max() >= inputs.shape[0] or inputs.shape[0] != targets.size:
        raise ValueError("clean training coordinates fall outside the timelines")
    x_values = inputs[x_indices]
    y_values = targets[y_indices]
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("clean scaler fitting values must be finite")
    normalized_method = str(method).lower()
    if normalized_method == "minmax":
        x_location = x_values.min(axis=0)
        x_scale = x_values.max(axis=0) - x_location
        y_location = float(y_values.min())
        y_scale = float(y_values.max() - y_location)
    elif normalized_method == "zscore":
        x_location = x_values.mean(axis=0)
        x_scale = x_values.std(axis=0)
        y_location = float(y_values.mean())
        y_scale = float(y_values.std())
    else:
        raise ValueError("normalization method must be zscore or minmax")

    shared_target_indices = np.asarray([], dtype=int)
    shared_target_union_hash = None
    if share_target_scaler:
        shared_target_indices = np.union1d(x_indices, y_indices)
        overlap = np.intersect1d(x_indices, y_indices)
        if not np.allclose(
            inputs[overlap, target_feature_index],
            targets[overlap],
            rtol=0.0,
            atol=eps,
        ):
            raise ValueError(
                "a shared target scaler requires X and y to represent the same clean "
                "physical variable at overlapping training coordinates"
            )
        shared_target_values = inputs[shared_target_indices, target_feature_index].copy()
        y_only = np.setdiff1d(y_indices, x_indices)
        if y_only.size:
            positions = np.searchsorted(shared_target_indices, y_only)
            shared_target_values[positions] = targets[y_only]
        if normalized_method == "minmax":
            shared_location = float(shared_target_values.min())
            shared_scale = float(shared_target_values.max() - shared_location)
        else:
            shared_location = float(shared_target_values.mean())
            shared_scale = float(shared_target_values.std())
        if shared_scale < eps:
            shared_scale = 1.0
        x_location = np.asarray(x_location, dtype=float)
        x_scale = np.asarray(x_scale, dtype=float)
        x_location[target_feature_index] = shared_location
        x_scale[target_feature_index] = shared_scale
        y_location = shared_location
        y_scale = shared_scale
        shared_target_union_hash = _array_hash(shared_target_indices, shared_target_values)
    x_scale = np.where(x_scale < eps, 1.0, x_scale)
    if y_scale < eps:
        y_scale = 1.0
    return CleanTrainingScalers(
        method=normalized_method,
        x_location=np.asarray(x_location, dtype=float),
        x_scale=np.asarray(x_scale, dtype=float),
        y_location=y_location,
        y_scale=y_scale,
        target_scaling_strategy="shared" if share_target_scaler else "separate",
        x_observation_indices=tuple(int(item) for item in x_indices),
        y_target_indices=tuple(int(item) for item in y_indices),
        shared_target_indices=tuple(int(item) for item in shared_target_indices),
        x_union_hash=_array_hash(x_indices, x_values),
        y_union_hash=_array_hash(y_indices, y_values),
        shared_target_union_hash=shared_target_union_hash,
    )


def normalize_windows(
    windows: ForecastWindows,
    scalers: CleanTrainingScalers,
) -> ForecastWindows:
    X = (np.asarray(windows.X, dtype=float) - scalers.x_location) / scalers.x_scale
    y = (np.asarray(windows.y, dtype=float) - scalers.y_location) / scalers.y_scale
    return ForecastWindows(
        X,
        y,
        windows.window_start_indices,
        windows.window_end_indices,
        windows.target_indices,
    )
