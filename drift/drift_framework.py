"""Compatibility wrapper for the former drift_framework module."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from drift.contracts import DriftConfig, DriftSource, DriftType, TransitionFunction
from drift.framework import active_interval_id, transform_value, transition_progress

compute_drift_weight = transition_progress


def apply_drift(X, y, mode: str, drift_cfg: Optional[DriftConfig], *, observation_index: Optional[int] = None):
    """Compatibility adapter requiring an explicit absolute observation index."""
    if drift_cfg is None or drift_cfg.drift_type is DriftType.NONE or mode != drift_cfg.mode:
        return X, y, False, {"regime_active": False, "transform_applied": False}
    if observation_index is None:
        raise ValueError("observation_index is required; mutable call-counter drift timing was removed")
    warnings.warn("drift_framework.apply_drift is deprecated; use drift.injector.inject_series", DeprecationWarning)
    weight = transition_progress(observation_index, drift_cfg)
    array = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    transformed = np.array(array, copy=True, dtype=float)
    changed = False
    rng = np.random.default_rng(drift_cfg.random_seed + observation_index)
    for index in np.ndindex(transformed.shape):
        transformed[index], item_changed = transform_value(
            transformed[index], weight=weight, config=drift_cfg, rng=rng
        )
        changed = changed or item_changed
    transformed_x = X.new_tensor(transformed) if hasattr(X, "new_tensor") else transformed
    metadata = {
        "regime_active": weight > 0.0,
        "transform_applied": changed,
        "drift_weight": weight,
        "observation_index": observation_index,
    }
    return transformed_x, y, changed, metadata


__all__ = [
    "DriftConfig", "DriftSource", "DriftType", "TransitionFunction",
    "active_interval_id", "apply_drift", "compute_drift_weight", "transform_value", "transition_progress",
]
