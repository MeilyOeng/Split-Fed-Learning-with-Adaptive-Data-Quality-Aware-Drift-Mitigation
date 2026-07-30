"""Compatibility wrapper for the former drift_injector module."""

from __future__ import annotations

import warnings

from drift.contracts import DriftConfig, DriftSource, DriftType, TransitionFunction
from drift.injector import inject_series


def inject_drift(series, drift_type="sudden", start_ratio=0.2, magnitude=0.5, end_ratio=0.5,
                 period=50, transition_function="linear", seed=42):
    """Return the legacy (runtime_values, start_index) tuple."""
    warnings.warn("drift_injector.inject_drift is deprecated; use drift.injector.inject_series", DeprecationWarning)
    dtype_text = str(drift_type).lower()
    if dtype_text == "seasonal":
        warnings.warn("'seasonal' is a deprecated alias for square-wave 'recurring' drift", DeprecationWarning)
        dtype_text = "recurring"
    dtype = DriftType.parse(dtype_text)
    transition = TransitionFunction.parse(transition_function)
    size = len(series)
    start = int(start_ratio * size)
    end = int(end_ratio * size) if dtype in (DriftType.GRADUAL, DriftType.INCREMENTAL) else None
    config = DriftConfig(
        run_id="legacy", configuration_id="legacy", cluster_id="legacy", client_id="test",
        drift_type=dtype, drift_source=DriftSource.INPUT, start_index=start, end_index=end,
        magnitude=magnitude, period=period if dtype is DriftType.RECURRING else None,
        transition=transition, random_seed=seed,
    )
    return inject_series(series, config).runtime_values, start


__all__ = ["inject_drift", "inject_series"]
