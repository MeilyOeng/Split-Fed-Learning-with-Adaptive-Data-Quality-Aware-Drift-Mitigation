"""Pure drift mathematics indexed by absolute observation coordinates."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .contracts import DriftConfig, DriftType, TransitionFunction


def transition_progress(observation_index: int, config: DriftConfig) -> float:
    """Return the transition weight for one absolute observation index."""
    if observation_index < 0:
        raise ValueError("observation_index must be >= 0")
    dtype = config.drift_type
    if dtype is DriftType.NONE:
        return 0.0
    start = int(config.start_index)  # validated for non-NONE configs
    if dtype is DriftType.SUDDEN:
        return 1.0 if observation_index >= start else 0.0
    if dtype is DriftType.RECURRING:
        if observation_index < start:
            return 0.0
        phase_index = (observation_index - start) % int(config.period)
        active_length = (int(config.period) + 1) // 2
        return 1.0 if phase_index < active_length else 0.0
    if observation_index <= start:
        return 0.0
    end = int(config.end_index)
    if observation_index >= end:
        return 1.0
    progress = (observation_index - start) / (end - start)
    if config.transition is TransitionFunction.LINEAR:
        return float(progress)
    if config.transition is TransitionFunction.SIGMOID:
        k = config.sigmoid_k
        raw = 1.0 / (1.0 + math.exp(-k * (progress - 0.5)))
        low = 1.0 / (1.0 + math.exp(k * 0.5))
        high = 1.0 / (1.0 + math.exp(-k * 0.5))
        return float((raw - low) / (high - low))
    alpha = config.exp_alpha
    if abs(alpha) < 1e-12:
        return float(progress)
    return float((math.exp(alpha * progress) - 1.0) / (math.exp(alpha) - 1.0))


def active_interval_id(observation_index: int, config: DriftConfig) -> Optional[int]:
    if config.drift_type is not DriftType.RECURRING or observation_index < int(config.start_index):
        return None
    phase_index = (observation_index - int(config.start_index)) % int(config.period)
    if phase_index >= (int(config.period) + 1) // 2:
        return None
    return (observation_index - int(config.start_index)) // int(config.period)


def transform_value(
    value: float,
    *,
    weight: float,
    config: DriftConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, bool]:
    """Apply the configured scalar transform and report whether the value changed."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    original = float(value)
    if weight == 0.0 or not config.has_input_effect:
        return original, False
    transformed = original * (1.0 + weight * config.magnitude)
    transformed += weight * config.offset_delta
    if config.noise_sigma > 0.0:
        if rng is None:
            raise ValueError("a local RNG is required when noise_sigma > 0")
        transformed += float(rng.normal(0.0, weight * config.noise_sigma))
    return transformed, not np.isclose(transformed, original, rtol=0.0, atol=1e-12)
