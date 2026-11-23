"""Deterministic whole-series drift injection."""

from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Optional, Sequence

import numpy as np

from .contracts import DriftConfig, DriftSource, DriftType, InjectionRecord, InjectionResult, validate_series
from .framework import active_interval_id, transform_value, transition_progress


def derive_client_seed(base_seed: int, *, run_id: str, cluster_id: str, client_id: str) -> int:
    """Derive a stable local NumPy seed without touching process-global RNG state."""
    material = f"{int(base_seed)}\0{run_id}\0{cluster_id}\0{client_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def _timestamp_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def inject_series(
    base_values: Sequence[float] | np.ndarray,
    config: DriftConfig,
    *,
    timestamps: Optional[Sequence[object]] = None,
) -> InjectionResult:
    """Return an immutable base copy, a separate runtime series, and one record per observation."""
    base = validate_series(base_values).copy()
    base.setflags(write=False)
    runtime = np.array(base, copy=True)
    if timestamps is not None and len(timestamps) != len(base):
        raise ValueError(f"timestamps length {len(timestamps)} does not match series length {len(base)}")
    rng = np.random.default_rng(config.random_seed)
    records: list[InjectionRecord] = []

    for index, original in enumerate(base):
        weight = transition_progress(index, config)
        regime_active = weight > 0.0
        applied_weight = weight
        if config.drift_type is DriftType.GRADUAL and 0.0 < weight < 1.0:
            applied_weight = 1.0 if bool(rng.random() < weight) else 0.0

        transformed = float(original)
        changed = False
        if config.drift_source is DriftSource.MEASUREMENT:
            transformed, changed = transform_value(
                transformed,
                weight=applied_weight,
                config=config,
                rng=rng,
            )
        if config.drift_source is DriftSource.TARGET_MAPPING and applied_weight > 0.0 and config.has_target_effect:
            transformed += applied_weight * config.label_delta
            changed = not np.isclose(transformed, float(original), rtol=0.0, atol=1e-12)

        runtime[index] = transformed
        records.append(
            InjectionRecord(
                run_id=config.run_id,
                configuration_id=config.configuration_id,
                cluster_id=config.cluster_id,
                client_id=config.client_id,
                mode=config.mode,
                observation_index=index,
                timestamp=_timestamp_text(timestamps[index]) if timestamps is not None else None,
                base_value=float(original),
                runtime_value=float(transformed),
                drift_type=config.drift_type.value,
                drift_source=config.drift_source.value,
                configured_start_index=config.start_index,
                configured_end_index=config.end_index,
                drift_weight=float(weight),
                regime_active=regime_active,
                transform_applied=bool(changed),
                affected_sample_count=int(changed),
                magnitude=config.magnitude,
                offset_delta=config.offset_delta,
                noise_sigma=config.noise_sigma,
                label_delta=config.label_delta,
                period=config.period,
                transition_function=config.transition.value,
                random_seed=config.random_seed,
                active_interval_id=active_interval_id(index, config),
                scope=config.scope,
                requested_start_ratio=config.requested_start_ratio,
                requested_end_ratio=config.requested_end_ratio,
                segment_start_index=config.segment_start_index,
                segment_end_index_exclusive=config.segment_end_index_exclusive,
            )
        )

    return InjectionResult(base_values=base, runtime_values=runtime, records=tuple(records))
