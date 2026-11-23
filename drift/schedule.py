"""Absolute-coordinate schedule and client-selection resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .configuration import sha256_canonical
from .contracts import (
    ClientSelectionRecord,
    DriftExperimentConfig,
    DriftType,
    SchedulePoint,
    TransitionFunction,
)


def derive_stable_seed(base_seed: int, *parts: object) -> int:
    material = "\0".join([str(int(base_seed)), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


def _timestamp_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _resolve_timestamp_coordinate(
    requested: Optional[str],
    absolute_indices: Sequence[int],
    timestamps: Sequence[object],
    name: str,
) -> Optional[int]:
    if requested is None:
        return None
    target = pd.Timestamp(requested)
    matches = [
        int(index)
        for index, timestamp in zip(absolute_indices, timestamps)
        if not pd.isna(timestamp) and pd.Timestamp(timestamp) >= target
    ]
    if not matches:
        raise ValueError(f"{name}={requested!r} lies after the available timeline")
    return matches[0]


def resolve_temporal_bounds(
    config: DriftExperimentConfig,
    absolute_indices: Sequence[int],
    timestamps: Sequence[object],
) -> tuple[Optional[int], Optional[int]]:
    if len(absolute_indices) != len(timestamps):
        raise ValueError("absolute_indices and timestamps must have identical lengths")
    if not absolute_indices:
        raise ValueError("timeline coordinates must be non-empty")
    if len(set(int(item) for item in absolute_indices)) != len(absolute_indices):
        raise ValueError("absolute indices must be unique")
    if any(int(right) <= int(left) for left, right in zip(absolute_indices, absolute_indices[1:])):
        raise ValueError("absolute indices must be strictly increasing")
    coordinate_set = {int(item) for item in absolute_indices}
    temporal = config.temporal
    start = temporal.start_index
    if start is None:
        start = _resolve_timestamp_coordinate(
            temporal.start_timestamp, absolute_indices, timestamps, "start_timestamp"
        )
    end = temporal.end_index
    if end is None:
        end = _resolve_timestamp_coordinate(
            temporal.end_timestamp, absolute_indices, timestamps, "end_timestamp"
        )
    if temporal.drift_type is not DriftType.NONE:
        if start is None or start < int(absolute_indices[0]) or start > int(absolute_indices[-1]):
            raise ValueError("resolved drift start must lie within the raw timeline")
        if int(start) not in coordinate_set:
            raise ValueError("resolved drift start must match an absolute timeline index")
    if temporal.drift_type in (DriftType.INCREMENTAL, DriftType.GRADUAL):
        if end is None or end <= int(start):
            raise ValueError("resolved transition end must be greater than its start")
        if end > int(absolute_indices[-1]):
            raise ValueError("resolved transition end lies after the raw timeline")
        if int(end) not in coordinate_set:
            raise ValueError("resolved transition end must match an absolute timeline index")
    return start, end


def _transition_weight(index: int, start: int, end: int, config: DriftExperimentConfig) -> float:
    if index <= start:
        return 0.0
    if index >= end:
        return 1.0
    progress = (index - start) / (end - start)
    transition = config.temporal.transition
    if transition is TransitionFunction.LINEAR:
        return float(progress)
    if transition is TransitionFunction.SIGMOID:
        k = config.temporal.sigmoid_k
        raw = 1.0 / (1.0 + math.exp(-k * (progress - 0.5)))
        low = 1.0 / (1.0 + math.exp(k * 0.5))
        high = 1.0 / (1.0 + math.exp(-k * 0.5))
        return float((raw - low) / (high - low))
    alpha = config.temporal.exp_alpha
    if abs(alpha) < 1e-12:
        return float(progress)
    return float((math.exp(alpha * progress) - 1.0) / (math.exp(alpha) - 1.0))


def build_drift_schedule(
    config: DriftExperimentConfig,
    *,
    client_id: str,
    absolute_indices: Sequence[int],
    timestamps: Sequence[object],
    client_seed: Optional[int] = None,
) -> tuple[SchedulePoint, ...]:
    """Materialize timing and every gradual Bernoulli draw exactly once."""
    start, end = resolve_temporal_bounds(config, absolute_indices, timestamps)
    dtype = config.temporal.drift_type
    seed = client_seed if client_seed is not None else derive_stable_seed(
        config.experiment_seed, config.experiment_id, client_id
    )
    gradual_rng = np.random.default_rng(derive_stable_seed(seed, "gradual-mask"))
    points: list[SchedulePoint] = []

    for raw_index, timestamp in zip(absolute_indices, timestamps):
        index = int(raw_index)
        weight = 0.0
        regime_active = False
        change_event = False
        configured = "old"
        active = "old"
        selected = False

        if dtype is DriftType.SUDDEN:
            regime_active = index >= int(start)
            weight = 1.0 if regime_active else 0.0
            configured = "new" if regime_active else "old"
            active = configured
            selected = regime_active
            change_event = index == int(start)
        elif dtype in (DriftType.INCREMENTAL, DriftType.GRADUAL):
            weight = _transition_weight(index, int(start), int(end), config)
            regime_active = index >= int(start)
            change_event = index == int(start)
            configured = "old" if weight == 0.0 else ("new" if weight == 1.0 else "transition")
            if dtype is DriftType.GRADUAL:
                selected = bool(weight >= 1.0 or (weight > 0.0 and gradual_rng.random() < weight))
                active = "new" if selected else "old"
            else:
                selected = weight >= 1.0
                active = "new" if weight >= 1.0 else ("old" if weight <= 0.0 else "blend")
        elif dtype is DriftType.RECURRING:
            if index >= int(start):
                regime_active = True
                phase = (index - int(start)) % int(config.temporal.cycle_length)
                selected = phase < int(config.temporal.new_duration)
                weight = 1.0 if selected else 0.0
                configured = "recurring_new" if selected else "recurring_old"
                active = "new" if selected else "old"
                change_event = phase in {0, int(config.temporal.new_duration)}
        points.append(
            SchedulePoint(
                absolute_index=index,
                timestamp=_timestamp_text(timestamp),
                configured_regime=configured,
                old_concept_id="old",
                new_concept_id="new",
                active_concept_id=active,
                drift_weight=float(weight),
                regime_active=regime_active,
                change_point_event=change_event,
                new_regime_selected=selected,
            )
        )
    return tuple(points)


@dataclass(frozen=True)
class ResolvedExperimentSchedule:
    config: DriftExperimentConfig
    schedules: Mapping[tuple[str, str], tuple[SchedulePoint, ...]]
    selections: tuple[ClientSelectionRecord, ...]
    resolved_schedule_hash: str

    def for_client(self, cluster_id: str, client_id: str) -> tuple[SchedulePoint, ...]:
        return self.schedules.get((str(cluster_id), str(client_id)), ())

    def is_affected(self, cluster_id: str, client_id: str) -> bool:
        return bool(self.for_client(cluster_id, client_id))


def resolve_experiment_schedule(
    config: DriftExperimentConfig,
    client_timelines: Mapping[tuple[str, str], tuple[Sequence[int], Sequence[object]]],
) -> ResolvedExperimentSchedule:
    """Resolve affected clients, client seeds, coordinates, and masks for the experiment."""
    candidates = sorted((str(cluster), str(client)) for cluster, client in client_timelines)
    if config.enabled and not candidates:
        raise ValueError("enabled drift requires at least one available client timeline")
    requested = set(config.client_ids)
    if not config.enabled:
        affected: set[tuple[str, str]] = set()
        method = "disabled"
    elif requested:
        affected = {
            item for item in candidates if item[1] in requested or f"{item[0]}:{item[1]}" in requested
        }
        unresolved = requested - {client for _, client in affected} - {
            f"{cluster}:{client}" for cluster, client in affected
        }
        if unresolved:
            raise ValueError(f"configured client IDs were not found: {sorted(unresolved)}")
        method = "explicit_ids"
    else:
        fraction = 0.0 if config.client_fraction is None else config.client_fraction
        count = min(len(candidates), max(1, int(round(len(candidates) * fraction))))
        rng = np.random.default_rng(
            derive_stable_seed(config.experiment_seed, config.experiment_id, "client-selection")
        )
        chosen = rng.choice(len(candidates), size=count, replace=False).tolist() if count else []
        affected = {candidates[index] for index in chosen}
        method = "fraction"

    schedules: dict[tuple[str, str], tuple[SchedulePoint, ...]] = {}
    seed_by_client: dict[tuple[str, str], int] = {}
    for cluster_id, client_id in candidates:
        seed = derive_stable_seed(
            config.experiment_seed, config.experiment_id, cluster_id, client_id
        )
        seed_by_client[(cluster_id, client_id)] = seed
        if (cluster_id, client_id) not in affected:
            continue
        indices, timestamps = client_timelines[(cluster_id, client_id)]
        schedules[(cluster_id, client_id)] = build_drift_schedule(
            config,
            client_id=client_id,
            absolute_indices=indices,
            timestamps=timestamps,
            client_seed=seed,
        )

    schedule_payload = {
        "config_hash": config.config_hash,
        "clients": [
            {
                "cluster_id": cluster,
                "client_id": client,
                "affected": (cluster, client) in affected,
                "client_seed": seed_by_client[(cluster, client)],
            }
            for cluster, client in candidates
        ],
        "schedule": [
            {
                "cluster_id": cluster,
                "client_id": client,
                **asdict(point),
            }
            for (cluster, client), points in sorted(schedules.items())
            for point in points
        ],
    }
    resolved_hash = sha256_canonical(schedule_payload)
    selections = tuple(
        ClientSelectionRecord(
            experiment_id=config.experiment_id,
            cluster_id=cluster,
            client_id=client,
            affected=(cluster, client) in affected,
            selection_method=method,
            experiment_seed=config.experiment_seed,
            client_seed=seed_by_client[(cluster, client)],
            config_hash=config.config_hash,
            resolved_schedule_hash=resolved_hash,
        )
        for cluster, client in candidates
    )
    return ResolvedExperimentSchedule(config, schedules, selections, resolved_hash)
