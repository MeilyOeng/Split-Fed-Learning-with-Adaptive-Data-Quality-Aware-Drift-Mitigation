"""Offline perfect-oracle labels for controlled diagnostic experiments.

This module intentionally has no runtime-stream integration.  Oracle truth is
derived only from a resolved condition manifest and exact window coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


@dataclass(frozen=True)
class OracleLabel:
    experiment_id: str
    condition_id: str
    cluster_id: str
    client_id: str
    mode: str
    window_start_index: int
    window_end_index: int
    detector_endpoint_index: int
    endpoint_timestamp: str
    affected_client: bool
    true_drift_active: bool
    oracle_alarm: bool
    drift_onset_index: int | None
    oracle_source: str = "condition_manifest"
    oracle_role: str = "primary_ground_truth"
    controls_runtime: bool = False


@dataclass(frozen=True)
class OracleOnsetEvent:
    experiment_id: str
    condition_id: str
    cluster_id: str
    client_id: str
    mode: str
    event_index: int
    observation_index: int
    timestamp: str
    event_type: str = "sudden_drift_onset"
    detector_name: str = "perfect_oracle"
    detection_delay: int = 0
    controls_runtime: bool = False


def _normalize_client_map(raw: object) -> dict[str, str]:
    if isinstance(raw, Mapping):
        return {str(client_id): str(cluster_id) for client_id, cluster_id in raw.items()}
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("condition manifest must provide clients as a mapping or sequence")
    result: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("each condition-manifest client row must be a mapping")
        client_id = str(row.get("client_id", "")).strip()
        cluster_id = str(row.get("cluster_id", "")).strip()
        if not client_id or not cluster_id:
            raise ValueError("condition-manifest clients require client_id and cluster_id")
        result[client_id] = cluster_id
    return result


def build_oracle_tables(
    condition_manifest: Mapping[str, object],
    timestamps: Sequence[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact offline labels and onset events from one condition manifest."""
    experiment_id = str(condition_manifest["experiment_id"])
    condition_id = str(condition_manifest["condition_id"])
    window_size = int(condition_manifest["window_size"])
    first_start = int(condition_manifest["test_window_start_first"])
    last_start = int(condition_manifest["test_window_start_last"])
    drift_enabled = bool(condition_manifest["true_drift"])
    onset = int(condition_manifest["drift_onset_index"])
    clients = _normalize_client_map(condition_manifest["clients"])
    affected = {str(value) for value in condition_manifest.get("affected_clients", [])}

    if window_size <= 0 or first_start < 0 or last_start < first_start:
        raise ValueError("invalid oracle window coordinates")
    last_endpoint = last_start + window_size - 1
    if last_endpoint >= len(timestamps):
        raise ValueError(
            f"oracle endpoint {last_endpoint} exceeds timestamp range 0..{len(timestamps) - 1}"
        )
    unknown_affected = sorted(affected - set(clients))
    if unknown_affected:
        raise ValueError(f"affected clients are absent from condition clients: {unknown_affected}")

    labels: list[OracleLabel] = []
    events: list[OracleOnsetEvent] = []
    for client_id, cluster_id in sorted(
        clients.items(), key=lambda item: (int(item[1]) if item[1].isdigit() else item[1], item[0])
    ):
        client_affected = drift_enabled and client_id in affected
        event_index = 0
        for start in range(first_start, last_start + 1):
            endpoint = start + window_size - 1
            active = client_affected and endpoint >= onset
            alarm = client_affected and endpoint == onset
            timestamp = pd.Timestamp(timestamps[endpoint]).isoformat()
            labels.append(
                OracleLabel(
                    experiment_id=experiment_id,
                    condition_id=condition_id,
                    cluster_id=cluster_id,
                    client_id=client_id,
                    mode="test",
                    window_start_index=start,
                    window_end_index=endpoint,
                    detector_endpoint_index=endpoint,
                    endpoint_timestamp=timestamp,
                    affected_client=client_id in affected,
                    true_drift_active=active,
                    oracle_alarm=alarm,
                    drift_onset_index=onset if client_affected else None,
                )
            )
            if alarm:
                events.append(
                    OracleOnsetEvent(
                        experiment_id=experiment_id,
                        condition_id=condition_id,
                        cluster_id=cluster_id,
                        client_id=client_id,
                        mode="test",
                        event_index=event_index,
                        observation_index=endpoint,
                        timestamp=timestamp,
                    )
                )
                event_index += 1

    labels_df = pd.DataFrame(asdict(row) for row in labels)
    events_df = pd.DataFrame(asdict(row) for row in events)
    if events_df.empty:
        events_df = pd.DataFrame(columns=[field for field in OracleOnsetEvent.__dataclass_fields__])

    expected_events = len(affected) if drift_enabled else 0
    if len(events_df) != expected_events:
        raise AssertionError(
            f"oracle event count mismatch for {condition_id}: expected {expected_events}, got {len(events_df)}"
        )
    if not events_df.empty and not (events_df["observation_index"] == onset).all():
        raise AssertionError("oracle onset events must occur exactly at the configured onset")
    return labels_df, events_df


def write_oracle_tables(
    condition_manifest: Mapping[str, object],
    timestamps: Sequence[object],
    *,
    labels_path: str | Path,
    events_path: str | Path,
) -> tuple[Path, Path]:
    labels, events = build_oracle_tables(condition_manifest, timestamps)
    labels_target = Path(labels_path)
    events_target = Path(events_path)
    labels_target.parent.mkdir(parents=True, exist_ok=True)
    events_target.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_target, index=False)
    events.to_csv(events_target, index=False)
    return labels_target, events_target


def validate_oracle_tables(
    labels: pd.DataFrame,
    events: pd.DataFrame,
    condition_manifest: Mapping[str, object],
) -> None:
    """Validate coverage, exact-once events, and non-controlling oracle metadata."""
    clients = _normalize_client_map(condition_manifest["clients"])
    first_start = int(condition_manifest["test_window_start_first"])
    last_start = int(condition_manifest["test_window_start_last"])
    expected_windows = last_start - first_start + 1
    expected_rows = len(clients) * expected_windows
    if len(labels) != expected_rows:
        raise AssertionError(f"oracle labels expected {expected_rows} rows, got {len(labels)}")
    keys = ["condition_id", "client_id", "mode", "window_start_index", "window_end_index"]
    if labels.duplicated(keys).any():
        raise AssertionError("oracle labels contain duplicate window keys")
    if _boolean_series(labels["controls_runtime"]).any():
        raise AssertionError("offline oracle must never control runtime")
    drift_enabled = bool(condition_manifest["true_drift"])
    affected = {str(value) for value in condition_manifest.get("affected_clients", [])}
    expected_events = len(affected) if drift_enabled else 0
    if len(events) != expected_events:
        raise AssertionError(f"oracle events expected {expected_events} rows, got {len(events)}")
    if not events.empty:
        if events["client_id"].astype(str).nunique() != expected_events:
            raise AssertionError("oracle events are not exactly once per affected client")
        if set(events["client_id"].astype(str)) != affected:
            raise AssertionError("oracle event clients differ from the affected cohort")
        if not (events["observation_index"].astype(int) == int(condition_manifest["drift_onset_index"])).all():
            raise AssertionError("oracle events do not match the configured onset")
        if _boolean_series(events["controls_runtime"]).any():
            raise AssertionError("offline oracle events must never control runtime")


def oracle_event_records(events: pd.DataFrame) -> Iterable[OracleOnsetEvent]:
    """Yield typed event rows, primarily for tests and downstream validation."""
    for row in events.to_dict(orient="records"):
        yield OracleOnsetEvent(**row)
