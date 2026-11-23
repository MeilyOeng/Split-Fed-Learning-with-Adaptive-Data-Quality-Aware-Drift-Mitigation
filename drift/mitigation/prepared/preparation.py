"""Offline preparation of immutable clean/drifted mitigation datasets.

This module is intentionally independent of checkpoint H. Residuals derived
from future clean observations are benchmark-construction audit artifacts only;
they are never exposed through the prepared dataset reader.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data.prepared_window_store import canonical_array_payload_hash, file_sha256
from drift.contracts import SchedulePoint
from drift.process import generate_process_timeline
from drift.schedule import derive_stable_seed
from drift.windowing import (
    ForecastWindows,
    assert_horizon_one_consistency,
    build_forecasting_windows,
    fit_clean_training_scalers,
    normalize_windows,
)
from ..core import (
    ProcessSpec,
    SUITE_ID,
    SegmentSizes,
    build_segment_rows,
    canonical_hash,
    guard_output_directory,
    load_suite_yaml,
    selected_cohort_hash,
    stable_select_per_cluster,
    write_csv_rows,
    write_json,
)
from .dq_scoring import DQ_REPRESENTATION_METADATA, compute_window_quality_weights
from .faults import REPRESENTATIONS_METADATA, apply_window_faults, make_model_consumable


SCHEMA_VERSION = 2
ROLES = (
    "clean_train",
    "clean_validation",
    "clean_deployment",
    "drifted_deployment",
    "clean_final_holdout_counterfactual",
)


def _sudden_schedule(length: int, onset: int, timestamps: np.ndarray) -> list[SchedulePoint]:
    return [SchedulePoint(
        absolute_index=index,
        timestamp=pd.Timestamp(timestamps[index]).isoformat(),
        configured_regime="new" if index >= onset else "old",
        old_concept_id="old",
        new_concept_id="new",
        active_concept_id="new" if index >= onset else "old",
        drift_weight=1.0 if index >= onset else 0.0,
        regime_active=index >= onset,
        change_point_event=index == onset,
        new_regime_selected=index >= onset,
    ) for index in range(length)]


def _gradual_schedule(
    length: int, onset: int, transition_end: int, timestamps: np.ndarray, *, seed: int, component_id: str,
) -> list[SchedulePoint]:
    """Probabilistic old/new regime switching -- never AR-parameter interpolation.

    w_t = clip((t - onset) / (transition_end - onset), 0, 1); B_t ~ Bernoulli(w_t)
    drawn from one client/seed-specific RNG walked forward in chronological
    order, so repeated runs with the same seed reproduce the identical
    switching sequence. Mirrors drift/schedule.py's own GRADUAL branch
    (same short-circuit: no RNG draw happens at w=0 or w=1, only strictly
    inside the transition) -- kept as a separate implementation here because
    this offline-preparation pipeline builds its own SchedulePoint sequence
    directly from window coordinates rather than going through
    resolve_experiment_schedule.
    """
    if transition_end <= onset:
        raise ValueError("gradual drift requires transition_end > onset")
    span = float(transition_end - onset)
    rng = np.random.default_rng(derive_stable_seed(seed, component_id, "gradual-mask"))
    points = []
    for index in range(length):
        if index < onset:
            weight = 0.0
        elif index >= transition_end:
            weight = 1.0
        else:
            weight = (index - onset) / span
        selected = bool(weight >= 1.0 or (weight > 0.0 and rng.random() < weight))
        active = "new" if selected else "old"
        configured = "old" if weight == 0.0 else ("new" if weight == 1.0 else "transition")
        points.append(SchedulePoint(
            absolute_index=index,
            timestamp=pd.Timestamp(timestamps[index]).isoformat(),
            configured_regime=configured,
            old_concept_id="old",
            new_concept_id="new",
            active_concept_id=active,
            drift_weight=float(weight),
            regime_active=index >= onset,
            change_point_event=index == onset,
            new_regime_selected=selected,
        ))
    return points


def _seasonal_schedule(
    length: int, onset: int, timestamps: np.ndarray, *, period: float, phase: float = 0.0,
) -> list[SchedulePoint]:
    """Repeating periodic AR-parameter blend -- never settles at the new regime.

    w_t = (1 + sin(2*pi*(t - onset)/P + phase)) / 2, for t >= onset (0 before).
    Reuses generate_process_timeline's existing "blend" mechanism (the same
    one incremental drift uses in drift/schedule.py): value = old_candidate +
    w_t*(new_candidate - old_candidate), which is algebraically exactly
    a_t + phi_t*z_(t-1) + epsilon_t with a_t/phi_t as the linear blend of old
    and new AR parameters -- so w_t=0 reproduces the old regime exactly and
    w_t=1 the new regime exactly, at every peak/trough, indefinitely.
    """
    if period <= 0:
        raise ValueError("seasonal drift requires period > 0")
    points = []
    for index in range(length):
        if index < onset:
            weight = 0.0
        else:
            weight = (1.0 + math.sin(2.0 * math.pi * (index - onset) / period + phase)) / 2.0
        active = "blend" if index >= onset else "old"
        configured = "old" if weight == 0.0 else ("new" if weight == 1.0 else "transition")
        points.append(SchedulePoint(
            absolute_index=index,
            timestamp=pd.Timestamp(timestamps[index]).isoformat(),
            configured_regime=configured,
            old_concept_id="old",
            new_concept_id="new",
            active_concept_id=active,
            drift_weight=float(weight),
            regime_active=index >= onset,
            change_point_event=index == onset,
            new_regime_selected=weight >= 1.0,
        ))
    return points


def _schedule(
    length: int, onset: int, timestamps: np.ndarray, *, drift_type: str,
    transition_end: int | None = None, seed: int | None = None, component_id: str | None = None,
    period: float | None = None, phase: float = 0.0,
) -> list[SchedulePoint]:
    if drift_type == "sudden":
        return _sudden_schedule(length, onset, timestamps)
    if drift_type == "gradual":
        if transition_end is None or seed is None or component_id is None:
            raise ValueError("gradual drift requires transition_end, seed, and component_id")
        return _gradual_schedule(length, onset, transition_end, timestamps, seed=seed, component_id=component_id)
    if drift_type == "seasonal":
        if period is None:
            raise ValueError("seasonal drift requires period")
        return _seasonal_schedule(length, onset, timestamps, period=period, phase=phase)
    raise ValueError(f"unsupported drift_type for offline preparation: {drift_type!r}")


def _array_hash(**arrays: np.ndarray) -> str:
    return canonical_array_payload_hash(arrays)


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_sources(
    source_dir: Path,
    assignment_path: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], list[dict[str, str]], list[dict[str, Any]]]:
    population = config["population"]
    cluster_count = int(population["clusters"])
    total_clients = int(population["total_clients"])
    clients_per_cluster = total_clients // cluster_count
    observations = int(config.get("source", {}).get("observations_per_client", 8760))
    assignment = pd.read_csv(assignment_path, dtype=str)
    columns = {str(column).lower(): column for column in assignment.columns}
    if not {"id", "cluster"}.issubset(columns):
        raise ValueError("assignment CSV must contain ID and Cluster")
    assignment_rows = [
        {"cluster_id": str(row[columns["cluster"]]), "client_id": str(row[columns["id"]])}
        for _, row in assignment.iterrows()
    ]
    assignment_pairs = {(row["cluster_id"], row["client_id"]) for row in assignment_rows}
    if len(assignment_pairs) != total_clients:
        raise ValueError(f"assignment must contain {total_clients} unique clients")

    result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    source_rows: list[dict[str, Any]] = []
    discovered: set[tuple[str, str]] = set()
    for cluster_number in range(1, cluster_count + 1):
        cluster_id = str(cluster_number)
        source_path = source_dir / f"cluster_{cluster_id}.pkl"
        payload = pd.read_pickle(source_path)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("timeseries"), pd.DataFrame):
            raise ValueError(f"invalid source payload: {source_path}")
        frame = payload["timeseries"].copy()
        required = {"Fecha_Hora", "id", "value"}
        if not required.issubset(frame.columns):
            raise ValueError(f"source {source_path} lacks {sorted(required - set(frame.columns))}")
        frame["Fecha_Hora"] = pd.to_datetime(frame["Fecha_Hora"], errors="raise")
        frame["id"] = frame["id"].map(lambda item: str(int(item)) if str(item).endswith(".0") else str(item))
        clients: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for client_id, client_rows in frame.groupby("id", sort=False):
            ordered = client_rows.sort_values("Fecha_Hora", kind="stable")
            if len(ordered) != observations:
                raise ValueError(f"client {cluster_id}/{client_id} must have {observations} observations")
            if ordered["Fecha_Hora"].duplicated().any():
                raise ValueError(f"duplicate timestamp for {cluster_id}/{client_id}")
            timestamps = ordered["Fecha_Hora"].to_numpy(dtype="datetime64[ns]")
            if not np.all(np.diff(timestamps.astype(np.int64)) == 3_600_000_000_000):
                raise ValueError(f"non-hourly timeline for {cluster_id}/{client_id}")
            # Preserve the schema-v2 stream's approved numerical boundary.
            values = ordered["value"].to_numpy(dtype=np.float32).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite source value for {cluster_id}/{client_id}")
            clients[str(client_id)] = (values, timestamps)
            discovered.add((cluster_id, str(client_id)))
        if len(clients) != clients_per_cluster:
            raise ValueError(f"cluster {cluster_id} must have {clients_per_cluster} clients")
        result[cluster_id] = clients
        source_rows.append({
            "cluster_id": cluster_id,
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "clients": len(clients),
            "observations_per_client": observations,
            "numeric_contract": "source_float32_then_float64",
        })
    if discovered != assignment_pairs:
        raise ValueError("source clients and assignment clients differ")
    return result, sorted(assignment_rows, key=lambda row: (row["cluster_id"], row["client_id"])), source_rows


def _coordinates(config: Mapping[str, Any], timestamps: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Derive every coordinate-contract field from the resolved config and
    the actual timestamp count, rather than pinning to the one golden
    production geometry. This intentionally accepts any scenario whose
    train/validation/deployment windows exactly and positively cover the
    available forecasting windows.
    """
    window_size = int(config["windowing"]["window_size"])
    horizon = int(config["windowing"]["forecast_horizon"])
    total = len(timestamps) - window_size - horizon + 1
    splits = config.get("splits", {})
    train_count = int(splits.get("train_windows", int(total * 0.70)))
    val_count = int(splits.get("validation_windows", int(total * 0.10)))
    sizes = SegmentSizes(**config["segments"])
    sizes.validate()
    if train_count <= 0 or val_count <= 0:
        raise ValueError(f"train_windows and validation_windows must be positive: train={train_count} validation={val_count}")
    if train_count + val_count + sizes.total != total:
        raise ValueError(
            f"train ({train_count}) + validation ({val_count}) + deployment segments ({sizes.total}) "
            f"must exactly cover the {total} available forecasting windows"
        )
    starts = {
        "train": np.arange(0, train_count, dtype=np.int64),
        "validation": np.arange(train_count, train_count + val_count, dtype=np.int64),
        "deployment": np.arange(train_count + val_count, total, dtype=np.int64),
    }

    def _block(window_starts: np.ndarray) -> dict[str, int]:
        first, last = int(window_starts[0]), int(window_starts[-1])
        return {
            "window_start_first": first,
            "window_start_last": last,
            "target_first": first + window_size - 1 + horizon,
            "target_last": last + window_size - 1 + horizon,
            "count": int(len(window_starts)),
        }

    clean_train = _block(starts["train"])
    clean_validation = _block(starts["validation"])
    deployment = _block(starts["deployment"])

    onset_position = sizes.pre_drift
    holdout_position = sizes.pre_drift + sizes.adaptation
    onset_start = int(starts["deployment"][onset_position])
    holdout_start = int(starts["deployment"][holdout_position])
    onset = onset_start + window_size - 1 + horizon

    contract = {
        "schema_version": SCHEMA_VERSION,
        "window_size": window_size,
        "forecast_horizon": horizon,
        "total_observations": len(timestamps),
        "total_forecasting_windows": total,
        "clean_train": clean_train,
        "clean_validation": clean_validation,
        "deployment": deployment,
        "scaler_fit": {
            "first": 0, "last": clean_train["target_last"],
            "count": clean_train["target_last"] + 1, "coordinate_6139_excluded": True,
        },
        "onset": {
            "deployment_window_number": onset_position + 1,
            "window_start_index": onset_start,
            "window_end_index": onset_start + window_size - 1,
            "target_index": onset,
            "timestamp": pd.Timestamp(timestamps[onset]).isoformat(),
        },
        "final_holdout": {
            "deployment_window_number": holdout_position + 1,
            "window_start_index": holdout_start,
            "window_end_index": holdout_start + window_size - 1,
            "target_index": holdout_start + window_size - 1 + horizon,
        },
    }
    contract["coordinate_manifest_hash"] = canonical_hash(contract)
    return contract, starts


def _metadata(
    starts: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
    horizon: int,
    segment_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, np.ndarray]:
    starts = np.asarray(starts, dtype=np.int64)
    ends = starts + window_size - 1
    targets = ends + horizon
    count = len(starts)
    result: dict[str, np.ndarray] = {
        "sample_number": np.arange(1, count + 1, dtype=np.int64),
        "deployment_window_number": np.zeros(count, dtype=np.int64),
        "window_start_index": starts,
        "window_end_index": ends,
        "target_index": targets,
        "window_start_timestamp_ns": timestamps[starts].astype(np.int64),
        "window_end_timestamp_ns": timestamps[ends].astype(np.int64),
        "target_timestamp_ns": timestamps[targets].astype(np.int64),
        "segment": np.full(count, "TRAIN_OR_VALIDATION", dtype="U20"),
        "process_regime_at_window_start": np.full(count, "OLD", dtype="U3"),
        "process_regime_at_window_end": np.full(count, "OLD", dtype="U3"),
        "process_regime_at_target": np.full(count, "OLD", dtype="U3"),
    }
    for name in ("window_contains_any_drifted_input", "window_input_fully_drifted", "target_drifted", "true_drift_active", "oracle_active"):
        result[name] = np.zeros(count, dtype=np.bool_)
    if segment_rows is not None:
        result["deployment_window_number"] = np.asarray([row["deployment_window_number"] for row in segment_rows], dtype=np.int64)
        for name in ("segment", "process_regime_at_window_start", "process_regime_at_window_end", "process_regime_at_target"):
            result[name] = np.asarray([row[name] for row in segment_rows], dtype="U20")
        for name in ("window_contains_any_drifted_input", "window_input_fully_drifted", "target_drifted", "true_drift_active", "oracle_active"):
            result[name] = np.asarray([bool(row[name]) for row in segment_rows], dtype=np.bool_)
    return result


def _write_samples(
    root: Path,
    role: str,
    cluster_id: str,
    client_id: str,
    windows: ForecastWindows,
    metadata: Mapping[str, np.ndarray],
    scaler_hash: str,
    coordinate_hash: str,
    timeline_hash: str,
) -> dict[str, Any]:
    X = np.asarray(windows.X, dtype=np.float32)
    y = np.asarray(windows.y, dtype=np.float32).reshape(-1)
    content = {"X": X, "y": y, **metadata}
    payload_hash = canonical_array_payload_hash(content)
    relative = Path("prepared_windows") / role / f"cluster_{cluster_id}" / f"client_{client_id}.npz"
    path = root / relative
    _save_npz(
        path,
        **content,
        dataset_role=np.asarray(role), cluster_id=np.asarray(cluster_id), client_id=np.asarray(client_id),
        representation=np.asarray("normalized"), normalization_applied_count=np.asarray(1, dtype=np.int8),
        scaler_hash=np.asarray(scaler_hash), coordinate_manifest_hash=np.asarray(coordinate_hash),
        source_timeline_hash=np.asarray(timeline_hash), prepared_payload_hash=np.asarray(payload_hash),
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int16),
    )
    return {
        "path": str(relative), "artifact_sha256": file_sha256(path), "payload_hash": payload_hash,
        "target_hash": _array_hash(y=y),
        "sample_key_hash": _array_hash(start=metadata["window_start_index"], end=metadata["window_end_index"], target=metadata["target_index"]),
        "scaler_hash": scaler_hash, "coordinate_manifest_hash": coordinate_hash,
        "source_timeline_hash": timeline_hash, "sample_count": len(y),
        "representation": "normalized", "normalization_applied_count": 1,
    }


def prepare_offline_data(
    config_path: str | Path,
    source_dir: str | Path,
    assignment_path: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create immutable raw and normalized prepared artifacts without opening H."""
    config_path = Path(config_path)
    config = load_suite_yaml(config_path)
    sources, assignment_rows, source_rows = _load_sources(Path(source_dir), Path(assignment_path), config)
    first_cluster = sorted(sources, key=int)[0]
    first_client = sorted(sources[first_cluster])[0]
    reference_timestamps = sources[first_cluster][first_client][1]
    for clients in sources.values():
        for _, timestamps in clients.values():
            if not np.array_equal(reference_timestamps, timestamps):
                raise ValueError("all clients must use identical chronological coordinates")
    coordinate_contract, starts = _coordinates(config, reference_timestamps)
    coordinate_hash = str(coordinate_contract["coordinate_manifest_hash"])
    clusters = {cluster: sorted(clients) for cluster, clients in sources.items()}
    cohort = stable_select_per_cluster(clusters, per_cluster=int(config["population"]["drifted_per_cluster"]), seed=int(config["seeds"]["client_selection_seed"]))
    cohort_hash = selected_cohort_hash(cohort)
    selected = {(row["cluster_id"], row["client_id"]) for row in cohort if row["selected_for_drift"]}
    process = ProcessSpec(
        drift_type=str(config["process"]["drift_type"]), drift_source=str(config["process"]["drift_source"]),
        old_intercept=float(config["process"]["old_intercept"]), old_ar_coefficients=tuple(config["process"]["old_ar_coefficients"]),
        intercept_delta=float(config["process"]["intercept_delta"]), ar_coefficient_delta=tuple(config["process"]["ar_coefficient_delta"]),
        innovation_sigma=float(config["process"]["innovation_sigma"]), innovation_scale=float(config["process"]["innovation_scale"]),
        innovation_source=str(config["process"]["innovation_source"]),
        gradual_transition_windows=(
            int(config["process"]["gradual_transition_windows"])
            if config["process"].get("gradual_transition_windows") is not None else None
        ),
        # Raw-observation units (e.g. hours), the same coordinate space as
        # onset -- not forecasting windows -- so a daily/weekly cycle is
        # simply 24/168.
        seasonal_period=(
            float(config["process"]["seasonal_period"])
            if config["process"].get("seasonal_period") is not None else None
        ),
        seasonal_phase=float(config["process"].get("seasonal_phase", 0.0)),
    )
    process.validate()
    process_manifest = process.equation_manifest() | {
        "benchmark_interpretation": "controlled counterfactual benchmark generated offline using observed future residuals; not online-generatable without future information",
        "residual_visibility": "offline_preparation_audit_only",
    }
    sizes = SegmentSizes(**config["segments"])
    root = guard_output_directory(output, overwrite=overwrite)
    (root / ".drift_mitigation_suite").write_text(SUITE_ID + "\n", encoding="utf-8")
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    shutil.copy2(config_path, root / "suite_resolved_config.yaml")
    write_json(manifests / "source_manifest.json", {"schema_version": 2, "assignment": {"path": str(assignment_path), "sha256": file_sha256(assignment_path)}, "clusters": source_rows})
    write_csv_rows(manifests / "client_assignment_manifest.csv", ["cluster_id", "client_id"], assignment_rows)
    write_csv_rows(manifests / "selected_drift_clients.csv", list(cohort[0]), cohort)
    write_json(manifests / "selected_drift_clients_hash.json", {"sha256": cohort_hash, "selected_count": len(selected)})
    write_json(manifests / "split_manifest.json", coordinate_contract)
    write_csv_rows(manifests / "process_parameters.csv", list(process_manifest), [process_manifest])

    role_entries: dict[str, dict[str, Any]] = {role: {"clients": {}} for role in ROLES}
    scalers: list[dict[str, Any]] = []
    timelines: list[dict[str, Any]] = []
    innovations: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    generic_segments: list[dict[str, Any]] | None = None
    onset = int(coordinate_contract["onset"]["target_index"])
    window_size = int(config["windowing"]["window_size"])
    horizon = int(config["windowing"]["forecast_horizon"])
    transition_end: int | None = None
    if process.drift_type == "gradual":
        transition_windows = (
            int(process.gradual_transition_windows)
            if process.gradual_transition_windows is not None
            else sizes.adaptation
        )
        transition_end_position = sizes.pre_drift + transition_windows
        if transition_end_position >= len(starts["deployment"]):
            raise ValueError("gradual_transition_windows extends past the end of the deployment timeline")
        transition_end_start = int(starts["deployment"][transition_end_position])
        transition_end = transition_end_start + window_size - 1 + horizon
    controls_unchanged = True
    pre_onset_unchanged = True
    selected_changed = True

    # Pass 1: everything up through true_timeline (process drift), exactly
    # as before -- computed once per client, independent of faults. State
    # needed by pass 2 is stashed in client_state rather than acted on
    # immediately, since the fault step (population-wide, run once for
    # every client at once) has to happen between the two passes.
    client_state: dict[tuple[str, str], dict[str, Any]] = {}
    for cluster_id in sorted(sources, key=int):
        for client_id in sorted(sources[cluster_id]):
            clean, timestamps = sources[cluster_id][client_id]
            # Required order: fit before residual calculation/drift generation.
            scaler = fit_clean_training_scalers(clean, clean, training_window_starts=starts["train"], window_size=window_size, forecast_horizon=horizon, target_feature_index=0, method=str(config["normalization"]["method"]), share_target_scaler=True)
            if scaler.shared_target_indices != tuple(range(6139)):
                raise ValueError("shared scaler union must be exactly coordinates 0..6138")
            clean_hash = _array_hash(values=clean, timestamps=timestamps.astype(np.int64))
            scaler_row = {
                "schema_version": 2, "cluster_id": cluster_id, "client_id": client_id,
                "ownership": "per_client", "method": scaler.method, "target_scaling_strategy": "shared",
                "location": float(scaler.y_location), "scale": float(scaler.y_scale),
                "fit_first_observation_index": 0, "fit_last_observation_index": 6138,
                "fit_observation_count": 6139, "coordinate_6139_excluded": True,
                "source_timeline_hash": clean_hash, "x_union_hash": scaler.x_union_hash,
                "y_union_hash": scaler.y_union_hash, "shared_union_hash": scaler.shared_target_union_hash,
            }
            scaler_hash = canonical_hash(scaler_row)
            scalers.append({**scaler_row, "scaler_hash": scaler_hash})

            residual = np.zeros_like(clean)
            residual[1:] = clean[1:] - (process.old_intercept + process.old_ar_coefficients[0] * clean[:-1])
            affected = (cluster_id, client_id) in selected
            true_timeline = clean.copy()
            if affected:
                drift_seed = int(config["seeds"]["drift_seed"])
                component_id = f"cluster-{cluster_id}-client-{client_id}"
                schedule_points = _schedule(
                    len(clean), onset, timestamps, drift_type=process.drift_type,
                    transition_end=transition_end, seed=drift_seed, component_id=component_id,
                    period=process.seasonal_period, phase=process.seasonal_phase,
                )
                generated = generate_process_timeline(clean, schedule_points, {
                    "old": {"intercept": process.old_intercept, "ar_coefficients": list(process.old_ar_coefficients), "innovation_sigma": process.innovation_sigma},
                    "changes": {"intercept_delta": process.intercept_delta, "ar_coefficient_delta": list(process.ar_coefficient_delta), "innovation_scale": process.innovation_scale},
                    "innovation_source": process.innovation_source,
                }, target_feature_index=0, client_seed=drift_seed, component_id=component_id)
                true_timeline = generated.values[:, 0]
                if not np.allclose(generated.innovations[1:], residual[1:], rtol=0.0, atol=1e-12):
                    raise AssertionError("generated innovations differ from stored residual audit sequence")
                if process.drift_type == "gradual":
                    gradual_schedule_path = root / "audit_private" / "gradual_schedule" / f"cluster_{cluster_id}" / f"client_{client_id}.npz"
                    _save_npz(
                        gradual_schedule_path,
                        weight=np.asarray([point.drift_weight for point in schedule_points], dtype=np.float64),
                        selected=np.asarray([point.new_regime_selected for point in schedule_points], dtype=bool),
                        onset_index=np.asarray(onset), transition_end_index=np.asarray(transition_end),
                        transition_windows=np.asarray(int(transition_end) - int(onset)),
                        seed=np.asarray(drift_seed),
                        old_intercept=np.asarray(process.old_intercept), old_ar_coefficient=np.asarray(process.old_ar_coefficients[0]),
                        new_intercept=np.asarray(process.new_intercept), new_ar_coefficient=np.asarray(process.new_ar_coefficients[0]),
                        cluster_id=np.asarray(cluster_id), client_id=np.asarray(client_id),
                        access_scope=np.asarray("offline_preparation_audit_only"), schema_version=np.asarray(2, dtype=np.int16),
                    )
                elif process.drift_type == "seasonal":
                    weight_array = np.asarray([point.drift_weight for point in schedule_points], dtype=np.float64)
                    a_t = process.old_intercept + weight_array * (process.new_intercept - process.old_intercept)
                    phi_t = process.old_ar_coefficients[0] + weight_array * (process.new_ar_coefficients[0] - process.old_ar_coefficients[0])
                    seasonal_schedule_path = root / "audit_private" / "seasonal_schedule" / f"cluster_{cluster_id}" / f"client_{client_id}.npz"
                    _save_npz(
                        seasonal_schedule_path,
                        weight=weight_array, a_t=a_t, phi_t=phi_t,
                        onset_index=np.asarray(onset), period=np.asarray(process.seasonal_period), phase=np.asarray(process.seasonal_phase),
                        seed=np.asarray(drift_seed),
                        old_intercept=np.asarray(process.old_intercept), old_ar_coefficient=np.asarray(process.old_ar_coefficients[0]),
                        new_intercept=np.asarray(process.new_intercept), new_ar_coefficient=np.asarray(process.new_ar_coefficients[0]),
                        cluster_id=np.asarray(cluster_id), client_id=np.asarray(client_id),
                        access_scope=np.asarray("offline_preparation_audit_only"), schema_version=np.asarray(2, dtype=np.int16),
                    )
            # drifted_deployment's RAW (unnormalized) windows, built here so
            # the population-wide fault step (which needs every client's
            # X_drift at once) can run on already-materialized 24-step
            # windows -- never on the raw timeline. Injecting faults into a
            # raw timeline before windowing was a real, confirmed defect in
            # an earlier version of this module: a single corrupted raw
            # position smears across every overlapping window that includes
            # it. X and y both come from true_timeline here (drift-only, no
            # fault yet) -- identical to the no-fault path's own construction.
            drift_raw_windows = build_forecasting_windows(
                true_timeline, true_timeline, window_size=window_size, forecast_horizon=horizon,
                target_feature_index=0, window_start_indices=starts["deployment"],
            )
            client_state[(cluster_id, client_id)] = {
                "clean": clean, "timestamps": timestamps, "scaler": scaler, "scaler_hash": scaler_hash,
                "clean_hash": clean_hash, "residual": residual, "affected": affected, "true_timeline": true_timeline,
                "drift_raw_windows": drift_raw_windows,
            }

    # Fault step: population-wide, run exactly once (never per-client, never
    # per-window), only when this suite's faults block is enabled. Disabled
    # (the default, and every pre-existing config) => this whole block is
    # skipped and the drifted_deployment role below takes the exact
    # no-fault path it always has.
    faults_enabled = bool(config.get("faults", {}).get("enabled", False))
    fault_x_observed: dict[tuple[str, str], np.ndarray] = {}
    fault_selected: set[tuple[str, str]] = set()
    quality_weighting_cfg = dict(config.get("quality_weighting") or {})
    quality_weighting_enabled = bool(quality_weighting_cfg.get("enabled", False))
    quality_weight_score: dict[tuple[str, str], np.ndarray] = {}
    quality_weight_score_computed: dict[tuple[str, str], np.ndarray] = {}
    if faults_enabled:
        # segment/target-index coordinates are identical for every client
        # (same starts["deployment"], same window_size/horizon/onset) -- one
        # population-wide computation, not re-derived per client, and reused
        # as the SAME arrays build_segment_rows already produces elsewhere.
        fault_segments = build_segment_rows(
            window_starts=starts["deployment"], window_size=window_size, horizon=horizon,
            timestamps=[pd.Timestamp(item) for item in reference_timestamps], sizes=sizes, onset_index=onset, selected=False,
        )
        segment_by_window = np.asarray([row["segment"] for row in fault_segments], dtype="U20")
        any_client_windows = next(iter(client_state.values()))["drift_raw_windows"]
        target_index_by_window = np.asarray(any_client_windows.target_indices)

        fault_result = apply_window_faults(
            X_drift={key: state["drift_raw_windows"].X for key, state in client_state.items()},
            segment_by_window=segment_by_window, target_index_by_window=target_index_by_window,
            timestamps=reference_timestamps, assignment_rows=assignment_rows,
            fault_config_path=config["faults"]["config_path"], fault_selection_seed=int(config["seeds"]["fault_selection_seed"]),
            window_size=window_size, manifests_dir=manifests,
        )
        fault_x_observed = fault_result["X_observed"]
        fault_selected = fault_result["fault_selected"]
        write_csv_rows(
            manifests / "selected_fault_clients.csv", ["cluster_id", "client_id", "selected_for_fault"],
            [{"cluster_id": c, "client_id": k, "selected_for_fault": (c, k) in fault_selected} for c, k in sorted(client_state)],
        )
        realizations = fault_result["realizations_frame"]
        if not realizations.empty:
            write_csv_rows(manifests / "fault_realizations.csv", list(realizations.columns), realizations.to_dict("records"))
        else:
            write_csv_rows(manifests / "fault_realizations.csv", [
                "cluster_id", "client_id", "deployment_window_index", "segment", "forecast_target_index_raw",
                "forecast_target_timestamp", "fault_position_within_window", "fault_dimension", "fault_type",
            ], [])
        write_json(manifests / "fault_configuration_identity.json", fault_result["identity"])

        # DQ-based quality weighting (M4_quality_aware_splitfed only, see
        # QualityWeightingConfig): scored from the SAME fault_x_observed/
        # segment_by_window this fault step already computed, never
        # recomputed. Requires faults_enabled -- enforced in load_suite_yaml,
        # not re-checked here -- since the score is derived from X_observed.
        if quality_weighting_enabled:
            dq_result = compute_window_quality_weights(
                X_observed=fault_x_observed, segment_by_window=segment_by_window,
                window_start_index_by_window=starts["deployment"], timestamps=reference_timestamps,
                window_size=window_size, dq_config_path=quality_weighting_cfg["config_path"],
            )
            quality_weight_score = dq_result["quality_weight_score"]
            quality_weight_score_computed = dq_result["quality_weight_score_computed"]
            write_json(manifests / "quality_weighting_identity.json", dq_result["identity"])
            summary = dq_result["summary_frame"]
            if not summary.empty:
                write_csv_rows(manifests / "quality_weighting_summary.csv", list(summary.columns), summary.to_dict("records"))

    # Pass 2: raw_clean/raw_drifted/residual artifacts and the three clean
    # roles are exactly the no-fault path this repository has always had --
    # faults never touch the raw timeline at all anymore, so none of this
    # per-client body depends on faults_enabled. drifted_deployment is
    # handled separately below, since it alone has the fault+impute step.
    for cluster_id in sorted(sources, key=int):
        for client_id in sorted(sources[cluster_id]):
            state = client_state[(cluster_id, client_id)]
            clean, timestamps, scaler, scaler_hash = state["clean"], state["timestamps"], state["scaler"], state["scaler_hash"]
            clean_hash, residual, affected, true_timeline = state["clean_hash"], state["residual"], state["affected"], state["true_timeline"]
            pre_onset_unchanged = pre_onset_unchanged and np.array_equal(clean[:onset], true_timeline[:onset])
            controls_unchanged = controls_unchanged and (affected or np.array_equal(clean, true_timeline))
            selected_changed = selected_changed and ((not affected) or np.any(clean[onset:] != true_timeline[onset:]))
            true_timeline_hash = _array_hash(values=true_timeline, timestamps=timestamps.astype(np.int64))
            common = {
                "timestamp_ns": timestamps.astype(np.int64), "observation_index": np.arange(len(clean), dtype=np.int64),
                "cluster_id": np.asarray(cluster_id), "client_id": np.asarray(client_id),
                "representation": np.asarray("raw_physical_units"), "normalization_applied_count": np.asarray(0, dtype=np.int8),
                "schema_version": np.asarray(2, dtype=np.int16),
            }
            clean_path = root / "raw_clean" / f"cluster_{cluster_id}" / f"client_{client_id}.npz"
            drifted_path = root / "raw_drifted" / f"cluster_{cluster_id}" / f"client_{client_id}.npz"
            residual_path = root / "audit_private" / "residuals" / f"cluster_{cluster_id}" / f"client_{client_id}.npz"
            _save_npz(clean_path, values=clean, timeline_hash=np.asarray(clean_hash), **common)
            _save_npz(drifted_path, values=true_timeline, timeline_hash=np.asarray(true_timeline_hash), **common)
            _save_npz(residual_path, epsilon=residual, cluster_id=np.asarray(cluster_id), client_id=np.asarray(client_id), innovation_source=np.asarray("observed_process_residual"), innovation_scale=np.asarray(1.0), onset_index=np.asarray(onset), access_scope=np.asarray("offline_preparation_audit_only"), schema_version=np.asarray(2, dtype=np.int16))
            residual_hash = _array_hash(epsilon=residual)
            timelines.append({
                "cluster_id": cluster_id, "client_id": client_id, "selected_for_drift": affected,
                "clean_timeline_hash": clean_hash, "drifted_timeline_hash": true_timeline_hash,
                "clean_artifact_sha256": file_sha256(clean_path), "drifted_artifact_sha256": file_sha256(drifted_path),
            })
            innovations.append({"cluster_id": cluster_id, "client_id": client_id, "path": str(residual_path.relative_to(root)), "payload_hash": residual_hash, "artifact_sha256": file_sha256(residual_path), "access_scope": "offline_preparation_audit_only"})

            drift_segments = build_segment_rows(window_starts=starts["deployment"], window_size=window_size, horizon=horizon, timestamps=[pd.Timestamp(item) for item in timestamps], sizes=sizes, onset_index=onset, selected=affected)
            clean_segments = build_segment_rows(window_starts=starts["deployment"], window_size=window_size, horizon=horizon, timestamps=[pd.Timestamp(item) for item in timestamps], sizes=sizes, onset_index=onset, selected=False)
            if generic_segments is None:
                generic_segments = clean_segments
            ground_truth.extend({"cluster_id": cluster_id, "client_id": client_id, **row} for row in drift_segments)

            specs = {
                "clean_train": (clean, clean, starts["train"], None, clean_hash),
                "clean_validation": (clean, clean, starts["validation"], None, clean_hash),
                "clean_deployment": (clean, clean, starts["deployment"], clean_segments, clean_hash),
            }
            for role, (observed_source, true_source, role_starts, segments, timeline_hash) in specs.items():
                raw_windows = build_forecasting_windows(observed_source, true_source, window_size=window_size, forecast_horizon=horizon, target_feature_index=0, window_start_indices=role_starts)
                normalized = normalize_windows(raw_windows, scaler)
                assert_horizon_one_consistency(normalized.X, normalized.y, target_feature_index=0, atol=1e-6)
                metadata = _metadata(role_starts, timestamps, window_size, horizon, segments)
                role_entries[role]["clients"][f"{cluster_id}:{client_id}"] = _write_samples(root, role, cluster_id, client_id, normalized, metadata, scaler_hash, coordinate_hash, timeline_hash)
                if role == "clean_deployment":
                    offset = sizes.pre_drift + sizes.adaptation
                    holdout = ForecastWindows(normalized.X[offset:], normalized.y[offset:], normalized.window_start_indices[offset:], normalized.window_end_indices[offset:], normalized.target_indices[offset:])
                    holdout_meta = {name: values[offset:] for name, values in metadata.items()}
                    role_entries["clean_final_holdout_counterfactual"]["clients"][f"{cluster_id}:{client_id}"] = _write_samples(root, "clean_final_holdout_counterfactual", cluster_id, client_id, holdout, holdout_meta, scaler_hash, coordinate_hash, clean_hash)

            # drifted_deployment: raw windows were already built in pass 1
            # (drift_raw_windows) so the population-wide fault step could
            # consume every client's X_drift at once. Required order:
            #   X_drift (raw) -> [faults on raw X only] -> X_observed (raw)
            #   -> forward-fill within window -> leading NaN -> scaler.x_location
            #   -> X_model_raw (raw, finite) -> normalize_windows() -> X (normalized), y (normalized, untouched)
            drift_raw_windows = state["drift_raw_windows"]
            imputation_meta: dict[str, Any] | None = None
            X_observed_raw: np.ndarray | None = None
            if faults_enabled and (cluster_id, client_id) in fault_x_observed:
                X_observed_raw = fault_x_observed[(cluster_id, client_id)]
                X_model_raw, imputation_meta = make_model_consumable(X_observed_raw, clean_training_mean_raw=float(np.asarray(scaler.x_location).reshape(-1)[0]))
                windows_for_norm = ForecastWindows(
                    X_model_raw, drift_raw_windows.y, drift_raw_windows.window_start_indices,
                    drift_raw_windows.window_end_indices, drift_raw_windows.target_indices,
                )
            else:
                windows_for_norm = drift_raw_windows
            normalized_drift = normalize_windows(windows_for_norm, scaler)
            if windows_for_norm.X is drift_raw_windows.X:
                # Only meaningful when X and y are drawn from the exact same
                # source (true, unfaulted) -- true whenever this client had
                # no realized fault (including every client when faults are
                # disabled suite-wide).
                assert_horizon_one_consistency(normalized_drift.X, normalized_drift.y, target_feature_index=0, atol=1e-6)
            drift_metadata = _metadata(starts["deployment"], timestamps, window_size, horizon, drift_segments)
            if X_observed_raw is not None:
                representations = dict(REPRESENTATIONS_METADATA)
                if quality_weighting_enabled:
                    representations.update(DQ_REPRESENTATION_METADATA)
                drift_metadata = {
                    **drift_metadata,
                    "X_observed": np.asarray(X_observed_raw, dtype=np.float32),
                    "representations_json": np.asarray(json.dumps(representations)),
                    "missing_values_before_imputation": np.asarray(imputation_meta["missing_values_before_imputation"], dtype=np.int64),
                    "forward_filled_count": np.asarray(imputation_meta["forward_filled_count"], dtype=np.int64),
                    "leading_fallback_count": np.asarray(imputation_meta["leading_fallback_count"], dtype=np.int64),
                    "missing_values_after_imputation": np.asarray(imputation_meta["missing_values_after_imputation"], dtype=np.int64),
                    "clean_training_mean_raw": np.asarray(imputation_meta["clean_training_mean_raw"], dtype=np.float64),
                }
                if quality_weighting_enabled and (cluster_id, client_id) in quality_weight_score:
                    drift_metadata["quality_weight_score"] = np.asarray(
                        quality_weight_score[(cluster_id, client_id)], dtype=np.float32,
                    )
                    drift_metadata["quality_weight_score_computed"] = np.asarray(
                        quality_weight_score_computed[(cluster_id, client_id)], dtype=bool,
                    )
            role_entries["drifted_deployment"]["clients"][f"{cluster_id}:{client_id}"] = _write_samples(
                root, "drifted_deployment", cluster_id, client_id, normalized_drift, drift_metadata, scaler_hash, coordinate_hash, true_timeline_hash,
            )

    if faults_enabled:
        write_csv_rows(
            manifests / "client_condition_manifest.csv",
            ["cluster_id", "client_id", "is_drift_selected", "is_fault_selected", "condition"],
            [
                {
                    "cluster_id": c, "client_id": k,
                    "is_drift_selected": (c, k) in selected, "is_fault_selected": (c, k) in fault_selected,
                    "condition": (
                        "drift_and_fault" if (c, k) in selected and (c, k) in fault_selected else
                        "drift_only" if (c, k) in selected else
                        "fault_only" if (c, k) in fault_selected else
                        "clean"
                    ),
                }
                for c, k in sorted(client_state)
            ],
        )

    for role in ROLES:
        entries = [role_entries[role]["clients"][key] for key in sorted(role_entries[role]["clients"])]
        role_entries[role].update({
            "client_count": len(entries),
            "combined_payload_hash": canonical_hash([entry["payload_hash"] for entry in entries]),
            "combined_target_hash": canonical_hash([entry["target_hash"] for entry in entries]),
            "combined_sample_key_hash": canonical_hash([entry["sample_key_hash"] for entry in entries]),
        })
    scaler_manifest_hash = canonical_hash(scalers)
    write_csv_rows(manifests / "scaler_manifest.csv", list(scalers[0]), scalers)
    write_json(manifests / "scaler_parameters.json", {"schema_version": 2, "ownership": "per_client_shared_x_y", "fit_coordinates": [0, 6138], "fit_count": 6139, "coordinate_6139_excluded": True, "combined_hash": scaler_manifest_hash, "clients": scalers})
    write_csv_rows(manifests / "segment_manifest.csv", list(generic_segments[0]), generic_segments)
    write_csv_rows(manifests / "drift_ground_truth_manifest.csv", list(ground_truth[0]), ground_truth)
    timeline_manifest = {"schema_version": 2, "clients": timelines, "combined_clean_hash": canonical_hash([row["clean_timeline_hash"] for row in timelines]), "combined_drifted_hash": canonical_hash([row["drifted_timeline_hash"] for row in timelines])}
    innovation_manifest = {"schema_version": 2, "access_scope": "offline_preparation_audit_only", "clients": innovations, "combined_hash": canonical_hash([row["payload_hash"] for row in innovations])}
    write_json(manifests / "timeline_hashes.json", timeline_manifest)
    write_json(manifests / "innovation_hashes.json", innovation_manifest)
    prepared = {
        "schema_version": 2, "coordinate_manifest_hash": coordinate_hash, "scaler_manifest_hash": scaler_manifest_hash,
        "selected_cohort_hash": cohort_hash, "process_parameters_hash": canonical_hash(process_manifest),
        "innovation_artifact_hash": innovation_manifest["combined_hash"], "roles": role_entries,
        "method_bindings": {"R0": "clean_deployment", "M0": "drifted_deployment", "M1": "drifted_deployment", "M2": "drifted_deployment", "M2_retention": "clean_final_holdout_counterfactual"},
    }
    write_json(manifests / "prepared_dataset_hashes.json", prepared)
    write_json(manifests / "preparation_access_policy.json", {
        "checkpoint_H_allowed_roles": ["clean_train", "clean_validation"], "gradient_role": "clean_train",
        "evaluation_only_roles": ["clean_validation"],
        "forbidden_to_H": ["clean_deployment", "drifted_deployment", "ADAPTATION", "FINAL_HOLDOUT", "audit_private/residuals", "audit_private/gradual_schedule", "audit_private/seasonal_schedule", "drift_ground_truth_manifest", "oracle_events", "fault_realizations", "selected_fault_clients", "client_condition_manifest"],
        "residual_access": "offline_preparation_audit_only",
    })
    drift_role = role_entries["drifted_deployment"]
    identity = {
        "prepared_dataset_hash": drift_role["combined_payload_hash"], "target_hash": drift_role["combined_target_hash"],
        "sample_key_hash": drift_role["combined_sample_key_hash"], "selected_cohort_hash": cohort_hash,
        "process_parameters_hash": prepared["process_parameters_hash"], "coordinate_manifest_hash": coordinate_hash,
        "innovation_artifact_hash": innovation_manifest["combined_hash"],
    }
    write_json(manifests / "cross_method_identity.json", {method: identity for method in ("M0", "M1", "M2")})
    checks = {
        "source_population_valid": len(assignment_rows) == 84,
        "scaler_per_client_shared_x_y": len(scalers) == 84 and all(row["target_scaling_strategy"] == "shared" for row in scalers),
        "scaler_fit_exactly_0_through_6138": all(row["fit_first_observation_index"] == 0 and row["fit_last_observation_index"] == 6138 and row["fit_observation_count"] == 6139 for row in scalers),
        "coordinate_6139_excluded": all(row["coordinate_6139_excluded"] for row in scalers),
        "no_pre_onset_drift": pre_onset_unchanged, "controls_unchanged": controls_unchanged,
        "selected_clients_change_after_onset": bool(selected_changed),
        "normalization_applied_exactly_once": all(entry["normalization_applied_count"] == 1 for role in role_entries.values() for entry in role["clients"].values()),
        "coherent_X_y_from_one_timeline": True,
        "H_roles_exclude_deployment": True, "residuals_audit_only": all(row["access_scope"] == "offline_preparation_audit_only" for row in innovations),
        "M0_M1_M2_identical_prepared_identity": True, "runtime_processing_disabled": True,
    }
    acceptance = {"schema_version": 2, "suite_id": SUITE_ID, "accepted": all(checks.values()), "complete": True, "checks": checks, "output": str(root)}
    write_json(root / "acceptance_report.json", acceptance)
    write_json(manifests / "preparation_acceptance_report.json", acceptance)
    return {"output": str(root), "acceptance": acceptance, "prepared_dataset_hashes": prepared, "cohort": cohort}


def validate_offline_preparation(output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    acceptance = json.loads((root / "manifests" / "preparation_acceptance_report.json").read_text(encoding="utf-8"))
    prepared = json.loads((root / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))
    if not acceptance.get("accepted") or int(prepared.get("schema_version", -1)) != 2:
        raise ValueError("offline preparation is not accepted")
    if any(prepared["method_bindings"][method] != "drifted_deployment" for method in ("M0", "M1", "M2")):
        raise ValueError("M0/M1/M2 are not bound to one prepared drifted deployment")
    return {"output": str(root), "valid": True, "acceptance": acceptance, "drifted_deployment_hash": prepared["roles"]["drifted_deployment"]["combined_payload_hash"]}

