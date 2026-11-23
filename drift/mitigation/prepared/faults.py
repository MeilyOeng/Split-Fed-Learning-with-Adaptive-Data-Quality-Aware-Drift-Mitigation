"""Window-level controlled fault injection for the prepared mitigation
pipeline, wired onto the existing, independently-owned ``data quality/``
fault-injection subsystem.

Required semantics (see the approved implementation plan this module
follows):

  - Faults are injected into already-materialized raw-scale forecasting
    windows (``X_drift``), never into the raw hourly timeline. Injecting
    into the raw timeline before windowing would let one corrupted raw
    position smear across every overlapping 24-step window that includes
    it -- confirmed as a real defect in an earlier version of this module,
    not a hypothetical concern.
  - ``y`` is never touched by any fault -- it always comes straight from
    ``true_timeline``/``y_drift``.
  - ``data quality/configs/fault_injection_controlled_window.yaml`` is read
    read-only and never modified or duplicated. The only in-memory
    deviation from the literal file is a ``window_faulting.window_size``/
    ``stride`` override, which is a *mechanical* adaptation (aligning the
    injector's own internal windowing granularity to exactly one real
    24-step forecasting window per candidate) -- not a scientific
    suppression. ``total_faulty_window_fraction`` and every
    ``dimension_window_fractions`` value, including ``completeness``,
    are used exactly as configured.
  - There is exactly one fault-selection seed: ``seeds.fault_selection_seed``.
  - The synthetic time axis this module builds to satisfy
    ``WideDemandFaultInjector``'s wide-DataFrame API is purely an internal
    implementation detail -- it is never persisted. Every realization gets
    mapped back to real experiment coordinates
    (``deployment_window_index``, ``segment``, ``forecast_target_index_raw``,
    ``forecast_target_timestamp``, ``fault_position_within_window``) before
    anything is written to disk.
  - NaN produced by a real fault is never "fixed" by looking at the true
    value -- that would erase the fault. The only missing-value handling is
    the deterministic, deliberately simple rule in ``make_model_consumable``:
    forward-fill within each window, then any remaining leading NaN falls
    back to the client's own raw-scale clean-training mean
    (``scaler.x_location``, already fitted -- never re-derived or assumed
    to be exactly zero on the raw scale).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DATA_QUALITY_ROOT = Path(__file__).resolve().parents[3] / "data quality"
if str(_DATA_QUALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_QUALITY_ROOT))

from modules.fault_injection import WideDemandFaultInjector  # noqa: E402
from modules.io import load_yaml_config  # noqa: E402
from modules.io import load_wide_demand_dataframe  # noqa: E402

_DQ_METADATA_PREFIX = "__dq_"

REPRESENTATIONS_METADATA = {
    "X_observed": {"role": "dq_input", "scale": "raw", "faulted": True, "missing_values_allowed": True},
    "X": {"role": "model_input", "scale": "normalized", "imputed": True, "missing_values_allowed": False},
    "y": {"role": "model_target", "scale": "normalized", "faulted": False},
}


def _hash_yaml_file(path: str | Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_filtered_assignment_csv(assignment_rows: list[dict[str, str]], output_path: Path) -> Path:
    """Writes ID/Cluster columns from the SAME assignment_rows
    preparation.py already validated against the loaded 84-client
    population -- never the raw assignment file path directly, so the
    injector's per_cluster_random selection universe is exactly the
    prepared population, never a larger/different source population.
    """
    frame = pd.DataFrame({
        "ID": [row["client_id"] for row in assignment_rows],
        "Cluster": [row["cluster_id"] for row in assignment_rows],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def _build_synthetic_wide_frame(
    X_drift: dict[tuple[str, str], np.ndarray], eligible_window_positions: np.ndarray, window_size: int,
) -> tuple[pd.DataFrame, str]:
    """One column per real client, built by concatenating that client's
    ELIGIBLE (non-PRE_DRIFT) forecasting windows end to end. With
    window_faulting.stride == window_size, the injector's own windowing
    then produces exactly one non-overlapping candidate window per real
    forecasting window -- verified empirically in the approved prototype
    (2 clients x 3 windows, confirmed 1:1 mapping and confirmed that
    corrupting one window never leaks into its raw-timeline neighbor).

    The row axis is a purely SYNTHETIC position index -- it has no
    relationship to real calendar time (concatenated windows are not
    contiguous in the real timeline) and must never be persisted as if it
    were a real timestamp.
    """
    ts_col = "_synthetic_position"
    n_windows = len(eligible_window_positions)
    total_rows = n_windows * window_size
    data: dict[str, Any] = {ts_col: pd.date_range("2000-01-01", periods=total_rows, freq="h")}
    for (cluster_id, client_id), X in X_drift.items():
        eligible = np.asarray(X, dtype=float)[eligible_window_positions]  # [n_windows, window_size, ...]
        data[client_id] = eligible.reshape(-1)
    return pd.DataFrame(data), ts_col


def _tensor_stats_finite_check(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array) | np.isnan(array)):
        raise ValueError(f"{name} contains non-finite, non-NaN values (e.g. Inf) -- fail-closed")


def apply_window_faults(
    *,
    X_drift: dict[tuple[str, str], np.ndarray],
    segment_by_window: np.ndarray,
    target_index_by_window: np.ndarray,
    timestamps: np.ndarray,
    assignment_rows: list[dict[str, str]],
    fault_config_path: str | Path,
    fault_selection_seed: int,
    window_size: int,
    manifests_dir: Path,
) -> dict[str, Any]:
    """Population-wide, window-level fault application. Runs the existing
    controlled-window injector exactly once for the whole population.

    X_drift: {(cluster_id, client_id): array of shape [n_deployment_windows, window_size, 1]}
        (raw scale, unfaulted -- built by the caller from true_timeline
        exactly as the no-fault path already does).
    segment_by_window: array of length n_deployment_windows, values in
        {"PRE_DRIFT", "ADAPTATION", "FINAL_HOLDOUT"} -- from the existing
        build_segment_rows output, never hard-coded window counts.
    target_index_by_window / timestamps: used only to map realizations back
        to real coordinates -- never fed into the injector itself.

    Returns X_observed (raw scale, NaN where faults landed; identical to
    X_drift outside realized faults and entirely for PRE_DRIFT), fault
    cohort membership, and a realization DataFrame in real coordinates only.
    """
    fault_config = load_yaml_config(fault_config_path)
    eligible_window_positions = np.flatnonzero(segment_by_window != "PRE_DRIFT")
    if eligible_window_positions.size == 0:
        raise ValueError("no fault-eligible (non-PRE_DRIFT) windows found")

    expected_client_ids = {client_id for _, client_id in X_drift}
    assignment_client_ids = {row["client_id"] for row in assignment_rows}
    if assignment_client_ids != expected_client_ids:
        missing = expected_client_ids - assignment_client_ids
        extra = assignment_client_ids - expected_client_ids
        raise ValueError(
            f"fault-injection assignment_rows do not exactly match the prepared population: "
            f"missing={sorted(missing)} extra={sorted(extra)} (fail-closed)"
        )
    filtered_assignment_path = _build_filtered_assignment_csv(assignment_rows, manifests_dir / "fault_cluster_assignment_filtered.csv")

    wide_frame, ts_col = _build_synthetic_wide_frame(X_drift, eligible_window_positions, window_size)
    expected_client_ids_set = expected_client_ids

    # Mechanical granularity override only: align the injector's own
    # candidate-window size/stride to exactly one real forecasting window,
    # non-overlapping within each client's concatenated column. Every other
    # field (total_faulty_window_fraction, every dimension_window_fraction
    # including completeness, min/max_faults_per_faulty_window,
    # multi_fault_probability, clients.selection_mode, per-dimension
    # severity params) is used exactly as configured in the shared file.
    windowed_config = copy.deepcopy(fault_config)
    window_faulting = dict(windowed_config.get("window_faulting") or {})
    granularity_override = {
        "window_size": {"original": window_faulting.get("window_size"), "overridden": window_size},
        "stride": {"original": window_faulting.get("stride"), "overridden": window_size},
    }
    window_faulting["window_size"] = window_size
    window_faulting["stride"] = window_size
    windowed_config["window_faulting"] = window_faulting

    dataset = load_wide_demand_dataframe(wide_frame, timestamp_col=ts_col)
    if set(dataset.client_cols) != expected_client_ids_set:
        raise ValueError(
            f"synthetic fault-injection frame's detected client columns do not match the prepared "
            f"population: detected={sorted(dataset.client_cols)} expected={sorted(expected_client_ids_set)} (fail-closed)"
        )

    injector = WideDemandFaultInjector(config=windowed_config)
    faulted_frame, report_df = injector.inject(
        dataset, seed=int(fault_selection_seed), cluster_assignment_path=str(filtered_assignment_path),
    )
    selected_clients_df = getattr(injector, "last_controlled_selected_clients_df", pd.DataFrame())
    window_summary_df = getattr(injector, "last_controlled_window_summary_df", pd.DataFrame())

    if len(faulted_frame) != len(wide_frame):
        raise ValueError(f"fault injector changed row count: {len(wide_frame)} -> {len(faulted_frame)} (fail-closed)")
    faulted_client_cols = {
        str(col) for col in faulted_frame.columns
        if col != dataset.timestamp_col and not str(col).startswith(_DQ_METADATA_PREFIX)
    }
    if faulted_client_cols != expected_client_ids_set:
        missing = expected_client_ids_set - faulted_client_cols
        extra = faulted_client_cols - expected_client_ids_set
        raise ValueError(f"fault injector changed the client column set: missing={sorted(missing)} extra={sorted(extra)} (fail-closed)")
    faulted_frame = faulted_frame.reset_index(drop=True)

    # last_controlled_selected_clients_df (per_cluster_random mode) has one
    # row per CANDIDATE client with a "selected" boolean -- not pre-filtered.
    fault_selected = {
        (str(row.get("cluster_id")), str(row.get("client_id")))
        for _, row in selected_clients_df.iterrows()
        if bool(row.get("selected", False))
    } if not selected_clients_df.empty and "selected" in selected_clients_df.columns else set()

    n_windows = len(eligible_window_positions)
    X_observed: dict[tuple[str, str], np.ndarray] = {}
    for key, X in X_drift.items():
        cluster_id, client_id = key
        observed = np.asarray(X, dtype=float).copy()
        faulted_eligible = faulted_frame[client_id].to_numpy(dtype=float).reshape(n_windows, window_size, *observed.shape[2:])
        observed[eligible_window_positions] = faulted_eligible
        _tensor_stats_finite_check(f"X_observed[{key}]", observed)
        X_observed[key] = observed

    realizations = _map_realizations_to_real_coordinates(
        report_df, eligible_window_positions=eligible_window_positions, window_size=window_size,
        segment_by_window=segment_by_window, target_index_by_window=target_index_by_window, timestamps=timestamps,
    )

    identity = {
        "fault_config_path": str(fault_config_path),
        "fault_config_sha256": _hash_yaml_file(fault_config_path),
        "fault_selection_seed": int(fault_selection_seed),
        "window_size": int(window_size),
        "eligible_window_count": int(n_windows),
        "granularity_override": granularity_override,
        "note": (
            "faults are injected on raw-scale, already-materialized forecasting windows, one "
            "non-overlapping candidate window per real 24-step window; the only override versus "
            "the shared config file is window_size/stride (mechanical granularity alignment) -- "
            "total_faulty_window_fraction and every dimension_window_fraction, including "
            "completeness, are used exactly as configured"
        ),
    }

    return {
        "X_observed": X_observed,
        "fault_selected": fault_selected,
        "selected_clients_frame": selected_clients_df,
        "window_summary_frame": window_summary_df,
        "realizations_frame": realizations,
        "identity": identity,
    }


def _map_realizations_to_real_coordinates(
    report_df: pd.DataFrame, *, eligible_window_positions: np.ndarray, window_size: int,
    segment_by_window: np.ndarray, target_index_by_window: np.ndarray, timestamps: np.ndarray,
) -> pd.DataFrame:
    """Converts the injector's synthetic-column-local positions into real
    experiment coordinates. Synthetic timestamps (start_time/end_time/
    affected_timestamps, meaningless outside the synthetic concatenated
    column) are deliberately dropped, never persisted.
    """
    if report_df.empty or "window_start_index" not in report_df.columns:
        return pd.DataFrame(columns=[
            "cluster_id", "client_id", "deployment_window_index", "segment",
            "forecast_target_index_raw", "forecast_target_timestamp",
            "fault_position_within_window", "fault_dimension", "fault_type",
        ])

    rows: list[dict[str, Any]] = []
    for _, event in report_df.iterrows():
        window_start = event.get("window_start_index")
        if pd.isna(window_start):
            continue  # summary rows (final_net_corruption) carry no window at all
        column_local_window_index = int(window_start) // window_size
        if column_local_window_index >= len(eligible_window_positions):
            continue
        deployment_window_index = int(eligible_window_positions[column_local_window_index])
        segment = str(segment_by_window[deployment_window_index])
        forecast_target_index_raw = int(target_index_by_window[deployment_window_index])
        forecast_target_timestamp = str(pd.Timestamp(timestamps[forecast_target_index_raw]))

        try:
            positions = json.loads(event.get("affected_positions")) if isinstance(event.get("affected_positions"), str) else None
        except (TypeError, json.JSONDecodeError):
            positions = None
        window_start_int = int(window_start)
        fault_positions_within_window = (
            [int(p) - window_start_int for p in positions] if positions else [None]
        )
        for position_within_window in fault_positions_within_window:
            rows.append({
                "cluster_id": event.get("cluster_id"), "client_id": event.get("client_id"),
                "deployment_window_index": deployment_window_index, "segment": segment,
                "forecast_target_index_raw": forecast_target_index_raw,
                "forecast_target_timestamp": forecast_target_timestamp,
                "fault_position_within_window": position_within_window,
                "fault_dimension": event.get("dimension"), "fault_type": event.get("fault_type"),
            })
    return pd.DataFrame(rows)


def make_model_consumable(
    X_observed: np.ndarray, clean_training_mean_raw: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic missing-value handling, on the raw scale, applied
    AFTER fault injection and BEFORE normalization.

    Step 1: forward-fill NaN left-to-right, within each window
    independently (never across windows, never using values from a
    different window).
    Step 2: any remaining leading NaN (no valid value yet seen in that
    window) falls back to clean_training_mean_raw -- the client's own
    fitted raw-scale clean-training mean (never the true drifted value,
    never a global constant).

    X_observed is never mutated -- a fresh array is always returned.
    """
    X_observed = np.asarray(X_observed, dtype=float)
    original_shape = X_observed.shape
    flat = X_observed.reshape(original_shape[0], -1).copy()
    n_rows, n_cols = flat.shape

    missing_before = int(np.isnan(flat).sum())
    valid_mask = ~np.isnan(flat)
    idx = np.where(valid_mask, np.arange(n_cols)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    row_indices = np.arange(n_rows)[:, None]
    filled = flat[row_indices, idx]
    seen_valid = np.maximum.accumulate(valid_mask.astype(np.int8), axis=1).astype(bool)
    still_missing = ~seen_valid

    forward_filled_count = int(np.isnan(flat).sum()) - int(still_missing.sum())
    leading_fallback_count = int(still_missing.sum())

    result_flat = np.where(still_missing, float(clean_training_mean_raw), filled)
    X_model = result_flat.reshape(original_shape)
    missing_after = int(np.isnan(X_model).sum()) + int(np.isinf(X_model).sum())
    if missing_after != 0:
        raise AssertionError(f"make_model_consumable produced {missing_after} non-finite values -- fail-closed")

    metadata = {
        "missing_values_before_imputation": missing_before,
        "forward_filled_count": forward_filled_count,
        "leading_fallback_count": leading_fallback_count,
        "missing_values_after_imputation": int(np.isnan(X_model).sum()),
        "clean_training_mean_raw": float(clean_training_mean_raw),
    }
    return X_model, metadata
