#!/usr/bin/env python3
import os
import math
import pickle
import sys
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.comm_logging import append_comm_row, tensor_shape
from utils.runtime_utils import build_quality_anchor_config, derive_dq_mode, quality_scores_needed
from drift.artifacts import CsvArtifactWriter, write_csv, write_json
from drift.contracts import ClientSelectionRecord, DetectorEvent, DetectorKey, DetectorName, DetectorParameters, DriftConfig, DriftManifestRow, DriftSource, DriftType, InjectionRecord, TransitionFunction
from drift.detectors import DetectorRegistry
from drift.injector import derive_client_seed, inject_series
from drift.configuration import load_drift_config
from drift.pipeline import inject_timeline
from drift.schedule import resolve_experiment_schedule
from drift.windowing import fit_clean_training_scalers

app = Flask(__name__)

DEFAULT_CLUSTER_DIR = os.path.join(BASE_DIR, "dataset", "output")
CLUSTER_DIR = os.environ.get("CLUSTER_DIR", DEFAULT_CLUSTER_DIR)
TRAIN_CLUSTER_DATA_DIR = os.environ.get("TRAIN_CLUSTER_DATA_DIR", CLUSTER_DIR)
VAL_CLUSTER_DATA_DIR = os.environ.get("VAL_CLUSTER_DATA_DIR", CLUSTER_DIR)
TEST_CLUSTER_DATA_DIR = os.environ.get("TEST_CLUSTER_DATA_DIR", CLUSTER_DIR)
TRAIN_TARGET_CLUSTER_DATA_DIR = os.environ.get("TRAIN_TARGET_CLUSTER_DATA_DIR", TRAIN_CLUSTER_DATA_DIR)
VAL_TARGET_CLUSTER_DATA_DIR = os.environ.get("VAL_TARGET_CLUSTER_DATA_DIR", VAL_CLUSTER_DATA_DIR)
TEST_TARGET_CLUSTER_DATA_DIR = os.environ.get("TEST_TARGET_CLUSTER_DATA_DIR", TEST_CLUSTER_DATA_DIR)
MODE_CLUSTER_DIRS = {
    "train": TRAIN_CLUSTER_DATA_DIR,
    "val": VAL_CLUSTER_DATA_DIR,
    "test": TEST_CLUSTER_DATA_DIR,
}
MODE_TARGET_CLUSTER_DIRS = {
    "train": TRAIN_TARGET_CLUSTER_DATA_DIR,
    "val": VAL_TARGET_CLUSTER_DATA_DIR,
    "test": TEST_TARGET_CLUSTER_DATA_DIR,
}

STREAM_CLUSTER_ID = os.environ.get("STREAM_CLUSTER_ID")
STREAM_PORT = int(os.environ.get("STREAM_PORT", "5001"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "24"))
FORECAST_HORIZON = int(os.environ.get("FORECAST_HORIZON", "1"))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.7"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.15"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.15"))
NORMALIZE = os.environ.get("NORMALIZE", "false").lower() in ("1", "true", "yes", "y")
NORMALIZE_METHOD = os.environ.get("NORMALIZE_METHOD", "zscore").lower()
NORMALIZE_EPS = float(os.environ.get("NORMALIZE_EPS", "1e-8"))
COMM_LOG_STREAM = os.environ.get("COMM_LOG_STREAM", "false").lower() in ("1", "true", "yes", "y")
USE_STATIC_FEATURES = os.environ.get("USE_STATIC_FEATURES", "false").lower() in ("1", "true", "yes", "y")
STATIC_FEATURE_COLUMNS_RAW = os.environ.get("STATIC_FEATURE_COLUMNS", "")
STATIC_FEATURE_GROUPS_RAW = os.environ.get("STATIC_FEATURE_GROUPS", "")
USE_QUALITY_WEIGHTS = os.environ.get("USE_QUALITY_WEIGHTS", "false").lower() in ("1", "true", "yes", "y")
USE_QUALITY_AS_INPUT = os.environ.get("USE_QUALITY_AS_INPUT", "false").lower() in ("1", "true", "yes", "y")
USE_QUALITY_AGGREGATION_WEIGHTS = os.environ.get("USE_QUALITY_AGGREGATION_WEIGHTS", "false").lower() in ("1", "true", "yes", "y")
QUALITY_ANCHOR_CONFIG = build_quality_anchor_config(os.environ)
USE_QUALITY_ANCHOR_AGGREGATION = QUALITY_ANCHOR_CONFIG.use_quality_anchor_aggregation
USE_GRADIENT_ALIGNMENT_FILTER = QUALITY_ANCHOR_CONFIG.use_gradient_alignment_filter
USE_HIGH_QUALITY_VALIDATION_GUARDRAIL = QUALITY_ANCHOR_CONFIG.use_high_quality_validation_guardrail
QUALITY_SCORE_PATH = str(os.environ.get("QUALITY_SCORE_PATH", "")).strip()
QUALITY_INPUT_COLUMNS_RAW = str(os.environ.get("QUALITY_INPUT_COLUMNS", "quality_score")).strip()
QUALITY_INPUT_COLUMNS = [col.strip() for col in QUALITY_INPUT_COLUMNS_RAW.split(",") if col.strip()] or ["quality_score"]
QUALITY_WEIGHT_COLUMN = str(os.environ.get("QUALITY_WEIGHT_COLUMN", "quality_score")).strip() or "quality_score"
QUALITY_SCORE_GRANULARITY = str(os.environ.get("QUALITY_SCORE_GRANULARITY", "window")).strip().lower()
STRICT_QUALITY_SCORE_MATCHING = os.environ.get("STRICT_QUALITY_SCORE_MATCHING", "true").lower() in ("1", "true", "yes", "y")
QUALITY_SCORE_FALLBACK = float(os.environ.get("QUALITY_SCORE_FALLBACK", "1.0"))
DQ_MODE = derive_dq_mode(
    USE_QUALITY_WEIGHTS,
    USE_QUALITY_AS_INPUT,
    USE_QUALITY_AGGREGATION_WEIGHTS,
    USE_QUALITY_ANCHOR_AGGREGATION,
    USE_GRADIENT_ALIGNMENT_FILTER,
    USE_HIGH_QUALITY_VALIDATION_GUARDRAIL,
)
QUALITY_NEEDED = quality_scores_needed(
    USE_QUALITY_WEIGHTS,
    USE_QUALITY_AS_INPUT,
    USE_QUALITY_AGGREGATION_WEIGHTS,
    USE_QUALITY_ANCHOR_AGGREGATION,
)
MODEL_INPUT_SIZE_RAW = str(os.environ.get("MODEL_INPUT_SIZE", "")).strip()
DRIFT_INJECTION_ENABLED = os.environ.get("DRIFT_INJECTION_ENABLED", "false").lower() in ("1", "true", "yes", "y")
DRIFT_DETECTION_ENABLED = os.environ.get("DRIFT_DETECTION_ENABLED", "false").lower() in ("1", "true", "yes", "y")
DRIFT_CONFIG_PATH = str(os.environ.get("DRIFT_CONFIG_PATH", "")).strip()
DRIFT_SOURCE_RAW = os.environ.get("DRIFT_SOURCE", "input").strip().lower()
DRIFT_SCOPE = os.environ.get("DRIFT_SCOPE", "test").strip().lower()
DRIFT_RUN_ID = os.environ.get("DRIFT_RUN_ID", "drift-validation").strip()
DRIFT_CONFIGURATION_ID = os.environ.get("DRIFT_CONFIGURATION_ID", "default").strip()
DRIFT_TYPE_RAW = os.environ.get("DRIFT_TYPE", "sudden").strip().lower()
DRIFT_TRANSITION_RAW = os.environ.get("DRIFT_TRANSITION", "linear").strip().lower()
DRIFT_MAGNITUDE = float(os.environ.get("DRIFT_MAGNITUDE", "0.5"))
DRIFT_OFFSET_DELTA = float(os.environ.get("DRIFT_OFFSET_DELTA", "0.0"))
DRIFT_NOISE_SIGMA = float(os.environ.get("DRIFT_NOISE_SIGMA", "0.0"))
DRIFT_START_RATIO = float(os.environ.get("DRIFT_START_RATIO", "0.5"))
DRIFT_END_RATIO_RAW = os.environ.get("DRIFT_END_RATIO", "").strip()
DRIFT_END_RATIO = float(DRIFT_END_RATIO_RAW) if DRIFT_END_RATIO_RAW else None
DRIFT_PERIOD = int(os.environ.get("DRIFT_PERIOD", "168"))
DRIFT_RANDOM_SEED = int(os.environ.get("DRIFT_RANDOM_SEED", "42"))
DRIFT_CLIENT_IDS = set(value.strip() for value in os.environ.get("DRIFT_CLIENT_IDS", "").split(",") if value.strip())
DRIFT_CLIENT_FRACTION = float(os.environ.get("DRIFT_CLIENT_FRACTION", "0.5"))
DRIFT_DETECTORS_RAW = os.environ.get("DRIFT_DETECTORS", "page_hinkley,adwin")
DRIFT_STREAM_NAME = "standardized_window_endpoint"
DRIFT_EVENT_BUFFER_SIZE = int(os.environ.get("DRIFT_EVENT_BUFFER_SIZE", "4096"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))
try:
    MODEL_INPUT_SIZE = int(MODEL_INPUT_SIZE_RAW) if MODEL_INPUT_SIZE_RAW else None
except ValueError:
    MODEL_INPUT_SIZE = None

if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")
if QUALITY_SCORE_GRANULARITY != "window":
    raise ValueError("Only QUALITY_SCORE_GRANULARITY=window is supported in the current runtime.")

client_series: Dict[Tuple[str, str, int], torch.Tensor] = {}
client_timestamps: Dict[Tuple[str, str, int], List[pd.Timestamp]] = {}
client_frequency_deltas: Dict[Tuple[str, str, int], pd.Timedelta] = {}
client_target_series: Dict[Tuple[str, str, int], torch.Tensor] = {}
client_target_timestamps: Dict[Tuple[str, str, int], List[pd.Timestamp]] = {}
client_target_lookup: Dict[Tuple[str, str, int], Dict[pd.Timestamp, torch.Tensor]] = {}
client_static_features: Dict[Tuple[str, str, int], torch.Tensor] = {}
client_norm_stats: Dict[Tuple[str, int], Tuple[float, float]] = {}
client_split_bounds: Dict[Tuple[str, str, int], Dict[str, Tuple[int, int]]] = {}
client_window_counts: Dict[Tuple[str, str, int], Dict[str, int]] = {}
client_input_sizes: Dict[Tuple[str, int], int] = {}
client_offsets: Dict[Tuple[str, int, str], int] = {}
client_runtime_series: Dict[Tuple[str, str, int], torch.Tensor] = {}
client_schema_v2_scalers: Dict[Tuple[str, int], Any] = {}
schema_v2_config: Optional[Any] = None
schema_v2_resolved: Optional[Any] = None
drift_injection_records: Dict[Tuple[str, str, int], Dict[int, InjectionRecord]] = {}
drift_selected_clients: set[Tuple[str, int]] = set()
drift_detector_registry: Optional[DetectorRegistry] = None
drift_event_writer: Optional[CsvArtifactWriter] = None
drift_event_counts: Dict[Tuple[str, str], int] = {}
quality_score_index: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
quality_score_summary: Dict[str, Any] = {
    "enabled": QUALITY_NEEDED,
    "loaded": False,
    "path": QUALITY_SCORE_PATH or None,
    "quality_score_path": QUALITY_SCORE_PATH or None,
    "use_quality_weights": USE_QUALITY_WEIGHTS,
    "use_quality_as_input": USE_QUALITY_AS_INPUT,
    "use_quality_aggregation_weights": USE_QUALITY_AGGREGATION_WEIGHTS,
    "use_quality_anchor_aggregation": USE_QUALITY_ANCHOR_AGGREGATION,
    "use_gradient_alignment_filter": USE_GRADIENT_ALIGNMENT_FILTER,
    "use_high_quality_validation_guardrail": USE_HIGH_QUALITY_VALIDATION_GUARDRAIL,
    "quality_needed": QUALITY_NEEDED,
    "dq_mode": DQ_MODE,
    "granularity": QUALITY_SCORE_GRANULARITY,
    "quality_input_columns": QUALITY_INPUT_COLUMNS if USE_QUALITY_AS_INPUT else [],
    "quality_input_dim": len(QUALITY_INPUT_COLUMNS) if USE_QUALITY_AS_INPUT else 0,
    "quality_weight_column": QUALITY_WEIGHT_COLUMN,
}
offset_lock = Lock()


def _stream_comm_log_path(cluster_id: str | None) -> str:
    target_cluster = str(cluster_id) if cluster_id not in (None, "") else "global"
    return os.path.join(BASE_DIR, "output", f"cluster_{target_cluster}", "stream_comm.csv")


def _append_stream_comm(row: Dict[str, object], cluster_id: str | None) -> None:
    if not COMM_LOG_STREAM:
        return
    append_comm_row(_stream_comm_log_path(cluster_id), row)


def _normalize_client_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _parse_optional_csv(raw: str | None) -> List[str]:
    if raw in (None, ""):
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _feature_columns_for_groups(available_columns: list[str], groups: list[str]) -> list[str]:
    normalized_groups = [group.strip().lower() for group in groups if group.strip()]
    if not normalized_groups:
        return []
    if "all" in normalized_groups:
        return list(available_columns)

    selected: list[str] = []
    for column in available_columns:
        if "numeric" in normalized_groups and column.startswith("meta_num_"):
            selected.append(column)
            continue
        if "category" in normalized_groups and column.startswith("meta_cat_categoria_"):
            selected.append(column)
            continue
        if "subcategory" in normalized_groups and column.startswith("meta_cat_subcategoria_"):
            selected.append(column)
            continue
        if "categorical" in normalized_groups and column.startswith("meta_cat_"):
            selected.append(column)
            continue
    return selected


def _resolve_static_feature_columns(available_columns: list[str]) -> list[str]:
    if not USE_STATIC_FEATURES:
        return []
    requested_columns = _parse_optional_csv(STATIC_FEATURE_COLUMNS_RAW)
    if not requested_columns:
        requested_groups = _parse_optional_csv(STATIC_FEATURE_GROUPS_RAW)
        if requested_groups:
            selected_columns = _feature_columns_for_groups(available_columns, requested_groups)
            if not selected_columns:
                raise ValueError(
                    "STATIC_FEATURE_GROUPS did not match any available columns: "
                    + ", ".join(requested_groups)
                )
            return selected_columns
        return list(available_columns)
    missing = [col for col in requested_columns if col not in available_columns]
    if missing:
        raise ValueError(
            "Requested STATIC_FEATURE_COLUMNS were not found in the cluster payload: "
            + ", ".join(missing)
        )
    return requested_columns


def _series_key(mode: str, cluster_id: str, client_id: int) -> Tuple[str, str, int]:
    return (str(mode), str(cluster_id), int(client_id))


def _client_key(cluster_id: str, client_id: int) -> Tuple[str, int]:
    return (str(cluster_id), int(client_id))


def _num_windows(num_values: int) -> int:
    return max(0, num_values - WINDOW_SIZE - FORECAST_HORIZON + 1)


def _get_split_bounds(num_values: int) -> Dict[str, Tuple[int, int]]:
    total_windows = _num_windows(num_values)
    train_windows = int(total_windows * TRAIN_RATIO)
    val_windows = int(total_windows * VAL_RATIO)
    test_windows = total_windows - train_windows - val_windows
    train_end = train_windows
    val_end = train_end + val_windows
    test_end = val_end + test_windows
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, test_end)}


def _compute_norm_stats(values: torch.Tensor) -> Tuple[float, float]:
    split_bounds = _get_split_bounds(len(values))
    _, train_end = split_bounds["train"]
    if train_end <= 0:
        return 0.0, 1.0
    # Training window starts occupy [0, train_end). The final training target
    # is therefore at train_end + WINDOW_SIZE + FORECAST_HORIZON - 2, making
    # the following value the exclusive slice boundary. Advancing one element
    # further would include the target of the first validation window.
    train_value_end = min(
        len(values),
        train_end + WINDOW_SIZE + FORECAST_HORIZON - 1,
    )
    train_values = values[:train_value_end]
    if train_values.numel() == 0:
        return 0.0, 1.0
    if NORMALIZE_METHOD == "minmax":
        min_v = float(train_values.min().item())
        max_v = float(train_values.max().item())
        return min_v, max(max_v - min_v, NORMALIZE_EPS)
    mean = float(train_values.mean().item())
    scale = float(train_values.std().item())
    if scale < NORMALIZE_EPS:
        scale = 1.0
    return mean, scale


def _normalize_tensor(tensor: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
    return (tensor - mean) / scale


def _infer_frequency_delta(timestamps: List[pd.Timestamp]) -> pd.Timedelta:
    valid = [pd.Timestamp(ts) for ts in timestamps if not pd.isna(ts)]
    if len(valid) < 2:
        raise ValueError("Cannot infer timestamp frequency from fewer than two valid timestamps")
    deltas = []
    previous = valid[0]
    for current in valid[1:]:
        delta = current - previous
        if delta > pd.Timedelta(0):
            deltas.append(delta)
        previous = current
    if not deltas:
        raise ValueError("Cannot infer positive timestamp frequency from input series")
    return pd.Series(deltas).median()


def _target_timestamp_for_window(timestamps: List[pd.Timestamp], window_end_idx: int, frequency_delta: pd.Timedelta) -> pd.Timestamp:
    if window_end_idx >= len(timestamps) or pd.isna(timestamps[window_end_idx]):
        raise KeyError(f"Missing input timestamp at window_end_idx={window_end_idx}")
    return pd.Timestamp(timestamps[window_end_idx]) + (frequency_delta * FORECAST_HORIZON)


def _clean_optional_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _series_counts_for_client(cluster_id: str, client_id: int) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for mode in ("train", "val", "test"):
        series_key = _series_key(mode, cluster_id, client_id)
        mode_counts = client_window_counts.get(series_key)
        if mode_counts is None:
            continue
        counts[mode] = int(mode_counts.get(mode, 0))
    return counts


def _prepare_from_dataframe(
    df: pd.DataFrame,
    cluster_id: str,
    source_mode: str,
    *,
    timestamp_col: str = "Fecha_Hora",
    value_col: str = "value",
    client_metadata: pd.DataFrame | None = None,
    static_feature_columns: list[str] | None = None,
) -> None:
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col]).sort_values(["id", timestamp_col], kind="stable")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    feature_lookup: dict[str, torch.Tensor] = {}
    static_feature_columns = _resolve_static_feature_columns(list(static_feature_columns or []))
    if client_metadata is not None and not client_metadata.empty:
        meta = client_metadata.copy()
        meta["id"] = meta["id"].map(_normalize_client_id)
        selected_static_feature_columns = [col for col in static_feature_columns if col in meta.columns]
        if selected_static_feature_columns:
            meta[selected_static_feature_columns] = (
                meta[selected_static_feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            )
        for _, row in meta.iterrows():
            raw_id = row.get("id")
            if raw_id is None:
                continue
            normalized_id = _normalize_client_id(raw_id)
            if selected_static_feature_columns:
                values = row.loc[selected_static_feature_columns].to_numpy(dtype=float, copy=True)
                feature_lookup[normalized_id] = torch.as_tensor(values, dtype=torch.float32)
            else:
                feature_lookup[normalized_id] = torch.empty(0, dtype=torch.float32)

    for client_id, rows in df.groupby("id", sort=False):
        rows = rows.copy()
        rows[value_col] = rows[value_col].ffill().bfill()
        if rows[value_col].isna().any():
            raise ValueError(
                f"NaNs remain after fill for source_mode={source_mode}, cluster={cluster_id}, client={client_id}"
            )
        normalized_client_id = _normalize_client_id(client_id)
        client_int = int(normalized_client_id)
        mode_series_key = _series_key(source_mode, cluster_id, client_int)
        stable_client_key = _client_key(cluster_id, client_int)

        values = torch.as_tensor(rows[value_col].to_numpy(dtype=float), dtype=torch.float32)
        client_series[mode_series_key] = values.contiguous()
        if timestamp_col in rows.columns:
            timestamps = pd.to_datetime(rows[timestamp_col], errors="coerce").tolist()
        else:
            timestamps = [pd.NaT] * len(rows)
        client_timestamps[mode_series_key] = timestamps
        client_frequency_deltas[mode_series_key] = _infer_frequency_delta(timestamps)
        client_split_bounds[mode_series_key] = _get_split_bounds(len(values))
        client_window_counts[mode_series_key] = {
            k: max(0, end - start) for k, (start, end) in client_split_bounds[mode_series_key].items()
        }
        if source_mode == "train" or stable_client_key not in client_norm_stats:
            client_norm_stats[stable_client_key] = _compute_norm_stats(values)

        static_features = feature_lookup.get(normalized_client_id, torch.empty(0, dtype=torch.float32))
        current_input_size = 1 + int(static_features.numel())
        previous_input_size = client_input_sizes.get(stable_client_key)
        if previous_input_size is not None and previous_input_size != current_input_size:
            raise ValueError(
                f"Input feature mismatch for cluster={cluster_id}, client={client_int}: "
                f"existing={previous_input_size}, incoming={current_input_size}, source_mode={source_mode}"
            )
        client_static_features[mode_series_key] = static_features
        client_input_sizes[stable_client_key] = current_input_size


def _prepare_target_from_dataframe(
    df: pd.DataFrame,
    cluster_id: str,
    source_mode: str,
    *,
    timestamp_col: str = "Fecha_Hora",
    value_col: str = "value",
) -> None:
    if timestamp_col not in df.columns:
        raise ValueError(f"Target data for mode={source_mode}, cluster={cluster_id} must include {timestamp_col}")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(["id", timestamp_col], kind="stable")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    for client_id, rows in df.groupby("id", sort=False):
        rows = rows.copy()
        rows[value_col] = rows[value_col].ffill().bfill()
        if rows[value_col].isna().any():
            raise ValueError(
                f"NaNs remain after target fill for source_mode={source_mode}, cluster={cluster_id}, client={client_id}"
            )
        normalized_client_id = _normalize_client_id(client_id)
        client_int = int(normalized_client_id)
        mode_series_key = _series_key(source_mode, cluster_id, client_int)
        values = torch.as_tensor(rows[value_col].to_numpy(dtype=float), dtype=torch.float32)
        timestamps = pd.to_datetime(rows[timestamp_col], errors="coerce").tolist()
        lookup: Dict[pd.Timestamp, torch.Tensor] = {}
        for idx, timestamp in enumerate(timestamps):
            if pd.isna(timestamp):
                continue
            key = pd.Timestamp(timestamp)
            if key in lookup:
                raise ValueError(
                    f"Duplicate target timestamp for mode={source_mode}, cluster={cluster_id}, client={client_int}: {key}"
                )
            lookup[key] = values[idx]
        client_target_series[mode_series_key] = values.contiguous()
        client_target_timestamps[mode_series_key] = timestamps
        client_target_lookup[mode_series_key] = lookup


def _prepare_target_from_payload(payload: Any, cluster_id: str, source_mode: str) -> None:
    if isinstance(payload, pd.DataFrame):
        _prepare_target_from_dataframe(payload, cluster_id, source_mode)
        return
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported target cluster pickle payload type: {type(payload).__name__}")

    timeseries = payload.get("timeseries")
    if not isinstance(timeseries, pd.DataFrame):
        raise ValueError("Target cluster pickle payload must contain a pandas DataFrame under the timeseries key.")
    timestamp_col = str(payload.get("timestamp_col", "Fecha_Hora"))
    value_col = str(payload.get("value_col", "value"))
    _prepare_target_from_dataframe(
        timeseries,
        cluster_id,
        source_mode,
        timestamp_col=timestamp_col,
        value_col=value_col,
    )


def _prepare_from_payload(payload: Any, cluster_id: str, source_mode: str) -> None:
    if isinstance(payload, pd.DataFrame):
        _prepare_from_dataframe(payload, cluster_id, source_mode)
        return
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported cluster pickle payload type: {type(payload).__name__}")

    timeseries = payload.get("timeseries")
    if not isinstance(timeseries, pd.DataFrame):
        raise ValueError("Cluster pickle payload must contain a pandas DataFrame under 'timeseries'.")
    client_metadata = payload.get("client_metadata")
    if client_metadata is not None and not isinstance(client_metadata, pd.DataFrame):
        raise ValueError("Cluster pickle payload field 'client_metadata' must be a pandas DataFrame when present.")
    static_feature_columns = payload.get("static_feature_columns", []) or []
    timestamp_col = str(payload.get("timestamp_col", "Fecha_Hora"))
    value_col = str(payload.get("value_col", "value"))
    _prepare_from_dataframe(
        timeseries,
        cluster_id,
        source_mode,
        timestamp_col=timestamp_col,
        value_col=value_col,
        client_metadata=client_metadata,
        static_feature_columns=[str(col) for col in static_feature_columns],
    )


def _load_cluster_data_for_mode(source_mode: str, cluster_dir: str) -> None:
    loaded_any = False
    if os.path.isdir(cluster_dir):
        for file in sorted(os.listdir(cluster_dir)):
            if not file.endswith(".pkl"):
                continue
            cluster_id = file.split("_")[1].split(".")[0]
            if STREAM_CLUSTER_ID and str(cluster_id) != str(STREAM_CLUSTER_ID):
                continue
            payload = pd.read_pickle(os.path.join(cluster_dir, file))
            _prepare_from_payload(payload, str(cluster_id), source_mode)
            loaded_any = True
    if loaded_any:
        return

    if os.path.abspath(cluster_dir) != os.path.abspath(CLUSTER_DIR):
        raise FileNotFoundError(
            f"No cluster pickle files found for mode={source_mode} in {cluster_dir}. "
            "Run data/clustering_processing.py for that dataset first."
        )

    dataset_path = os.path.join(BASE_DIR, "dataset", "dataset.csv")
    assignment_path = os.path.join(BASE_DIR, "dataset", "cluster-assignment.csv")
    dataset = pd.read_csv(dataset_path)
    cluster_assignments = pd.read_csv(assignment_path)
    merged = dataset.melt(id_vars=["Fecha_Hora"], var_name="id", value_name="value")
    merged["id"] = merged["id"].astype(int)
    merged["Fecha_Hora"] = pd.to_datetime(merged["Fecha_Hora"], errors="coerce")
    merged = merged.merge(cluster_assignments, left_on="id", right_on="ID")
    merged = merged.sort_values(["Cluster", "id", "Fecha_Hora"], kind="stable")
    for cluster_id, rows in merged.groupby("Cluster", sort=False):
        if STREAM_CLUSTER_ID and str(cluster_id) != str(STREAM_CLUSTER_ID):
            continue
        _prepare_from_dataframe(rows.reset_index(drop=True), str(cluster_id), source_mode)


def _load_target_cluster_data_for_mode(source_mode: str, cluster_dir: str) -> None:
    loaded_any = False
    if os.path.isdir(cluster_dir):
        for file in sorted(os.listdir(cluster_dir)):
            if not file.endswith(".pkl"):
                continue
            cluster_id = file.split("_")[1].split(".")[0]
            if STREAM_CLUSTER_ID and str(cluster_id) != str(STREAM_CLUSTER_ID):
                continue
            payload = pd.read_pickle(os.path.join(cluster_dir, file))
            _prepare_target_from_payload(payload, str(cluster_id), source_mode)
            loaded_any = True
    if loaded_any:
        return

    if os.path.abspath(cluster_dir) != os.path.abspath(CLUSTER_DIR):
        raise FileNotFoundError(
            f"No target cluster pickle files found for mode={source_mode} in {cluster_dir}. "
            "Run data/clustering_processing.py for that dataset first."
        )

    dataset_path = os.path.join(BASE_DIR, "dataset", "dataset.csv")
    assignment_path = os.path.join(BASE_DIR, "dataset", "cluster-assignment.csv")
    dataset = pd.read_csv(dataset_path)
    cluster_assignments = pd.read_csv(assignment_path)
    merged = dataset.melt(id_vars=["Fecha_Hora"], var_name="id", value_name="value")
    merged["id"] = merged["id"].astype(int)
    merged["Fecha_Hora"] = pd.to_datetime(merged["Fecha_Hora"], errors="coerce")
    merged = merged.merge(cluster_assignments, left_on="id", right_on="ID")
    merged = merged.sort_values(["Cluster", "id", "Fecha_Hora"], kind="stable")
    for cluster_id, rows in merged.groupby("Cluster", sort=False):
        if STREAM_CLUSTER_ID and str(cluster_id) != str(STREAM_CLUSTER_ID):
            continue
        _prepare_target_from_dataframe(rows.reset_index(drop=True), str(cluster_id), source_mode)


def load_cluster_data() -> None:
    for mode in ("train", "val", "test"):
        _load_cluster_data_for_mode(mode, MODE_CLUSTER_DIRS[mode])
        _load_target_cluster_data_for_mode(mode, MODE_TARGET_CLUSTER_DIRS[mode])


def _load_quality_score_index() -> None:
    global quality_score_summary
    if not QUALITY_NEEDED:
        quality_score_summary = {
            "enabled": False,
            "loaded": False,
            "path": QUALITY_SCORE_PATH or None,
            "quality_score_path": QUALITY_SCORE_PATH or None,
            "granularity": QUALITY_SCORE_GRANULARITY,
            "use_quality_weights": USE_QUALITY_WEIGHTS,
            "use_quality_as_input": USE_QUALITY_AS_INPUT,
            "use_quality_aggregation_weights": USE_QUALITY_AGGREGATION_WEIGHTS,
            "use_quality_anchor_aggregation": USE_QUALITY_ANCHOR_AGGREGATION,
            "use_gradient_alignment_filter": USE_GRADIENT_ALIGNMENT_FILTER,
            "use_high_quality_validation_guardrail": USE_HIGH_QUALITY_VALIDATION_GUARDRAIL,
            "quality_needed": False,
            "dq_mode": DQ_MODE,
            "quality_input_columns": QUALITY_INPUT_COLUMNS if USE_QUALITY_AS_INPUT else [],
            "quality_input_dim": len(QUALITY_INPUT_COLUMNS) if USE_QUALITY_AS_INPUT else 0,
            "quality_weight_column": QUALITY_WEIGHT_COLUMN,
        }
        return
    if not QUALITY_SCORE_PATH:
        raise ValueError(
            "USE_QUALITY_WEIGHTS=true, USE_QUALITY_AS_INPUT=true, or "
            "USE_QUALITY_AGGREGATION_WEIGHTS=true, or USE_QUALITY_ANCHOR_AGGREGATION=true "
            "requires QUALITY_SCORE_PATH to be set."
        )

    quality_path = os.path.join(BASE_DIR, QUALITY_SCORE_PATH) if not os.path.isabs(QUALITY_SCORE_PATH) else QUALITY_SCORE_PATH
    if not os.path.exists(quality_path):
        raise FileNotFoundError(f"Quality score CSV not found: {quality_path}")

    df = pd.read_csv(quality_path)
    required = {"client_id", "mode", "window_start_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Quality score CSV must contain columns {sorted(required)}; missing={missing}. "
            f"Path: {quality_path}"
        )
    score_column = "quality_score" if "quality_score" in df.columns else "y_final"
    if score_column not in df.columns:
        raise ValueError(
            f"Quality score CSV must contain either 'quality_score' or 'y_final'. Path: {quality_path}"
        )
    missing_input_columns = [col for col in QUALITY_INPUT_COLUMNS if col not in df.columns]
    if USE_QUALITY_AS_INPUT and missing_input_columns:
        raise ValueError(
            f"QUALITY_INPUT_COLUMNS contains missing column(s): {missing_input_columns}. Path: {quality_path}"
        )
    if USE_QUALITY_WEIGHTS and QUALITY_WEIGHT_COLUMN not in df.columns:
        raise ValueError(
            f"QUALITY_WEIGHT_COLUMN={QUALITY_WEIGHT_COLUMN!r} is not present in {quality_path}"
        )

    for _, row in df.iterrows():
        client_id = _normalize_client_id(row["client_id"])
        mode = str(row["mode"]).strip().lower()
        window_start_index = int(row["window_start_index"])
        key = (client_id, mode, window_start_index)
        if key in quality_score_index:
            raise ValueError(f"Duplicate quality score key found: {key} in {quality_path}")
        quality_score = float(row[score_column])
        quality_score_index[key] = {
            "quality_score": quality_score,
            "quality_input_values": [float(row[col]) for col in QUALITY_INPUT_COLUMNS],
            "quality_weight_score": float(row[QUALITY_WEIGHT_COLUMN]) if QUALITY_WEIGHT_COLUMN in row else quality_score,
            "window_start_index": window_start_index,
            "window_end_index": (None if "window_end_index" not in df.columns or pd.isna(row["window_end_index"]) else int(row["window_end_index"])),
            "window_start": (None if "window_start" not in df.columns or pd.isna(row["window_start"]) else str(row["window_start"])),
            "window_end": (None if "window_end" not in df.columns or pd.isna(row["window_end"]) else str(row["window_end"])),
            "dimension_completeness": _clean_optional_float(row.get("dimension_completeness", row.get("y_completeness"))),
            "dimension_validity": _clean_optional_float(row.get("dimension_validity", row.get("y_validity"))),
            "dimension_consistency": _clean_optional_float(row.get("dimension_consistency", row.get("y_consistency"))),
            "dimension_accuracy": _clean_optional_float(row.get("dimension_accuracy", row.get("y_accuracy"))),
            "dimension_freshness": _clean_optional_float(row.get("dimension_freshness", row.get("y_freshness"))),
        }

    quality_score_summary = {
        "enabled": True,
        "loaded": True,
        "path": quality_path,
        "quality_score_path": quality_path,
        "granularity": QUALITY_SCORE_GRANULARITY,
        "use_quality_weights": USE_QUALITY_WEIGHTS,
        "use_quality_as_input": USE_QUALITY_AS_INPUT,
        "use_quality_aggregation_weights": USE_QUALITY_AGGREGATION_WEIGHTS,
        "use_quality_anchor_aggregation": USE_QUALITY_ANCHOR_AGGREGATION,
        "use_gradient_alignment_filter": USE_GRADIENT_ALIGNMENT_FILTER,
        "use_high_quality_validation_guardrail": USE_HIGH_QUALITY_VALIDATION_GUARDRAIL,
        "quality_needed": True,
        "dq_mode": DQ_MODE,
        "row_count": int(len(df)),
        "client_count": int(df["client_id"].astype(str).nunique()),
        "modes": sorted({str(mode).strip().lower() for mode in df["mode"].dropna().astype(str)}),
        "quality_input_columns": QUALITY_INPUT_COLUMNS if USE_QUALITY_AS_INPUT else [],
        "quality_input_dim": len(QUALITY_INPUT_COLUMNS) if USE_QUALITY_AS_INPUT else 0,
        "quality_weight_column": QUALITY_WEIGHT_COLUMN,
    }


def _quality_record_for_window(client_id: int, mode: str, window_start_index: int) -> Dict[str, Any]:
    if not QUALITY_NEEDED:
        return {
            "quality_score": 1.0,
            "quality_input_values": [1.0 for _ in QUALITY_INPUT_COLUMNS],
            "quality_weight_score": 1.0,
            "match_status": "disabled",
        }

    key = (_normalize_client_id(client_id), str(mode), int(window_start_index))
    record = quality_score_index.get(key)
    if record is not None:
        return {**record, "match_status": "matched"}

    if mode == "train" and STRICT_QUALITY_SCORE_MATCHING:
        raise KeyError(
            f"Missing quality score for client_id={client_id}, mode={mode}, "
            f"window_start_index={window_start_index}"
        )

    return {
        "quality_score": QUALITY_SCORE_FALLBACK,
        "quality_input_values": [QUALITY_SCORE_FALLBACK for _ in QUALITY_INPUT_COLUMNS],
        "quality_weight_score": QUALITY_SCORE_FALLBACK,
        "match_status": "fallback",
    }


def _drift_visible_bounds(series_key: Tuple[str, str, int]) -> Tuple[int, int]:
    split_start, split_end = client_split_bounds[series_key]["test"]
    return split_start + WINDOW_SIZE - 1, split_end + WINDOW_SIZE - 1


def _resolve_ratio_index(start: int, end_exclusive: int, ratio: float, *, allow_endpoint: bool = False) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("drift ratios must be in [0, 1]")
    resolved = start + int(math.floor(ratio * (end_exclusive - start)))
    return min(resolved, end_exclusive if allow_endpoint else end_exclusive - 1)


def _drift_event_sink(event: DetectorEvent) -> None:
    if drift_event_writer is not None:
        drift_event_writer.append(event)
    key = (event.client_id, event.detector_name)
    drift_event_counts[key] = drift_event_counts.get(key, 0) + 1


def _initialize_drift_runtime() -> None:
    global drift_detector_registry, drift_event_writer
    if not (DRIFT_INJECTION_ENABLED or DRIFT_DETECTION_ENABLED):
        return
    if DRIFT_SOURCE_RAW != "input" or DRIFT_SCOPE != "test":
        raise ValueError("this milestone supports only DRIFT_SOURCE=input and DRIFT_SCOPE=test")
    if DRIFT_DETECTION_ENABLED and not NORMALIZE:
        raise ValueError("DRIFT_DETECTION_ENABLED requires NORMALIZE=true for standardized endpoint detection")
    if not DRIFT_RUN_ID or not DRIFT_CONFIGURATION_ID:
        raise ValueError("DRIFT_RUN_ID and DRIFT_CONFIGURATION_ID must be non-empty")
    if not 0.0 <= DRIFT_CLIENT_FRACTION <= 1.0:
        raise ValueError("DRIFT_CLIENT_FRACTION must be in [0, 1]")
    drift_type = DriftType.parse(DRIFT_TYPE_RAW)
    transition = TransitionFunction.parse(DRIFT_TRANSITION_RAW)
    candidates = sorted({(cluster_id, client_id) for mode, cluster_id, client_id in client_series if mode == "test"})
    requested = set(DRIFT_CLIENT_IDS)
    if not DRIFT_INJECTION_ENABLED:
        selected = set()
    elif requested:
        selected = {(cluster_id, client_id) for cluster_id, client_id in candidates if str(client_id) in requested or f"{cluster_id}:{client_id}" in requested}
    else:
        local_rng = __import__("numpy").random.default_rng(derive_client_seed(DRIFT_RANDOM_SEED, run_id=DRIFT_RUN_ID, cluster_id=str(STREAM_CLUSTER_ID or "all"), client_id="selection"))
        count = int(round(len(candidates) * DRIFT_CLIENT_FRACTION))
        chosen = local_rng.choice(len(candidates), size=count, replace=False).tolist() if count else []
        selected = {candidates[index] for index in chosen}
    drift_selected_clients.update(selected)

    drift_dir = os.path.join(OUTPUT_DIR, "drift")
    os.makedirs(drift_dir, exist_ok=True)
    manifest_rows = []
    if DRIFT_INJECTION_ENABLED and drift_type is not DriftType.NONE:
        for cluster_id, client_id in sorted(selected):
            key = _series_key("test", cluster_id, client_id)
            base = client_series[key]
            visible_start, visible_end = _drift_visible_bounds(key)
            start_index = _resolve_ratio_index(visible_start, visible_end, DRIFT_START_RATIO)
            end_index = None
            if drift_type in (DriftType.GRADUAL, DriftType.INCREMENTAL):
                if DRIFT_END_RATIO is None:
                    raise ValueError(f"{drift_type.value} drift requires DRIFT_END_RATIO")
                end_index = _resolve_ratio_index(visible_start, visible_end, DRIFT_END_RATIO, allow_endpoint=True)
            seed = derive_client_seed(DRIFT_RANDOM_SEED, run_id=DRIFT_RUN_ID, cluster_id=cluster_id, client_id=str(client_id))
            cfg = DriftConfig(
                run_id=DRIFT_RUN_ID, configuration_id=DRIFT_CONFIGURATION_ID, cluster_id=cluster_id, client_id=str(client_id),
                mode="test", scope="test", drift_type=drift_type, drift_source=DriftSource.INPUT,
                start_index=start_index, end_index=end_index, magnitude=DRIFT_MAGNITUDE, offset_delta=DRIFT_OFFSET_DELTA,
                noise_sigma=DRIFT_NOISE_SIGMA, period=DRIFT_PERIOD if drift_type is DriftType.RECURRING else None,
                transition=transition, random_seed=seed, requested_start_ratio=DRIFT_START_RATIO,
                requested_end_ratio=DRIFT_END_RATIO, segment_start_index=visible_start, segment_end_index_exclusive=visible_end,
            )
            injected = inject_series(base.cpu().numpy(), cfg, timestamps=client_timestamps[key])
            runtime = base.clone()
            runtime[visible_start:visible_end] = torch.as_tensor(injected.runtime_values[visible_start:visible_end], dtype=runtime.dtype)
            if not torch.equal(runtime[:visible_start], base[:visible_start]) or not torch.equal(runtime[visible_end:], base[visible_end:]):
                raise AssertionError("drift injection changed observations outside the detector-visible test segment")
            client_runtime_series[key] = runtime.contiguous()
            visible_records = tuple(injected.records[visible_start:visible_end])
            drift_injection_records[key] = {record.observation_index: record for record in visible_records}
            manifest_rows.extend(visible_records)
        write_csv(os.path.join(drift_dir, "injection_manifest.csv"), manifest_rows, InjectionRecord)

    if DRIFT_DETECTION_ENABLED:
        params = DetectorParameters(
            ph_min_instances=int(os.environ.get("PH_MIN_INSTANCES", "300")),
            ph_delta=float(os.environ.get("PH_DELTA", "0.005")), ph_threshold=float(os.environ.get("PH_THRESHOLD", "50.0")),
            ph_alpha=float(os.environ.get("PH_ALPHA", "0.9999")), adwin_delta=float(os.environ.get("ADWIN_DELTA", "0.002")),
            adwin_clock=int(os.environ.get("ADWIN_CLOCK", "32")), adwin_max_buckets=int(os.environ.get("ADWIN_MAX_BUCKETS", "5")),
            adwin_min_window_length=int(os.environ.get("ADWIN_MIN_WINDOW_LENGTH", "5")), adwin_grace_period=int(os.environ.get("ADWIN_GRACE_PERIOD", "10")),
        )
        names = tuple(DetectorName.parse(value) for value in DRIFT_DETECTORS_RAW.split(",") if value.strip())
        if not names:
            raise ValueError("DRIFT_DETECTORS must contain at least one detector")
        drift_event_writer = CsvArtifactWriter.for_dataclass(os.path.join(drift_dir, "detector_events.csv"), DetectorEvent)
        drift_event_writer.reset()
        drift_detector_registry = DetectorRegistry(params, max_buffered_events=DRIFT_EVENT_BUFFER_SIZE, event_sink=_drift_event_sink)
        app.config["DRIFT_DETECTOR_NAMES"] = names
    write_json(os.path.join(drift_dir, "resolved_config.json"), {
        "run_id": DRIFT_RUN_ID, "configuration_id": DRIFT_CONFIGURATION_ID, "source": DRIFT_SOURCE_RAW, "scope": DRIFT_SCOPE,
        "injection_enabled": DRIFT_INJECTION_ENABLED, "detection_enabled": DRIFT_DETECTION_ENABLED,
        "drift_type": drift_type.value, "transition": transition.value, "start_ratio": DRIFT_START_RATIO,
        "end_ratio": DRIFT_END_RATIO, "period": DRIFT_PERIOD, "random_seed": DRIFT_RANDOM_SEED,
        "selected_clients": [{"cluster_id": cluster, "client_id": client} for cluster, client in sorted(selected)],
        "clean_clients": [{"cluster_id": cluster, "client_id": client} for cluster, client in candidates if (cluster, client) not in selected],
        "coordinate_system": "absolute observation index; detector-visible test endpoints",
    })


def _initialize_schema_v2_drift_runtime() -> None:
    """Wire the source-aware timeline drift library (drift.pipeline.inject_timeline)
    into the runtime stream, independent of the legacy v1 DRIFT_TYPE injector above.
    """
    global schema_v2_config, schema_v2_resolved
    drift_dir = os.path.join(OUTPUT_DIR, "drift")
    if not (DRIFT_CONFIG_PATH or DRIFT_INJECTION_ENABLED or DRIFT_DETECTION_ENABLED):
        os.makedirs(drift_dir, exist_ok=True)
        write_json(os.path.join(drift_dir, "resolved_config_v2.json"), {
            "schema_version": None,
            "injection_enabled": False,
            "detection_enabled": False,
            "drift_config_path": None,
        })
        return
    if not DRIFT_CONFIG_PATH:
        return

    config = load_drift_config(DRIFT_CONFIG_PATH)
    client_timelines: Dict[Tuple[str, str], Tuple[List[int], List[pd.Timestamp]]] = {}
    for mode, cluster_id, client_id in list(client_series.keys()):
        if mode != "test":
            continue
        series_key = _series_key(mode, cluster_id, client_id)
        length = len(client_series[series_key])
        client_timelines[(str(cluster_id), str(client_id))] = (
            list(range(length)), client_timestamps[series_key],
        )
    resolved = resolve_experiment_schedule(config, client_timelines)
    schema_v2_config, schema_v2_resolved = config, resolved

    os.makedirs(drift_dir, exist_ok=True)
    all_manifest_rows: list[DriftManifestRow] = []
    normalization_entries: list[dict] = []
    for selection in resolved.selections:
        if not selection.affected:
            continue
        cluster_id, client_id_str = selection.cluster_id, selection.client_id
        client_id = int(client_id_str)
        series_key = _series_key("test", cluster_id, client_id)
        clean_values = client_series[series_key].cpu().numpy()
        timestamps = client_timestamps[series_key]
        result = inject_timeline(
            clean_values, timestamps=timestamps, config=config, resolved_schedule=resolved,
            cluster_id=cluster_id, client_id=client_id_str,
        )
        client_runtime_series[series_key] = torch.as_tensor(
            result.observed_timeline[:, 0], dtype=torch.float32
        ).contiguous()
        client_target_lookup[series_key] = {
            pd.Timestamp(timestamp): torch.tensor(float(result.target_timeline[index, 0]), dtype=torch.float32)
            for index, timestamp in enumerate(timestamps) if not pd.isna(timestamp)
        }
        all_manifest_rows.extend(result.manifest)

        train_key = _series_key("train", cluster_id, client_id)
        if train_key in client_series:
            train_values = client_series[train_key].cpu().numpy()
            train_start, train_end = client_split_bounds[train_key]["train"]
            scaler = fit_clean_training_scalers(
                train_values, train_values, training_window_starts=range(train_start, train_end),
                window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON, target_feature_index=0,
                method=NORMALIZE_METHOD, share_target_scaler=True,
            )
            client_schema_v2_scalers[_client_key(cluster_id, client_id)] = scaler
            normalization_entries.append({
                "cluster_id": cluster_id, "client_id": client_id_str,
                "target_scaling_strategy": scaler.target_scaling_strategy,
                "method": scaler.method,
                "x_location": list(scaler.x_location), "x_scale": list(scaler.x_scale),
                "y_location": scaler.y_location, "y_scale": scaler.y_scale,
                "shared_target_indices": list(scaler.shared_target_indices),
                "shared_target_union_hash": scaler.shared_target_union_hash,
            })

    if all_manifest_rows:
        write_csv(os.path.join(drift_dir, "component_manifest.csv"), all_manifest_rows, DriftManifestRow)
    write_csv(
        os.path.join(drift_dir, "client_selection_manifest.csv"),
        list(resolved.selections), ClientSelectionRecord,
    )
    write_json(os.path.join(drift_dir, "normalization_manifest.json"), {"clients": normalization_entries})
    write_json(os.path.join(drift_dir, "resolved_config_v2.json"), {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_id,
        "drift_source": config.drift_source.value,
        "injection_enabled": True,
        "detection_enabled": DRIFT_DETECTION_ENABLED,
        "drift_config_path": DRIFT_CONFIG_PATH,
        "config_hash": config.config_hash,
        "resolved_schedule_hash": resolved.resolved_schedule_hash,
        "affected_clients": [
            {"cluster_id": item.cluster_id, "client_id": item.client_id}
            for item in resolved.selections if item.affected
        ],
    })


load_cluster_data()
_load_quality_score_index()
_initialize_drift_runtime()
_initialize_schema_v2_drift_runtime()


@app.route("/drift/config", methods=["GET"])
def drift_config():
    if schema_v2_config is None or schema_v2_resolved is None:
        return jsonify({"error": "no schema-v2 drift configuration loaded"}), 404
    return jsonify({
        "schema_version": schema_v2_config.schema_version,
        "experiment_id": schema_v2_config.experiment_id,
        "drift_source": schema_v2_config.drift_source.value,
        "config_hash": schema_v2_config.config_hash,
        "resolved_schedule_hash": schema_v2_resolved.resolved_schedule_hash,
    })


@app.route("/health", methods=["GET"])
def health():
    cluster_ids = sorted({cluster_id for _, cluster_id, _ in client_series.keys()})
    input_sizes = sorted({input_size for input_size in client_input_sizes.values()})
    loaded_modes = sorted({mode for mode, _, _ in client_series.keys()})
    return jsonify(
        {
            "status": "healthy",
            "clusters_loaded": len(cluster_ids),
            "cluster_ids": cluster_ids,
            "clients_loaded": len(client_input_sizes),
            "loaded_modes": loaded_modes,
            "window_size": WINDOW_SIZE,
            "forecast_horizon": FORECAST_HORIZON,
            "normalize": NORMALIZE,
            "normalize_method": NORMALIZE_METHOD,
            "use_static_features": USE_STATIC_FEATURES,
            "use_quality_as_input": USE_QUALITY_AS_INPUT,
            "use_quality_weights": USE_QUALITY_WEIGHTS,
            "use_quality_aggregation_weights": USE_QUALITY_AGGREGATION_WEIGHTS,
            "use_quality_anchor_aggregation": USE_QUALITY_ANCHOR_AGGREGATION,
            "use_gradient_alignment_filter": USE_GRADIENT_ALIGNMENT_FILTER,
            "use_high_quality_validation_guardrail": USE_HIGH_QUALITY_VALIDATION_GUARDRAIL,
            "dq_mode": DQ_MODE,
            "quality_needed": QUALITY_NEEDED,
            "requested_static_feature_groups": _parse_optional_csv(STATIC_FEATURE_GROUPS_RAW),
            "requested_static_feature_columns": _parse_optional_csv(STATIC_FEATURE_COLUMNS_RAW),
            "input_sizes": input_sizes,
            "base_input_sizes": input_sizes,
            "stream_input_sizes": input_sizes,
            "model_input_size": MODEL_INPUT_SIZE,
            "quality_input_enabled": USE_QUALITY_AS_INPUT,
            "quality_weights_enabled": USE_QUALITY_WEIGHTS,
            "quality_aggregation_weights_enabled": USE_QUALITY_AGGREGATION_WEIGHTS,
            "quality_score_path": quality_score_summary.get("path"),
            "cluster_data_dirs": MODE_CLUSTER_DIRS,
            "target_cluster_data_dirs": MODE_TARGET_CLUSTER_DIRS,
            "quality_scores": quality_score_summary,
        }
    )


@app.route("/stream/norm_stats", methods=["GET"])
def stream_norm_stats():
    t0 = time.perf_counter()
    client_id = int(request.args["client_id"])
    cluster_id = str(request.args["cluster_id"])
    key = _client_key(cluster_id, client_id)
    if key not in client_norm_stats:
        return jsonify({"error": f"Unknown cluster/client: {cluster_id}/{client_id}"}), 404
    mean, scale = client_norm_stats[key]
    counts = _series_counts_for_client(cluster_id, client_id)
    base_input_size = int(client_input_sizes.get(key, 1))
    static_feature_count = int(
        client_static_features.get(_series_key("train", cluster_id, client_id), torch.empty(0)).numel()
    )
    response = {
        "cluster_id": cluster_id,
        "client_id": client_id,
        "normalize": NORMALIZE,
        "method": NORMALIZE_METHOD,
        "mean": mean,
        "scale": scale,
        "use_static_features": USE_STATIC_FEATURES,
        "use_quality_as_input": USE_QUALITY_AS_INPUT,
        "use_quality_weights": USE_QUALITY_WEIGHTS,
        "use_quality_anchor_aggregation": USE_QUALITY_ANCHOR_AGGREGATION,
        "use_gradient_alignment_filter": USE_GRADIENT_ALIGNMENT_FILTER,
        "use_high_quality_validation_guardrail": USE_HIGH_QUALITY_VALIDATION_GUARDRAIL,
        "dq_mode": DQ_MODE,
        "quality_needed": QUALITY_NEEDED,
        "requested_static_feature_groups": _parse_optional_csv(STATIC_FEATURE_GROUPS_RAW),
        "requested_static_feature_columns": _parse_optional_csv(STATIC_FEATURE_COLUMNS_RAW),
        "input_size": base_input_size,
        "base_input_size": base_input_size,
        "stream_input_size": base_input_size,
        "model_input_size": MODEL_INPUT_SIZE,
        "static_feature_count": static_feature_count,
        "num_train_windows": counts["train"],
        "num_val_windows": counts["val"],
        "num_test_windows": counts["test"],
        "quality_input_enabled": USE_QUALITY_AS_INPUT,
        "quality_weights_enabled": USE_QUALITY_WEIGHTS,
        "quality_score_path": quality_score_summary.get("path"),
        "quality_score_granularity": QUALITY_SCORE_GRANULARITY,
        "quality_input_columns": QUALITY_INPUT_COLUMNS if USE_QUALITY_AS_INPUT else [],
        "quality_input_dim": len(QUALITY_INPUT_COLUMNS) if USE_QUALITY_AS_INPUT else 0,
        "quality_weight_column": QUALITY_WEIGHT_COLUMN,
        "cluster_data_dirs": MODE_CLUSTER_DIRS,
        "target_cluster_data_dirs": MODE_TARGET_CLUSTER_DIRS,
    }
    schema_v2_scaler = client_schema_v2_scalers.get(key)
    if schema_v2_scaler is not None:
        response.update({
            "target_scaling_strategy": schema_v2_scaler.target_scaling_strategy,
            "x_location": list(schema_v2_scaler.x_location),
            "x_scale": list(schema_v2_scaler.x_scale),
            "y_location": schema_v2_scaler.y_location,
            "y_scale": schema_v2_scaler.y_scale,
        })
    if COMM_LOG_STREAM:
        _append_stream_comm(
            {
                "level": "stream",
                "observer": "stream",
                "cluster_id": cluster_id,
                "client_id": str(client_id),
                "event": "norm_stats",
                "route": "/stream/norm_stats",
                "direction": "stream_to_client",
                "payload_type": "norm_stats",
                "payload_bytes": len(jsonify(response).get_data()),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "status": "ok",
            },
            cluster_id,
        )
    return jsonify(response)


@app.route("/stream/reset", methods=["POST"])
def stream_reset():
    t0 = time.perf_counter()
    data = request.get_json() or {}
    client_id = int(data["client_id"])
    cluster_id = str(data["cluster_id"])
    mode = str(data.get("mode", "train")).strip().lower()
    series_key = _series_key(mode, cluster_id, client_id)
    if series_key not in client_series:
        return jsonify({"error": f"Unknown cluster/client/mode: {cluster_id}/{client_id}/{mode}"}), 404
    split_start, split_end = client_split_bounds[series_key].get(mode, (0, 0))
    if split_end <= split_start:
        return jsonify({"error": f"No windows available for mode={mode}"}), 400
    window_offset = max(0, int(data.get("window_offset", 0)))
    new_offset = min(split_end, split_start + window_offset)
    with offset_lock:
        client_offsets[(cluster_id, client_id, mode)] = new_offset
    counts = client_window_counts[series_key]
    response = {
        "status": "reset",
        "cluster_id": cluster_id,
        "client_id": client_id,
        "mode": mode,
        "window_offset": window_offset,
        "start_offset": new_offset,
        "windows_available": max(0, split_end - new_offset),
        "num_train_windows": counts["train"],
        "num_val_windows": counts["val"],
        "num_test_windows": counts["test"],
    }
    if COMM_LOG_STREAM:
        _append_stream_comm(
            {
                "level": "stream",
                "observer": "stream",
                "cluster_id": cluster_id,
                "client_id": str(client_id),
                "event": "stream_reset",
                "route": "/stream/reset",
                "direction": "client_to_stream",
                "payload_type": "stream_reset",
                "payload_bytes": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "status": "ok",
                "batch_size_requested": data.get("batch_size"),
                "start_idx": new_offset,
            },
            cluster_id,
        )
    return jsonify(response)


@app.route("/stream/status", methods=["GET"])
def stream_status():
    client_id = int(request.args["client_id"])
    cluster_id = str(request.args["cluster_id"])
    mode = str(request.args.get("mode", "train")).strip().lower()
    series_key = _series_key(mode, cluster_id, client_id)
    if series_key not in client_series:
        return jsonify({"error": f"Unknown cluster/client/mode: {cluster_id}/{client_id}/{mode}"}), 404
    split_start, split_end = client_split_bounds[series_key].get(mode, (0, 0))
    offset_key = (cluster_id, client_id, mode)
    with offset_lock:
        current_offset = client_offsets.get(offset_key, split_start)
    return jsonify(
        {
            "cluster_id": cluster_id,
            "client_id": client_id,
            "mode": mode,
            "split_start": split_start,
            "split_end": split_end,
            "current_offset": current_offset,
            "remaining_windows": max(0, split_end - current_offset),
        }
    )


@app.route("/stream/next", methods=["POST"])
def stream_next():
    t0 = time.perf_counter()
    data = request.get_json() or {}
    client_id = int(data["client_id"])
    cluster_id = str(data["cluster_id"])
    mode = str(data.get("mode", "train")).strip().lower()
    batch_size = int(data["batch_size"])
    series_key = _series_key(mode, cluster_id, client_id)
    if series_key not in client_series:
        return jsonify({"error": f"Unknown cluster/client/mode: {cluster_id}/{client_id}/{mode}"}), 404

    values = client_runtime_series.get(series_key, client_series[series_key])
    timestamps = client_timestamps.get(series_key, [])
    frequency_delta = client_frequency_deltas.get(series_key)
    target_lookup = client_target_lookup.get(series_key)
    if frequency_delta is None:
        return jsonify({"error": f"Missing inferred frequency for cluster/client/mode: {cluster_id}/{client_id}/{mode}"}), 500
    if target_lookup is None:
        return jsonify({"error": f"Missing target series for cluster/client/mode: {cluster_id}/{client_id}/{mode}"}), 500
    split_start, split_end = client_split_bounds[series_key].get(mode, (0, 0))
    if split_end <= split_start:
        return jsonify({"error": f"No windows for mode={mode}"}), 400

    offset_key = (cluster_id, client_id, mode)
    with offset_lock:
        start_idx = client_offsets.get(offset_key, split_start)
        if start_idx < split_start:
            start_idx = split_start
        if start_idx >= split_end:
            return Response(status=204)
        end_idx = min(split_end, start_idx + batch_size)
        client_offsets[offset_key] = end_idx

    start_indices = list(range(start_idx, end_idx))
    window_end_indices = [idx + WINDOW_SIZE - 1 for idx in start_indices]
    windows = [values[idx : idx + WINDOW_SIZE] for idx in start_indices]
    target_timestamps = [
        _target_timestamp_for_window(timestamps, window_end_idx, frequency_delta)
        for window_end_idx in window_end_indices
    ]
    missing_target_timestamps = [ts for ts in target_timestamps if ts not in target_lookup]
    if missing_target_timestamps:
        return jsonify({
            "error": (
                f"Missing clean target timestamp(s) for cluster={cluster_id} client={client_id} mode={mode}; "
                f"first_missing={missing_target_timestamps[0]} count={len(missing_target_timestamps)}"
            )
        }), 400
    targets = [target_lookup[ts] for ts in target_timestamps]

    X_batch = torch.stack(windows).unsqueeze(-1).float()
    static_features = client_static_features.get(series_key, torch.empty(0, dtype=torch.float32))
    if static_features.numel() > 0:
        static_block = static_features.view(1, 1, -1).expand(X_batch.size(0), X_batch.size(1), -1)
        X_batch = torch.cat([X_batch, static_block], dim=-1)
    y_batch = torch.stack(targets).float()

    quality_records = [_quality_record_for_window(client_id, mode, idx) for idx in start_indices]
    quality_input_scores = torch.tensor(
        [record.get("quality_input_values", [float(record.get("quality_score", QUALITY_SCORE_FALLBACK))]) for record in quality_records],
        dtype=torch.float32,
    )
    if quality_input_scores.dim() == 2 and quality_input_scores.size(1) == 1:
        quality_input_scores = quality_input_scores.squeeze(1)
    quality_weight_scores = torch.tensor(
        [float(record.get("quality_weight_score", record.get("quality_score", QUALITY_SCORE_FALLBACK))) for record in quality_records],
        dtype=torch.float32,
    )
    quality_anchor_scores = None
    if USE_QUALITY_ANCHOR_AGGREGATION:
        quality_anchor_scores = torch.tensor(
            [float(record.get("quality_score", QUALITY_SCORE_FALLBACK)) for record in quality_records],
            dtype=torch.float32,
        )

    if NORMALIZE:
        schema_v2_scaler = client_schema_v2_scalers.get(_client_key(cluster_id, client_id))
        mean, scale = client_norm_stats[_client_key(cluster_id, client_id)]
        X_batch = X_batch.clone()
        if schema_v2_scaler is not None:
            X_batch[:, :, 0] = _normalize_tensor(
                X_batch[:, :, 0], float(schema_v2_scaler.x_location[0]), float(schema_v2_scaler.x_scale[0])
            )
            y_batch = _normalize_tensor(y_batch, float(schema_v2_scaler.y_location), float(schema_v2_scaler.y_scale))
        else:
            X_batch[:, :, 0] = _normalize_tensor(X_batch[:, :, 0], mean, scale)
            y_batch = _normalize_tensor(y_batch, mean, scale)

    drift_batch_metadata = []
    if DRIFT_DETECTION_ENABLED and mode == "test" and drift_detector_registry is not None:
        names = app.config.get("DRIFT_DETECTOR_NAMES", ())
        record_lookup = drift_injection_records.get(series_key, {})
        for i, observation_index in enumerate(window_end_indices):
            timestamp = None if observation_index >= len(timestamps) or pd.isna(timestamps[observation_index]) else pd.Timestamp(timestamps[observation_index]).isoformat()
            record = record_lookup.get(observation_index)
            for detector_name in names:
                drift_detector_registry.update(
                    DetectorKey(DRIFT_RUN_ID, cluster_id, str(client_id), mode, DRIFT_STREAM_NAME, detector_name),
                    observation_index=observation_index, value=float(X_batch[i, -1, 0].item()), timestamp=timestamp,
                    regime_active=record.regime_active if record else False, transform_applied=record.transform_applied if record else False,
                )
            drift_batch_metadata.append({"observation_index": observation_index, "timestamp": timestamp, "drifted_client": _client_key(cluster_id, client_id) in drift_selected_clients})

    payload = {
        "X": X_batch,
        "y": y_batch,
        "cluster_id": cluster_id,
        "client_id": str(client_id),
        "mode": mode,
        "window_start_indices": start_indices,
        "window_end_indices": window_end_indices,
        "window_start_timestamps": [
            None if idx >= len(timestamps) or pd.isna(timestamps[idx]) else pd.Timestamp(timestamps[idx]).isoformat()
            for idx in start_indices
        ],
        "window_end_timestamps": [
            None
            if idx >= len(timestamps) or pd.isna(timestamps[idx])
            else pd.Timestamp(timestamps[idx]).isoformat()
            for idx in window_end_indices
        ],
        "target_timestamps": [ts.isoformat() for ts in target_timestamps],
        "quality_scores": quality_input_scores,
        "quality_weight_scores": quality_weight_scores,
        "quality_score_metadata": {
            "granularity": QUALITY_SCORE_GRANULARITY,
            "match_key": "client_id+mode+window_start_index",
            "match_statuses": [record.get("match_status") for record in quality_records],
            "score_path": quality_score_summary.get("path"),
            "quality_input_columns": QUALITY_INPUT_COLUMNS if USE_QUALITY_AS_INPUT else [],
            "quality_input_dim": len(QUALITY_INPUT_COLUMNS) if USE_QUALITY_AS_INPUT else 0,
            "quality_weight_column": QUALITY_WEIGHT_COLUMN,
            "dimension_completeness": [record.get("dimension_completeness") for record in quality_records],
            "dimension_validity": [record.get("dimension_validity") for record in quality_records],
            "dimension_consistency": [record.get("dimension_consistency") for record in quality_records],
            "dimension_accuracy": [record.get("dimension_accuracy") for record in quality_records],
            "dimension_freshness": [record.get("dimension_freshness") for record in quality_records],
        },
    }
    if DRIFT_CONFIG_PATH or DRIFT_INJECTION_ENABLED or DRIFT_DETECTION_ENABLED:
        payload["drift_metadata"] = drift_batch_metadata
    if USE_QUALITY_ANCHOR_AGGREGATION:
        payload["quality_anchor_scores"] = quality_anchor_scores
    payload_blob = pickle.dumps(payload)

    if COMM_LOG_STREAM:
        matched_count = sum(1 for record in quality_records if record.get("match_status") == "matched")
        fallback_count = sum(1 for record in quality_records if record.get("match_status") == "fallback")
        _append_stream_comm(
            {
                "level": "stream",
                "observer": "stream",
                "cluster_id": cluster_id,
                "client_id": str(client_id),
                "event": "stream_next",
                "route": "/stream/next",
                "direction": "stream_to_client",
                "payload_type": "stream_batch",
                "payload_bytes": len(payload_blob),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "status": "ok",
                "batch_size_requested": batch_size,
                "batch_size_returned": end_idx - start_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "x_shape": tensor_shape(X_batch),
                "y_shape": tensor_shape(y_batch),
                "payload_shape": f"X={tensor_shape(X_batch)};y={tensor_shape(y_batch)}",
                "quality_score_mean": float(quality_weight_scores.mean().item()) if quality_weight_scores.numel() else None,
                "quality_score_min": float(quality_weight_scores.min().item()) if quality_weight_scores.numel() else None,
                "quality_score_max": float(quality_weight_scores.max().item()) if quality_weight_scores.numel() else None,
                "quality_score_matches": matched_count,
                "quality_score_fallbacks": fallback_count,
            },
            cluster_id,
        )
    return Response(payload_blob, content_type="application/octet-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=STREAM_PORT)
