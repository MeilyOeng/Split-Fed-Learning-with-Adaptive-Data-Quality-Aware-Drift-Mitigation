#!/usr/bin/env python3
import os
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

app = Flask(__name__)

DEFAULT_CLUSTER_DIR = os.path.join(BASE_DIR, "dataset", "output")
CLUSTER_DIR = os.environ.get("CLUSTER_DIR", DEFAULT_CLUSTER_DIR)
TRAIN_CLUSTER_DATA_DIR = os.environ.get("TRAIN_CLUSTER_DATA_DIR", CLUSTER_DIR)
VAL_CLUSTER_DATA_DIR = os.environ.get("VAL_CLUSTER_DATA_DIR", CLUSTER_DIR)
TEST_CLUSTER_DATA_DIR = os.environ.get("TEST_CLUSTER_DATA_DIR", CLUSTER_DIR)
MODE_CLUSTER_DIRS = {
    "train": TRAIN_CLUSTER_DATA_DIR,
    "val": VAL_CLUSTER_DATA_DIR,
    "test": TEST_CLUSTER_DATA_DIR,
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
QUALITY_SCORE_PATH = str(os.environ.get("QUALITY_SCORE_PATH", "")).strip()
QUALITY_SCORE_GRANULARITY = str(os.environ.get("QUALITY_SCORE_GRANULARITY", "window")).strip().lower()
STRICT_QUALITY_SCORE_MATCHING = os.environ.get("STRICT_QUALITY_SCORE_MATCHING", "true").lower() in ("1", "true", "yes", "y")
QUALITY_SCORE_FALLBACK = float(os.environ.get("QUALITY_SCORE_FALLBACK", "1.0"))

if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")
if QUALITY_SCORE_GRANULARITY != "window":
    raise ValueError("Only QUALITY_SCORE_GRANULARITY=window is supported in the current runtime.")

client_series: Dict[Tuple[str, str, int], torch.Tensor] = {}
client_timestamps: Dict[Tuple[str, str, int], List[pd.Timestamp]] = {}
client_static_features: Dict[Tuple[str, str, int], torch.Tensor] = {}
client_norm_stats: Dict[Tuple[str, int], Tuple[float, float]] = {}
client_split_bounds: Dict[Tuple[str, str, int], Dict[str, Tuple[int, int]]] = {}
client_window_counts: Dict[Tuple[str, str, int], Dict[str, int]] = {}
client_input_sizes: Dict[Tuple[str, int], int] = {}
client_offsets: Dict[Tuple[str, int, str], int] = {}
quality_score_index: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
quality_score_summary: Dict[str, Any] = {
    "enabled": USE_QUALITY_WEIGHTS,
    "loaded": False,
    "path": QUALITY_SCORE_PATH or None,
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
    train_value_end = min(len(values), train_end + WINDOW_SIZE + FORECAST_HORIZON)
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


def load_cluster_data() -> None:
    for mode in ("train", "val", "test"):
        _load_cluster_data_for_mode(mode, MODE_CLUSTER_DIRS[mode])


def _load_quality_score_index() -> None:
    global quality_score_summary
    if not USE_QUALITY_WEIGHTS:
        quality_score_summary = {
            "enabled": False,
            "loaded": False,
            "path": QUALITY_SCORE_PATH or None,
            "granularity": QUALITY_SCORE_GRANULARITY,
        }
        return
    if not QUALITY_SCORE_PATH:
        raise ValueError("USE_QUALITY_WEIGHTS=true requires QUALITY_SCORE_PATH to be set.")

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

    for _, row in df.iterrows():
        client_id = _normalize_client_id(row["client_id"])
        mode = str(row["mode"]).strip().lower()
        window_start_index = int(row["window_start_index"])
        key = (client_id, mode, window_start_index)
        if key in quality_score_index:
            raise ValueError(f"Duplicate quality score key found: {key} in {quality_path}")
        quality_score_index[key] = {
            "quality_score": float(row[score_column]),
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
        "granularity": QUALITY_SCORE_GRANULARITY,
        "row_count": int(len(df)),
        "client_count": int(df["client_id"].astype(str).nunique()),
        "modes": sorted({str(mode).strip().lower() for mode in df["mode"].dropna().astype(str)}),
    }


def _quality_record_for_window(client_id: int, mode: str, window_start_index: int) -> Dict[str, Any]:
    if not USE_QUALITY_WEIGHTS:
        return {
            "quality_score": 1.0,
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
        "match_status": "fallback",
    }


load_cluster_data()
_load_quality_score_index()


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
            "requested_static_feature_groups": _parse_optional_csv(STATIC_FEATURE_GROUPS_RAW),
            "requested_static_feature_columns": _parse_optional_csv(STATIC_FEATURE_COLUMNS_RAW),
            "input_sizes": input_sizes,
            "cluster_data_dirs": MODE_CLUSTER_DIRS,
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
        "requested_static_feature_groups": _parse_optional_csv(STATIC_FEATURE_GROUPS_RAW),
        "requested_static_feature_columns": _parse_optional_csv(STATIC_FEATURE_COLUMNS_RAW),
        "input_size": int(client_input_sizes.get(key, 1)),
        "static_feature_count": static_feature_count,
        "num_train_windows": counts["train"],
        "num_val_windows": counts["val"],
        "num_test_windows": counts["test"],
        "quality_weights_enabled": USE_QUALITY_WEIGHTS,
        "quality_score_granularity": QUALITY_SCORE_GRANULARITY,
    }
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

    values = client_series[series_key]
    timestamps = client_timestamps.get(series_key, [])
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
    target_indices = [idx + WINDOW_SIZE + FORECAST_HORIZON - 1 for idx in start_indices]
    windows = [values[idx : idx + WINDOW_SIZE] for idx in start_indices]
    targets = [values[target_idx] for target_idx in target_indices]

    X_batch = torch.stack(windows).unsqueeze(-1).float()
    static_features = client_static_features.get(series_key, torch.empty(0, dtype=torch.float32))
    if static_features.numel() > 0:
        static_block = static_features.view(1, 1, -1).expand(X_batch.size(0), X_batch.size(1), -1)
        X_batch = torch.cat([X_batch, static_block], dim=-1)
    y_batch = torch.stack(targets).float()

    quality_records = [_quality_record_for_window(client_id, mode, idx) for idx in start_indices]
    quality_scores = torch.tensor(
        [float(record.get("quality_score", QUALITY_SCORE_FALLBACK)) for record in quality_records],
        dtype=torch.float32,
    )

    if NORMALIZE:
        mean, scale = client_norm_stats[_client_key(cluster_id, client_id)]
        X_batch = X_batch.clone()
        X_batch[:, :, 0] = _normalize_tensor(X_batch[:, :, 0], mean, scale)
        y_batch = _normalize_tensor(y_batch, mean, scale)

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
        "target_timestamps": [
            None
            if idx >= len(timestamps) or pd.isna(timestamps[idx])
            else pd.Timestamp(timestamps[idx]).isoformat()
            for idx in target_indices
        ],
        "quality_scores": quality_scores,
        "quality_score_metadata": {
            "granularity": QUALITY_SCORE_GRANULARITY,
            "match_key": "client_id+mode+window_start_index",
            "match_statuses": [record.get("match_status") for record in quality_records],
            "score_path": quality_score_summary.get("path"),
            "dimension_completeness": [record.get("dimension_completeness") for record in quality_records],
            "dimension_validity": [record.get("dimension_validity") for record in quality_records],
            "dimension_consistency": [record.get("dimension_consistency") for record in quality_records],
            "dimension_accuracy": [record.get("dimension_accuracy") for record in quality_records],
            "dimension_freshness": [record.get("dimension_freshness") for record in quality_records],
        },
    }
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
                "quality_score_mean": float(quality_scores.mean().item()) if quality_scores.numel() else None,
                "quality_score_min": float(quality_scores.min().item()) if quality_scores.numel() else None,
                "quality_score_max": float(quality_scores.max().item()) if quality_scores.numel() else None,
                "quality_score_matches": matched_count,
                "quality_score_fallbacks": fallback_count,
            },
            cluster_id,
        )
    return Response(payload_blob, content_type="application/octet-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=STREAM_PORT)
