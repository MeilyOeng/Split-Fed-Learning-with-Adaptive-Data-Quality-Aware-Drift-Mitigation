#!/usr/bin/env python3
import os
import pickle
import sys
import time
from threading import Lock
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.comm_logging import append_comm_row, tensor_shape

app = Flask(__name__)
CLUSTER_DIR = os.environ.get("CLUSTER_DIR", os.path.join(BASE_DIR, "dataset", "output"))
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
SIGN_FLIP = os.environ.get("SIGN_FLIP", "true").lower() in ("1", "true", "yes", "y")
COMM_LOG_STREAM = os.environ.get("COMM_LOG_STREAM", "false").lower() in ("1", "true", "yes", "y")
USE_STATIC_FEATURES = os.environ.get("USE_STATIC_FEATURES", "false").lower() in ("1", "true", "yes", "y")
STATIC_FEATURE_COLUMNS_RAW = os.environ.get("STATIC_FEATURE_COLUMNS", "")
STATIC_FEATURE_GROUPS_RAW = os.environ.get("STATIC_FEATURE_GROUPS", "")

if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

client_series: Dict[Tuple[str, int], torch.Tensor] = {}
client_static_features: Dict[Tuple[str, int], torch.Tensor] = {}
client_norm_stats: Dict[Tuple[str, int], Tuple[float, float]] = {}
client_split_bounds: Dict[Tuple[str, int], Dict[str, Tuple[int, int]]] = {}
client_window_counts: Dict[Tuple[str, int], Dict[str, int]] = {}
client_input_sizes: Dict[Tuple[str, int], int] = {}
client_sign_flipped: Dict[Tuple[str, int], bool] = {}
client_offsets: Dict[Tuple[str, int, str], int] = {}
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


def _transform_series(values) -> tuple[torch.Tensor, bool]:
    out = torch.as_tensor(values, dtype=torch.float32)
    should_flip = False
    if SIGN_FLIP:
        finite = torch.isfinite(out)
        valid = out[finite]
        should_flip = bool(valid.numel() > 0 and torch.all(valid < 0).item())
    if should_flip:
        out = -out
    return out, should_flip


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
    train_values = values[:train_value_end] # only compute on the train set
    if train_values.numel() == 0:
        return 0.0, 1.0
    if NORMALIZE_METHOD == "minmax":
        min_v = float(train_values.min().item())
        max_v = float(train_values.max().item())
        return min_v, max(max_v - min_v, NORMALIZE_EPS) # to avoid zero division
    mean = float(train_values.mean().item())
    scale = float(train_values.std().item())
    if scale < NORMALIZE_EPS: # to avoid zero division
        scale = 1.0
    return mean, scale


def _normalize_tensor(tensor: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
    return (tensor - mean) / scale


def _prepare_from_dataframe(
    df: pd.DataFrame,
    cluster_id: str,
    *,
    timestamp_col: str = "Fecha_Hora",
    value_col: str = "value",
    client_metadata: pd.DataFrame | None = None,
    static_feature_columns: list[str] | None = None,
) -> None:
    # prepare client tensors, static features, normalization stats, and split bounds from a cluster dataframe
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(["id", timestamp_col], kind="stable")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    feature_lookup: dict[str, torch.Tensor] = {}
    source_sign_flipped_lookup: dict[str, bool] = {}
    static_feature_columns = _resolve_static_feature_columns(list(static_feature_columns or []))
    if client_metadata is not None and not client_metadata.empty:
        meta = client_metadata.copy()
        meta["id"] = meta["id"].map(_normalize_client_id)
        selected_static_feature_columns = [col for col in static_feature_columns if col in meta.columns]
        if selected_static_feature_columns:
            meta[selected_static_feature_columns] = meta[selected_static_feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        for _, row in meta.iterrows():
            raw_id = row.get("id")
            if raw_id is None:
                continue
            normalized_id = _normalize_client_id(raw_id)
            source_sign_flipped_lookup[normalized_id] = bool(row.get("sign_flipped", False))
            if selected_static_feature_columns:
                values = row.loc[selected_static_feature_columns].to_numpy(dtype=float, copy=True)
                feature_lookup[normalized_id] = torch.as_tensor(values, dtype=torch.float32)
            else:
                feature_lookup[normalized_id] = torch.empty(0, dtype=torch.float32)

    for client_id, rows in df.groupby("id", sort=False):
        rows = rows.copy()
        rows[value_col] = rows[value_col].ffill().bfill()
        if rows[value_col].isna().any():
            raise ValueError(f"NaNs remain after fill for cluster={cluster_id}, client={client_id}")
        normalized_client_id = _normalize_client_id(client_id)
        key = (str(cluster_id), int(normalized_client_id))
        values, sign_flipped = _transform_series(rows[value_col].to_numpy(dtype=float))
        client_series[key] = values.contiguous()
        client_sign_flipped[key] = bool(source_sign_flipped_lookup.get(normalized_client_id, False) or sign_flipped)
        client_split_bounds[key] = _get_split_bounds(len(values))
        client_window_counts[key] = {k: max(0, end - start) for k, (start, end) in client_split_bounds[key].items()}
        client_norm_stats[key] = _compute_norm_stats(values)
        static_features = feature_lookup.get(normalized_client_id, torch.empty(0, dtype=torch.float32))
        client_static_features[key] = static_features
        client_input_sizes[key] = 1 + int(static_features.numel())


def _prepare_from_payload(payload: Any, cluster_id: str) -> None:
    if isinstance(payload, pd.DataFrame):
        _prepare_from_dataframe(payload, cluster_id)
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
        timestamp_col=timestamp_col,
        value_col=value_col,
        client_metadata=client_metadata,
        static_feature_columns=[str(col) for col in static_feature_columns],
    )


def load_cluster_data() -> None:
    loaded_any = False
    if os.path.exists(CLUSTER_DIR) and os.listdir(CLUSTER_DIR):
        for file in sorted(os.listdir(CLUSTER_DIR)):
            if not file.endswith(".pkl"):
                continue
            cluster_id = file.split("_")[1].split(".")[0]
            if STREAM_CLUSTER_ID and str(cluster_id) != str(STREAM_CLUSTER_ID):
                continue
            payload = pd.read_pickle(os.path.join(CLUSTER_DIR, file))
            _prepare_from_payload(payload, str(cluster_id))
            loaded_any = True
    if loaded_any:
        return

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
        _prepare_from_dataframe(rows.reset_index(drop=True), str(cluster_id))


load_cluster_data()


@app.route("/health", methods=["GET"])
def health():
    cluster_ids = sorted({cluster_id for cluster_id, _ in client_series.keys()})
    input_sizes = sorted({input_size for input_size in client_input_sizes.values()})
    return jsonify({
        "status": "healthy",
        "clusters_loaded": len(cluster_ids),
        "cluster_ids": cluster_ids,
        "clients_loaded": len(client_series),
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "normalize": NORMALIZE,
        "normalize_method": NORMALIZE_METHOD,
        "sign_flip": SIGN_FLIP,
        "sign_flip_rule": "flip_only_when_all_non_null_values_are_strictly_negative",
        "use_static_features": USE_STATIC_FEATURES,
        "requested_static_feature_groups": _parse_optional_csv(STATIC_FEATURE_GROUPS_RAW),
        "requested_static_feature_columns": _parse_optional_csv(STATIC_FEATURE_COLUMNS_RAW),
        "input_sizes": input_sizes,
    })


@app.route("/stream/norm_stats", methods=["GET"]) 
def stream_norm_stats():                            # send normalization stats to clients to invert predictions back to original scale 
    t0 = time.perf_counter()
    client_id = int(request.args["client_id"])
    cluster_id = str(request.args["cluster_id"])
    key = (cluster_id, client_id)
    if key not in client_series:
        return jsonify({"error": f"Unknown cluster/client: {cluster_id}/{client_id}"}), 404
    mean, scale = client_norm_stats[key]
    counts = client_window_counts[key]
    static_feature_count = int(client_static_features.get(key, torch.empty(0)).numel())
    response = {
        "cluster_id": cluster_id,
        "client_id": client_id,
        "normalize": NORMALIZE,
        "method": NORMALIZE_METHOD,
        "mean": mean,
        "scale": scale,
        "sign_flip": SIGN_FLIP,
        "sign_flipped_client": bool(client_sign_flipped.get(key, False)),
        "sign_flip_rule": "flip_only_when_all_non_null_values_are_strictly_negative",
        "use_static_features": USE_STATIC_FEATURES,
        "requested_static_feature_groups": _parse_optional_csv(STATIC_FEATURE_GROUPS_RAW),
        "requested_static_feature_columns": _parse_optional_csv(STATIC_FEATURE_COLUMNS_RAW),
        "input_size": int(client_input_sizes.get(key, 1)),
        "static_feature_count": static_feature_count,
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
    mode = str(data.get("mode", "train"))
    key = (cluster_id, client_id)
    if key not in client_series:
        return jsonify({"error": f"Unknown cluster/client: {cluster_id}/{client_id}"}), 404
    split_start, split_end = client_split_bounds[key].get(mode, (0, 0))
    if split_end <= split_start:
        return jsonify({"error": f"No windows available for mode={mode}"}), 400
    window_offset = max(0, int(data.get("window_offset", 0)))
    new_offset = min(split_end, split_start + window_offset)
    with offset_lock:
        client_offsets[(cluster_id, client_id, mode)] = new_offset
    counts = client_window_counts[key]
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
    mode = str(request.args.get("mode", "train"))
    key = (cluster_id, client_id)
    if key not in client_series:
        return jsonify({"error": f"Unknown cluster/client: {cluster_id}/{client_id}"}), 404
    split_start, split_end = client_split_bounds[key].get(mode, (0, 0))
    offset_key = (cluster_id, client_id, mode)
    with offset_lock:
        current_offset = client_offsets.get(offset_key, split_start)
    return jsonify({
        "cluster_id": cluster_id,
        "client_id": client_id,
        "mode": mode,
        "split_start": split_start,
        "split_end": split_end,
        "current_offset": current_offset,
        "remaining_windows": max(0, split_end - current_offset),
    })

@app.route("/stream/next", methods=["POST"])
def stream_next():
    t0 = time.perf_counter()
    data = request.get_json() or {}
    client_id = int(data["client_id"])
    cluster_id = str(data["cluster_id"])
    mode = str(data.get("mode", "train"))
    batch_size = int(data["batch_size"])
    key = (cluster_id, client_id)
    if key not in client_series:
        return jsonify({"error": f"Unknown cluster/client: {cluster_id}/{client_id}"}), 404

    values = client_series[key]
    split_start, split_end = client_split_bounds[key].get(mode, (0, 0))
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

    windows = [values[i : i + WINDOW_SIZE] for i in range(start_idx, end_idx)]
    targets = [values[i + WINDOW_SIZE + FORECAST_HORIZON - 1] for i in range(start_idx, end_idx)]
    X_batch = torch.stack(windows).unsqueeze(-1).float()
    static_features = client_static_features.get(key, torch.empty(0, dtype=torch.float32))
    if static_features.numel() > 0:
        static_block = static_features.view(1, 1, -1).expand(X_batch.size(0), X_batch.size(1), -1)
        X_batch = torch.cat([X_batch, static_block], dim=-1)
    y_batch = torch.stack(targets).float()

    if NORMALIZE:
        mean, scale = client_norm_stats[key]
        X_batch = X_batch.clone()
        X_batch[:, :, 0] = _normalize_tensor(X_batch[:, :, 0], mean, scale)
        y_batch = _normalize_tensor(y_batch, mean, scale)

    payload = pickle.dumps({"X": X_batch, "y": y_batch})
    if COMM_LOG_STREAM:
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
                "payload_bytes": len(payload),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "status": "ok",
                "batch_size_requested": batch_size,
                "batch_size_returned": end_idx - start_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "x_shape": tensor_shape(X_batch),
                "y_shape": tensor_shape(y_batch),
                "payload_shape": f"X={tensor_shape(X_batch)};y={tensor_shape(y_batch)}",
            },
            cluster_id,
        )
    return Response(payload, content_type="application/octet-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=STREAM_PORT)
