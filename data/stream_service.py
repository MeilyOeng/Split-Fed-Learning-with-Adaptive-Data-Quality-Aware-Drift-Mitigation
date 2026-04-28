#!/usr/bin/env python3
"""Efficient client-partitioned streaming service with precomputed client tensors."""

import os
import pickle
from threading import Lock
from typing import Dict, Tuple

import pandas as pd
import torch
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

client_series: Dict[Tuple[str, int], torch.Tensor] = {}
client_norm_stats: Dict[Tuple[str, int], Tuple[float, float]] = {}
client_split_bounds: Dict[Tuple[str, int], Dict[str, Tuple[int, int]]] = {}
client_window_counts: Dict[Tuple[str, int], Dict[str, int]] = {}
client_offsets: Dict[Tuple[str, int, str], int] = {}
offset_lock = Lock()


def _transform_series(values) -> torch.Tensor:
    out = torch.as_tensor(values, dtype=torch.float32)
    if SIGN_FLIP:
        out = -out
    return out


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


def _prepare_from_dataframe(df: pd.DataFrame, cluster_id: str) -> None: # prepare client tensors, normalization stats, and split bounds from a cluster dataframe
    if "Fecha_Hora" in df.columns:
        df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"], errors="coerce")
        df = df.sort_values(["id", "Fecha_Hora"], kind="stable")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    for client_id, rows in df.groupby("id", sort=False):
        rows = rows.copy()
        rows["value"] = rows["value"].ffill().bfill()
        if rows["value"].isna().any():
            raise ValueError(f"NaNs remain after fill for cluster={cluster_id}, client={client_id}")
        key = (str(cluster_id), int(client_id))
        values = _transform_series(rows["value"].to_numpy(dtype=float))
        client_series[key] = values.contiguous()
        client_split_bounds[key] = _get_split_bounds(len(values))
        client_window_counts[key] = {k: max(0, end - start) for k, (start, end) in client_split_bounds[key].items()}
        client_norm_stats[key] = _compute_norm_stats(values)


def load_cluster_data() -> None:
    loaded_any = False
    if os.path.exists(CLUSTER_DIR) and os.listdir(CLUSTER_DIR):
        for file in sorted(os.listdir(CLUSTER_DIR)):
            if not file.endswith(".pkl"):
                continue
            cluster_id = file.split("_")[1].split(".")[0]
            if STREAM_CLUSTER_ID and str(cluster_id) != str(STREAM_CLUSTER_ID):
                continue
            df = pd.read_pickle(os.path.join(CLUSTER_DIR, file))
            _prepare_from_dataframe(df, str(cluster_id))
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
    })


@app.route("/stream/norm_stats", methods=["GET"]) 
def stream_norm_stats():                            # send normalization stats to clients to invert predictions back to original scale 
    client_id = int(request.args["client_id"])
    cluster_id = str(request.args["cluster_id"])
    key = (cluster_id, client_id)
    if key not in client_series:
        return jsonify({"error": f"Unknown cluster/client: {cluster_id}/{client_id}"}), 404
    mean, scale = client_norm_stats[key]
    counts = client_window_counts[key]
    return jsonify({
        "cluster_id": cluster_id,
        "client_id": client_id,
        "normalize": NORMALIZE,
        "method": NORMALIZE_METHOD,
        "mean": mean,
        "scale": scale,
        "sign_flip": SIGN_FLIP,
        "num_train_windows": counts["train"],
        "num_val_windows": counts["val"],
        "num_test_windows": counts["test"],
    })


@app.route("/stream/reset", methods=["POST"])
def stream_reset():
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
    return jsonify({
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
    })




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
    y_batch = torch.stack(targets).float()

    if NORMALIZE:
        mean, scale = client_norm_stats[key]
        X_batch = _normalize_tensor(X_batch, mean, scale)
        y_batch = _normalize_tensor(y_batch, mean, scale)

    payload = pickle.dumps({"X": X_batch, "y": y_batch})
    return Response(payload, content_type="application/octet-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=STREAM_PORT)
