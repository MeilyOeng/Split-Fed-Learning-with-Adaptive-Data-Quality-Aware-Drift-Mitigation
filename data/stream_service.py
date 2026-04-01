#!/usr/bin/env python3
"""
Client-partitioned streaming service.

Fixes:
- consistent sign flip for both norm stats and streamed windows
- explicit ratio validation
- explicit sorting by client/time
- richer metadata on /stream/reset and /stream/norm_stats
"""

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
    raise ValueError(f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}")

cluster_batches: Dict[str, pd.DataFrame] = {}
client_offsets: Dict[Tuple[str, int, str], int] = {}
offset_lock = Lock()
norm_stats: Dict[Tuple[str, int], Tuple[float, float]] = {}


def _transform_series(values):
    out = torch.as_tensor(values, dtype=torch.float32)
    if SIGN_FLIP:
        out = -out
    return out


def _num_windows(num_values: int, window_size: int, horizon: int) -> int:
    return max(0, num_values - window_size - horizon + 1)


def _get_split_bounds(num_values: int, window_size: int, horizon: int) -> Dict[str, Tuple[int, int]]:
    total_windows = _num_windows(num_values, window_size, horizon)
    if total_windows <= 0:
        return {"train": (0, 0), "val": (0, 0), "test": (0, 0)}

    train_windows = int(total_windows * TRAIN_RATIO)
    val_windows = int(total_windows * VAL_RATIO)
    test_windows = total_windows - train_windows - val_windows

    train_end = train_windows
    val_end = train_end + val_windows
    test_end = val_end + test_windows

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }


def _window_counts(num_values: int) -> Dict[str, int]:
    bounds = _get_split_bounds(num_values, WINDOW_SIZE, FORECAST_HORIZON)
    return {k: max(0, end - start) for k, (start, end) in bounds.items()}


def _get_or_compute_norm_stats(cluster_id: str, client_id: int, transformed_values: torch.Tensor) -> Tuple[float, float]:
    key = (cluster_id, client_id)
    if key in norm_stats:
        return norm_stats[key]

    values = torch.as_tensor(transformed_values, dtype=torch.float32)
    split_bounds = _get_split_bounds(len(values), WINDOW_SIZE, FORECAST_HORIZON)
    _, train_end = split_bounds["train"]
    train_value_end = min(len(values), train_end + WINDOW_SIZE + FORECAST_HORIZON)
    train_values = values[:train_value_end]

    if train_values.numel() == 0:
        mean, scale = 0.0, 1.0
    elif NORMALIZE_METHOD == "minmax":
        min_v = float(train_values.min().item())
        max_v = float(train_values.max().item())
        mean = min_v
        scale = max(max_v - min_v, NORMALIZE_EPS)
    else:
        mean = float(train_values.mean().item())
        scale = float(train_values.std().item())
        if scale < NORMALIZE_EPS:
            scale = 1.0

    norm_stats[key] = (mean, scale)
    return mean, scale


def _normalize_tensor(tensor: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
    return (tensor - mean) / scale


def _cluster_rows(cluster_id: str, client_id: int) -> pd.DataFrame:
    if cluster_id not in cluster_batches:
        raise KeyError(f"Cluster {cluster_id} not found")

    rows = cluster_batches[cluster_id]
    client_rows = rows[rows["id"] == client_id].copy()

    if len(client_rows) == 0:
        raise KeyError(f"No rows for client {client_id}")

    if "Fecha_Hora" in client_rows.columns:
        client_rows["Fecha_Hora"] = pd.to_datetime(client_rows["Fecha_Hora"], errors="coerce")
        client_rows = client_rows.sort_values(["id", "Fecha_Hora"], kind="stable")

    # Force value column to numeric. Invalid parsing becomes NaN.
    client_rows["value"] = pd.to_numeric(client_rows["value"], errors="coerce")

    # Fill missing values in time order.
    client_rows["value"] = client_rows["value"].ffill().bfill()

    # Fail loudly if anything is still invalid.
    if client_rows["value"].isna().any():
        n_missing = int(client_rows["value"].isna().sum())
        raise ValueError(
            f"NaNs remain after fill for cluster={cluster_id}, client={client_id}, "
            f"missing={n_missing}"
        )

    return client_rows.reset_index(drop=True)


def load_cluster_data() -> None:
    global cluster_batches
    if os.path.exists(CLUSTER_DIR) and os.listdir(CLUSTER_DIR):
        for file in sorted(os.listdir(CLUSTER_DIR)):
            if not file.endswith(".pkl"):
                continue
            cluster_id = file.split("_")[1].split(".")[0]
            if STREAM_CLUSTER_ID and str(cluster_id) != str(STREAM_CLUSTER_ID):
                continue
            df = pd.read_pickle(os.path.join(CLUSTER_DIR, file))
            if "Fecha_Hora" in df.columns:
                df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"])
                sort_cols = ["id", "Fecha_Hora"] if "id" in df.columns else ["Fecha_Hora"]
                df = df.sort_values(sort_cols, kind="stable")
            cluster_batches[cluster_id] = df.reset_index(drop=True)
        if cluster_batches:
            return

    dataset_path = os.path.join(BASE_DIR, "dataset", "dataset.csv")
    assignment_path = os.path.join(BASE_DIR, "dataset", "cluster-assignment.csv")
    dataset = pd.read_csv(dataset_path)
    cluster_assignments = pd.read_csv(assignment_path)
    merged = dataset.melt(id_vars=["Fecha_Hora"], var_name="id", value_name="value")
    merged["id"] = merged["id"].astype(int)
    merged["Fecha_Hora"] = pd.to_datetime(merged["Fecha_Hora"])
    merged = merged.merge(cluster_assignments, left_on="id", right_on="ID")
    merged = merged.sort_values(["Cluster", "id", "Fecha_Hora"], kind="stable")
    for cluster_id, rows in merged.groupby("Cluster"):
        if STREAM_CLUSTER_ID and str(cluster_id) != str(STREAM_CLUSTER_ID):
            continue
        cluster_batches[str(cluster_id)] = rows.reset_index(drop=True)


load_cluster_data()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "clusters_loaded": len(cluster_batches),
            "cluster_ids": list(cluster_batches.keys()),
            "window_size": WINDOW_SIZE,
            "forecast_horizon": FORECAST_HORIZON,
            "normalize": NORMALIZE,
            "normalize_method": NORMALIZE_METHOD,
            "sign_flip": SIGN_FLIP,
        }
    )


@app.route("/stream/norm_stats", methods=["GET"])
def stream_norm_stats():
    client_id = int(request.args["client_id"])
    cluster_id = str(request.args["cluster_id"])
    try:
        rows = _cluster_rows(cluster_id, client_id)
    except KeyError as e:
        return jsonify({"error": str(e)}), 404

    values = rows["value"].to_numpy(dtype=float)
    transformed_values = _transform_series(values)
    mean, scale = _get_or_compute_norm_stats(cluster_id, client_id, transformed_values)
    counts = _window_counts(len(transformed_values))
    return jsonify(
        {
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
        }
    )


@app.route("/stream/reset", methods=["POST"])
def stream_reset():
    data = request.get_json() or {}
    client_id = int(data["client_id"])
    cluster_id = str(data["cluster_id"])
    mode = data.get("mode", "train")

    try:
        rows = _cluster_rows(cluster_id, client_id)
    except KeyError as e:
        return jsonify({"error": str(e)}), 404

    values = _transform_series(rows["value"].to_numpy(dtype=float))
    split_bounds = _get_split_bounds(len(values), WINDOW_SIZE, FORECAST_HORIZON)
    split_start, split_end = split_bounds.get(mode, (0, 0))
    if split_end <= split_start:
        return jsonify({"error": f"No windows available for mode={mode}"}), 400

    counts = _window_counts(len(values))
    key = (cluster_id, client_id, mode)
    with offset_lock:
        client_offsets[key] = split_start

    return jsonify(
        {
            "status": "reset",
            "client_id": client_id,
            "cluster_id": cluster_id,
            "mode": mode,
            "windows_available": split_end - split_start,
            "num_train_windows": counts["train"],
            "num_val_windows": counts["val"],
            "num_test_windows": counts["test"],
            "sign_flip": SIGN_FLIP,
            "normalize": NORMALIZE,
            "method": NORMALIZE_METHOD,
        }
    )


@app.route("/stream/next", methods=["POST"])
def stream_next():
    data = request.get_json() or {}
    client_id = int(data["client_id"])
    cluster_id = str(data["cluster_id"])
    mode = data.get("mode", "train")
    batch_size = int(data["batch_size"])

    try:
        rows = _cluster_rows(cluster_id, client_id)
    except KeyError as e:
        return jsonify({"error": str(e)}), 404

    values = _transform_series(rows["value"].to_numpy(dtype=float))
    split_bounds = _get_split_bounds(len(values), WINDOW_SIZE, FORECAST_HORIZON)
    split_start, split_end = split_bounds.get(mode, (0, 0))
    if split_end <= split_start:
        return jsonify({"error": f"No windows for mode={mode}"}), 400

    key = (cluster_id, client_id, mode)
    with offset_lock:
        start_idx = client_offsets.get(key, split_start)
        if start_idx < split_start:
            start_idx = split_start
        if start_idx >= split_end:
            return Response(status=204)
        end_idx = min(split_end, start_idx + batch_size)
        client_offsets[key] = end_idx

    X_windows, y_targets = [], []
    for i in range(start_idx, end_idx):
        X_windows.append(values[i : i + WINDOW_SIZE])
        y_targets.append(values[i + WINDOW_SIZE + FORECAST_HORIZON - 1])

    X_batch = torch.stack(X_windows).float().unsqueeze(-1)
    y_batch = torch.stack(y_targets).float()

    if NORMALIZE:
        mean, scale = _get_or_compute_norm_stats(cluster_id, client_id, values)
        X_batch = _normalize_tensor(X_batch, mean, scale)
        y_batch = _normalize_tensor(y_batch, mean, scale)

    payload = pickle.dumps({"X": X_batch, "y": y_batch})
    return Response(payload, content_type="application/octet-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=STREAM_PORT)
