#!/usr/bin/env python3
"""
Predict and plot split-learning forecasts from saved checkpoints.

What this script does
---------------------
1. Loads the saved client and server checkpoints for each cluster/client.
2. Rebuilds the exact test windows from the dataset.
3. Recomputes normalization using TRAIN data only.
4. Runs local inference (client-side split model -> server-side split model).
5. Inverse-transforms predictions/targets back to the original scale.
6. Saves:
   - per-client forecast plots
   - one overview plot per cluster
   - per-client metrics CSV/JSON
   - per-cluster summary CSV/JSON
   - global summary JSON

Default paths/ratios are aligned with your current split-learning setup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
# This script is intended to live in the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.split_lstm import LSTMConfig, build_split_lstm  # type: ignore


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_ids(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    import csv

    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else None


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------
def load_cluster_ids(assignments_path: Path) -> List[str]:
    df = pd.read_csv(assignments_path)
    out = sorted(df["Cluster"].astype(str).unique().tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    return out


def load_client_ids_for_cluster(assignments_path: Path, cluster_id: str) -> List[str]:
    df = pd.read_csv(assignments_path)
    rows = df[df["Cluster"].astype(str) == str(cluster_id)]
    return rows["ID"].astype(str).tolist()


class ClusterDataRepository:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.cluster_output_dir = dataset_dir / "output"
        self.raw_csv_path = dataset_dir / "dataset.csv"
        self.assignments_path = dataset_dir / "cluster-assignment.csv"
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_cluster_df(self, cluster_id: str) -> pd.DataFrame:
        cluster_id = str(cluster_id)
        if cluster_id in self._cache:
            return self._cache[cluster_id]

        pkl_path = self.cluster_output_dir / f"cluster_{cluster_id}.pkl"
        if pkl_path.exists():
            df = pd.read_pickle(pkl_path)
            df = df.copy()
        else:
            if not self.raw_csv_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.raw_csv_path}")
            if not self.assignments_path.exists():
                raise FileNotFoundError(f"Assignments not found: {self.assignments_path}")

            dataset = pd.read_csv(self.raw_csv_path)
            assignments = pd.read_csv(self.assignments_path)
            merged = dataset.melt(id_vars=["Fecha_Hora"], var_name="id", value_name="value")
            merged["id"] = merged["id"].astype(int)
            merged = merged.merge(assignments, left_on="id", right_on="ID")
            df = merged[merged["Cluster"].astype(str) == cluster_id].copy()

        if "Fecha_Hora" in df.columns:
            df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"])
        df["id"] = df["id"].astype(int)
        df = df.sort_values(["id", "Fecha_Hora"] if "Fecha_Hora" in df.columns else ["id"]).reset_index(drop=True)
        self._cache[cluster_id] = df
        return df


def get_split_bounds(num_values: int, window_size: int, horizon: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, Tuple[int, int]]:
    total_windows = num_values - window_size - horizon + 1
    if total_windows <= 0:
        return {"train": (0, 0), "val": (0, 0), "test": (0, 0)}

    train_end = int(total_windows * train_ratio)
    val_end = train_end + int(total_windows * val_ratio)
    test_end = total_windows
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, test_end),
    }


def compute_norm_stats(values: np.ndarray, window_size: int, horizon: int, train_ratio: float, val_ratio: float, test_ratio: float, normalize_method: str = "zscore", normalize_eps: float = 1e-8) -> Tuple[float, float]:
    bounds = get_split_bounds(len(values), window_size, horizon, train_ratio, val_ratio, test_ratio)
    _, train_end = bounds["train"]
    train_value_end = min(len(values), train_end + window_size + horizon)
    train_values = values[:train_value_end]

    if train_values.size == 0:
        return 0.0, 1.0

    if normalize_method == "minmax":
        min_v = float(np.min(train_values))
        max_v = float(np.max(train_values))
        scale = max(max_v - min_v, normalize_eps)
        return min_v, scale

    mean = float(np.mean(train_values))
    scale = float(np.std(train_values))
    if scale < normalize_eps:
        scale = 1.0
    return mean, scale


def build_windows(values: np.ndarray, timestamps: np.ndarray, split_name: str, window_size: int, horizon: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = get_split_bounds(len(values), window_size, horizon, train_ratio, val_ratio, test_ratio)
    start_idx, end_idx = bounds[split_name]

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ts_list: List[np.datetime64] = []

    for i in range(start_idx, end_idx):
        X_list.append(values[i : i + window_size])
        y_list.append(values[i + window_size + horizon - 1])
        ts_list.append(timestamps[i + window_size + horizon - 1])

    if not X_list:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype="datetime64[ns]")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    ts = np.asarray(ts_list)
    return X, y, ts


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def checkpoint_path_client(output_dir: Path, cluster_id: str, client_id: str, prefer_best: bool = True) -> Path:
    best = output_dir / f"cluster_{cluster_id}" / f"client_{client_id}_best_checkpoint.pt"
    latest = output_dir / f"cluster_{cluster_id}" / f"client_{client_id}_checkpoint.pt"
    if prefer_best and best.exists():
        return best
    if latest.exists():
        return latest
    return best


def checkpoint_path_server(output_dir: Path, cluster_id: str, prefer_best: bool = True) -> Path:
    best = output_dir / f"cluster_{cluster_id}" / "server_best.pt"
    latest = output_dir / f"cluster_{cluster_id}" / "server_latest.pt"
    if prefer_best and best.exists():
        return best
    if latest.exists():
        return latest
    return best


def load_checkpoint_state(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Unexpected checkpoint format in {path}")
    return ckpt


def build_models(model_cfg: LSTMConfig, split_layer: int, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    client_model, server_model = build_split_lstm(model_cfg, split_layer)
    client_model = client_model.to(device)
    server_model = server_model.to(device)
    client_model.eval()
    server_model.eval()
    return client_model, server_model


def load_weights_into_models(client_model: torch.nn.Module, server_model: torch.nn.Module, client_ckpt_path: Path, server_ckpt_path: Path, device: torch.device) -> Tuple[Dict, Dict]:
    client_ckpt = load_checkpoint_state(client_ckpt_path, device)
    server_ckpt = load_checkpoint_state(server_ckpt_path, device)
    client_model.load_state_dict(client_ckpt["model_state_dict"])
    server_model.load_state_dict(server_ckpt["model_state_dict"])
    return client_ckpt, server_ckpt




def build_inference_config(base_cfg: LSTMConfig, split_layer: int) -> LSTMConfig:
    """Avoid PyTorch LSTM dropout warnings for 1-layer split partitions.

    PyTorch only applies built-in LSTM dropout when a local LSTM has >1 layers.
    With total_layers=3 and split_layer=2, one side has 1 layer, so using
    dropout>0 triggers a warning even though inference still works. For this
    evaluation script, set dropout=0.0 when either split side has only 1 layer.
    Dropout is inactive in eval mode anyway, so this does not change inference.
    """
    client_layers = split_layer
    server_layers = base_cfg.num_layers - split_layer
    dropout = 0.0 if (client_layers <= 1 or server_layers <= 1) else base_cfg.dropout
    return LSTMConfig(
        input_size=base_cfg.input_size,
        hidden_size=base_cfg.hidden_size,
        num_layers=base_cfg.num_layers,
        output_size=base_cfg.output_size,
        dropout=dropout,
        bidirectional=base_cfg.bidirectional,
    )

# -----------------------------------------------------------------------------
# Metrics and inference
# -----------------------------------------------------------------------------
def inverse_transform(arr: np.ndarray, normalize: bool, method: str, mean: float, scale: float) -> np.ndarray:
    if not normalize:
        return arr
    if method.lower() in ("zscore", "minmax"):
        return arr * scale + mean
    return arr


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Optional[float]]:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    err = preds - targets
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    denom = np.where(np.abs(targets) < 1e-8, np.nan, np.abs(targets))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0) if np.any(~np.isnan(denom)) else None

    smape_denom = np.abs(preds) + np.abs(targets)
    smape = float(np.nanmean(np.where(smape_denom < 1e-8, np.nan, 2.0 * np.abs(err) / smape_denom)) * 100.0) if np.any(smape_denom >= 1e-8) else None

    target_abs_sum = float(np.sum(np.abs(targets)))
    wape = float(np.sum(np.abs(err)) / target_abs_sum * 100.0) if target_abs_sum >= 1e-8 else None

    ss_res = float(np.sum(np.square(err)))
    target_mean = float(np.mean(targets))
    ss_tot = float(np.sum(np.square(targets - target_mean)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot >= 1e-12 else None

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "r2": r2,
        "num_samples": int(targets.shape[0]),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "pred_mean": float(np.mean(preds)),
        "pred_std": float(np.std(preds)),
    }


def run_local_inference(client_model: torch.nn.Module, server_model: torch.nn.Module, X_norm: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    preds_all: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_norm), batch_size):
            batch = X_norm[start : start + batch_size]
            X = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(-1)
            activations = client_model(X)
            if activations.dim() == 3:
                activations = activations[:, -1, :]
            if activations.dim() == 2:
                activations = activations.unsqueeze(1)
            preds = server_model(activations)
            preds_all.append(preds.detach().cpu().numpy().reshape(-1, 1))
    if not preds_all:
        return np.empty((0, 1), dtype=np.float32)
    return np.concatenate(preds_all, axis=0)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_client_forecast(save_path: Path, timestamps: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, cluster_id: str, client_id: str, metrics: Dict[str, Optional[float]], last_n: int = 200) -> None:
    ensure_dir(save_path.parent)
    n = min(last_n, len(y_true))
    ts_plot = pd.to_datetime(timestamps[-n:])
    yt = y_true[-n:].reshape(-1)
    yp = y_pred[-n:].reshape(-1)

    plt.figure(figsize=(12, 4.5))
    plt.plot(ts_plot, yt, label="True")
    plt.plot(ts_plot, yp, label="Pred")
    plt.title(
        f"Cluster {cluster_id} | Client {client_id} | Test forecast\n"
        f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} WAPE={metrics['wape']:.2f}%"
        if metrics.get("mae") is not None and metrics.get("rmse") is not None and metrics.get("wape") is not None
        else f"Cluster {cluster_id} | Client {client_id} | Test forecast"
    )
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_cluster_overview(save_path: Path, client_payloads: List[Dict], max_clients: Optional[int] = None, last_n: int = 168) -> None:
    ensure_dir(save_path.parent)
    payloads = client_payloads[:max_clients] if max_clients is not None else client_payloads
    if not payloads:
        return

    n_clients = len(payloads)
    ncols = 2 if n_clients > 1 else 1
    nrows = int(math.ceil(n_clients / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, payload in zip(axes_flat, payloads):
        timestamps = pd.to_datetime(payload["timestamps"][-last_n:])
        y_true = np.asarray(payload["y_true"][-last_n:], dtype=np.float32).reshape(-1)
        y_pred = np.asarray(payload["y_pred"][-last_n:], dtype=np.float32).reshape(-1)
        m = payload["metrics"]
        ax.plot(timestamps, y_true, label="True")
        ax.plot(timestamps, y_pred, label="Pred")
        ax.set_title(f"Client {payload['client_id']} | MAE={m['mae']:.3f} RMSE={m['rmse']:.3f}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=25)

    for ax in axes_flat[n_clients:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main evaluation logic
# -----------------------------------------------------------------------------
def evaluate_one_client(
    repo: ClusterDataRepository,
    output_dir: Path,
    cluster_id: str,
    client_id: str,
    model_cfg: LSTMConfig,
    split_layer: int,
    device: torch.device,
    window_size: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    batch_size: int,
    normalize: bool,
    normalize_method: str,
    normalize_eps: float,
    plots_dir: Path,
    prefer_best: bool,
    last_n_plot: int,
) -> Dict:
    cluster_df = repo.load_cluster_df(cluster_id)
    cdf = cluster_df[cluster_df["id"].astype(str) == str(client_id)].copy()
    if cdf.empty:
        raise ValueError(f"No data found for client {client_id} in cluster {cluster_id}")

    values = cdf["value"].to_numpy(dtype=np.float32)
    timestamps = cdf["Fecha_Hora"].to_numpy()
    X_test, y_test_raw, ts_test = build_windows(
        values=values,
        timestamps=timestamps,
        split_name="test",
        window_size=window_size,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    if len(X_test) == 0:
        raise ValueError(f"No test windows for client {client_id} in cluster {cluster_id}")

    mean, scale = compute_norm_stats(
        values=values,
        window_size=window_size,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize_method=normalize_method,
        normalize_eps=normalize_eps,
    )

    X_test_norm = X_test.copy()
    y_test_norm = y_test_raw.copy().reshape(-1, 1)
    if normalize:
        X_test_norm = (X_test_norm - mean) / scale
        y_test_norm = (y_test_norm - mean) / scale

    inference_cfg = build_inference_config(model_cfg, split_layer)
    client_model, server_model = build_models(inference_cfg, split_layer, device)
    client_ckpt_path = checkpoint_path_client(output_dir, cluster_id, client_id, prefer_best=prefer_best)
    server_ckpt_path = checkpoint_path_server(output_dir, cluster_id, prefer_best=prefer_best)
    client_ckpt, server_ckpt = load_weights_into_models(client_model, server_model, client_ckpt_path, server_ckpt_path, device)

    preds_norm = run_local_inference(client_model, server_model, X_test_norm, batch_size=batch_size, device=device)
    preds_raw = inverse_transform(preds_norm, normalize=normalize, method=normalize_method, mean=mean, scale=scale)
    y_raw = y_test_raw.reshape(-1, 1)

    metrics = compute_metrics(preds_raw, y_raw)
    plot_path = plots_dir / f"cluster_{cluster_id}" / f"client_{client_id}_forecast.png"
    plot_client_forecast(plot_path, ts_test, y_raw, preds_raw, cluster_id, client_id, metrics, last_n=last_n_plot)

    return {
        "cluster_id": str(cluster_id),
        "client_id": str(client_id),
        "num_test_samples": int(len(y_raw)),
        "checkpoint_client": str(client_ckpt_path),
        "checkpoint_server": str(server_ckpt_path),
        "client_checkpoint_epoch": int(client_ckpt.get("epoch", -1)) if isinstance(client_ckpt, dict) else None,
        "server_global_step": int(server_ckpt.get("global_step", -1)) if isinstance(server_ckpt, dict) else None,
        "normalize": bool(normalize),
        "normalize_method": normalize_method,
        "norm_mean": float(mean),
        "norm_scale": float(scale),
        "plot_path": str(plot_path),
        **metrics,
        "_plot_payload": {
            "cluster_id": str(cluster_id),
            "client_id": str(client_id),
            "timestamps": ts_test.astype("datetime64[ns]").astype(str).tolist(),
            "y_true": y_raw.reshape(-1).tolist(),
            "y_pred": preds_raw.reshape(-1).tolist(),
            "metrics": metrics,
        },
    }


def summarize_by_cluster(rows: List[Dict]) -> List[Dict]:
    from collections import defaultdict

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)

    out: List[Dict] = []
    for cluster_id, items in sorted(grouped.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        out.append(
            {
                "cluster_id": cluster_id,
                "num_clients": len(items),
                "avg_num_test_samples": safe_mean([r.get("num_test_samples") for r in items]),
                "avg_mae": safe_mean([r.get("mae") for r in items]),
                "avg_rmse": safe_mean([r.get("rmse") for r in items]),
                "avg_mse": safe_mean([r.get("mse") for r in items]),
                "avg_mape": safe_mean([r.get("mape") for r in items]),
                "avg_smape": safe_mean([r.get("smape") for r in items]),
                "avg_wape": safe_mean([r.get("wape") for r in items]),
                "avg_r2": safe_mean([r.get("r2") for r in items]),
                "best_client_by_mae": min(items, key=lambda r: float("inf") if r.get("mae") is None else r["mae"])["client_id"],
                "worst_client_by_mae": max(items, key=lambda r: -float("inf") if r.get("mae") is None else r["mae"])["client_id"],
            }
        )
    return out


def build_global_summary(rows: List[Dict]) -> Dict:
    return {
        "num_clients": len(rows),
        "avg_num_test_samples": safe_mean([r.get("num_test_samples") for r in rows]),
        "avg_mae": safe_mean([r.get("mae") for r in rows]),
        "avg_rmse": safe_mean([r.get("rmse") for r in rows]),
        "avg_mse": safe_mean([r.get("mse") for r in rows]),
        "avg_mape": safe_mean([r.get("mape") for r in rows]),
        "avg_smape": safe_mean([r.get("smape") for r in rows]),
        "avg_wape": safe_mean([r.get("wape") for r in rows]),
        "avg_r2": safe_mean([r.get("r2") for r in rows]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict and plot saved split-learning checkpoints.")
    p.add_argument("--project-root", type=Path, default=Path("."), help="Project root containing dataset/, output/, models/")
    p.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Dataset directory")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Training output directory containing checkpoints")
    p.add_argument("--report-dir", type=Path, default=Path("output") / "prediction_report", help="Where to save plots and reports")
    p.add_argument("--clusters", type=str, default=None, help="Comma-separated cluster IDs. Default: all")
    p.add_argument("--clients", type=str, default=None, help="Comma-separated client IDs. Default: all clients in selected clusters")
    p.add_argument("--window-size", type=int, default=24)
    p.add_argument("--forecast-horizon", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--normalize-method", type=str, default="zscore", choices=["zscore", "minmax"])
    p.add_argument("--normalize-eps", type=float, default=1e-8)
    p.add_argument("--model-input-size", type=int, default=1)
    p.add_argument("--model-hidden-size", type=int, default=128)
    p.add_argument("--model-num-layers", type=int, default=3)
    p.add_argument("--model-output-size", type=int, default=1)
    p.add_argument("--model-dropout", type=float, default=0.4)
    p.add_argument("--model-bidirectional", action="store_true", default=False)
    p.add_argument("--model-split-layer", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--prefer-best", action="store_true", default=True, help="Prefer *_best_checkpoint.pt if available")
    p.add_argument("--prefer-latest", dest="prefer_best", action="store_false")
    p.add_argument("--last-n-plot", type=int, default=200, help="How many last test points to show in each client plot")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    dataset_dir = (project_root / args.dataset_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    report_dir = (project_root / args.report_dir).resolve()
    plots_dir = report_dir / "plots"
    ensure_dir(report_dir)
    ensure_dir(plots_dir)

    assignments_path = dataset_dir / "cluster-assignment.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    selected_clusters = parse_csv_ids(args.clusters)
    selected_clients = parse_csv_ids(args.clients)

    repo = ClusterDataRepository(dataset_dir)
    all_clusters = load_cluster_ids(assignments_path)
    cluster_ids = selected_clusters if selected_clusters is not None else all_clusters

    model_cfg = LSTMConfig(
        input_size=args.model_input_size,
        hidden_size=args.model_hidden_size,
        num_layers=args.model_num_layers,
        output_size=args.model_output_size,
        dropout=args.model_dropout,
        bidirectional=args.model_bidirectional,
    )

    if not (1 <= args.model_split_layer < args.model_num_layers):
        raise ValueError(f"model_split_layer must be in [1, num_layers-1], got split={args.model_split_layer}, layers={args.model_num_layers}")

    log(f"Device: {device}")
    log(f"Project root: {project_root}")
    log(f"Dataset dir: {dataset_dir}")
    log(f"Output dir: {output_dir}")
    log(f"Report dir: {report_dir}")

    per_client_rows: List[Dict] = []
    cluster_plot_payloads: Dict[str, List[Dict]] = {}

    for cluster_id in cluster_ids:
        client_ids = load_client_ids_for_cluster(assignments_path, cluster_id)
        if selected_clients is not None:
            client_ids = [cid for cid in client_ids if cid in selected_clients]
        if not client_ids:
            log(f"Skipping cluster {cluster_id}: no selected clients")
            continue

        log(f"\n[Cluster {cluster_id}] clients={len(client_ids)}")
        cluster_plot_payloads[str(cluster_id)] = []

        for client_id in client_ids:
            log(f"  Evaluating client {client_id} ...")
            row = evaluate_one_client(
                repo=repo,
                output_dir=output_dir,
                cluster_id=str(cluster_id),
                client_id=str(client_id),
                model_cfg=model_cfg,
                split_layer=args.model_split_layer,
                device=device,
                window_size=args.window_size,
                horizon=args.forecast_horizon,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                batch_size=args.batch_size,
                normalize=args.normalize,
                normalize_method=args.normalize_method,
                normalize_eps=args.normalize_eps,
                plots_dir=plots_dir,
                prefer_best=args.prefer_best,
                last_n_plot=args.last_n_plot,
            )
            cluster_plot_payloads[str(cluster_id)].append(row.pop("_plot_payload"))
            per_client_rows.append(row)
            log(f"    MAE={row['mae']:.6f} RMSE={row['rmse']:.6f} WAPE={row['wape']:.3f}% R2={row['r2']}")

    if not per_client_rows:
        raise RuntimeError("No client results were produced.")

    cluster_rows = summarize_by_cluster(per_client_rows)
    global_summary = build_global_summary(per_client_rows)

    for cluster_id, payloads in cluster_plot_payloads.items():
        plot_cluster_overview(
            save_path=plots_dir / f"cluster_{cluster_id}" / f"cluster_{cluster_id}_overview.png",
            client_payloads=payloads,
            last_n=args.last_n_plot,
        )

    # JSON-safe copy without large plot payloads.
    report = {
        "config": {
            "project_root": str(project_root),
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "window_size": args.window_size,
            "forecast_horizon": args.forecast_horizon,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "normalize": args.normalize,
            "normalize_method": args.normalize_method,
            "model": asdict(model_cfg),
            "model_split_layer": args.model_split_layer,
            "device": str(device),
            "prefer_best": args.prefer_best,
        },
        "clients": per_client_rows,
        "clusters": cluster_rows,
        "global_summary": global_summary,
        "note": "MAPE can explode when true target values are close to zero; WAPE and sMAPE are included for a more stable view.",
    }

    save_json(report_dir / "prediction_report.json", report)
    save_csv(
        report_dir / "prediction_report_clients.csv",
        per_client_rows,
        [
            "cluster_id",
            "client_id",
            "num_test_samples",
            "client_checkpoint_epoch",
            "server_global_step",
            "mae",
            "rmse",
            "mse",
            "mape",
            "smape",
            "wape",
            "r2",
            "target_mean",
            "target_std",
            "pred_mean",
            "pred_std",
            "plot_path",
            "checkpoint_client",
            "checkpoint_server",
        ],
    )
    save_csv(
        report_dir / "prediction_report_clusters.csv",
        cluster_rows,
        [
            "cluster_id",
            "num_clients",
            "avg_num_test_samples",
            "avg_mae",
            "avg_rmse",
            "avg_mse",
            "avg_mape",
            "avg_smape",
            "avg_wape",
            "avg_r2",
            "best_client_by_mae",
            "worst_client_by_mae",
        ],
    )

    log("\nCluster summary:")
    for row in cluster_rows:
        log(str(row))
    log("\nGlobal summary:")
    log(str(global_summary))
    log(f"\nSaved report to: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
