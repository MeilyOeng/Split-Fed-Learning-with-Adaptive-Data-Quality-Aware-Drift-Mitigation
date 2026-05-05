#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "output" / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from models.split_lstm import LSTMConfig, build_split_lstm  # type: ignore


CLIENT_CONFIG_RE = re.compile(r"client_(.+?)_config\.json$")
CLIENT_ARTIFACT_RE = re.compile(
    r"client_(.+?)_(?:config\.json|best_checkpoint\.pt|checkpoint\.pt|final\.pt|test_metrics\.json|train_metrics\.json)$"
)

ERROR_METRICS = ("mae", "rmse", "mse", "mape", "smape", "wape")
SCORE_METRICS = ("r2",)
DEFAULT_TEST_REPORT_CANDIDATES = (
    Path("output") / "results" / "test_report.csv",
    Path("output") / "result" / "test_report.csv",
    Path("dataset") / "test_report.csv",
)
DEFAULT_BASELINE_CANDIDATES = (
    Path("dataset") / "lstm_results_updated.csv",
    Path("dataset") / "lstm_result_updated.csv",
)


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_csv_ids(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def default_report_dir_for_output(output_dir: Path) -> Path:
    name = output_dir.name.strip()
    if name == "output":
        return Path("report")
    if name.startswith("output-"):
        return Path("report-" + name[len("output-"):])
    if name.startswith("output"):
        return Path(name.replace("output", "report", 1))
    return Path(f"report-{name}")


def resolve_input_path(
    project_root: Path,
    explicit_path: Optional[Path],
    candidates: Sequence[Path] | Path,
    *,
    label: str,
) -> Path:
    search_paths: List[Path] = []
    if explicit_path is not None:
        search_paths.append(explicit_path if explicit_path.is_absolute() else project_root / explicit_path)
    candidate_paths = (candidates,) if isinstance(candidates, Path) else candidates
    for candidate in candidate_paths:
        search_paths.append(candidate if candidate.is_absolute() else project_root / candidate)
    for path in search_paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"Could not find {label}. Checked: {', '.join(str(path) for path in search_paths)}"
    )


def default_test_report_candidates_for_output_dir(output_dir: Path) -> List[Path]:
    return [
        output_dir / "results" / "test_report.csv",
        output_dir / "result" / "test_report.csv",
        *DEFAULT_TEST_REPORT_CANDIDATES,
    ]


def normalize_id_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else None


def safe_quantile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q))


def metric_higher_is_better(metric: str) -> bool:
    return metric in SCORE_METRICS


def clean_metric_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def get_metric_value(row: Dict, metric: str) -> Optional[float]:
    return clean_metric_value(row.get(metric))


def select_best_row(rows: Sequence[Dict], metric: str) -> Optional[Dict]:
    valid = [row for row in rows if get_metric_value(row, metric) is not None]
    if not valid:
        return None
    if metric_higher_is_better(metric):
        return max(valid, key=lambda row: float(row[metric]))
    return min(valid, key=lambda row: float(row[metric]))


def select_worst_row(rows: Sequence[Dict], metric: str) -> Optional[Dict]:
    valid = [row for row in rows if get_metric_value(row, metric) is not None]
    if not valid:
        return None
    if metric_higher_is_better(metric):
        return min(valid, key=lambda row: float(row[metric]))
    return max(valid, key=lambda row: float(row[metric]))


def sort_rows_by_metric(rows: Sequence[Dict], metric: str, *, best_first: bool) -> List[Dict]:
    higher_is_better = metric_higher_is_better(metric)
    valid = [row for row in rows if get_metric_value(row, metric) is not None]
    reverse = higher_is_better if best_first else not higher_is_better
    return sorted(valid, key=lambda row: float(row[metric]), reverse=reverse)


def discover_clusters(output_dir: Path) -> List[str]:
    cluster_ids: List[str] = []
    if not output_dir.exists():
        return cluster_ids
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and child.name.startswith("cluster_"):
            cluster_ids.append(child.name.split("cluster_", 1)[1])
    return cluster_ids


def discover_clients(output_dir: Path, cluster_id: str) -> List[str]:
    cluster_dir = output_dir / f"cluster_{cluster_id}"
    if not cluster_dir.exists():
        return []
    discovered = set()
    for path in cluster_dir.iterdir():
        match = CLIENT_ARTIFACT_RE.match(path.name)
        if match:
            discovered.add(match.group(1))
    return sorted(discovered, key=lambda x: int(x) if x.isdigit() else x)


def client_config_path(output_dir: Path, cluster_id: str, client_id: str) -> Path:
    return output_dir / f"cluster_{cluster_id}" / f"client_{client_id}_config.json"


def client_test_metrics_path(output_dir: Path, cluster_id: str, client_id: str) -> Path:
    return output_dir / f"cluster_{cluster_id}" / f"client_{client_id}_test_metrics.json"


def client_checkpoint_candidates(output_dir: Path, cluster_id: str, client_id: str, checkpoint_mode: str) -> List[Path]:
    cluster_dir = output_dir / f"cluster_{cluster_id}"
    best = cluster_dir / f"client_{client_id}_best_checkpoint.pt"
    latest = cluster_dir / f"client_{client_id}_checkpoint.pt"
    final = cluster_dir / f"client_{client_id}_final.pt"
    if checkpoint_mode == "paired_best":
        return [best, latest, final]
    if checkpoint_mode in {"final_shared", "final"}:
        return [final, latest, best]
    if checkpoint_mode == "latest":
        return [latest, final, best]
    raise ValueError(f"Unsupported checkpoint mode: {checkpoint_mode}")


def server_checkpoint_candidates(output_dir: Path, cluster_id: str, checkpoint_mode: str) -> List[Path]:
    cluster_dir = output_dir / f"cluster_{cluster_id}"
    best = cluster_dir / "server_best.pt"
    latest = cluster_dir / "server_latest.pt"
    final = cluster_dir / "server_final.pt"
    if checkpoint_mode == "paired_best":
        return [best, latest, final]
    if checkpoint_mode in {"final_shared", "final"}:
        return [final, latest, best]
    if checkpoint_mode == "latest":
        return [latest, final, best]
    raise ValueError(f"Unsupported checkpoint mode: {checkpoint_mode}")


def first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


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
            try:
                df = pd.read_pickle(pkl_path).copy()
            except Exception as exc:
                log(f"[WARN] Failed to read {pkl_path}; falling back to raw CSV merge: {exc}")
                df = self._load_cluster_df_from_csv(cluster_id)
        else:
            df = self._load_cluster_df_from_csv(cluster_id)

        if "Fecha_Hora" in df.columns:
            df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"], errors="coerce")
        df["id"] = df["id"].astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values(["id", "Fecha_Hora"] if "Fecha_Hora" in df.columns else ["id"]).reset_index(drop=True)
        self._cache[cluster_id] = df
        return df

    def _load_cluster_df_from_csv(self, cluster_id: str) -> pd.DataFrame:
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.raw_csv_path}")
        if not self.assignments_path.exists():
            raise FileNotFoundError(f"Assignments not found: {self.assignments_path}")

        dataset = pd.read_csv(self.raw_csv_path)
        assignments = pd.read_csv(self.assignments_path)
        merged = dataset.melt(id_vars=["Fecha_Hora"], var_name="id", value_name="value")
        merged["id"] = merged["id"].astype(int)
        merged = merged.merge(assignments, left_on="id", right_on="ID")
        return merged[merged["Cluster"].astype(str) == str(cluster_id)].copy()


def build_all_windows(values: np.ndarray, timestamps: np.ndarray, window_size: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_windows = len(values) - window_size - horizon + 1
    if total_windows <= 0:
        return (
            np.empty((0, window_size), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ts_list: List[np.datetime64] = []
    for start in range(total_windows):
        X_list.append(values[start : start + window_size])
        target_idx = start + window_size + horizon - 1
        y_list.append(values[target_idx])
        ts_list.append(timestamps[target_idx])

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(ts_list),
    )


def split_windows_by_counts(
    X_all: np.ndarray,
    y_all: np.ndarray,
    ts_all: np.ndarray,
    *,
    num_train_windows: int,
    num_val_windows: int,
    num_test_windows: int,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_map = {
        "train": 0,
        "val": num_train_windows,
        "test": num_train_windows + num_val_windows,
    }
    count_map = {
        "train": num_train_windows,
        "val": num_val_windows,
        "test": num_test_windows,
    }
    if split_name not in start_map:
        raise ValueError(f"Unsupported split: {split_name}")
    start = start_map[split_name]
    end = start + count_map[split_name]
    return X_all[start:end], y_all[start:end], ts_all[start:end]


def get_split_bounds(num_values: int, window_size: int, horizon: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, Tuple[int, int]]:
    total_windows = num_values - window_size - horizon + 1
    if total_windows <= 0:
        return {"train": (0, 0), "val": (0, 0), "test": (0, 0)}
    train_end = int(total_windows * train_ratio)
    val_end = train_end + int(total_windows * val_ratio)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_windows),
    }


def split_windows_by_ratios(
    X_all: np.ndarray,
    y_all: np.ndarray,
    ts_all: np.ndarray,
    *,
    window_size: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = get_split_bounds(len(X_all) + window_size + horizon - 1, window_size, horizon, train_ratio, val_ratio, test_ratio)
    start, end = bounds[split_name]
    return X_all[start:end], y_all[start:end], ts_all[start:end]


def normalize_array(arr: np.ndarray, normalize: bool, method: str, mean: float, scale: float) -> np.ndarray:
    if not normalize:
        return arr
    if method.lower() in {"zscore", "minmax"}:
        return (arr - mean) / scale
    return arr


def inverse_transform(arr: np.ndarray, normalize: bool, method: str, mean: float, scale: float) -> np.ndarray:
    if not normalize:
        return arr
    if method.lower() in {"zscore", "minmax"}:
        return arr * scale + mean
    return arr


def apply_sign_flip(arr: np.ndarray, sign_flip: bool) -> np.ndarray:
    return (-arr) if sign_flip else arr


def undo_sign_flip(arr: np.ndarray, sign_flip: bool) -> np.ndarray:
    return (-arr) if sign_flip else arr


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Optional[float]]:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    if preds.size == 0:
        return {metric: None for metric in ("mse", "mae", "rmse", "mape", "smape", "wape", "r2")}
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


def load_checkpoint_state(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format in {path}")
    return checkpoint


def build_models(model_cfg: LSTMConfig, split_layer: int, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    client_model, server_model = build_split_lstm(model_cfg, split_layer)
    client_model = client_model.to(device).eval()
    server_model = server_model.to(device).eval()
    return client_model, server_model


def load_weights_into_models(
    client_model: torch.nn.Module,
    server_model: torch.nn.Module,
    *,
    client_ckpt_path: Path,
    server_ckpt_path: Path,
    device: torch.device,
) -> Tuple[Dict, Dict]:
    client_ckpt = load_checkpoint_state(client_ckpt_path, device)
    server_ckpt = load_checkpoint_state(server_ckpt_path, device)
    client_model.load_state_dict(client_ckpt["model_state_dict"])
    server_model.load_state_dict(server_ckpt["model_state_dict"])
    return client_ckpt, server_ckpt


def build_inference_config(base_cfg: LSTMConfig, split_layer: int) -> LSTMConfig:
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


def run_local_inference(
    client_model: torch.nn.Module,
    server_model: torch.nn.Module,
    X_norm: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
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


@dataclass
class ClientArtifactConfig:
    cluster_id: str
    client_id: str
    model_cfg: LSTMConfig
    split_layer: int
    eval_mode: str
    normalize: bool
    normalize_method: str
    sign_flip: bool
    mean: float
    scale: float
    num_train_windows: Optional[int]
    num_val_windows: Optional[int]
    num_test_windows: Optional[int]


def load_client_artifact_config(output_dir: Path, cluster_id: str, client_id: str) -> ClientArtifactConfig:
    cfg_path = client_config_path(output_dir, cluster_id, client_id)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing client config artifact: {cfg_path}")
    cfg = load_json(cfg_path)
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    normalization = cfg.get("normalization", {})
    return ClientArtifactConfig(
        cluster_id=str(cluster_id),
        client_id=str(client_id),
        model_cfg=LSTMConfig(
            input_size=int(model.get("input_size", 1)),
            hidden_size=int(model.get("hidden_size", 128)),
            num_layers=int(model.get("num_layers", 3)),
            output_size=int(model.get("output_size", 1)),
            dropout=float(model.get("dropout", 0.4)),
            bidirectional=bool(model.get("bidirectional", False)),
        ),
        split_layer=int(model.get("split_layer", 2)),
        eval_mode=str(training.get("eval_mode", "final_shared")),
        normalize=bool(normalization.get("normalize", True)),
        normalize_method=str(normalization.get("method", "zscore")),
        sign_flip=bool(normalization.get("sign_flip", False)),
        mean=float(normalization.get("mean", 0.0)),
        scale=float(normalization.get("scale", 1.0)),
        num_train_windows=(None if normalization.get("num_train_windows") is None else int(normalization["num_train_windows"])),
        num_val_windows=(None if normalization.get("num_val_windows") is None else int(normalization["num_val_windows"])),
        num_test_windows=(None if normalization.get("num_test_windows") is None else int(normalization["num_test_windows"])),
    )


def resolve_checkpoint_mode(requested_mode: str, artifact_cfg: ClientArtifactConfig) -> str:
    if requested_mode != "auto":
        return requested_mode
    mode = str(artifact_cfg.eval_mode).strip().lower()
    if mode in {"paired_best", "final_shared", "final", "latest"}:
        return mode
    return "final_shared"


def maybe_load_saved_test_metrics(output_dir: Path, cluster_id: str, client_id: str) -> Optional[Dict]:
    path = client_test_metrics_path(output_dir, cluster_id, client_id)
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def save_prediction_points_csv(path: Path, timestamps: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    ensure_dir(path.parent)
    errors = (y_pred.reshape(-1) - y_true.reshape(-1))
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps).astype(str),
            "y_true": y_true.reshape(-1),
            "y_pred": y_pred.reshape(-1),
            "abs_error": np.abs(errors),
            "sq_error": np.square(errors),
        }
    )
    df.to_csv(path, index=False)


def plot_client_forecast(
    save_path: Path,
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    cluster_id: str,
    client_id: str,
    metrics: Dict[str, Optional[float]],
    last_n: int,
) -> None:
    ensure_dir(save_path.parent)
    n = min(last_n, len(y_true))
    ts_plot = pd.to_datetime(timestamps[-n:])
    yt = y_true[-n:].reshape(-1)
    yp = y_pred[-n:].reshape(-1)

    plt.figure(figsize=(12, 4.8))
    plt.plot(ts_plot, yt, label="True", linewidth=1.4)
    plt.plot(ts_plot, yp, label="Pred", linewidth=1.2)
    title = (
        f"Cluster {cluster_id} | Client {client_id} | MAE={metrics.get('mae'):.4f} "
        f"RMSE={metrics.get('rmse'):.4f} WAPE={metrics.get('wape'):.2f}% R2={metrics.get('r2'):.4f}"
        if metrics.get("mae") is not None and metrics.get("rmse") is not None and metrics.get("wape") is not None and metrics.get("r2") is not None
        else f"Cluster {cluster_id} | Client {client_id}"
    )
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_cluster_overview(save_path: Path, payloads: List[Dict], *, max_clients: Optional[int], last_n: int) -> None:
    ensure_dir(save_path.parent)
    if not payloads:
        return
    sorted_payloads = sorted(payloads, key=lambda item: float(item["metrics"].get("mae", float("inf"))))
    payloads = sorted_payloads[:max_clients] if max_clients is not None else sorted_payloads
    n_clients = len(payloads)
    ncols = 1 if n_clients == 1 else (2 if n_clients <= 8 else 3)
    nrows = int(math.ceil(n_clients / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, payload in zip(axes_flat, payloads):
        timestamps = pd.to_datetime(payload["timestamps"][-last_n:])
        y_true = np.asarray(payload["y_true"][-last_n:], dtype=np.float32)
        y_pred = np.asarray(payload["y_pred"][-last_n:], dtype=np.float32)
        m = payload["metrics"]
        ax.plot(timestamps, y_true, label="True", linewidth=1.2)
        ax.plot(timestamps, y_pred, label="Pred", linewidth=1.0)
        ax.set_title(f"Client {payload['client_id']} | MAE={m.get('mae'):.3f} RMSE={m.get('rmse'):.3f}")
        ax.tick_params(axis="x", rotation=25)

    for ax in axes_flat[n_clients:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_worst_pair(save_path: Path, *, best_payload: Dict, worst_payload: Dict, metric: str, last_n: int, title: str) -> None:
    ensure_dir(save_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), squeeze=False)
    axes_flat = axes.flatten()
    for ax, payload, label in zip(axes_flat, [best_payload, worst_payload], ["Best", "Worst"]):
        timestamps = pd.to_datetime(payload["timestamps"][-last_n:])
        y_true = np.asarray(payload["y_true"][-last_n:], dtype=np.float32)
        y_pred = np.asarray(payload["y_pred"][-last_n:], dtype=np.float32)
        value = payload["metrics"].get(metric)
        ax.plot(timestamps, y_true, label="True", linewidth=1.3)
        ax.plot(timestamps, y_pred, label="Pred", linewidth=1.1)
        ax.set_title(
            f"{label}: cluster={payload['cluster_id']} client={payload['client_id']} "
            f"{metric.upper()}={value:.4f}" if value is not None else label
        )
        ax.tick_params(axis="x", rotation=25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_metric_boxplot(save_path: Path, rows: List[Dict], metric: str) -> None:
    ensure_dir(save_path.parent)
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        value = get_metric_value(row, metric)
        if value is None:
            continue
        grouped.setdefault(str(row["cluster_id"]), []).append(float(value))
    if not grouped:
        return
    clusters = sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)
    data = [grouped[c] for c in clusters]
    plt.figure(figsize=(max(8, 1.4 * len(clusters)), 5))
    plt.boxplot(data, tick_labels=clusters, showmeans=True)
    plt.xlabel("Cluster")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} distribution across clients")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_cluster_avg_worst(save_path: Path, cluster_rows: List[Dict], metric: str) -> None:
    ensure_dir(save_path.parent)
    if not cluster_rows:
        return
    avg_key = f"avg_{metric}"
    worst_key = f"worst_{metric}"
    labels = [str(row["cluster_id"]) for row in cluster_rows]
    avg_values = [row.get(avg_key) or 0.0 for row in cluster_rows]
    worst_values = [row.get(worst_key) or 0.0 for row in cluster_rows]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(max(8, 1.5 * len(labels)), 5))
    plt.bar(x - width / 2, avg_values, width=width, label="Average")
    plt.bar(x + width / 2, worst_values, width=width, label="Worst")
    plt.xticks(x, labels)
    plt.xlabel("Cluster")
    plt.ylabel(metric.upper())
    plt.title(f"Average vs worst-client {metric.upper()} by cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_metric_ecdf(save_path: Path, rows: List[Dict], metric: str) -> None:
    ensure_dir(save_path.parent)
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        value = get_metric_value(row, metric)
        if value is None:
            continue
        grouped.setdefault(str(row["cluster_id"]), []).append(float(value))
    if not grouped:
        return
    plt.figure(figsize=(8.5, 5))
    for cluster_id in sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x):
        values = np.sort(np.asarray(grouped[cluster_id], dtype=np.float64))
        y = np.arange(1, len(values) + 1) / len(values)
        plt.step(values, y, where="post", label=f"Cluster {cluster_id}")
    plt.xlabel(metric.upper())
    plt.ylabel("Empirical CDF")
    plt.title(f"Client-level {metric.upper()} distribution by cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def summarize_error_metric(rows: List[Dict], metric: str) -> Dict[str, Optional[float]]:
    values = [get_metric_value(row, metric) for row in rows]
    mean = safe_mean(values)
    median = safe_quantile(values, 0.50)
    p90 = safe_quantile(values, 0.90)
    p95 = safe_quantile(values, 0.95)
    best = None
    worst = None
    best_row = select_best_row(rows, metric)
    worst_row = select_worst_row(rows, metric)
    if best_row is not None:
        best = float(best_row[metric])
    if worst_row is not None:
        worst = float(worst_row[metric])
    return {
        f"avg_{metric}": mean,
        f"median_{metric}": median,
        f"p90_{metric}": p90,
        f"p95_{metric}": p95,
        f"best_{metric}": best,
        f"worst_{metric}": worst,
        f"worst_over_avg_{metric}": (None if mean in (None, 0.0) or worst is None else float(worst / mean)),
    }


def summarize_score_metric(rows: List[Dict], metric: str) -> Dict[str, Optional[float]]:
    values = [get_metric_value(row, metric) for row in rows]
    mean = safe_mean(values)
    median = safe_quantile(values, 0.50)
    p10 = safe_quantile(values, 0.10)
    p05 = safe_quantile(values, 0.05)
    best = None
    worst = None
    best_row = select_best_row(rows, metric)
    worst_row = select_worst_row(rows, metric)
    if best_row is not None:
        best = float(best_row[metric])
    if worst_row is not None:
        worst = float(worst_row[metric])
    return {
        f"avg_{metric}": mean,
        f"median_{metric}": median,
        f"p10_{metric}": p10,
        f"p05_{metric}": p05,
        f"best_{metric}": best,
        f"worst_{metric}": worst,
        f"avg_minus_worst_{metric}": (None if mean is None or worst is None else float(mean - worst)),
    }


def summarize_cluster_rows(rows: List[Dict]) -> List[Dict]:
    from collections import defaultdict

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)

    out: List[Dict] = []
    for cluster_id, items in sorted(grouped.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        cluster_row = {
            "cluster_id": cluster_id,
            "num_clients": len(items),
            "avg_num_test_samples": safe_mean([row.get("num_test_samples") for row in items]),
            "best_client_by_mae": None,
            "worst_client_by_mae": None,
            "best_client_by_r2": None,
            "worst_client_by_r2": None,
        }
        cluster_row.update(summarize_error_metric(items, "mae"))
        cluster_row.update(summarize_error_metric(items, "rmse"))
        cluster_row.update(summarize_error_metric(items, "wape"))
        cluster_row.update(summarize_error_metric(items, "smape"))
        cluster_row.update(summarize_score_metric(items, "r2"))

        best_mae = select_best_row(items, "mae")
        worst_mae = select_worst_row(items, "mae")
        best_r2 = select_best_row(items, "r2")
        worst_r2 = select_worst_row(items, "r2")
        cluster_row["best_client_by_mae"] = None if best_mae is None else best_mae["client_id"]
        cluster_row["worst_client_by_mae"] = None if worst_mae is None else worst_mae["client_id"]
        cluster_row["best_client_by_r2"] = None if best_r2 is None else best_r2["client_id"]
        cluster_row["worst_client_by_r2"] = None if worst_r2 is None else worst_r2["client_id"]
        out.append(cluster_row)
    return out


def build_global_summary(rows: List[Dict]) -> Dict:
    summary = {
        "num_clients": len(rows),
        "avg_num_test_samples": safe_mean([row.get("num_test_samples") for row in rows]),
    }
    summary.update(summarize_error_metric(rows, "mae"))
    summary.update(summarize_error_metric(rows, "rmse"))
    summary.update(summarize_error_metric(rows, "wape"))
    summary.update(summarize_error_metric(rows, "smape"))
    summary.update(summarize_score_metric(rows, "r2"))
    return summary


def build_rank_report(rows: List[Dict], metric: str, top_k: int) -> Dict[str, List[Dict]]:
    best_rows = sort_rows_by_metric(rows, metric, best_first=True)[:top_k]
    worst_rows = sort_rows_by_metric(rows, metric, best_first=False)[:top_k]
    fields = [
        "cluster_id",
        "client_id",
        "num_test_samples",
        "mae",
        "rmse",
        "wape",
        "smape",
        "r2",
        "prediction_csv",
        "plot_path",
    ]
    return {
        "best": [{field: row.get(field) for field in fields} for row in best_rows],
        "worst": [{field: row.get(field) for field in fields} for row in worst_rows],
    }


def load_test_report_df(report_path: Path) -> pd.DataFrame:
    df = pd.read_csv(report_path)
    required_cols = {"cluster_id", "client_id", "mae", "rmse"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {report_path}: {missing}")

    df = df.copy()
    df["cluster_id"] = normalize_id_series(df["cluster_id"])
    df["client_id"] = normalize_id_series(df["client_id"])
    for col in (
        "mae",
        "rmse",
        "mse",
        "mape",
        "smape",
        "r2",
        "num_test_batches",
        "num_test_samples",
        "test_duration_sec",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "test_ok" in df.columns:
        df = df[parse_bool_series(df["test_ok"])].copy()
    return df.sort_values(["cluster_id", "client_id"]).reset_index(drop=True)


def load_lstm_baseline_df(baseline_path: Path, assignments_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    assignments = pd.read_csv(assignments_path)
    if "cell_id" in baseline.columns and "client_id" not in baseline.columns:
        baseline = baseline.rename(columns={"cell_id": "client_id"})
    if "client_id" not in baseline.columns:
        raise ValueError(f"Baseline file {baseline_path} must contain 'client_id' or 'cell_id'")
    if not {"ID", "Cluster"}.issubset(assignments.columns):
        raise ValueError(f"Assignments file {assignments_path} must contain 'ID' and 'Cluster'")

    baseline = baseline.copy()
    assignments = assignments.copy()
    baseline["client_id"] = normalize_id_series(baseline["client_id"])
    assignments["client_id"] = normalize_id_series(assignments["ID"])
    assignments["cluster_id"] = normalize_id_series(assignments["Cluster"])
    merged = baseline.merge(assignments[["client_id", "cluster_id"]], on="client_id", how="left")
    for col in ("mae", "rmse", "mse", "r2", "msle"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged.sort_values(["cluster_id", "client_id"]).reset_index(drop=True)


def filter_metrics_df(
    df: pd.DataFrame,
    *,
    selected_clusters: Optional[Sequence[str]],
    selected_clients: Optional[Sequence[str]],
) -> pd.DataFrame:
    out = df.copy()
    if selected_clusters is not None:
        cluster_set = {str(x) for x in selected_clusters}
        out = out[out["cluster_id"].isin(cluster_set)].copy()
    if selected_clients is not None:
        client_set = {str(x) for x in selected_clients}
        out = out[out["client_id"].isin(client_set)].copy()
    return out.reset_index(drop=True)


def build_cluster_best_worst_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["cluster_id"]), []).append(row)

    out: List[Dict] = []
    for cluster_id, items in sorted(grouped.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        best_mae = select_best_row(items, "mae")
        worst_mae = select_worst_row(items, "mae")
        best_rmse = select_best_row(items, "rmse")
        worst_rmse = select_worst_row(items, "rmse")
        out.append(
            {
                "cluster_id": cluster_id,
                "num_clients": len(items),
                "best_client_by_mae": None if best_mae is None else best_mae["client_id"],
                "best_mae": None if best_mae is None else best_mae.get("mae"),
                "worst_client_by_mae": None if worst_mae is None else worst_mae["client_id"],
                "worst_mae": None if worst_mae is None else worst_mae.get("mae"),
                "best_client_by_rmse": None if best_rmse is None else best_rmse["client_id"],
                "best_rmse": None if best_rmse is None else best_rmse.get("rmse"),
                "worst_client_by_rmse": None if worst_rmse is None else worst_rmse["client_id"],
                "worst_rmse": None if worst_rmse is None else worst_rmse.get("rmse"),
            }
        )
    return out


def build_splitfed_vs_baseline_rows(current_rows: List[Dict], baseline_df: pd.DataFrame) -> List[Dict]:
    current_df = pd.DataFrame(current_rows).copy()
    if current_df.empty:
        return []

    current_df["cluster_id"] = normalize_id_series(current_df["cluster_id"])
    current_df["client_id"] = normalize_id_series(current_df["client_id"])
    baseline_subset = baseline_df.copy()
    baseline_subset["cluster_id"] = normalize_id_series(baseline_subset["cluster_id"])
    baseline_subset["client_id"] = normalize_id_series(baseline_subset["client_id"])
    keep_cols = ["cluster_id", "client_id"] + [col for col in ("mae", "rmse", "mse", "r2", "msle") if col in baseline_subset.columns]
    baseline_subset = baseline_subset[keep_cols]
    baseline_subset = baseline_subset.rename(columns={col: f"baseline_{col}" for col in keep_cols if col not in {"cluster_id", "client_id"}})

    merged = current_df.merge(baseline_subset, on=["cluster_id", "client_id"], how="left")
    for metric in ("mae", "rmse", "mse", "r2"):
        baseline_col = f"baseline_{metric}"
        if baseline_col in merged.columns:
            merged[f"delta_vs_lstm_{metric}"] = merged[metric] - merged[baseline_col]
    return merged.sort_values(["cluster_id", "client_id"]).to_dict(orient="records")


def _cluster_subplot_layout(n_clusters: int) -> Tuple[int, int]:
    if n_clusters <= 1:
        return 1, 1
    if n_clusters <= 4:
        return 2, 2
    ncols = 3
    nrows = int(math.ceil(n_clusters / ncols))
    return nrows, ncols


def plot_client_metric_bars_by_cluster(save_path: Path, rows: List[Dict], metric: str, *, title: str) -> None:
    ensure_dir(save_path.parent)
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        if get_metric_value(row, metric) is None:
            continue
        grouped.setdefault(str(row["cluster_id"]), []).append(row)
    if not grouped:
        return

    cluster_ids = sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)
    nrows, ncols = _cluster_subplot_layout(len(cluster_ids))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.4 * nrows), squeeze=False, sharey=False)
    axes_flat = axes.flatten()

    for ax, cluster_id in zip(axes_flat, cluster_ids):
        items = sorted(grouped[cluster_id], key=lambda row: (float(row[metric]), str(row["client_id"])))
        labels = [str(row["client_id"]) for row in items]
        values = [float(row[metric]) for row in items]
        x = np.arange(len(labels))
        ax.bar(x, values, color="#2E6F95")
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_xlabel("Client")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    for ax in axes_flat[len(cluster_ids):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_baseline_by_cluster(
    save_path: Path,
    comparison_rows: List[Dict],
    metric: str,
    *,
    current_label: str,
    baseline_label: str,
    title: str,
) -> None:
    ensure_dir(save_path.parent)
    grouped: Dict[str, List[Dict]] = {}
    baseline_metric = f"baseline_{metric}"
    for row in comparison_rows:
        if get_metric_value(row, metric) is None or get_metric_value(row, baseline_metric) is None:
            continue
        grouped.setdefault(str(row["cluster_id"]), []).append(row)
    if not grouped:
        return

    cluster_ids = sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)
    nrows, ncols = _cluster_subplot_layout(len(cluster_ids))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.6 * ncols, 4.6 * nrows), squeeze=False, sharey=False)
    axes_flat = axes.flatten()

    for ax, cluster_id in zip(axes_flat, cluster_ids):
        items = sorted(grouped[cluster_id], key=lambda row: (float(row[metric]), str(row["client_id"])))
        labels = [str(row["client_id"]) for row in items]
        current_vals = [float(row[metric]) for row in items]
        baseline_vals = [float(row[baseline_metric]) for row in items]
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, current_vals, width=width, label=current_label, color="#2E6F95")
        ax.bar(x + width / 2, baseline_vals, width=width, label=baseline_label, color="#D98E04")
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_xlabel("Client")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    for ax in axes_flat[len(cluster_ids):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_one_client(
    repo: ClusterDataRepository,
    *,
    output_dir: Path,
    cluster_id: str,
    client_id: str,
    device: torch.device,
    batch_size: int,
    checkpoint_mode_arg: str,
    window_size: int,
    forecast_horizon: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    plots_dir: Path,
    predictions_dir: Path,
    last_n_plot: int,
) -> Dict:
    artifact_cfg = load_client_artifact_config(output_dir, cluster_id, client_id)
    checkpoint_mode = resolve_checkpoint_mode(checkpoint_mode_arg, artifact_cfg)
    if not (1 <= artifact_cfg.split_layer < artifact_cfg.model_cfg.num_layers):
        raise ValueError(
            f"Invalid split layer in client config for cluster={cluster_id} client={client_id}: "
            f"split={artifact_cfg.split_layer}, num_layers={artifact_cfg.model_cfg.num_layers}"
        )

    cluster_df = repo.load_cluster_df(cluster_id)
    cdf = cluster_df[cluster_df["id"].astype(str) == str(client_id)].copy()
    if cdf.empty:
        raise ValueError(f"No data found for client {client_id} in cluster {cluster_id}")

    original_values = cdf["value"].to_numpy(dtype=np.float32)
    timestamps = cdf["Fecha_Hora"].to_numpy()
    working_values = apply_sign_flip(original_values, artifact_cfg.sign_flip)

    X_all, y_all_working, ts_all = build_all_windows(working_values, timestamps, window_size, forecast_horizon)
    if len(X_all) == 0:
        raise ValueError(f"No windows available for cluster={cluster_id} client={client_id}")

    if (
        artifact_cfg.num_train_windows is not None
        and artifact_cfg.num_val_windows is not None
        and artifact_cfg.num_test_windows is not None
    ):
        X_test, y_test_working, ts_test = split_windows_by_counts(
            X_all,
            y_all_working,
            ts_all,
            num_train_windows=artifact_cfg.num_train_windows,
            num_val_windows=artifact_cfg.num_val_windows,
            num_test_windows=artifact_cfg.num_test_windows,
            split_name="test",
        )
    else:
        X_test, y_test_working, ts_test = split_windows_by_ratios(
            X_all,
            y_all_working,
            ts_all,
            window_size=window_size,
            horizon=forecast_horizon,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_name="test",
        )

    if len(X_test) == 0:
        raise ValueError(f"No test windows for client {client_id} in cluster {cluster_id}")

    X_test_norm = normalize_array(
        X_test,
        normalize=artifact_cfg.normalize,
        method=artifact_cfg.normalize_method,
        mean=artifact_cfg.mean,
        scale=artifact_cfg.scale,
    )

    inference_cfg = build_inference_config(artifact_cfg.model_cfg, artifact_cfg.split_layer)
    client_model, server_model = build_models(inference_cfg, artifact_cfg.split_layer, device)

    client_ckpt_path = first_existing(client_checkpoint_candidates(output_dir, cluster_id, client_id, checkpoint_mode))
    server_ckpt_path = first_existing(server_checkpoint_candidates(output_dir, cluster_id, checkpoint_mode))
    client_ckpt, server_ckpt = load_weights_into_models(
        client_model,
        server_model,
        client_ckpt_path=client_ckpt_path,
        server_ckpt_path=server_ckpt_path,
        device=device,
    )

    preds_norm = run_local_inference(client_model, server_model, X_test_norm, batch_size=batch_size, device=device)
    preds_working = inverse_transform(
        preds_norm,
        normalize=artifact_cfg.normalize,
        method=artifact_cfg.normalize_method,
        mean=artifact_cfg.mean,
        scale=artifact_cfg.scale,
    ).reshape(-1, 1)
    y_true_working = y_test_working.reshape(-1, 1)

    preds_original = undo_sign_flip(preds_working, artifact_cfg.sign_flip)
    y_true_original = undo_sign_flip(y_true_working, artifact_cfg.sign_flip)
    metrics = compute_metrics(preds_original, y_true_original)

    plot_path = plots_dir / "predictions" / f"cluster_{cluster_id}" / f"{client_id}.png"
    prediction_csv = predictions_dir / f"cluster_{cluster_id}" / f"client_{client_id}_predictions.csv"
    plot_client_forecast(
        plot_path,
        ts_test,
        y_true_original,
        preds_original,
        cluster_id=cluster_id,
        client_id=client_id,
        metrics=metrics,
        last_n=last_n_plot,
    )
    save_prediction_points_csv(prediction_csv, ts_test, y_true_original, preds_original)

    saved_test = maybe_load_saved_test_metrics(output_dir, cluster_id, client_id)
    saved_raw = None if saved_test is None else saved_test.get("raw_metrics", {})
    saved_mae = None if not saved_raw else clean_metric_value(saved_raw.get("mae"))
    saved_rmse = None if not saved_raw else clean_metric_value(saved_raw.get("rmse"))
    saved_r2 = None if not saved_raw else clean_metric_value(saved_raw.get("r2"))
    saved_smape = None if not saved_raw else clean_metric_value(saved_raw.get("smape"))

    row = {
        "cluster_id": str(cluster_id),
        "client_id": str(client_id),
        "checkpoint_mode": checkpoint_mode,
        "artifact_eval_mode": artifact_cfg.eval_mode,
        "num_test_samples": int(len(y_true_original)),
        "checkpoint_client": str(client_ckpt_path),
        "checkpoint_server": str(server_ckpt_path),
        "client_checkpoint_epoch": int(client_ckpt.get("epoch", -1)) if isinstance(client_ckpt, dict) else None,
        "server_global_step": int(server_ckpt.get("global_step", -1)) if isinstance(server_ckpt, dict) else None,
        "normalize": bool(artifact_cfg.normalize),
        "normalize_method": artifact_cfg.normalize_method,
        "sign_flip": bool(artifact_cfg.sign_flip),
        "norm_mean": float(artifact_cfg.mean),
        "norm_scale": float(artifact_cfg.scale),
        "plot_path": str(plot_path),
        "prediction_csv": str(prediction_csv),
        "saved_test_mae": saved_mae,
        "saved_test_rmse": saved_rmse,
        "saved_test_r2": saved_r2,
        "saved_test_smape": saved_smape,
        "delta_vs_saved_mae": (None if saved_mae is None or metrics["mae"] is None else float(metrics["mae"] - saved_mae)),
        "delta_vs_saved_rmse": (None if saved_rmse is None or metrics["rmse"] is None else float(metrics["rmse"] - saved_rmse)),
        "delta_vs_saved_r2": (None if saved_r2 is None or metrics["r2"] is None else float(metrics["r2"] - saved_r2)),
        "delta_vs_saved_smape": (None if saved_smape is None or metrics["smape"] is None else float(metrics["smape"] - saved_smape)),
        **metrics,
        "_plot_payload": {
            "cluster_id": str(cluster_id),
            "client_id": str(client_id),
            "timestamps": pd.to_datetime(ts_test).astype(str).tolist(),
            "y_true": y_true_original.reshape(-1).tolist(),
            "y_pred": preds_original.reshape(-1).tolist(),
            "metrics": metrics,
        },
    }
    return row


def parse_args() -> argparse.Namespace:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    p = argparse.ArgumentParser(description="Run local prediction and fairness analysis from saved split-learning checkpoints.")
    p.add_argument("--project-root", type=Path, default=Path("."), help="Project root containing dataset/ and output/")
    p.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Dataset directory")
    p.add_argument("--output-dir", type=Path, default=Path("output-fedavg-30EP5LEP"), help="Training output directory")
    p.add_argument("--test-report", type=Path, default=None, help="Per-client split-learning test report CSV. Defaults to output/results/test_report.csv with fallbacks.")
    p.add_argument("--baseline-report", type=Path, default=None, help="Baseline LSTM metrics CSV. Defaults to dataset/lstm_results_updated.csv with fallbacks.")
    p.add_argument("--assignments-path", type=Path, default=Path("dataset") / "cluster-assignment.csv", help="Cluster assignment CSV used to map baseline rows to clusters.")
    p.add_argument("--report-dir", type=Path, default=None, help="Directory for analysis outputs. Default: derived from output dir name, e.g. output-ep30-noES -> report-ep30-noES")
    p.add_argument("--clusters", type=str, default=None, help="Comma-separated cluster IDs. Default: discover from output/")
    p.add_argument("--clients", type=str, default=None, help="Comma-separated client IDs. Default: discover from output/")
    p.add_argument("--checkpoint-mode", type=str, default="auto", choices=["auto", "paired_best", "final_shared", "final", "latest"])
    p.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "wape", "smape", "r2"], help="Metric used for best/worst ranking plots")
    p.add_argument("--top-k", type=int, default=5, help="How many best/worst clients to export in ranking tables")
    p.add_argument("--window-size", type=int, default=int(os.environ.get("WINDOW_SIZE", "24")))
    p.add_argument("--forecast-horizon", type=int, default=int(os.environ.get("FORECAST_HORIZON", "1")))
    p.add_argument("--train-ratio", type=float, default=float(os.environ.get("TRAIN_RATIO", "0.7")), help="Only used if artifact window counts are missing")
    p.add_argument("--val-ratio", type=float, default=float(os.environ.get("VAL_RATIO", "0.1")), help="Only used if artifact window counts are missing")
    p.add_argument("--test-ratio", type=float, default=float(os.environ.get("TEST_RATIO", "0.2")), help="Only used if artifact window counts are missing")
    p.add_argument("--batch-size", type=int, default=128, help="Inference batch size for offline prediction")
    p.add_argument("--last-n-plot", type=int, default=200, help="How many final test points to display in forecast plots")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.resolve()
    dataset_dir = (project_root / args.dataset_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    test_report_path = resolve_input_path(
        project_root,
        args.test_report,
        default_test_report_candidates_for_output_dir(output_dir),
        label="test report CSV",
    )
    baseline_report_path = resolve_input_path(
        project_root,
        args.baseline_report,
        DEFAULT_BASELINE_CANDIDATES,
        label="baseline LSTM CSV",
    )
    assignments_path = (
        args.assignments_path.resolve()
        if args.assignments_path.is_absolute()
        else (project_root / args.assignments_path).resolve()
    )
    report_dir_arg = args.report_dir if args.report_dir is not None else default_report_dir_for_output(output_dir)
    report_dir = (project_root / report_dir_arg).resolve()
    plots_dir = report_dir / "plots"
    predictions_dir = report_dir / "predictions"
    rankings_dir = report_dir / "rankings"

    ensure_dir(report_dir)
    ensure_dir(plots_dir)
    ensure_dir(predictions_dir)
    ensure_dir(rankings_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    selected_clusters = parse_csv_ids(args.clusters)
    selected_clients = parse_csv_ids(args.clients)
    repo = ClusterDataRepository(dataset_dir)
    metrics_df = filter_metrics_df(
        load_test_report_df(test_report_path),
        selected_clusters=selected_clusters,
        selected_clients=selected_clients,
    )
    if metrics_df.empty:
        raise RuntimeError(f"No matching client rows found in {test_report_path}")
    baseline_df = filter_metrics_df(
        load_lstm_baseline_df(baseline_report_path, assignments_path),
        selected_clusters=selected_clusters,
        selected_clients=selected_clients,
    )

    cluster_ids = sorted(
        metrics_df["cluster_id"].astype(str).unique().tolist(),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )
    per_client_rows: List[Dict] = metrics_df.to_dict(orient="records")
    payloads_by_cluster: Dict[str, List[Dict]] = {str(cluster_id): [] for cluster_id in cluster_ids}
    prediction_meta: Dict[Tuple[str, str], Dict] = {}
    prediction_failures = 0
    log(f"Device: {device}")
    log(f"Dataset dir: {dataset_dir}")
    log(f"Output dir: {output_dir}")
    log(f"Test report: {test_report_path}")
    log(f"Baseline report: {baseline_report_path}")
    log(f"Assignments: {assignments_path}")
    log(f"Report dir: {report_dir}")
    log(f"Checkpoint mode: {args.checkpoint_mode}")
    log(f"Ranking metric: {args.metric}")
    log(f"Clusters in report: {cluster_ids}")

    for cluster_id in cluster_ids:
        client_ids = (
            metrics_df[metrics_df["cluster_id"].astype(str) == str(cluster_id)]["client_id"]
            .astype(str)
            .tolist()
        )
        if not client_ids:
            log(f"Skipping cluster {cluster_id}: no matching clients")
            continue

        log(f"\n[Cluster {cluster_id}] building prediction plots for {len(client_ids)} clients")

        for client_id in client_ids:
            log(f"  -> client {client_id}")
            try:
                prediction_row = evaluate_one_client(
                    repo,
                    output_dir=output_dir,
                    cluster_id=str(cluster_id),
                    client_id=str(client_id),
                    device=device,
                    batch_size=args.batch_size,
                    checkpoint_mode_arg=args.checkpoint_mode,
                    window_size=args.window_size,
                    forecast_horizon=args.forecast_horizon,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    plots_dir=plots_dir,
                    predictions_dir=predictions_dir,
                    last_n_plot=args.last_n_plot,
                )
            except Exception as exc:
                prediction_failures += 1
                log(f"     [WARN] skipped prediction plot: {exc}")
                continue

            payloads_by_cluster[str(cluster_id)].append(prediction_row.pop("_plot_payload"))
            prediction_meta[(str(cluster_id), str(client_id))] = prediction_row
            log(
                f"     recomputed MAE={prediction_row['mae']:.6f} "
                f"RMSE={prediction_row['rmse']:.6f} R2={prediction_row['r2']}"
            )

    if not per_client_rows:
        raise RuntimeError("No client evaluations were produced.")

    for row in per_client_rows:
        key = (str(row["cluster_id"]), str(row["client_id"]))
        meta = prediction_meta.get(key)
        row["prediction_csv"] = None
        row["plot_path"] = None
        row["checkpoint_client"] = None
        row["checkpoint_server"] = None
        row["client_checkpoint_epoch"] = None
        row["server_global_step"] = None
        row["recomputed_mae"] = None
        row["recomputed_rmse"] = None
        row["recomputed_r2"] = None
        row["recomputed_smape"] = None
        row["delta_recomputed_vs_report_mae"] = None
        row["delta_recomputed_vs_report_rmse"] = None
        if meta is None:
            continue
        row["prediction_csv"] = meta.get("prediction_csv")
        row["plot_path"] = meta.get("plot_path")
        row["checkpoint_client"] = meta.get("checkpoint_client")
        row["checkpoint_server"] = meta.get("checkpoint_server")
        row["client_checkpoint_epoch"] = meta.get("client_checkpoint_epoch")
        row["server_global_step"] = meta.get("server_global_step")
        row["recomputed_mae"] = meta.get("mae")
        row["recomputed_rmse"] = meta.get("rmse")
        row["recomputed_r2"] = meta.get("r2")
        row["recomputed_smape"] = meta.get("smape")
        if meta.get("mae") is not None and row.get("mae") is not None:
            row["delta_recomputed_vs_report_mae"] = float(meta["mae"] - row["mae"])
        if meta.get("rmse") is not None and row.get("rmse") is not None:
            row["delta_recomputed_vs_report_rmse"] = float(meta["rmse"] - row["rmse"])

    cluster_rows = summarize_cluster_rows(per_client_rows)
    global_summary = build_global_summary(per_client_rows)
    ranking_report = build_rank_report(per_client_rows, args.metric, args.top_k)
    cluster_best_worst_rows = build_cluster_best_worst_rows(per_client_rows)
    comparison_rows = build_splitfed_vs_baseline_rows(per_client_rows, baseline_df)

    for cluster_id, payloads in payloads_by_cluster.items():
        if not payloads:
            continue
        cluster_client_rows = [row for row in per_client_rows if str(row["cluster_id"]) == str(cluster_id)]
        best_row = select_best_row(cluster_client_rows, args.metric)
        worst_row = select_worst_row(cluster_client_rows, args.metric)
        if best_row is not None and worst_row is not None:
            best_payload = next((payload for payload in payloads if payload["client_id"] == best_row["client_id"]), None)
            worst_payload = next((payload for payload in payloads if payload["client_id"] == worst_row["client_id"]), None)
            if best_payload is None or worst_payload is None:
                continue
            plot_best_worst_pair(
                plots_dir / "best_worst" / f"cluster_{cluster_id}_best_worst_{args.metric}.png",
                best_payload=best_payload,
                worst_payload=worst_payload,
                metric=args.metric,
                last_n=args.last_n_plot,
                title=f"Cluster {cluster_id}: best vs worst client by {args.metric.upper()}",
            )

    global_best = select_best_row(per_client_rows, args.metric)
    global_worst = select_worst_row(per_client_rows, args.metric)
    if global_best is not None and global_worst is not None:
        best_payload = next(
            (
                payload
                for payloads in payloads_by_cluster.values()
                for payload in payloads
                if payload["client_id"] == global_best["client_id"] and payload["cluster_id"] == global_best["cluster_id"]
            ),
            None,
        )
        worst_payload = next(
            (
                payload
                for payloads in payloads_by_cluster.values()
                for payload in payloads
                if payload["client_id"] == global_worst["client_id"] and payload["cluster_id"] == global_worst["cluster_id"]
            ),
            None,
        )
        if best_payload is not None and worst_payload is not None:
            plot_best_worst_pair(
                plots_dir / "best_worst" / f"global_best_worst_{args.metric}.png",
                best_payload=best_payload,
                worst_payload=worst_payload,
                metric=args.metric,
                last_n=args.last_n_plot,
                title=f"Global best vs worst client by {args.metric.upper()}",
            )

    plot_client_metric_bars_by_cluster(
        plots_dir / "metrics" / "clients_mae_by_cluster.png",
        per_client_rows,
        "mae",
        title="Per-client MAE by cluster",
    )
    plot_client_metric_bars_by_cluster(
        plots_dir / "metrics" / "clients_rmse_by_cluster.png",
        per_client_rows,
        "rmse",
        title="Per-client RMSE by cluster",
    )
    if comparison_rows:
        plot_metric_vs_baseline_by_cluster(
            plots_dir / "metrics_vs_lstm" / "clients_mae_vs_lstm_by_cluster.png",
            comparison_rows,
            "mae",
            current_label="SplitFed",
            baseline_label="LSTM",
            title="Per-client MAE: SplitFed vs LSTM by cluster",
        )
        plot_metric_vs_baseline_by_cluster(
            plots_dir / "metrics_vs_lstm" / "clients_rmse_vs_lstm_by_cluster.png",
            comparison_rows,
            "rmse",
            current_label="SplitFed",
            baseline_label="LSTM",
            title="Per-client RMSE: SplitFed vs LSTM by cluster",
        )

    save_json(report_dir / "global_summary.json", global_summary)
    save_json(report_dir / "cluster_best_worst_clients.json", cluster_best_worst_rows)
    save_csv(
        report_dir / "prediction_clusters.csv",
        cluster_rows,
        [
            "cluster_id",
            "num_clients",
            "avg_num_test_samples",
            "avg_mae",
            "median_mae",
            "p90_mae",
            "p95_mae",
            "worst_mae",
            "best_mae",
            "worst_over_avg_mae",
            "avg_rmse",
            "median_rmse",
            "p90_rmse",
            "p95_rmse",
            "worst_rmse",
            "best_rmse",
            "worst_over_avg_rmse",
            "avg_wape",
            "p90_wape",
            "worst_wape",
            "worst_over_avg_wape",
            "avg_smape",
            "p90_smape",
            "worst_smape",
            "worst_over_avg_smape",
            "avg_r2",
            "median_r2",
            "p10_r2",
            "p05_r2",
            "worst_r2",
            "best_r2",
            "avg_minus_worst_r2",
            "best_client_by_mae",
            "worst_client_by_mae",
            "best_client_by_r2",
            "worst_client_by_r2",
        ],
    )
    save_csv(
        report_dir / "cluster_best_worst_clients.csv",
        cluster_best_worst_rows,
        [
            "cluster_id",
            "num_clients",
            "best_client_by_mae",
            "best_mae",
            "worst_client_by_mae",
            "worst_mae",
            "best_client_by_rmse",
            "best_rmse",
            "worst_client_by_rmse",
            "worst_rmse",
        ],
    )
    if comparison_rows:
        save_csv(
            report_dir / "splitfed_vs_lstm_clients.csv",
            comparison_rows,
            [
                "cluster_id",
                "client_id",
                "mae",
                "baseline_mae",
                "delta_vs_lstm_mae",
                "rmse",
                "baseline_rmse",
                "delta_vs_lstm_rmse",
                "mse",
                "baseline_mse",
                "delta_vs_lstm_mse",
                "r2",
                "baseline_r2",
                "delta_vs_lstm_r2",
                "baseline_msle",
                "prediction_csv",
                "plot_path",
            ],
        )
    save_csv(report_dir / "rankings" / f"best_clients_by_{args.metric}.csv", ranking_report["best"], ranking_report["best"][0].keys() if ranking_report["best"] else ["cluster_id", "client_id"])
    save_csv(report_dir / "rankings" / f"worst_clients_by_{args.metric}.csv", ranking_report["worst"], ranking_report["worst"][0].keys() if ranking_report["worst"] else ["cluster_id", "client_id"])

    log("\nCluster summary:")
    for row in cluster_rows:
        log(str(row))
    log("\nBest/worst clients by cluster:")
    for row in cluster_best_worst_rows:
        log(str(row))
    log("\nGlobal summary:")
    log(str(global_summary))
    if prediction_failures:
        log(f"\nPrediction plot failures: {prediction_failures}")
    log(f"\nSaved analysis to: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
