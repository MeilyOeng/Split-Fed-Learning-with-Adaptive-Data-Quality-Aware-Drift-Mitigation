#!/usr/bin/env python3
"""Long-lived split-learning client worker.

The process keeps model, optimizer, norm stats, and stream iterators in memory.
The scheduler talks to it over HTTP instead of starting a fresh process per batch.
"""

import json
import os
import pickle
import random
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
import math
import numpy as np
import requests
import torch
from flask import Flask, jsonify
from torch import nn, optim
torch.backends.cudnn.enabled = False
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.data_streaming import StreamingIterableDataset
from models.split_lstm import LSTMConfig, build_split_lstm

app = Flask(__name__)


def parse_bool(env_value: Optional[str], default: bool = False) -> bool:
    if env_value is None:
        return default
    return env_value.strip().lower() in ("1", "true", "yes", "y")


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val


def resolve_device() -> torch.device:
    gpu_enabled = parse_bool(_get_env("GPU_ENABLED", "true"), default=True)
    gpu_id = _get_env("GPU_ID", "")
    if gpu_enabled and torch.cuda.is_available():
        if gpu_id not in (None, ""):
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "selected_device": str(device),
        "gpu_enabled": parse_bool(_get_env("GPU_ENABLED", "true"), default=True),
        "gpu_id_env": _get_env("GPU_ID", ""),
        "cuda_visible_devices": _get_env("CUDA_VISIBLE_DEVICES", ""),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "process_role": _get_env("PROCESS_ROLE", "client"),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_index": idx,
                "cuda_name": torch.cuda.get_device_name(idx),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
            }
        )
    return info


def build_lstm_config_from_env() -> Tuple[LSTMConfig, int]:
    cfg = LSTMConfig(
        input_size=int(_get_env("MODEL_INPUT_SIZE", "1")),
        hidden_size=int(_get_env("MODEL_HIDDEN_SIZE", "128")),
        num_layers=int(_get_env("MODEL_NUM_LAYERS", "3")),
        output_size=int(_get_env("MODEL_OUTPUT_SIZE", "1")),
        dropout=float(_get_env("MODEL_DROPOUT", "0.4")),
        bidirectional=parse_bool(_get_env("MODEL_BIDIRECTIONAL", "false")),
    )
    split_layer = int(_get_env("MODEL_SPLIT_LAYER", "2"))
    if split_layer <= 0 or split_layer >= cfg.num_layers:
        raise ValueError(f"MODEL_SPLIT_LAYER must be in [1, num_layers-1], got split={split_layer}, num_layers={cfg.num_layers}")
    return cfg, split_layer


def _cluster_output_dir(cluster_id: str) -> str:
    path = os.path.join("output", f"cluster_{cluster_id}")
    os.makedirs(path, exist_ok=True)
    return path


def client_checkpoint_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_checkpoint.pt")


def client_best_checkpoint_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_best_checkpoint.pt")


def client_final_checkpoint_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_final.pt")


def client_train_metrics_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_train_metrics.json")


def client_test_metrics_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_test_metrics.json")


def client_progress_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_progress.json")


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_torch_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], epoch: int, path: str) -> None:
    payload = {"model_state_dict": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"[Client] Checkpoint -> {path}", flush=True)


def try_load_model_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], path: str, *, strict: bool = True) -> Tuple[Optional[int], bool]:
    if not os.path.exists(path):
        return None, False
    try:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[Client] WARNING: optimizer state incompatible for {path}: {e}", flush=True)
        return int(ckpt.get("epoch", 0)), True
    except Exception as e:
        print(f"[Client] WARNING: skipping incompatible checkpoint {path}: {e}", flush=True)
        return None, False


@dataclass
class ClientRuntimeConfig:
    client_id: str
    cluster_id: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    data_stream_api_url: str
    worker_port: int
    random_seed: int
    eval_mode: str
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_metric: str
    early_stopping_start_from_epoch: int
    server_port: int


class ClusteredClientRunner:
    def __init__(self) -> None:
        self.runtime_cfg = ClientRuntimeConfig(
            client_id=_get_env("CLIENT_ID", "9900002"),
            cluster_id=str(_get_env("CLUSTER_ID", "0")).replace("cluster_", ""),
            batch_size=int(_get_env("BATCH_SIZE", "32")),
            num_epochs=int(_get_env("NUM_EPOCHS", "100")),
            learning_rate=float(_get_env("CLIENT_LR", _get_env("LEARNING_RATE", "0.0005"))),
            data_stream_api_url=_get_env("DATA_STREAM_API_URL", "http://localhost:5001"),
            worker_port=int(_get_env("CLIENT_WORKER_PORT", "6100")),
            random_seed=int(_get_env("RANDOM_SEED", "42")),
            eval_mode=_get_env("EVAL_MODE", "final_shared"),
            early_stopping=parse_bool(_get_env("EARLY_STOPPING", "true")),
            early_stopping_patience=int(_get_env("EARLY_STOPPING_PATIENCE", "15")),
            early_stopping_min_delta=float(_get_env("EARLY_STOPPING_MIN_DELTA", "0.0")),
            early_stopping_metric=_get_env("EARLY_STOPPING_METRIC", "val_raw_mae"),
            early_stopping_start_from_epoch=int(_get_env("EARLY_STOPPING_START_FROM_EPOCH", "30")),
            server_port=int(_get_env("SERVER_PORT", "5002")),
        )
        self.device = resolve_device()
        self.device_info = get_device_info(self.device)
        seed = self.runtime_cfg.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.current_server_url = f"http://localhost:{self.runtime_cfg.server_port}"
        self.session = requests.Session()
        self.norm_stats = self._fetch_norm_stats()

        self.client_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.client_model.parameters(), lr=self.runtime_cfg.learning_rate)
        self.progress = self._load_or_init_progress()
        self.best_val_score = float(self.progress.get("best_val_score", float("inf")) if self.progress.get("best_val_score") is not None else float("inf"))
        self.best_epoch = int(self.progress.get("best_epoch", 0))
        self.train_history = list(self.progress.get("history", []))
        self.epochs_without_improvement = int(self.progress.get("epochs_without_improvement", 0))
        self.early_stopped = bool(self.progress.get("early_stopped", False))
        self.stop_reason = self.progress.get("stop_reason")
        self.start_epoch = self._maybe_load_checkpoint()

        self.train_dataset = self._build_stream_dataset("train")
        self.val_dataset = self._build_stream_dataset("val")
        self.test_dataset = self._build_stream_dataset("test")
        self.train_iter: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.train_stream_active = bool(self.progress.get("train_stream_active", False))
        self.prefetched_train_batch = None
        self.train_epoch_exhausted = False
        if self.train_stream_active:
            self._reset_stream("train", window_offset=int(self.progress.get("stream_window_offset", 0)))
            self.train_iter = iter(self.train_dataset)

        self._save_run_config()
        print(f"[Client {self.runtime_cfg.client_id}] device_info={self.device_info} client_lr={self.runtime_cfg.learning_rate}", flush=True)

    def _build_model(self) -> nn.Module:
        cfg, split_layer = build_lstm_config_from_env()
        client_model, _ = build_split_lstm(cfg, split_layer)
        return client_model

    def _save_run_config(self) -> None:
        cfg, split_layer = build_lstm_config_from_env()
        write_json(
            os.path.join(_cluster_output_dir(self.runtime_cfg.cluster_id), f"client_{self.runtime_cfg.client_id}_config.json"),
            {
                "component": "client_worker",
                "client_id": self.runtime_cfg.client_id,
                "cluster_id": self.runtime_cfg.cluster_id,
                "worker_port": self.runtime_cfg.worker_port,
                "server_port": self.runtime_cfg.server_port,
                "model": {
                    "input_size": cfg.input_size,
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "split_layer": split_layer,
                    "dropout": cfg.dropout,
                    "bidirectional": cfg.bidirectional,
                    "output_size": cfg.output_size,
                },
                "training": {
                    "batch_size": self.runtime_cfg.batch_size,
                    "num_epochs": self.runtime_cfg.num_epochs,
                    "learning_rate": self.runtime_cfg.learning_rate,
                    "random_seed": self.runtime_cfg.random_seed,
                    "early_stopping": self.runtime_cfg.early_stopping,
                    "early_stopping_patience": self.runtime_cfg.early_stopping_patience,
                    "early_stopping_min_delta": self.runtime_cfg.early_stopping_min_delta,
                    "early_stopping_metric": self.runtime_cfg.early_stopping_metric,
                    "early_stopping_start_from_epoch": self.runtime_cfg.early_stopping_start_from_epoch,
                    "eval_mode": self.runtime_cfg.eval_mode,
                },
                "normalization": self.norm_stats,
                "device": self.device_info,
            },
        )

    def _load_or_init_progress(self) -> Dict[str, Any]:
        path = client_progress_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)
        if os.path.exists(path):
            data = read_json(path)
            data.setdefault("completed_epochs", 0)
            data.setdefault("current_epoch_loss_sum", 0.0)
            data.setdefault("current_epoch_batch_count", 0)
            data.setdefault("history", [])
            data.setdefault("best_val_score", None)
            data.setdefault("best_epoch", 0)
            data.setdefault("epochs_without_improvement", 0)
            data.setdefault("early_stopped", False)
            data.setdefault("stop_reason", None)
            data.setdefault("train_stream_active", False)
            data.setdefault("stream_window_offset", 0)
            return data
        return {
            "completed_epochs": 0,
            "current_epoch_loss_sum": 0.0,
            "current_epoch_batch_count": 0,
            "history": [],
            "best_val_score": None,
            "best_epoch": 0,
            "epochs_without_improvement": 0,
            "early_stopped": False,
            "stop_reason": None,
            "train_stream_active": False,
            "stream_window_offset": 0,
        }

    def _save_progress(self) -> None:
        self.progress["history"] = self.train_history
        self.progress["best_val_score"] = None if self.best_val_score == float("inf") else self.best_val_score
        self.progress["best_epoch"] = self.best_epoch
        self.progress["epochs_without_improvement"] = self.epochs_without_improvement
        self.progress["early_stopped"] = self.early_stopped
        self.progress["stop_reason"] = self.stop_reason
        self.progress["train_stream_active"] = self.train_stream_active
        write_json(client_progress_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id), self.progress)

    def _maybe_load_checkpoint(self) -> int:
        latest_path = client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)
        epoch, ok = try_load_model_checkpoint(self.client_model, self.optimizer, latest_path)
        if ok:
            print(f"[Client {self.runtime_cfg.client_id}] Loaded latest checkpoint <- {latest_path}", flush=True)
            return int(epoch or 0)
        print(f"[Client {self.runtime_cfg.client_id}] No compatible checkpoint - random init.", flush=True)
        return 0

    def _build_stream_dataset(self, mode: str) -> StreamingIterableDataset:
        return StreamingIterableDataset(
            api_url=self.runtime_cfg.data_stream_api_url,
            client_id=self.runtime_cfg.client_id,
            cluster_id=self.runtime_cfg.cluster_id,
            mode=mode,
            batch_size=self.runtime_cfg.batch_size,
            timeout=60.0,
            max_retries=5,
            backoff=1.0,
        )

    def _reset_stream(self, mode: str, window_offset: int = 0) -> None:
        payload = {"client_id": self.runtime_cfg.client_id, "cluster_id": self.runtime_cfg.cluster_id, "mode": mode}
        if window_offset > 0:
            payload["window_offset"] = int(window_offset)
        resp = requests.post(
            f"{self.runtime_cfg.data_stream_api_url}/stream/reset",
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        body = resp.json()
        print(
            f"[Client {self.runtime_cfg.client_id}] Stream reset ({mode}), {body.get('windows_available', '?')} windows, "
            f"window_offset={body.get('window_offset', 0)}"
        )

    def _ensure_train_stream(self) -> None:
        if self.train_iter is None:
            window_offset = int(self.progress.get("stream_window_offset", 0))
            self._reset_stream("train", window_offset=window_offset)
            self.train_iter = iter(self.train_dataset)
            self.train_stream_active = True
            self.train_epoch_exhausted = False
            self.prefetched_train_batch = None
            self._save_progress()

    def _prefetch_train_batch(self) -> bool:
        self._ensure_train_stream()
        if self.prefetched_train_batch is not None:
            return True
        if self.train_epoch_exhausted:
            return False
        assert self.train_iter is not None
        try:
            self.prefetched_train_batch = next(self.train_iter)
            return True
        except StopIteration:
            self.train_epoch_exhausted = True
            self.prefetched_train_batch = None
            return False

    def _consume_prefetched_train_batch(self):
        has_batch = self._prefetch_train_batch()
        if not has_batch:
            return None
        batch = self.prefetched_train_batch
        self.prefetched_train_batch = None
        return batch

    def _mark_epoch_finished(self) -> None:
        self.train_iter = None
        self.train_stream_active = False
        self.train_epoch_exhausted = False
        self.prefetched_train_batch = None
        self.progress["stream_window_offset"] = 0
        self._save_progress()

    def _fetch_norm_stats(self) -> Optional[Dict[str, float]]:
        resp = self.session.get(
            f"{self.runtime_cfg.data_stream_api_url}/stream/norm_stats",
            params={"client_id": self.runtime_cfg.client_id, "cluster_id": self.runtime_cfg.cluster_id},
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json()

    def _prepare_batch_for_lstm(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if X.dim() == 2:
            X = X.unsqueeze(-1)
        elif X.dim() == 4 and X.size(-1) == 1:
            X = X.squeeze(-1)
        X = X.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)
        return X, y

    def _pack_payload(self, client_id: str, activations: torch.Tensor, targets: Optional[torch.Tensor]) -> bytes:
        payload = {"client_id": str(client_id), "activations": activations.detach().cpu()}
        if targets is not None:
            payload["targets"] = targets.detach().cpu()
        return pickle.dumps(payload)

    def _post_pickle(self, url: str, data: bytes):
        resp = self.session.post(url, data=data, timeout=120.0)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        return pickle.loads(resp.content)

    def _inverse_tensor(self, arr: np.ndarray) -> np.ndarray:
        if not self.norm_stats or not bool(self.norm_stats.get("normalize", False)):
            return arr
        mean = float(self.norm_stats["mean"])
        scale = float(self.norm_stats["scale"])
        method = str(self.norm_stats.get("method", "zscore")).lower()
        if method == "minmax":
            return arr * scale + mean
        return arr * scale + mean

    def _compute_metrics(self, preds: np.ndarray, trues: np.ndarray) -> Dict[str, Optional[float]]:
        preds = preds.reshape(-1)
        trues = trues.reshape(-1)
        if preds.size == 0:
            return {"mse": None, "mae": None, "rmse": None, "r2": None, "smape": None}
        err = preds - trues
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(mse))
        denom = np.abs(preds) + np.abs(trues)
        mask = denom > 0
        smape = float(np.mean(2.0 * np.abs(err[mask]) / denom[mask]) * 100.0) if np.any(mask) else None
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((trues - np.mean(trues)) ** 2))
        r2 = None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "smape": smape}

    def _save_best_pair(self, epoch: int) -> None:
        save_torch_checkpoint(self.client_model, self.optimizer, epoch, client_best_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id))
        resp = self.session.post(f"{self.current_server_url}/save_best_checkpoint", timeout=30.0)
        resp.raise_for_status()

    def _save_latest(self, epoch: int) -> None:
        save_torch_checkpoint(self.client_model, self.optimizer, epoch, client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id))

    def _save_final(self, epoch: int) -> None:
        save_torch_checkpoint(self.client_model, self.optimizer, epoch, client_final_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id))

    def _run_eval(self, mode: str, epoch: Optional[int] = None, save_json_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if mode not in {"val", "test"}:
            raise ValueError(f"Unsupported eval mode: {mode}")
        dataset = self.val_dataset if mode == "val" else self.test_dataset
        self._reset_stream(mode)
        loader = iter(dataset)
        route = "/validate_step" if mode == "val" else "/inference_step"
        self.client_model.eval()
        preds_norm, trues_norm = [], []
        total_server_loss = 0.0
        num_batches = 0
        num_samples = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = self._prepare_batch_for_lstm(X, y)
                activations = self.client_model(X)
                if activations.dim() == 3:
                    activations = activations[:, -1, :]
                if mode == "val":
                    result = self._post_pickle(f"{self.current_server_url}{route}", self._pack_payload(self.runtime_cfg.client_id, activations, y))
                    total_server_loss += float(result["loss"])
                    batch_preds = np.asarray(result["predictions"], dtype=np.float32)
                else:
                    result = self._post_pickle(f"{self.current_server_url}{route}", self._pack_payload(self.runtime_cfg.client_id, activations, None))
                    batch_preds = np.asarray(result["predictions"], dtype=np.float32)
                batch_true = y.detach().cpu().numpy().astype(np.float32)
                preds_norm.append(batch_preds.reshape(-1, 1))
                trues_norm.append(batch_true.reshape(-1, 1))
                num_batches += 1
                num_samples += int(batch_true.shape[0])
        self.client_model.train()
        if num_batches == 0:
            return None
        preds_norm_np = np.concatenate(preds_norm, axis=0)
        trues_norm_np = np.concatenate(trues_norm, axis=0)
        preds_raw = self._inverse_tensor(preds_norm_np)
        trues_raw = self._inverse_tensor(trues_norm_np)
        result = {
            "mode": mode,
            "epoch": epoch,
            "client_id": self.runtime_cfg.client_id,
            "cluster_id": self.runtime_cfg.cluster_id,
            "num_batches": num_batches,
            "num_samples": num_samples,
            "server_loss": (total_server_loss / num_batches) if mode == "val" else None,
            "normalized_metrics": self._compute_metrics(preds_norm_np, trues_norm_np),
            "raw_metrics": self._compute_metrics(preds_raw, trues_raw),
            "normalization": self.norm_stats,
            "eval_mode": self.runtime_cfg.eval_mode,
        }
        if save_json_path:
            write_json(save_json_path, result)
        return result

    def _finalize_epoch(self, epoch_number: int, avg_train_loss: Optional[float]) -> Dict[str, Any]:
        val_result = self._run_eval("val", epoch=epoch_number)
        val_raw_mae = None if val_result is None else val_result["raw_metrics"]["mae"]
        val_server_loss = None if val_result is None else val_result["server_loss"]
        val_raw_rmse = None if val_result is None else val_result["raw_metrics"]["rmse"]
        self.train_history.append(
            {
                "epoch": epoch_number,
                "train_loss": avg_train_loss,
                "val_loss": val_server_loss,
                "val_raw_mae": val_raw_mae,
                "val_raw_rmse": val_raw_rmse,
            }
        )
        improved = False
        if val_raw_mae is not None and (self.best_val_score == float("inf") or val_raw_mae < (self.best_val_score - self.runtime_cfg.early_stopping_min_delta)):
            improved = True
            self.best_val_score = float(val_raw_mae)
            self.best_epoch = epoch_number
            self._save_best_pair(epoch_number)

        if epoch_number >= self.runtime_cfg.early_stopping_start_from_epoch:
            self.epochs_without_improvement = 0 if improved else self.epochs_without_improvement + 1
            if self.runtime_cfg.early_stopping and self.epochs_without_improvement >= self.runtime_cfg.early_stopping_patience:
                self.early_stopped = True
                self.stop_reason = (
                    f"no improvement in {self.runtime_cfg.early_stopping_metric} for "
                    f"{self.runtime_cfg.early_stopping_patience} epoch(s) after epoch "
                    f"{self.runtime_cfg.early_stopping_start_from_epoch}"
                )
        else:
            self.epochs_without_improvement = 0

        self.progress["completed_epochs"] = epoch_number
        self.progress["current_epoch_loss_sum"] = 0.0
        self.progress["current_epoch_batch_count"] = 0
        self.train_stream_active = False
        self.train_iter = None
        self._save_latest(epoch_number)
        if self.early_stopped or epoch_number >= self.runtime_cfg.num_epochs:
            self._save_final(epoch_number)
        summary = {
            "client_id": self.runtime_cfg.client_id,
            "cluster_id": self.runtime_cfg.cluster_id,
            "num_epochs": self.runtime_cfg.num_epochs,
            "completed_epochs": self.progress["completed_epochs"],
            "best_epoch": self.best_epoch,
            "best_val_raw_mae": None if self.best_val_score == float("inf") else self.best_val_score,
            "history": self.train_history,
            "final_train_loss": self.train_history[-1]["train_loss"] if self.train_history else None,
            "final_val_loss": self.train_history[-1]["val_loss"] if self.train_history else None,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        write_json(client_train_metrics_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id), summary)
        self._save_progress()
        return summary

    def _train_single_batch(self, X: torch.Tensor, y: torch.Tensor) -> float:
        X, y = self._prepare_batch_for_lstm(X, y)
        self.optimizer.zero_grad(set_to_none=True)

        use_cudnn = not (self.device.type == "cuda")
        with torch.backends.cudnn.flags(enabled=use_cudnn):
            activations = self.client_model(X)
            if activations.dim() == 3:
                activations = activations[:, -1, :]
            result = self._post_pickle(
                f"{self.current_server_url}/train_step",
                self._pack_payload(self.runtime_cfg.client_id, activations, y),
            )
            loss = float(result["loss"])
            if not math.isfinite(loss):
                raise ValueError(f"[Client {self.runtime_cfg.client_id}] Non-finite server loss: {loss}")

            grad = result["grad_activations"].to(self.device)
            if not torch.isfinite(grad).all():
                raise ValueError(f"[Client {self.runtime_cfg.client_id}] Non-finite gradient from server")

            activations.backward(gradient=grad)

        torch.nn.utils.clip_grad_norm_(self.client_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss

    def train_one_batch(self) -> Dict[str, Any]:
        if self.is_done():
            return {"status": "done", **self.status()}

        batch = self._consume_prefetched_train_batch()
        if batch is None:
            epoch_number = self.progress["completed_epochs"] + 1
            batch_count = int(self.progress.get("current_epoch_batch_count", 0))
            epoch_loss = float(self.progress.get("current_epoch_loss_sum", 0.0))
            avg_train_loss = (epoch_loss / batch_count) if batch_count > 0 else None
            self._mark_epoch_finished()
            summary = self._finalize_epoch(epoch_number, avg_train_loss)
            return {"status": "epoch_completed", "summary": summary, **self.status()}

        X, y = batch
        loss = self._train_single_batch(X, y)
        self.progress["current_epoch_loss_sum"] = float(self.progress.get("current_epoch_loss_sum", 0.0)) + loss
        self.progress["current_epoch_batch_count"] = int(self.progress.get("current_epoch_batch_count", 0)) + 1
        self.progress["stream_window_offset"] = int(self.progress.get("stream_window_offset", 0)) + int(self.runtime_cfg.batch_size)
        self._save_progress()
        return {
            "status": "batch_processed",
            "loss": loss,
            "current_epoch": self.progress["completed_epochs"] + 1,
            "current_epoch_batch_count": self.progress["current_epoch_batch_count"],
            **self.status(),
        }

    def train_one_epoch(self) -> Dict[str, Any]:
        while True:
            result = self.train_one_batch()
            if result["status"] != "batch_processed":
                return result

    def train_epochs(self) -> Dict[str, Any]:
        last = {}
        while not self.is_done():
            last = self.train_one_epoch()
        return {"status": "training_complete", "last": last, **self.status()}

    def is_done(self) -> bool:
        return self.early_stopped or self.progress["completed_epochs"] >= self.runtime_cfg.num_epochs

    def status(self) -> Dict[str, Any]:
        return {
            "client_id": self.runtime_cfg.client_id,
            "cluster_id": self.runtime_cfg.cluster_id,
            "completed_epochs": int(self.progress.get("completed_epochs", 0)),
            "current_epoch_batch_count": int(self.progress.get("current_epoch_batch_count", 0)),
            "train_stream_active": self.train_stream_active,
            "has_next_train_batch": (False if self.is_done() else self._prefetch_train_batch()),
            "stream_window_offset": int(self.progress.get("stream_window_offset", 0)),
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "best_epoch": self.best_epoch,
            "best_val_score": None if self.best_val_score == float("inf") else self.best_val_score,
            "is_done": self.is_done(),
            "device": self.device_info,
        }

    def finalize_epoch_if_needed(self) -> Dict[str, Any]:
        if self.is_done():
            return {"status": "done", **self.status()}

        if self._prefetch_train_batch():
            return {
                "status": "not_epoch_boundary",
                "message": "train batch is still available; epoch finalization is not allowed yet",
                **self.status(),
            }

        epoch_number = self.progress["completed_epochs"] + 1
        batch_count = int(self.progress.get("current_epoch_batch_count", 0))
        epoch_loss = float(self.progress.get("current_epoch_loss_sum", 0.0))
        avg_train_loss = (epoch_loss / batch_count) if batch_count > 0 else None
        self._mark_epoch_finished()
        summary = self._finalize_epoch(epoch_number, avg_train_loss)
        final_status = "training_complete" if self.is_done() else "epoch_completed"
        return {"status": final_status, "summary": summary, **self.status()}

    def test(self) -> Dict[str, Any]:
        mode = str(self.runtime_cfg.eval_mode).strip().lower()
        client_paths = []
        server_route = "/load_final_checkpoint"
        if mode == "paired_best":
            client_paths = [
                client_best_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
                client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
                client_final_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
            ]
            server_route = "/load_best_checkpoint"
        elif mode in {"final_shared", "final"}:
            client_paths = [
                client_final_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
                client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
            ]
            server_route = "/load_final_checkpoint"
        elif mode == "latest":
            client_paths = [client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)]
            server_route = "/load_latest_checkpoint"
        else:
            raise ValueError(f"Unsupported EVAL_MODE: {self.runtime_cfg.eval_mode}")

        ok = False
        for client_path in client_paths:
            _, ok = try_load_model_checkpoint(self.client_model, None, client_path)
            if ok:
                break
        if not ok:
            raise FileNotFoundError(f"Missing client checkpoint candidates: {client_paths}")

        resp = self.session.post(f"{self.current_server_url}{server_route}", timeout=30.0)
        resp.raise_for_status()
        result = self._run_eval("test", save_json_path=client_test_metrics_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id))
        if result is None:
            raise RuntimeError("Test evaluation produced no batches")
        return result


runner = ClusteredClientRunner()


def _error_response(exc: Exception, endpoint: str):
    tb = traceback.format_exc()
    print(f"[Client {runner.runtime_cfg.client_id}] ERROR endpoint={endpoint}\n{tb}", flush=True)
    try:
        status = runner.status()
    except Exception as status_exc:
        status = {
            "status_error_type": type(status_exc).__name__,
            "status_error": str(status_exc),
        }
    payload = {
        "status": "error",
        "endpoint": endpoint,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": tb,
        "worker_status": status,
    }
    return jsonify(payload), 500


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", **runner.status()})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(runner.status())


@app.route("/command/train_one_batch", methods=["POST"])
def command_train_one_batch():
    try:
        return jsonify(runner.train_one_batch())
    except Exception as exc:
        return _error_response(exc, "/command/train_one_batch")


@app.route("/command/train_one_epoch", methods=["POST"])
def command_train_one_epoch():
    try:
        return jsonify(runner.train_one_epoch())
    except Exception as exc:
        return _error_response(exc, "/command/train_one_epoch")


@app.route("/command/train", methods=["POST"])
def command_train():
    try:
        return jsonify(runner.train_epochs())
    except Exception as exc:
        return _error_response(exc, "/command/train")


@app.route("/command/finalize_epoch", methods=["POST"])
def command_finalize_epoch():
    try:
        return jsonify(runner.finalize_epoch_if_needed())
    except Exception as exc:
        return _error_response(exc, "/command/finalize_epoch")


@app.route("/command/test", methods=["POST"])
def command_test():
    try:
        return jsonify(runner.test())
    except Exception as exc:
        return _error_response(exc, "/command/test")


if __name__ == "__main__":
    print(f"[Client Worker] Starting client={runner.runtime_cfg.client_id} cluster={runner.runtime_cfg.cluster_id} port={runner.runtime_cfg.worker_port}", flush=True)
    app.run(host="0.0.0.0", port=runner.runtime_cfg.worker_port)