#!/usr/bin/env python3
import csv
import os
import pickle
import random
import sys
import traceback
import time
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
from models.split_lstm import build_split_lstm
from utils.comm_logging import append_comm_row, calc_bandwidth, should_log_index, tensor_nbytes, tensor_shape
from utils.runtime_utils import (
    _get_env,
    augment_sequence_inputs_with_quality,
    build_lstm_config_from_env,
    build_quality_anchor_config,
    compute_model_input_size,
    derive_dq_mode,
    get_device_info,
    parse_bool,
    read_json,
    resolve_device,
    save_torch_checkpoint,
    try_load_torch_checkpoint,
    write_json,
)

app = Flask(__name__)


def _cluster_output_dir(cluster_id: str) -> str:
    path = _get_env("OUTPUT_DIR", os.path.join("output", f"cluster_{cluster_id}"))
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


def client_test_predictions_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_test_predictions.csv")


def client_progress_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_progress.json")


def client_comm_log_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_comm.csv")


def write_prediction_points_csv(
    path: str,
    *,
    timestamps,
    window_start_indices,
    window_end_indices,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    true_values = y_true.reshape(-1)
    pred_values = y_pred.reshape(-1)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "window_start_index",
                "window_end_index",
                "y_true",
                "y_pred",
                "abs_error",
                "sq_error",
            ],
        )
        writer.writeheader()
        for idx, (true_value, pred_value) in enumerate(zip(true_values, pred_values)):
            err = float(pred_value - true_value)
            writer.writerow(
                {
                    "timestamp": timestamps[idx] if idx < len(timestamps) else "",
                    "window_start_index": window_start_indices[idx] if idx < len(window_start_indices) else "",
                    "window_end_index": window_end_indices[idx] if idx < len(window_end_indices) else "",
                    "y_true": float(true_value),
                    "y_pred": float(pred_value),
                    "abs_error": abs(err),
                    "sq_error": err * err,
                }
            )


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
    use_quality_as_input: bool
    use_quality_weights: bool
    use_quality_aggregation_weights: bool
    use_quality_anchor_aggregation: bool
    use_gradient_alignment_filter: bool
    use_high_quality_validation_guardrail: bool
    dq_mode: str
    quality_score_path: str
    quality_input_dim: int
    base_input_size: Optional[int]
    model_input_size: Optional[int]


class ClusteredClientRunner:
    def __init__(self) -> None:
        use_quality_as_input = parse_bool(_get_env("USE_QUALITY_AS_INPUT", "false"))
        use_quality_weights = parse_bool(_get_env("USE_QUALITY_WEIGHTS", "false"))
        use_quality_aggregation_weights = parse_bool(_get_env("USE_QUALITY_AGGREGATION_WEIGHTS", "false"))
        quality_anchor_config = build_quality_anchor_config(os.environ)
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
            use_quality_as_input=use_quality_as_input,
            use_quality_weights=use_quality_weights,
            use_quality_aggregation_weights=use_quality_aggregation_weights,
            use_quality_anchor_aggregation=quality_anchor_config.use_quality_anchor_aggregation,
            use_gradient_alignment_filter=quality_anchor_config.use_gradient_alignment_filter,
            use_high_quality_validation_guardrail=quality_anchor_config.use_high_quality_validation_guardrail,
            dq_mode=derive_dq_mode(
                use_quality_weights,
                use_quality_as_input,
                use_quality_aggregation_weights,
                quality_anchor_config.use_quality_anchor_aggregation,
                quality_anchor_config.use_gradient_alignment_filter,
                quality_anchor_config.use_high_quality_validation_guardrail,
            ),
            quality_score_path=str(_get_env("QUALITY_SCORE_PATH", "") or "").strip(),
            quality_input_dim=int(_get_env("QUALITY_INPUT_DIM", "1")),
            base_input_size=None,
            model_input_size=None,
        )
        self.device = resolve_device()
        self.device_info = get_device_info(self.device, process_role_default="client")
        seed = self.runtime_cfg.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.current_server_url = f"http://localhost:{self.runtime_cfg.server_port}"
        self.session = requests.Session()
        self.norm_stats = self._fetch_norm_stats()
        self.comm_log_train = parse_bool(_get_env("COMM_LOG_TRAIN", "true"), default=True)
        self.comm_log_eval = parse_bool(_get_env("COMM_LOG_EVAL", "false"), default=False)
        self.comm_log_every_n_batches = max(1, int(_get_env("COMM_LOG_EVERY_N_BATCHES", "1")))
        self.progress_save_every_n_batches = max(1, int(_get_env("SAVE_PROGRESS_EVERY_N_BATCHES", "1")))
        self.comm_log_path = client_comm_log_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)

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
        self.train_iter: Optional[Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]] = None
        self.train_stream_active = bool(self.progress.get("train_stream_active", False))
        self.prefetched_train_batch = None
        self.train_epoch_exhausted = False
        if self.train_stream_active:
            self._reset_stream("train", window_offset=int(self.progress.get("stream_window_offset", 0)))
            self.train_iter = iter(self.train_dataset)

        self._save_run_config()
        print(
            f"[Client {self.runtime_cfg.client_id}] device_info={self.device_info} "
            f"client_lr={self.runtime_cfg.learning_rate} dq_mode={self.runtime_cfg.dq_mode} "
            f"use_quality_as_input={self.runtime_cfg.use_quality_as_input} "
            f"use_quality_weights={self.runtime_cfg.use_quality_weights} "
            f"use_quality_aggregation_weights={self.runtime_cfg.use_quality_aggregation_weights} "
            f"use_quality_anchor_aggregation={self.runtime_cfg.use_quality_anchor_aggregation} "
            f"use_gradient_alignment_filter={self.runtime_cfg.use_gradient_alignment_filter} "
            f"base_input_size={self.runtime_cfg.base_input_size} "
            f"model_input_size={self.runtime_cfg.model_input_size}",
            flush=True,
        )

    def _build_model(self) -> nn.Module:
        cfg, split_layer = build_lstm_config_from_env()
        self._validate_model_input_contract(int(cfg.input_size))
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
                    "base_input_size": self.runtime_cfg.base_input_size,
                    "hidden_size": cfg.hidden_size,
                    "num_layers": cfg.num_layers,
                    "split_layer": split_layer,
                    "dropout": cfg.dropout,
                    "bidirectional": cfg.bidirectional,
                    "output_size": cfg.output_size,
                    "quality_feature_appended": self.runtime_cfg.use_quality_as_input,
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
                    "save_progress_every_n_batches": self.progress_save_every_n_batches,
                },
                "data_quality": {
                    "use_quality_as_input": self.runtime_cfg.use_quality_as_input,
                    "use_quality_weights": self.runtime_cfg.use_quality_weights,
                    "use_quality_aggregation_weights": self.runtime_cfg.use_quality_aggregation_weights,
                    "use_quality_anchor_aggregation": self.runtime_cfg.use_quality_anchor_aggregation,
                    "use_gradient_alignment_filter": self.runtime_cfg.use_gradient_alignment_filter,
                    "use_high_quality_validation_guardrail": self.runtime_cfg.use_high_quality_validation_guardrail,
                    "dq_mode": self.runtime_cfg.dq_mode,
                    "base_input_size": self.runtime_cfg.base_input_size,
                    "model_input_size": self.runtime_cfg.model_input_size,
                    "quality_feature_appended": self.runtime_cfg.use_quality_as_input,
                    "quality_score_path": self.runtime_cfg.quality_score_path or None,
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

    def _save_progress(self, *, force: bool = False) -> None:
        self.progress["history"] = self.train_history
        self.progress["best_val_score"] = None if self.best_val_score == float("inf") else self.best_val_score
        self.progress["best_epoch"] = self.best_epoch
        self.progress["epochs_without_improvement"] = self.epochs_without_improvement
        self.progress["early_stopped"] = self.early_stopped
        self.progress["stop_reason"] = self.stop_reason
        self.progress["train_stream_active"] = self.train_stream_active
        if not force:
            batch_count = int(self.progress.get("current_epoch_batch_count", 0))
            if batch_count <= 0 or not should_log_index(batch_count, self.progress_save_every_n_batches):
                return
        write_json(client_progress_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id), self.progress)

    def _maybe_load_checkpoint(self) -> int:
        latest_path = client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)
        extra, ok = try_load_torch_checkpoint(
            self.client_model,
            self.optimizer,
            latest_path,
            strict=True,
            log_prefix=f"Client {self.runtime_cfg.client_id}",
        )
        if ok:
            return int((extra or {}).get("epoch", 0) or 0)
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
            self._save_progress(force=True)

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
        self._save_progress(force=True)

    def _fetch_norm_stats(self) -> Dict[str, Any]:
        resp = self.session.get(
            f"{self.runtime_cfg.data_stream_api_url}/stream/norm_stats",
            params={"client_id": self.runtime_cfg.client_id, "cluster_id": self.runtime_cfg.cluster_id},
            timeout=20.0,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_bool(value: Any) -> Optional[bool]:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            return value
        return parse_bool(str(value))

    def _validate_model_input_contract(self, model_input_size: int) -> None:
        if not isinstance(self.norm_stats, dict):
            raise ValueError("Expected stream /stream/norm_stats to return a JSON object.")

        base_input_size = self._coerce_optional_int(self.norm_stats.get("base_input_size"))
        if base_input_size is None:
            base_input_size = self._coerce_optional_int(self.norm_stats.get("input_size"))
        if base_input_size is None:
            raise ValueError("Stream /stream/norm_stats did not provide base input size metadata.")

        self.runtime_cfg.base_input_size = int(base_input_size)
        self.runtime_cfg.model_input_size = int(model_input_size)
        stream_quality_input_dim = self._coerce_optional_int(self.norm_stats.get("quality_input_dim"))
        if stream_quality_input_dim is None:
            stream_quality_input_dim = self.runtime_cfg.quality_input_dim
        self.runtime_cfg.quality_input_dim = int(stream_quality_input_dim)
        expected_model_input_size = compute_model_input_size(
            base_input_size,
            self.runtime_cfg.use_quality_as_input,
            self.runtime_cfg.quality_input_dim,
        )
        if model_input_size != expected_model_input_size:
            suffix = (
                f"{base_input_size}+{self.runtime_cfg.quality_input_dim}"
                if self.runtime_cfg.use_quality_as_input
                else str(base_input_size)
            )
            raise ValueError(
                "MODEL_INPUT_SIZE mismatch: "
                f"stream/base input size is {base_input_size}, "
                f"USE_QUALITY_AS_INPUT={self.runtime_cfg.use_quality_as_input} requires {suffix}, "
                f"but MODEL_INPUT_SIZE is {model_input_size}."
            )

        stream_use_quality_as_input = self._coerce_optional_bool(self.norm_stats.get("use_quality_as_input"))
        if (
            stream_use_quality_as_input is not None
            and stream_use_quality_as_input != self.runtime_cfg.use_quality_as_input
        ):
            raise ValueError(
                "Runtime mismatch: client USE_QUALITY_AS_INPUT="
                f"{self.runtime_cfg.use_quality_as_input} but stream reports "
                f"use_quality_as_input={stream_use_quality_as_input}."
            )

        stream_model_input_size = self._coerce_optional_int(self.norm_stats.get("model_input_size"))
        if stream_model_input_size is not None and stream_model_input_size != model_input_size:
            raise ValueError(
                "MODEL_INPUT_SIZE mismatch between client and stream metadata: "
                f"stream model_input_size={stream_model_input_size}, client MODEL_INPUT_SIZE={model_input_size}."
            )

        stream_dq_mode = str(self.norm_stats.get("dq_mode", "") or "").strip()
        if stream_dq_mode and stream_dq_mode != self.runtime_cfg.dq_mode:
            raise ValueError(
                f"Runtime mismatch: client dq_mode={self.runtime_cfg.dq_mode}, stream dq_mode={stream_dq_mode}."
            )

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

    def _unpack_stream_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if isinstance(batch, tuple):
            if len(batch) == 3:
                X, y, meta = batch
                return X, y, (meta or {})
            if len(batch) == 2:
                X, y = batch
                return X, y, {}
        raise TypeError(f"Unsupported stream batch type: {type(batch).__name__}")

    def _augment_inputs_with_quality(self, X: torch.Tensor, batch_meta: Dict[str, Any], mode: str) -> torch.Tensor:
        X_aug = augment_sequence_inputs_with_quality(
            X,
            batch_meta.get("quality_scores"),
            enabled=self.runtime_cfg.use_quality_as_input,
            mode=mode,
        )
        if X_aug.dim() != 3:
            raise ValueError(f"Expected model input with 3 dims, got shape={tuple(X_aug.shape)} during mode={mode}")
        expected_input_size = self.runtime_cfg.model_input_size
        if expected_input_size is not None and int(X_aug.size(-1)) != int(expected_input_size):
            raise ValueError(
                f"Augmented input size mismatch during mode={mode}: "
                f"expected feature_dim={expected_input_size}, got {int(X_aug.size(-1))}"
            )
        return X_aug

    def _pack_payload(
        self,
        client_id: str,
        activations: torch.Tensor,
        targets: Optional[torch.Tensor],
        quality_scores: Optional[torch.Tensor] = None,
        quality_score_metadata: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        payload = {"client_id": str(client_id), "activations": activations.detach().cpu()}
        if targets is not None:
            payload["targets"] = targets.detach().cpu()
        if quality_scores is not None:
            payload["quality_scores"] = quality_scores.detach().cpu()
        if quality_score_metadata is not None:
            payload["quality_score_metadata"] = quality_score_metadata
        if meta:
            payload.update(meta)
        return pickle.dumps(payload)

    def _append_comm_row(self, row: Dict[str, Any]) -> None:
        append_comm_row(self.comm_log_path, row)

    def _post_pickle(self, url: str, data: bytes, *, log_meta: Optional[Dict[str, Any]] = None):
        route = "/" + url.rstrip("/").split("/")[-1]
        t0 = time.perf_counter()
        response_bytes = None
        latency_ms = None
        status = "ok"
        result = None
        try:
            resp = self.session.post(url, data=data, timeout=120.0)
            response_bytes = len(resp.content)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            if resp.status_code >= 400:
                status = f"http_{resp.status_code}"
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            result = pickle.loads(resp.content)
            if log_meta and log_meta.get("log_enabled", False):
                upload_MBps, upload_Mbps = calc_bandwidth(len(data), latency_ms)
                download_MBps, download_Mbps = calc_bandwidth(response_bytes, latency_ms)
                base_row = {
                    "level": "split",
                    "observer": "client",
                    "cluster_id": self.runtime_cfg.cluster_id,
                    "client_id": self.runtime_cfg.client_id,
                    "scheduler_epoch": log_meta.get("scheduler_epoch"),
                    "local_epoch": log_meta.get("local_epoch"),
                    "round_id": log_meta.get("round_id"),
                    "batch_id": log_meta.get("batch_id"),
                    "event": log_meta.get("event"),
                    "route": route,
                    "status": status,
                    "activation_shape": log_meta.get("activation_shape"),
                    "activation_bytes": log_meta.get("activation_bytes"),
                    "target_bytes": log_meta.get("target_bytes"),
                    "loss": result.get("loss") if isinstance(result, dict) else None,
                }
                self._append_comm_row(
                    {
                        **base_row,
                        "direction": "client_to_server",
                        "payload_type": log_meta.get("request_payload_type"),
                    "payload_shape": log_meta.get("activation_shape"),
                    "payload_bytes": len(data),
                    "latency_ms": latency_ms,
                    "bandwidth_MBps": upload_MBps,
                    "bandwidth_Mbps": upload_Mbps,
                    "weighted_loss": result.get("weighted_loss") if isinstance(result, dict) else None,
                    "quality_score_min": result.get("quality_score_min") if isinstance(result, dict) else None,
                    "quality_score_max": result.get("quality_score_max") if isinstance(result, dict) else None,
                    "quality_score_mean": result.get("quality_score_mean") if isinstance(result, dict) else None,
                }
                )
                response_payload_type = log_meta.get("response_payload_type")
                grad = result.get("grad_activations") if isinstance(result, dict) else None
                grad_shape = tensor_shape(grad)
                grad_bytes = tensor_nbytes(grad)
                response_shape = grad_shape
                if response_payload_type == "prediction_payload" and isinstance(result, dict):
                    preds = result.get("predictions")
                    if preds is not None:
                        try:
                            response_shape = f"{len(preds)}"
                        except Exception:
                            response_shape = None
                self._append_comm_row(
                    {
                        **base_row,
                        "direction": "server_to_client",
                        "payload_type": response_payload_type,
                        "payload_shape": response_shape,
                        "payload_bytes": response_bytes,
                        "response_payload_bytes": response_bytes,
                        "grad_shape": grad_shape,
                        "grad_bytes": grad_bytes,
                        "latency_ms": latency_ms,
                        "bandwidth_MBps": download_MBps,
                        "bandwidth_Mbps": download_Mbps,
                    }
                )
            return result
        except Exception:
            if log_meta and log_meta.get("log_enabled", False):
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3) if latency_ms is None else latency_ms
                upload_MBps, upload_Mbps = calc_bandwidth(len(data), latency_ms)
                self._append_comm_row(
                    {
                        "level": "split",
                        "observer": "client",
                        "cluster_id": self.runtime_cfg.cluster_id,
                        "client_id": self.runtime_cfg.client_id,
                        "scheduler_epoch": log_meta.get("scheduler_epoch"),
                        "local_epoch": log_meta.get("local_epoch"),
                        "round_id": log_meta.get("round_id"),
                        "batch_id": log_meta.get("batch_id"),
                        "event": log_meta.get("event"),
                        "route": route,
                        "direction": "client_to_server",
                        "payload_type": log_meta.get("request_payload_type"),
                        "payload_shape": log_meta.get("activation_shape"),
                        "payload_bytes": len(data),
                        "activation_shape": log_meta.get("activation_shape"),
                        "activation_bytes": log_meta.get("activation_bytes"),
                        "target_bytes": log_meta.get("target_bytes"),
                        "latency_ms": latency_ms,
                        "bandwidth_MBps": upload_MBps,
                        "bandwidth_Mbps": upload_Mbps,
                        "status": "error",
                    }
                )
            raise

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
        save_torch_checkpoint(
            self.client_model,
            self.optimizer,
            client_best_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
            extra={"epoch": epoch},
            log_prefix="Client",
        )
        resp = self.session.post(f"{self.current_server_url}/save_best_checkpoint", timeout=30.0)
        resp.raise_for_status()

    def _save_latest(self, epoch: int) -> None:
        save_torch_checkpoint(
            self.client_model,
            self.optimizer,
            client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
            extra={"epoch": epoch},
            log_prefix="Client",
        )

    def _save_final(self, epoch: int) -> None:
        save_torch_checkpoint(
            self.client_model,
            self.optimizer,
            client_final_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
            extra={"epoch": epoch},
            log_prefix="Client",
        )

    def _run_eval(self, mode: str, epoch: Optional[int] = None, save_json_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if mode not in {"val", "test"}:
            raise ValueError(f"Unsupported eval mode: {mode}")
        dataset = self.val_dataset if mode == "val" else self.test_dataset
        self._reset_stream(mode)
        loader = iter(dataset)
        route = "/validate_step" if mode == "val" else "/inference_step"
        self.client_model.eval()
        preds_norm, trues_norm = [], []
        target_timestamps, window_start_indices, window_end_indices = [], [], []
        total_server_loss = 0.0
        num_batches = 0
        num_samples = 0
        with torch.no_grad():
            batch_idx = 0
            for batch in loader:
                X, y, batch_meta = self._unpack_stream_batch(batch)
                batch_idx += 1
                X, y = self._prepare_batch_for_lstm(X, y)
                X = self._augment_inputs_with_quality(X, batch_meta, mode=mode)
                # The active split contract sends only the last hidden state [B, H].
                activations = self.client_model(X)[:, -1, :]
                log_enabled = self.comm_log_eval and should_log_index(batch_idx, self.comm_log_every_n_batches)
                log_meta = {
                    "log_enabled": log_enabled,
                    "scheduler_epoch": epoch,
                    "local_epoch": epoch,
                    "round_id": None,
                    "batch_id": batch_idx,
                    "event": "validate_step" if mode == "val" else "inference_step",
                    "request_payload_type": "validation_payload" if mode == "val" else "inference_payload",
                    "response_payload_type": "prediction_payload",
                    "activation_shape": tensor_shape(activations),
                    "activation_bytes": tensor_nbytes(activations),
                    "target_bytes": tensor_nbytes(y) if mode == "val" else None,
                }
                if mode == "val":
                    result = self._post_pickle(
                        f"{self.current_server_url}{route}",
                        self._pack_payload(
                            self.runtime_cfg.client_id,
                            activations,
                            y,
                            quality_scores=None,
                            quality_score_metadata=None,
                            meta={"scheduler_epoch": epoch, "local_epoch": epoch, "batch_id": batch_idx},
                        ),
                        log_meta=log_meta,
                    )
                    total_server_loss += float(result["loss"])
                    batch_preds = np.asarray(result["predictions"], dtype=np.float32)
                else:
                    result = self._post_pickle(
                        f"{self.current_server_url}{route}",
                        self._pack_payload(
                            self.runtime_cfg.client_id,
                            activations,
                            None,
                            quality_scores=None,
                            quality_score_metadata=None,
                            meta={"scheduler_epoch": epoch, "local_epoch": epoch, "batch_id": batch_idx},
                        ),
                        log_meta=log_meta,
                    )
                    batch_preds = np.asarray(result["predictions"], dtype=np.float32)
                batch_true = y.detach().cpu().numpy().astype(np.float32)
                preds_norm.append(batch_preds.reshape(-1, 1))
                trues_norm.append(batch_true.reshape(-1, 1))
                target_timestamps.extend(batch_meta.get("target_timestamps", []))
                window_start_indices.extend(batch_meta.get("window_start_indices", []))
                window_end_indices.extend(batch_meta.get("window_end_indices", []))
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
            "dq_mode": self.runtime_cfg.dq_mode,
            "use_quality_as_input": self.runtime_cfg.use_quality_as_input,
            "use_quality_weights": self.runtime_cfg.use_quality_weights,
            "use_quality_aggregation_weights": self.runtime_cfg.use_quality_aggregation_weights,
            "use_quality_anchor_aggregation": self.runtime_cfg.use_quality_anchor_aggregation,
            "use_gradient_alignment_filter": self.runtime_cfg.use_gradient_alignment_filter,
            "use_high_quality_validation_guardrail": self.runtime_cfg.use_high_quality_validation_guardrail,
            "base_input_size": self.runtime_cfg.base_input_size,
            "model_input_size": self.runtime_cfg.model_input_size,
            "num_batches": num_batches,
            "num_samples": num_samples,
            "server_loss": (total_server_loss / num_batches) if mode == "val" else None,
            "normalized_metrics": self._compute_metrics(preds_norm_np, trues_norm_np),
            "raw_metrics": self._compute_metrics(preds_raw, trues_raw),
            "normalization": self.norm_stats,
            "eval_mode": self.runtime_cfg.eval_mode,
            "prediction_points_path": client_test_predictions_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id) if mode == "test" else None,
        }
        if mode == "test":
            write_prediction_points_csv(
                client_test_predictions_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
                timestamps=target_timestamps,
                window_start_indices=window_start_indices,
                window_end_indices=window_end_indices,
                y_true=trues_raw,
                y_pred=preds_raw,
            )
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
            "dq_mode": self.runtime_cfg.dq_mode,
            "use_quality_as_input": self.runtime_cfg.use_quality_as_input,
            "use_quality_weights": self.runtime_cfg.use_quality_weights,
            "use_quality_aggregation_weights": self.runtime_cfg.use_quality_aggregation_weights,
            "use_quality_anchor_aggregation": self.runtime_cfg.use_quality_anchor_aggregation,
            "use_gradient_alignment_filter": self.runtime_cfg.use_gradient_alignment_filter,
            "use_high_quality_validation_guardrail": self.runtime_cfg.use_high_quality_validation_guardrail,
            "base_input_size": self.runtime_cfg.base_input_size,
            "model_input_size": self.runtime_cfg.model_input_size,
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
        self._save_progress(force=True)
        return summary

    def _train_single_batch(self, X: torch.Tensor, y: torch.Tensor, batch_meta: Dict[str, Any]) -> float:
        X, y = self._prepare_batch_for_lstm(X, y)
        mode = str(batch_meta.get("mode", "train"))
        X = self._augment_inputs_with_quality(X, batch_meta, mode=mode)
        self.optimizer.zero_grad(set_to_none=True)
        local_epoch = int(self.progress.get("completed_epochs", 0)) + 1
        batch_id = int(self.progress.get("current_epoch_batch_count", 0)) + 1
        if self.runtime_cfg.use_quality_anchor_aggregation:
            quality_anchor_scores = batch_meta.get("quality_anchor_scores")
            if quality_anchor_scores is None:
                raise ValueError(
                    "quality_anchor_scores missing while USE_QUALITY_ANCHOR_AGGREGATION=true"
                )
            if not torch.is_tensor(quality_anchor_scores):
                quality_anchor_scores = torch.tensor(quality_anchor_scores, dtype=torch.float32)
            send_quality_scores = quality_anchor_scores
            send_quality_score_metadata = batch_meta.get("quality_score_metadata", {})
        else:
            quality_weight_scores = batch_meta.get("quality_weight_scores")
            if quality_weight_scores is None:
                quality_weight_scores = batch_meta.get("quality_scores")
            if quality_weight_scores is not None and not torch.is_tensor(quality_weight_scores):
                quality_weight_scores = torch.tensor(quality_weight_scores, dtype=torch.float32)
            server_needs_quality = (
                self.runtime_cfg.use_quality_weights
                or self.runtime_cfg.use_quality_aggregation_weights
            )
            send_quality_scores = quality_weight_scores if server_needs_quality else None
            send_quality_score_metadata = (
                batch_meta.get("quality_score_metadata", {}) if server_needs_quality else None
            )

        use_cudnn = not (self.device.type == "cuda")
        with torch.backends.cudnn.flags(enabled=use_cudnn):
            # The active split contract sends only the last hidden state [B, H].
            activations = self.client_model(X)[:, -1, :]
            payload_meta = {
                "scheduler_epoch": local_epoch,
                "local_epoch": local_epoch,
                "batch_id": batch_id,
                "mode": mode,
                "window_start_indices": batch_meta.get("window_start_indices", []),
                "window_end_indices": batch_meta.get("window_end_indices", []),
                "window_start_timestamps": batch_meta.get("window_start_timestamps", []),
                "window_end_timestamps": batch_meta.get("window_end_timestamps", []),
                "target_timestamps": batch_meta.get("target_timestamps", []),
            }
            result = self._post_pickle(
                f"{self.current_server_url}/train_step",
                self._pack_payload(
                    self.runtime_cfg.client_id,
                    activations,
                    y,
                    quality_scores=send_quality_scores,
                    quality_score_metadata=send_quality_score_metadata,
                    meta=payload_meta,
                ),
                log_meta={
                    "log_enabled": self.comm_log_train and should_log_index(batch_id, self.comm_log_every_n_batches),
                    "scheduler_epoch": local_epoch,
                    "local_epoch": local_epoch,
                    "round_id": None,
                    "batch_id": batch_id,
                    "event": "train_step",
                    "request_payload_type": "activation_payload",
                    "response_payload_type": "gradient_payload",
                    "activation_shape": tensor_shape(activations),
                    "activation_bytes": tensor_nbytes(activations),
                    "target_bytes": tensor_nbytes(y),
                },
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

        X, y, batch_meta = self._unpack_stream_batch(batch)
        loss = self._train_single_batch(X, y, batch_meta)
        self.progress["current_epoch_loss_sum"] = float(self.progress.get("current_epoch_loss_sum", 0.0)) + loss
        self.progress["current_epoch_batch_count"] = int(self.progress.get("current_epoch_batch_count", 0)) + 1
        self.progress["stream_window_offset"] = int(self.progress.get("stream_window_offset", 0)) + int(X.shape[0])
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
            "dq_mode": self.runtime_cfg.dq_mode,
            "use_quality_as_input": self.runtime_cfg.use_quality_as_input,
            "use_quality_weights": self.runtime_cfg.use_quality_weights,
            "use_quality_aggregation_weights": self.runtime_cfg.use_quality_aggregation_weights,
            "use_quality_anchor_aggregation": self.runtime_cfg.use_quality_anchor_aggregation,
            "use_gradient_alignment_filter": self.runtime_cfg.use_gradient_alignment_filter,
            "use_high_quality_validation_guardrail": self.runtime_cfg.use_high_quality_validation_guardrail,
            "base_input_size": self.runtime_cfg.base_input_size,
            "model_input_size": self.runtime_cfg.model_input_size,
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
            _, ok = try_load_torch_checkpoint(
                self.client_model,
                None,
                client_path,
                strict=True,
                log_prefix="Client",
            )
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
