# # client/clustered_client.py
# """
# Modified split-learning client.

# Key changes:
# - Adds an explicit test mode: `python client.py test`
# - Evaluates on the real test split (not inference)
# - Inverse-transforms predictions and targets using train-only normalization stats
# - Writes structured JSON metrics files instead of relying on log parsing
# - Saves a paired best client checkpoint and asks the server to save the paired best server checkpoint
# - Logs [TRAIN]/[VAL]/[TEST] lines for easy human inspection
# """

# import json
# import math
# import os
# import pickle
# import random
# import sys
# from dataclasses import asdict, dataclass
# from typing import Any, Dict, Optional, Tuple

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# import numpy as np
# import requests
# import torch
# from prometheus_client import Gauge, Histogram, start_http_server
# from torch import nn, optim

# from models.split_lstm import LSTMConfig, build_split_lstm
# from data.data_streaming import StreamingIterableDataset


# def parse_bool(env_value: Optional[str], default: bool = False) -> bool:
#     if env_value is None:
#         return default
#     return env_value.strip().lower() in ("1", "true", "yes", "y")


# def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
#     val = os.environ.get(name)
#     if val is None or val.strip() == "":
#         return default
#     return val



# def resolve_device() -> torch.device:
#     gpu_enabled = parse_bool(_get_env("GPU_ENABLED", "true"), default=True)
#     gpu_id = _get_env("GPU_ID", "")
#     if gpu_enabled and torch.cuda.is_available():
#         if gpu_id not in (None, ""):
#             return torch.device(f"cuda:{gpu_id}")
#         return torch.device("cuda")
#     return torch.device("cpu")


# def get_device_info(device: torch.device) -> Dict[str, Any]:
#     info: Dict[str, Any] = {
#         "selected_device": str(device),
#         "gpu_enabled": parse_bool(_get_env("GPU_ENABLED", "true"), default=True),
#         "gpu_id_env": _get_env("GPU_ID", ""),
#         "cuda_visible_devices": _get_env("CUDA_VISIBLE_DEVICES", ""),
#         "cuda_available": torch.cuda.is_available(),
#         "cuda_device_count": torch.cuda.device_count(),
#         "process_role": _get_env("PROCESS_ROLE", "client"),
#     }
#     if device.type == "cuda" and torch.cuda.is_available():
#         idx = device.index if device.index is not None else torch.cuda.current_device()
#         props = torch.cuda.get_device_properties(idx)
#         info.update({
#             "cuda_index": idx,
#             "cuda_name": torch.cuda.get_device_name(idx),
#             "cuda_capability": f"{props.major}.{props.minor}",
#             "cuda_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
#         })
#     return info


# def build_lstm_config_from_env() -> Tuple[LSTMConfig, int]:
#     cfg = LSTMConfig(
#         input_size=int(_get_env("MODEL_INPUT_SIZE", "24")),
#         hidden_size=int(_get_env("MODEL_HIDDEN_SIZE", "128")),
#         num_layers=int(_get_env("MODEL_NUM_LAYERS", "3")),
#         output_size=int(_get_env("MODEL_OUTPUT_SIZE", "1")),
#         dropout=float(_get_env("MODEL_DROPOUT", "0.4")),
#         bidirectional=parse_bool(_get_env("MODEL_BIDIRECTIONAL", "false")),
#     )
#     split_layer = int(_get_env("MODEL_SPLIT_LAYER", "2"))
#     if split_layer <= 0 or split_layer >= cfg.num_layers:
#         raise ValueError(
#             f"MODEL_SPLIT_LAYER must be in [1, num_layers-1]. Got split={split_layer}, num_layers={cfg.num_layers}"
#         )
#     return cfg, split_layer


# def _cluster_output_dir(cluster_id: str) -> str:
#     return os.path.join("output", f"cluster_{cluster_id}")


# def client_checkpoint_path(cluster_id: str, client_id: str) -> str:
#     return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_checkpoint.pt")


# def client_best_checkpoint_path(cluster_id: str, client_id: str) -> str:
#     return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_best_checkpoint.pt")


# def client_train_metrics_path(cluster_id: str, client_id: str) -> str:
#     return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_train_metrics.json")


# def client_test_metrics_path(cluster_id: str, client_id: str) -> str:
#     return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_test_metrics.json")


# def save_torch_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], epoch: int, path: str) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     payload = {
#         "model_state_dict": model.state_dict(),
#         "epoch": epoch,
#     }
#     if optimizer is not None:
#         payload["optimizer_state_dict"] = optimizer.state_dict()
#     torch.save(payload, path)
#     print(f"[Client] Checkpoint -> {path}")


# def load_torch_checkpoint(path: str) -> Optional[dict]:
#     if os.path.exists(path):
#         return torch.load(path, map_location="cpu")
#     return None




# def try_load_model_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], path: str, *, strict: bool = True) -> Tuple[Optional[int], bool]:
#     """
#     Try to load a checkpoint. Returns (epoch, loaded_ok).

#     If the checkpoint exists but is incompatible with the current architecture
#     (for example after changing num_layers or split_layer), this function keeps
#     the run alive and returns (None, False) instead of crashing the whole client.
#     """
#     ckpt = load_torch_checkpoint(path)
#     if ckpt is None:
#         return None, False
#     try:
#         model.load_state_dict(ckpt["model_state_dict"], strict=strict)
#         if optimizer is not None and "optimizer_state_dict" in ckpt:
#             try:
#                 optimizer.load_state_dict(ckpt["optimizer_state_dict"])
#             except Exception as e:
#                 print(f"[Client] WARNING: optimizer state incompatible for {path}: {e}")
#         return int(ckpt.get("epoch", 0)), True
#     except Exception as e:
#         print(f"[Client] WARNING: skipping incompatible checkpoint {path}: {e}")
#         return None, False

# def write_json(path: str, obj: Dict[str, Any]) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2)


# @dataclass
# class ClientRuntimeConfig:
#     client_id: str
#     cluster_id: str
#     data_root: str
#     num_clusters: int
#     num_clients_per_cluster: int
#     batch_size: int
#     num_epochs: int
#     learning_rate: float
#     data_stream_api_url: Optional[str]
#     metrics_port: int
#     random_seed: int


# @dataclass
# class TrainingRecord:
#     ts: float
#     role: str
#     mode: str
#     cluster_id: str
#     client_id: str
#     epoch: Optional[int]
#     step: int
#     loss: float
#     extra: Optional[Dict[str, Any]] = None


# def build_runtime_config_from_env() -> ClientRuntimeConfig:
#     client_id = _get_env("CLIENT_ID", "9900002")
#     num_clients_per_cluster = int(_get_env("NUM_CLIENTS_PER_CLUSTER", "3"))
#     if "_" in client_id:
#         cluster_id = str((int(client_id.split("_")[-1]) - 1) // num_clients_per_cluster)
#     else:
#         cluster_id = _get_env("CLUSTER_ID", "0")
#     if cluster_id.startswith("cluster_"):
#         cluster_id = cluster_id.replace("cluster_", "")

#     return ClientRuntimeConfig(
#         client_id=client_id,
#         cluster_id=cluster_id,
#         data_root=_get_env("DATA_ROOT", "/mnt/shared/data"),
#         num_clusters=int(_get_env("NUM_CLUSTERS", "3")),
#         num_clients_per_cluster=num_clients_per_cluster,
#         batch_size=int(_get_env("BATCH_SIZE", "32")),
#         num_epochs=int(_get_env("NUM_EPOCHS", "100")),
#         learning_rate=float(_get_env("LEARNING_RATE", "1e-3")),
#         data_stream_api_url=_get_env("DATA_STREAM_API_URL", "http://localhost:5001"),
#         metrics_port=int(_get_env("CLIENT_METRICS_PORT", "9100")),
#         random_seed=int(_get_env("RANDOM_SEED", "42")),
#     )


# class ClusteredClientRunner:
#     def __init__(self, device: Optional[torch.device] = None):
#         self.runtime_cfg = build_runtime_config_from_env()
#         self.device = device or resolve_device()
#         self.device_info = get_device_info(self.device)

#         seed = self.runtime_cfg.random_seed
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)

#         self.current_server_url = self._build_server_url()
#         self.norm_stats = self._fetch_norm_stats()
#         self.best_val_score = float("inf")
#         self.best_epoch = 0
#         self.train_history = []

#         self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
#         self.client_model = self._build_model().to(self.device)
#         self.optimizer = optim.Adam(self.client_model.parameters(), lr=self.runtime_cfg.learning_rate)
#         self.start_epoch = self._maybe_load_checkpoint()
#         self._init_metrics()
#         self._save_run_config()
#         print(f"[Client {self.runtime_cfg.client_id}] device_info={self.device_info}", flush=True)

#     def _save_run_config(self) -> None:
#         cfg, split_layer = build_lstm_config_from_env()
#         write_json(
#             os.path.join(_cluster_output_dir(self.runtime_cfg.cluster_id), f"client_{self.runtime_cfg.client_id}_config.json"),
#             {
#                 "component": "client",
#                 "client_id": self.runtime_cfg.client_id,
#                 "cluster_id": self.runtime_cfg.cluster_id,
#                 "model": {
#                     "input_size": cfg.input_size,
#                     "hidden_size": cfg.hidden_size,
#                     "num_layers": cfg.num_layers,
#                     "split_layer": split_layer,
#                     "dropout": cfg.dropout,
#                     "bidirectional": cfg.bidirectional,
#                     "output_size": cfg.output_size,
#                 },
#                 "training": {
#                     "batch_size": self.runtime_cfg.batch_size,
#                     "num_epochs": self.runtime_cfg.num_epochs,
#                     "learning_rate": self.runtime_cfg.learning_rate,
#                     "random_seed": self.runtime_cfg.random_seed,
#                 },
#                 "device": self.device_info,
#             },
#         )

#     def _build_dataloaders(self):
#         cfg = self.runtime_cfg
#         if not cfg.data_stream_api_url:
#             raise ValueError("DATA_STREAM_API_URL must be set")

#         def ds(mode: str):
#             return StreamingIterableDataset(
#                 api_url=cfg.data_stream_api_url,
#                 client_id=str(cfg.client_id),
#                 cluster_id=str(cfg.cluster_id),
#                 mode=mode,
#                 batch_size=cfg.batch_size,
#             )

#         return ds("train"), ds("val"), ds("test")

#     def _build_server_url(self) -> str:
#         port = _get_env("SERVER_PORT", "5002")
#         if parse_bool(_get_env("USE_LOCALHOST"), default=True):
#             return f"http://localhost:{port}"
#         return f"http://server-cluster-{self.runtime_cfg.cluster_id}:{port}"

#     def _build_model(self) -> nn.Module:
#         cfg, split_layer = build_lstm_config_from_env()
#         client_model, _ = build_split_lstm(cfg, split_layer)
#         print(
#             f"[Client {self.runtime_cfg.client_id}] hidden={cfg.hidden_size}, layers={cfg.num_layers}, split={split_layer}"
#         )
#         return client_model

#     def _maybe_load_checkpoint(self) -> int:
#         local_path = client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)
#         epoch, ok = try_load_model_checkpoint(self.client_model, self.optimizer, local_path)
#         if ok:
#             print(f"[Client {self.runtime_cfg.client_id}] Loaded checkpoint <- {local_path} (epoch={epoch})")
#             return int(epoch or 0)
#         print(f"[Client {self.runtime_cfg.client_id}] No compatible checkpoint - random init.")
#         return 0

#     def _init_metrics(self) -> None:
#         try:
#             start_http_server(self.runtime_cfg.metrics_port, addr="0.0.0.0")
#             labels = ["client_id"]
#             self.m_train_loss = Histogram("split_client_train_loss", "Train loss", labels)
#             self.m_val_loss = Gauge("split_client_val_loss", "Val loss", labels)
#             self.m_test_mae = Gauge("split_client_test_mae", "Test MAE", labels)
#         except Exception:
#             self.m_train_loss = None
#             self.m_val_loss = None
#             self.m_test_mae = None

#     def _reset_stream(self, mode: str) -> None:
#         resp = requests.post(
#             f"{self.runtime_cfg.data_stream_api_url}/stream/reset",
#             json={"client_id": self.runtime_cfg.client_id, "cluster_id": self.runtime_cfg.cluster_id, "mode": mode},
#             timeout=10.0,
#         )
#         resp.raise_for_status()
#         body = resp.json()
#         print(
#             f"[Client {self.runtime_cfg.client_id}] Stream reset ({mode}), {body.get('windows_available', '?')} windows"
#         )

#     def _fetch_norm_stats(self) -> Optional[Dict[str, float]]:
#         try:
#             resp = requests.get(
#                 f"{self.runtime_cfg.data_stream_api_url}/stream/norm_stats",
#                 params={"client_id": self.runtime_cfg.client_id, "cluster_id": self.runtime_cfg.cluster_id},
#                 timeout=10.0,
#             )
#             if resp.status_code == 404:
#                 return None
#             resp.raise_for_status()
#             return resp.json()
#         except Exception as e:
#             print(f"[Client {self.runtime_cfg.client_id}] norm stats unavailable: {e}")
#             return None

#     def _prepare_batch_for_lstm(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
#         if X.dim() == 1:
#             X = X.unsqueeze(1).unsqueeze(2)
#         elif X.dim() == 2:
#             X = X.unsqueeze(2)
#         X = X.float().to(self.device)
#         if y is None:
#             return X
#         y = y.float().to(self.device)
#         if y.dim() == 1:
#             y = y.unsqueeze(1)
#         return X, y

#     @staticmethod
#     def _pack_payload(client_id: str, activations: torch.Tensor, targets: Optional[torch.Tensor] = None):
#         payload = {"client_id": client_id, "activations": activations.detach().cpu()}
#         if targets is not None:
#             payload["targets"] = targets.detach().cpu()
#         return pickle.dumps(payload)

#     @staticmethod
#     def _post_pickle(url: str, payload: bytes, timeout: float = 30.0):
#         resp = requests.post(url, data=payload, headers={"Content-Type": "application/octet-stream"}, timeout=timeout)
#         resp.raise_for_status()
#         return pickle.loads(resp.content)

#     def _inverse_tensor(self, arr: np.ndarray) -> np.ndarray:
#         if not self.norm_stats or not bool(self.norm_stats.get("normalize", False)):
#             return arr
#         mean = float(self.norm_stats["mean"])
#         scale = float(self.norm_stats["scale"])
#         method = str(self.norm_stats.get("method", "zscore")).lower()
#         if method in ("zscore", "minmax"):
#             return arr * scale + mean
#         return arr

#     def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
#         err = preds - targets
#         mse = float(np.mean(np.square(err)))
#         mae = float(np.mean(np.abs(err)))
#         rmse = float(math.sqrt(mse))
#         denom = np.where(np.abs(targets) < 1e-8, np.nan, np.abs(targets))
#         mape = float(np.nanmean(np.abs(err) / denom) * 100.0) if np.any(~np.isnan(denom)) else None
#         return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}

#     def _run_eval(self, mode: str, epoch: Optional[int] = None, save_json_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
#         if mode not in {"val", "test"}:
#             raise ValueError(f"Unsupported eval mode: {mode}")

#         loader = self.val_loader if mode == "val" else self.test_loader
#         route = "/validate_step" if mode == "val" else "/inference_step"

#         self.client_model.eval()
#         self._reset_stream(mode)

#         total_server_loss = 0.0
#         nb = 0
#         ns = 0
#         preds_norm = []
#         trues_norm = []

#         with torch.no_grad():
#             for batch_idx, (X, y) in enumerate(loader):
#                 X, y = self._prepare_batch_for_lstm(X, y)

#                 # ---- INPUT/TARGET DIAGNOSTICS ----
#                 if torch.isnan(X).any():
#                     print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] NaN in input batch {batch_idx}")
#                 if torch.isnan(y).any():
#                     print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] NaN in target batch {batch_idx}")
#                 if torch.isinf(X).any():
#                     print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] Inf in input batch {batch_idx}")
#                 if torch.isinf(y).any():
#                     print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] Inf in target batch {batch_idx}")

#                 activations = self.client_model(X)
#                 if activations.dim() == 3:
#                     activations = activations[:, -1, :]

#                 if torch.isnan(activations).any():
#                     print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] NaN in activations batch {batch_idx}")
#                 if torch.isinf(activations).any():
#                     print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] Inf in activations batch {batch_idx}")

#                 if mode == "val":
#                     result = self._post_pickle(
#                         f"{self.current_server_url}{route}",
#                         self._pack_payload(self.runtime_cfg.client_id, activations, y),
#                     )
#                     batch_preds = np.asarray(result["predictions"], dtype=np.float32)
#                     total_server_loss += float(result["loss"])
#                 else:
#                     result = self._post_pickle(
#                         f"{self.current_server_url}{route}",
#                         self._pack_payload(self.runtime_cfg.client_id, activations, None),
#                     )
#                     batch_preds = np.asarray(result["predictions"], dtype=np.float32)

#                 # ---- PREDICTION DIAGNOSTICS ----
#                 if np.isnan(batch_preds).any():
#                     print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] NaN in predictions batch {batch_idx}")
#                 if np.isinf(batch_preds).any():
#                     print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] Inf in predictions batch {batch_idx}")

#                 batch_true = y.detach().cpu().numpy().astype(np.float32)

#                 preds_norm.append(batch_preds.reshape(-1, 1))
#                 trues_norm.append(batch_true.reshape(-1, 1))
#                 nb += 1
#                 ns += int(batch_true.shape[0])

#         if nb == 0:
#             return None

#         preds_norm_np = np.concatenate(preds_norm, axis=0)
#         trues_norm_np = np.concatenate(trues_norm, axis=0)

#         # ---- FINAL ARRAY DIAGNOSTICS ----
#         print(
#             f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] "
#             f"preds_norm shape={preds_norm_np.shape}, trues_norm shape={trues_norm_np.shape}"
#         )
#         print(
#             f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] "
#             f"preds_norm nan={np.isnan(preds_norm_np).sum()}, trues_norm nan={np.isnan(trues_norm_np).sum()}"
#         )
#         print(
#             f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] "
#             f"preds_norm inf={np.isinf(preds_norm_np).sum()}, trues_norm inf={np.isinf(trues_norm_np).sum()}"
#         )

#         if np.isnan(preds_norm_np).any():
#             bad_idx = np.where(np.isnan(preds_norm_np.reshape(-1)))[0][:10]
#             print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] First NaN pred indices: {bad_idx}")
#         if np.isnan(trues_norm_np).any():
#             bad_idx = np.where(np.isnan(trues_norm_np.reshape(-1)))[0][:10]
#             print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] First NaN target indices: {bad_idx}")

#         preds_raw = self._inverse_tensor(preds_norm_np)
#         trues_raw = self._inverse_tensor(trues_norm_np)

#         print(
#             f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] "
#             f"preds_raw nan={np.isnan(preds_raw).sum()}, trues_raw nan={np.isnan(trues_raw).sum()}"
#         )
#         print(
#             f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] "
#             f"preds_raw inf={np.isinf(preds_raw).sum()}, trues_raw inf={np.isinf(trues_raw).sum()}"
#         )

#         metrics_norm = self._compute_metrics(preds_norm_np, trues_norm_np)
#         metrics_raw = self._compute_metrics(preds_raw, trues_raw)

#         result = {
#             "mode": mode,
#             "epoch": epoch,
#             "client_id": self.runtime_cfg.client_id,
#             "cluster_id": self.runtime_cfg.cluster_id,
#             "num_batches": nb,
#             "num_samples": ns,
#             "server_loss": (total_server_loss / nb) if mode == "val" else None,
#             "normalized_metrics": metrics_norm,
#             "raw_metrics": metrics_raw,
#             "normalization": self.norm_stats,
#         }

#         tag = mode.upper()
#         prefix = f"[Client {self.runtime_cfg.client_id}] [{tag}]"
#         if mode == "val":
#             print(
#                 f"{prefix} epoch={epoch} server_loss={result['server_loss']:.6f} "
#                 f"raw_mae={metrics_raw['mae']:.6f} raw_rmse={metrics_raw['rmse']:.6f} "
#                 f"norm_mae={metrics_norm['mae']:.6f}"
#             )
#             print(
#                 f"[VAL] epoch={epoch} val_loss={result['server_loss']:.6f} "
#                 f"raw_mae={metrics_raw['mae']:.6f} raw_rmse={metrics_raw['rmse']:.6f} "
#                 f"mse={metrics_raw['mse']:.6f}"
#             )
#             if self.m_val_loss is not None:
#                 self.m_val_loss.labels(client_id=self.runtime_cfg.client_id).set(result["server_loss"])
#         else:
#             print(
#                 f"{prefix} mae={metrics_raw['mae']:.6f} rmse={metrics_raw['rmse']:.6f} "
#                 f"mse={metrics_raw['mse']:.6f} mape={metrics_raw['mape']}"
#             )
#             print(
#                 f"[TEST] mae={metrics_raw['mae']:.6f} rmse={metrics_raw['rmse']:.6f} "
#                 f"mse={metrics_raw['mse']:.6f} mape={metrics_raw['mape']}"
#             )
#             if self.m_test_mae is not None and metrics_raw["mae"] is not None:
#                 self.m_test_mae.labels(client_id=self.runtime_cfg.client_id).set(metrics_raw["mae"])

#         if save_json_path:
#             write_json(save_json_path, result)

#         self.client_model.train()
#         return result

#     def _save_best_pair(self, epoch: int) -> None:
#         save_torch_checkpoint(
#             self.client_model,
#             self.optimizer,
#             epoch,
#             client_best_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
#         )
#         try:
#             resp = requests.post(f"{self.current_server_url}/save_best_checkpoint", timeout=20.0)
#             resp.raise_for_status()
#             print(f"[Client {self.runtime_cfg.client_id}] Paired best server checkpoint saved.")
#         except Exception as e:
#             print(f"[Client {self.runtime_cfg.client_id}] WARNING: could not save paired best server checkpoint: {e}")

#     def train_epochs(self) -> None:
#         cfg = self.runtime_cfg
#         print(f"[Client {cfg.client_id}] Training epoch {self.start_epoch + 1}->{cfg.num_epochs}")
#         total_loss = 0.0
#         total_batches = 0
#         last_epoch = self.start_epoch

#         for epoch in range(self.start_epoch, cfg.num_epochs):
#             last_epoch = epoch + 1
#             self.client_model.train()
#             self._reset_stream("train")
#             epoch_loss = 0.0
#             batch_count = 0

#             print(f"\n[Client {cfg.client_id}] Epoch {epoch + 1}/{cfg.num_epochs}")

#             for bidx, (X, y) in enumerate(self.train_loader):
#                 X, y = self._prepare_batch_for_lstm(X, y)
#                 self.optimizer.zero_grad(set_to_none=True)
#                 activations = self.client_model(X)
#                 if activations.dim() == 3:
#                     activations = activations[:, -1, :]

#                 result = self._post_pickle(
#                     f"{self.current_server_url}/train_step",
#                     self._pack_payload(cfg.client_id, activations, y),
#                 )
#                 grad = result["grad_activations"].to(self.device)
#                 loss = float(result["loss"])

#                 activations.backward(gradient=grad)
#                 self.optimizer.step()

#                 epoch_loss += loss
#                 total_loss += loss
#                 batch_count += 1
#                 total_batches += 1

#                 if self.m_train_loss is not None:
#                     self.m_train_loss.labels(client_id=cfg.client_id).observe(loss)

#                 if batch_count % 50 == 0:
#                     print(f"[Client {cfg.client_id}] batch={batch_count} loss={loss:.6f}")

#             avg_train_loss = epoch_loss / batch_count if batch_count else None
#             print(f"[TRAIN] epoch={epoch + 1} train_loss={avg_train_loss:.6f}" if avg_train_loss is not None else f"[TRAIN] epoch={epoch + 1} train_loss=None")

#             val_result = self._run_eval("val", epoch=epoch + 1)
#             val_raw_mae = None if val_result is None else float(val_result["raw_metrics"]["mae"])
#             val_server_loss = None if val_result is None else float(val_result["server_loss"])

#             self.train_history.append(
#                 {
#                     "epoch": epoch + 1,
#                     "train_loss": avg_train_loss,
#                     "val_loss": val_server_loss,
#                     "val_raw_mae": val_raw_mae,
#                     "val_raw_rmse": None if val_result is None else float(val_result["raw_metrics"]["rmse"]),
#                 }
#             )

#             if val_raw_mae is not None and val_raw_mae < self.best_val_score:
#                 self.best_val_score = val_raw_mae
#                 self.best_epoch = epoch + 1
#                 self._save_best_pair(epoch + 1)
#                 print(f"[Client {cfg.client_id}] New best checkpoint at epoch {epoch + 1} with raw_val_mae={val_raw_mae:.6f}")

#             save_torch_checkpoint(
#                 self.client_model,
#                 self.optimizer,
#                 epoch + 1,
#                 client_checkpoint_path(cfg.cluster_id, cfg.client_id),
#             )

#         summary = {
#             "client_id": cfg.client_id,
#             "cluster_id": cfg.cluster_id,
#             "num_epochs": cfg.num_epochs,
#             "start_epoch": self.start_epoch,
#             "last_epoch": last_epoch,
#             "avg_train_loss_over_run": (total_loss / total_batches) if total_batches else None,
#             "best_epoch": self.best_epoch,
#             "best_val_raw_mae": None if self.best_val_score == float("inf") else self.best_val_score,
#             "history": self.train_history,
#             "final_train_loss": self.train_history[-1]["train_loss"] if self.train_history else None,
#             "final_val_loss": self.train_history[-1]["val_loss"] if self.train_history else None,
#         }
#         write_json(client_train_metrics_path(cfg.cluster_id, cfg.client_id), summary)
#         print(f"[Client {cfg.client_id}] Training complete.")

#     def _load_best_checkpoints_for_test(self) -> None:
#         client_best = client_best_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)
#         epoch, ok = try_load_model_checkpoint(self.client_model, None, client_best)
#         if not ok:
#             raise FileNotFoundError(f"Compatible best client checkpoint not found: {client_best}")
#         print(f"[Client {self.runtime_cfg.client_id}] Loaded best client checkpoint <- {client_best}")
#         try:
#             resp = requests.post(f"{self.current_server_url}/load_best_checkpoint", timeout=20.0)
#             resp.raise_for_status()
#             print(f"[Client {self.runtime_cfg.client_id}] Loaded paired best server checkpoint.")
#         except Exception as e:
#             raise RuntimeError(f"Could not load paired best server checkpoint: {e}") from e

#     def test(self) -> Dict[str, Any]:
#         self._load_best_checkpoints_for_test()
#         result = self._run_eval("test", save_json_path=client_test_metrics_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id))
#         if result is None:
#             raise RuntimeError("Test evaluation produced no batches.")
#         return result


# if __name__ == "__main__":
#     runner = ClusteredClientRunner()
#     command = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "train"
#     if command == "train":
#         runner.train_epochs()
#     elif command == "test":
#         runner.test()
#     else:
#         raise SystemExit(f"Unsupported command: {command}. Use 'train' or 'test'.")
# client/client.py
"""
Split-learning client with:
- train / train_one_epoch / train_one_batch / test commands
- persistent progress for scheduler-driven training
- early stopping on validation raw MAE
- structured JSON train/test/step outputs
- eval_mode = final_shared (default) or paired_best
"""

import json
import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import requests
import torch
from prometheus_client import Gauge, Histogram, start_http_server
from torch import nn, optim

from models.split_lstm import LSTMConfig, build_split_lstm
from data.data_streaming import StreamingIterableDataset


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
        raise ValueError(f"MODEL_SPLIT_LAYER must be in [1, num_layers-1]. Got split={split_layer}, num_layers={cfg.num_layers}")
    return cfg, split_layer


def _cluster_output_dir(cluster_id: str) -> str:
    return os.path.join("output", f"cluster_{cluster_id}")


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


def client_step_metrics_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_step_metrics.json")


def client_progress_path(cluster_id: str, client_id: str) -> str:
    return os.path.join(_cluster_output_dir(cluster_id), f"client_{client_id}_progress.json")


def save_torch_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], epoch: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"[Client] Checkpoint -> {path}")


def load_torch_checkpoint(path: str) -> Optional[dict]:
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def try_load_model_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], path: str, *, strict: bool = True) -> Tuple[Optional[int], bool]:
    ckpt = load_torch_checkpoint(path)
    if ckpt is None:
        return None, False
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[Client] WARNING: optimizer state incompatible for {path}: {e}")
        return int(ckpt.get("epoch", 0)), True
    except Exception as e:
        print(f"[Client] WARNING: skipping incompatible checkpoint {path}: {e}")
        return None, False


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ClientRuntimeConfig:
    client_id: str
    cluster_id: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    data_stream_api_url: str
    metrics_port: int
    random_seed: int
    eval_mode: str
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_metric: str


def build_runtime_config_from_env() -> ClientRuntimeConfig:
    client_id = _get_env("CLIENT_ID", "9900002")
    cluster_id = _get_env("CLUSTER_ID", "0")
    if cluster_id.startswith("cluster_"):
        cluster_id = cluster_id.replace("cluster_", "")

    return ClientRuntimeConfig(
        client_id=client_id,
        cluster_id=cluster_id,
        batch_size=int(_get_env("BATCH_SIZE", "32")),
        num_epochs=int(_get_env("NUM_EPOCHS", "100")),
        learning_rate=float(_get_env("LEARNING_RATE", "0.0005")),
        data_stream_api_url=_get_env("DATA_STREAM_API_URL", "http://localhost:5001"),
        metrics_port=int(_get_env("CLIENT_METRICS_PORT", "9100")),
        random_seed=int(_get_env("RANDOM_SEED", "42")),
        eval_mode=_get_env("EVAL_MODE", "final_shared"),
        early_stopping=parse_bool(_get_env("EARLY_STOPPING", "true")),
        early_stopping_patience=int(_get_env("EARLY_STOPPING_PATIENCE", "10")),
        early_stopping_min_delta=float(_get_env("EARLY_STOPPING_MIN_DELTA", "0.0")),
        early_stopping_metric=_get_env("EARLY_STOPPING_METRIC", "val_raw_mae"),
    )


class ClusteredClientRunner:
    def __init__(self, device: Optional[torch.device] = None):
        self.runtime_cfg = build_runtime_config_from_env()
        self.device = device or resolve_device()
        self.device_info = get_device_info(self.device)

        seed = self.runtime_cfg.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.current_server_url = self._build_server_url()
        self.norm_stats = self._fetch_norm_stats()

        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()

        self.client_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.client_model.parameters(), lr=self.runtime_cfg.learning_rate)
        self._init_metrics()

        self.progress = self._load_or_init_progress()
        self.best_val_score = float(self.progress.get("best_val_score", float("inf")) if self.progress.get("best_val_score", None) is not None else float("inf"))
        self.best_epoch = int(self.progress.get("best_epoch", 0))
        self.train_history = list(self.progress.get("history", []))
        self.epochs_without_improvement = int(self.progress.get("epochs_without_improvement", 0))
        self.early_stopped = bool(self.progress.get("early_stopped", False))
        self.stop_reason = self.progress.get("stop_reason")

        self.start_epoch = self._maybe_load_checkpoint()
        if self.progress["completed_epochs"] < self.start_epoch:
            self.progress["completed_epochs"] = self.start_epoch
            self._save_progress()

        self._save_run_config()
        print(f"[Client {self.runtime_cfg.client_id}] device_info={self.device_info}", flush=True)

    def _save_run_config(self) -> None:
        cfg, split_layer = build_lstm_config_from_env()
        write_json(
            os.path.join(_cluster_output_dir(self.runtime_cfg.cluster_id), f"client_{self.runtime_cfg.client_id}_config.json"),
            {
                "component": "client",
                "client_id": self.runtime_cfg.client_id,
                "cluster_id": self.runtime_cfg.cluster_id,
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
                    "eval_mode": self.runtime_cfg.eval_mode,
                },
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
        }

    def _save_progress(self) -> None:
        self.progress["history"] = self.train_history
        self.progress["best_val_score"] = None if self.best_val_score == float("inf") else self.best_val_score
        self.progress["best_epoch"] = self.best_epoch
        self.progress["epochs_without_improvement"] = self.epochs_without_improvement
        self.progress["early_stopped"] = self.early_stopped
        self.progress["stop_reason"] = self.stop_reason
        write_json(client_progress_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id), self.progress)

    def _write_step_metrics(self, batch_processed: bool, epoch_completed: bool, done_all_epochs: bool) -> None:
        write_json(
            client_step_metrics_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
            {
                "client_id": self.runtime_cfg.client_id,
                "cluster_id": self.runtime_cfg.cluster_id,
                "batch_processed": batch_processed,
                "epoch_completed": epoch_completed,
                "done_all_epochs": done_all_epochs,
                "completed_epochs": self.progress["completed_epochs"],
                "early_stopped": self.early_stopped,
                "stop_reason": self.stop_reason,
            },
        )

    def _build_dataloaders(self):
        cfg = self.runtime_cfg
        if not cfg.data_stream_api_url:
            raise ValueError("DATA_STREAM_API_URL must be set")

        def ds(mode: str):
            return StreamingIterableDataset(
                api_url=cfg.data_stream_api_url,
                client_id=str(cfg.client_id),
                cluster_id=str(cfg.cluster_id),
                mode=mode,
                batch_size=cfg.batch_size,
            )

        return ds("train"), ds("val"), ds("test")

    def _build_server_url(self) -> str:
        port = _get_env("SERVER_PORT", "5002")
        return f"http://localhost:{port}"

    def _build_model(self) -> nn.Module:
        cfg, split_layer = build_lstm_config_from_env()
        client_model, _ = build_split_lstm(cfg, split_layer)
        print(f"[Client {self.runtime_cfg.client_id}] hidden={cfg.hidden_size}, layers={cfg.num_layers}, split={split_layer}")
        return client_model

    def _maybe_load_checkpoint(self) -> int:
        local_path = client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)
        epoch, ok = try_load_model_checkpoint(self.client_model, self.optimizer, local_path)
        if ok:
            print(f"[Client {self.runtime_cfg.client_id}] Loaded checkpoint <- {local_path} (epoch={epoch})")
            return int(epoch or 0)
        print(f"[Client {self.runtime_cfg.client_id}] No compatible checkpoint - random init.")
        return 0

    def _init_metrics(self) -> None:
        try:
            start_http_server(self.runtime_cfg.metrics_port, addr="0.0.0.0")
            labels = ["client_id"]
            self.m_train_loss = Histogram("split_client_train_loss", "Train loss", labels)
            self.m_val_loss = Gauge("split_client_val_loss", "Val loss", labels)
            self.m_test_mae = Gauge("split_client_test_mae", "Test MAE", labels)
        except Exception:
            self.m_train_loss = None
            self.m_val_loss = None
            self.m_test_mae = None

    def _reset_stream(self, mode: str) -> None:
        resp = requests.post(
            f"{self.runtime_cfg.data_stream_api_url}/stream/reset",
            json={"client_id": self.runtime_cfg.client_id, "cluster_id": self.runtime_cfg.cluster_id, "mode": mode},
            timeout=10.0,
        )
        resp.raise_for_status()
        body = resp.json()
        print(f"[Client {self.runtime_cfg.client_id}] Stream reset ({mode}), {body.get('windows_available', '?')} windows")

    def _fetch_norm_stats(self) -> Optional[Dict[str, float]]:
        try:
            resp = requests.get(
                f"{self.runtime_cfg.data_stream_api_url}/stream/norm_stats",
                params={"client_id": self.runtime_cfg.client_id, "cluster_id": self.runtime_cfg.cluster_id},
                timeout=10.0,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[Client {self.runtime_cfg.client_id}] norm stats unavailable: {e}")
            return None

    def _prepare_batch_for_lstm(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        if X.dim() == 1:
            X = X.unsqueeze(1).unsqueeze(2)
        elif X.dim() == 2:
            X = X.unsqueeze(2)
        X = X.float().to(self.device)
        if y is None:
            return X
        y = y.float().to(self.device)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        return X, y

    @staticmethod
    def _pack_payload(client_id: str, activations: torch.Tensor, targets: Optional[torch.Tensor] = None):
        payload = {"client_id": client_id, "activations": activations.detach().cpu()}
        if targets is not None:
            payload["targets"] = targets.detach().cpu()
        return pickle.dumps(payload)

    @staticmethod
    def _post_pickle(url: str, payload: bytes, timeout: float = 30.0):
        resp = requests.post(url, data=payload, headers={"Content-Type": "application/octet-stream"}, timeout=timeout)
        resp.raise_for_status()
        return pickle.loads(resp.content)

    def _inverse_tensor(self, arr: np.ndarray) -> np.ndarray:
        if not self.norm_stats or not bool(self.norm_stats.get("normalize", False)):
            return arr
        mean = float(self.norm_stats["mean"])
        scale = float(self.norm_stats["scale"])
        method = str(self.norm_stats.get("method", "zscore")).lower()
        if method in ("zscore", "minmax"):
            return arr * scale + mean
        return arr

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, Optional[float]]:
        preds = np.asarray(preds, dtype=np.float32).reshape(-1)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)

        valid = np.isfinite(preds) & np.isfinite(targets)
        if not np.any(valid):
            return {"mse": None, "mae": None, "rmse": None, "mape": None}

        preds = preds[valid]
        targets = targets[valid]

        err = preds - targets
        mse = float(np.mean(np.square(err)))
        mae = float(np.mean(np.abs(err)))
        rmse = float(math.sqrt(mse))

        denom = np.where(np.abs(targets) < 1e-8, np.nan, np.abs(targets))
        mape = float(np.nanmean(np.abs(err) / denom) * 100.0) if np.any(~np.isnan(denom)) else None

        return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}

    def _run_eval(self, mode: str, epoch: Optional[int] = None, save_json_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if mode not in {"val", "test"}:
            raise ValueError(f"Unsupported eval mode: {mode}")

        loader = self.val_loader if mode == "val" else self.test_loader
        route = "/validate_step" if mode == "val" else "/inference_step"

        self.client_model.eval()
        self._reset_stream(mode)

        total_server_loss = 0.0
        nb = 0
        ns = 0
        preds_norm = []
        trues_norm = []

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(loader):
                X, y = self._prepare_batch_for_lstm(X, y)

                if torch.isnan(X).any():
                    print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] NaN in input batch {batch_idx}")
                if torch.isnan(y).any():
                    print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] NaN in target batch {batch_idx}")
                if torch.isinf(X).any():
                    print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] Inf in input batch {batch_idx}")
                if torch.isinf(y).any():
                    print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] Inf in target batch {batch_idx}")

                activations = self.client_model(X)
                if activations.dim() == 3:
                    activations = activations[:, -1, :]

                if torch.isnan(activations).any():
                    print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] NaN in activations batch {batch_idx}")
                if torch.isinf(activations).any():
                    print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] Inf in activations batch {batch_idx}")

                if mode == "val":
                    result = self._post_pickle(
                        f"{self.current_server_url}{route}",
                        self._pack_payload(self.runtime_cfg.client_id, activations, y),
                    )
                    batch_preds = np.asarray(result["predictions"], dtype=np.float32)
                    total_server_loss += float(result["loss"])
                else:
                    result = self._post_pickle(
                        f"{self.current_server_url}{route}",
                        self._pack_payload(self.runtime_cfg.client_id, activations, None),
                    )
                    batch_preds = np.asarray(result["predictions"], dtype=np.float32)

                if np.isnan(batch_preds).any():
                    print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] NaN in predictions batch {batch_idx}")
                if np.isinf(batch_preds).any():
                    print(f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] Inf in predictions batch {batch_idx}")

                batch_true = y.detach().cpu().numpy().astype(np.float32)

                preds_norm.append(batch_preds.reshape(-1, 1))
                trues_norm.append(batch_true.reshape(-1, 1))
                nb += 1
                ns += int(batch_true.shape[0])

        if nb == 0:
            return None

        preds_norm_np = np.concatenate(preds_norm, axis=0)
        trues_norm_np = np.concatenate(trues_norm, axis=0)

        print(
            f"[Client {self.runtime_cfg.client_id}] [{mode.upper()}] "
            f"preds_norm nan={np.isnan(preds_norm_np).sum()}, trues_norm nan={np.isnan(trues_norm_np).sum()}"
        )

        preds_raw = self._inverse_tensor(preds_norm_np)
        trues_raw = self._inverse_tensor(trues_norm_np)

        metrics_norm = self._compute_metrics(preds_norm_np, trues_norm_np)
        metrics_raw = self._compute_metrics(preds_raw, trues_raw)

        result = {
            "mode": mode,
            "epoch": epoch,
            "client_id": self.runtime_cfg.client_id,
            "cluster_id": self.runtime_cfg.cluster_id,
            "num_batches": nb,
            "num_samples": ns,
            "server_loss": (total_server_loss / nb) if mode == "val" else None,
            "normalized_metrics": metrics_norm,
            "raw_metrics": metrics_raw,
            "normalization": self.norm_stats,
            "eval_mode": self.runtime_cfg.eval_mode,
        }

        tag = mode.upper()
        prefix = f"[Client {self.runtime_cfg.client_id}] [{tag}]"
        if mode == "val":
            print(
                f"{prefix} epoch={epoch} server_loss={result['server_loss']:.6f} "
                f"raw_mae={metrics_raw['mae']} raw_rmse={metrics_raw['rmse']} "
                f"norm_mae={metrics_norm['mae']}"
            )
            if self.m_val_loss is not None and result["server_loss"] is not None:
                self.m_val_loss.labels(client_id=self.runtime_cfg.client_id).set(result["server_loss"])
        else:
            print(
                f"{prefix} mae={metrics_raw['mae']} rmse={metrics_raw['rmse']} "
                f"mse={metrics_raw['mse']} mape={metrics_raw['mape']}"
            )
            if self.m_test_mae is not None and metrics_raw["mae"] is not None:
                self.m_test_mae.labels(client_id=self.runtime_cfg.client_id).set(metrics_raw["mae"])

        if save_json_path:
            write_json(save_json_path, result)

        self.client_model.train()
        return result

    def _save_best_pair(self, epoch: int) -> None:
        save_torch_checkpoint(
            self.client_model,
            self.optimizer,
            epoch,
            client_best_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
        )
        try:
            resp = requests.post(f"{self.current_server_url}/save_best_checkpoint", timeout=20.0)
            resp.raise_for_status()
            print(f"[Client {self.runtime_cfg.client_id}] Paired best server checkpoint saved.")
        except Exception as e:
            print(f"[Client {self.runtime_cfg.client_id}] WARNING: could not save paired best server checkpoint: {e}")

    def _save_latest(self, epoch: int) -> None:
        save_torch_checkpoint(
            self.client_model,
            self.optimizer,
            epoch,
            client_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
        )

    def _save_final(self, epoch: int) -> None:
        save_torch_checkpoint(
            self.client_model,
            self.optimizer,
            epoch,
            client_final_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
        )

    def _finalize_epoch(self, epoch_number: int, avg_train_loss: Optional[float]) -> None:
        cfg = self.runtime_cfg
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
        if val_raw_mae is not None:
            if self.best_val_score == float("inf") or val_raw_mae < (self.best_val_score - cfg.early_stopping_min_delta):
                improved = True
                self.best_val_score = float(val_raw_mae)
                self.best_epoch = epoch_number
                self._save_best_pair(epoch_number)
                print(
                    f"[Client {cfg.client_id}] New best checkpoint at epoch {epoch_number} "
                    f"with {cfg.early_stopping_metric}={val_raw_mae:.6f}"
                )

        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if cfg.early_stopping and self.epochs_without_improvement >= cfg.early_stopping_patience:
            self.early_stopped = True
            self.stop_reason = (
                f"no improvement in {cfg.early_stopping_metric} for "
                f"{cfg.early_stopping_patience} epoch(s)"
            )
            print(
                f"[Client {cfg.client_id}] Early stopping triggered at epoch {epoch_number}. "
                f"Best epoch={self.best_epoch}, best_val_raw_mae={self.best_val_score}"
            )

        self.progress["completed_epochs"] = epoch_number
        self.progress["current_epoch_loss_sum"] = 0.0
        self.progress["current_epoch_batch_count"] = 0
        self.progress["train_stream_active"] = False
        self._save_latest(epoch_number)

        if self.early_stopped or epoch_number >= cfg.num_epochs:
            self._save_final(epoch_number)

        summary = {
            "client_id": cfg.client_id,
            "cluster_id": cfg.cluster_id,
            "num_epochs": cfg.num_epochs,
            "completed_epochs": self.progress["completed_epochs"],
            "best_epoch": self.best_epoch,
            "best_val_raw_mae": None if self.best_val_score == float("inf") else self.best_val_score,
            "history": self.train_history,
            "final_train_loss": self.train_history[-1]["train_loss"] if self.train_history else None,
            "final_val_loss": self.train_history[-1]["val_loss"] if self.train_history else None,
            "early_stopping_enabled": cfg.early_stopping,
            "early_stopping_patience": cfg.early_stopping_patience,
            "early_stopping_min_delta": cfg.early_stopping_min_delta,
            "early_stopping_metric": cfg.early_stopping_metric,
            "early_stopped": self.early_stopped,
            "stop_reason": self.stop_reason,
            "epochs_without_improvement": self.epochs_without_improvement,
            "client_best_checkpoint_path": client_best_checkpoint_path(cfg.cluster_id, cfg.client_id),
            "client_final_checkpoint_path": client_final_checkpoint_path(cfg.cluster_id, cfg.client_id),
            "server_best_checkpoint_path": os.path.join(_cluster_output_dir(cfg.cluster_id), "server_best.pt"),
            "server_final_checkpoint_path": os.path.join(_cluster_output_dir(cfg.cluster_id), "server_final.pt"),
            "selection_metric": "raw MAE",
            "train_loss_name": "MSELoss",
        }
        write_json(client_train_metrics_path(cfg.cluster_id, cfg.client_id), summary)
        self._save_progress()

    def _train_single_batch(self, X: torch.Tensor, y: torch.Tensor) -> float:
        cfg = self.runtime_cfg
        X, y = self._prepare_batch_for_lstm(X, y)

        self.optimizer.zero_grad(set_to_none=True)
        activations = self.client_model(X)
        if activations.dim() == 3:
            activations = activations[:, -1, :]

        result = self._post_pickle(
            f"{self.current_server_url}/train_step",
            self._pack_payload(cfg.client_id, activations, y),
        )
        grad = result["grad_activations"].to(self.device)
        if torch.isnan(grad).any():
            raise ValueError(f"[Client {cfg.client_id}] NaN gradient from server")

        loss = float(result["loss"])
        activations.backward(gradient=grad)
        self.optimizer.step()

        for name, p in self.client_model.named_parameters():
            if torch.isnan(p).any():
                raise ValueError(f"[Client {cfg.client_id}] NaN parameter after optimizer step: {name}")

        if self.m_train_loss is not None:
            self.m_train_loss.labels(client_id=cfg.client_id).observe(loss)

        return loss

    def train_epochs(self) -> None:
        while not self.is_done():
            self.train_one_epoch()
        print(f"[Client {self.runtime_cfg.client_id}] Training complete.")

    def train_one_epoch(self) -> None:
        cfg = self.runtime_cfg
        if self.is_done():
            self._write_step_metrics(batch_processed=False, epoch_completed=False, done_all_epochs=True)
            return

        epoch_number = self.progress["completed_epochs"] + 1
        print(f"[Client {cfg.client_id}] train_one_epoch -> epoch {epoch_number}/{cfg.num_epochs}")

        self.client_model.train()
        self._reset_stream("train")

        epoch_loss = 0.0
        batch_count = 0
        for X, y in self.train_loader:
            loss = self._train_single_batch(X, y)
            epoch_loss += loss
            batch_count += 1

        avg_train_loss = (epoch_loss / batch_count) if batch_count else None
        self._finalize_epoch(epoch_number, avg_train_loss)
        self._write_step_metrics(batch_processed=batch_count > 0, epoch_completed=True, done_all_epochs=self.is_done())

    def train_one_batch(self) -> None:
        cfg = self.runtime_cfg
        if self.is_done():
            self._write_step_metrics(batch_processed=False, epoch_completed=False, done_all_epochs=True)
            return

        if not self.progress.get("train_stream_active", False):
            self._reset_stream("train")
            self.progress["train_stream_active"] = True
            self._save_progress()

        train_iter = iter(self.train_loader)
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch_number = self.progress["completed_epochs"] + 1
            batch_count = int(self.progress.get("current_epoch_batch_count", 0))
            epoch_loss = float(self.progress.get("current_epoch_loss_sum", 0.0))
            avg_train_loss = (epoch_loss / batch_count) if batch_count > 0 else None
            self._finalize_epoch(epoch_number, avg_train_loss)
            self._write_step_metrics(batch_processed=False, epoch_completed=True, done_all_epochs=self.is_done())
            return

        X, y = batch
        loss = self._train_single_batch(X, y)

        self.progress["current_epoch_loss_sum"] = float(self.progress.get("current_epoch_loss_sum", 0.0)) + loss
        self.progress["current_epoch_batch_count"] = int(self.progress.get("current_epoch_batch_count", 0)) + 1

        current_epoch = self.progress["completed_epochs"] + 1
        self._save_latest(current_epoch)
        self._save_progress()
        self._write_step_metrics(batch_processed=True, epoch_completed=False, done_all_epochs=False)

    def is_done(self) -> bool:
        return self.early_stopped or self.progress["completed_epochs"] >= self.runtime_cfg.num_epochs

    def _load_best_checkpoints_for_test(self) -> None:
        client_best = client_best_checkpoint_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id)
        epoch, ok = try_load_model_checkpoint(self.client_model, None, client_best)
        if not ok:
            raise FileNotFoundError(f"Compatible best client checkpoint not found: {client_best}")
        print(f"[Client {self.runtime_cfg.client_id}] Loaded best client checkpoint <- {client_best}")

        resp = requests.post(f"{self.current_server_url}/load_best_checkpoint", timeout=20.0)
        resp.raise_for_status()
        print(f"[Client {self.runtime_cfg.client_id}] Loaded paired best server checkpoint.")

    def _load_final_checkpoints_for_test(self) -> None:
        cfg = self.runtime_cfg
        client_final = client_final_checkpoint_path(cfg.cluster_id, cfg.client_id)
        client_latest = client_checkpoint_path(cfg.cluster_id, cfg.client_id)

        epoch, ok = try_load_model_checkpoint(self.client_model, None, client_final)
        if ok:
            print(f"[Client {cfg.client_id}] Loaded final client checkpoint <- {client_final}")
        else:
            epoch, ok = try_load_model_checkpoint(self.client_model, None, client_latest)
            if not ok:
                raise FileNotFoundError(
                    f"Compatible final or latest client checkpoint not found: "
                    f"{client_final} / {client_latest}"
                )
            print(
                f"[Client {cfg.client_id}] WARNING: final checkpoint missing; "
                f"using latest checkpoint instead <- {client_latest}"
            )

        resp = requests.post(f"{self.current_server_url}/load_final_checkpoint", timeout=20.0)
        resp.raise_for_status()
        print(f"[Client {cfg.client_id}] Loaded shared final server checkpoint.")
    def test(self) -> Dict[str, Any]:
        if self.runtime_cfg.eval_mode == "paired_best":
            self._load_best_checkpoints_for_test()
        else:
            self._load_final_checkpoints_for_test()

        result = self._run_eval(
            "test",
            save_json_path=client_test_metrics_path(self.runtime_cfg.cluster_id, self.runtime_cfg.client_id),
        )
        if result is None:
            raise RuntimeError("Test evaluation produced no batches.")
        return result


if __name__ == "__main__":
    runner = ClusteredClientRunner()
    command = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "train"

    if command == "train":
        runner.train_epochs()
    elif command == "train_one_epoch":
        runner.train_one_epoch()
    elif command == "train_one_batch":
        runner.train_one_batch()
    elif command == "test":
        runner.test()
    else:
        raise SystemExit("Unsupported command. Use 'train', 'train_one_epoch', 'train_one_batch', or 'test'.")