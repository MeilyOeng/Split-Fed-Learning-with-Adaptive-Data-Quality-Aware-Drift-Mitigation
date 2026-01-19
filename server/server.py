# server/server.py

import os
import sys
import pickle
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, Response
import torch
from torch import nn, optim
from threading import Lock
import requests
import numpy as np
import time
from prometheus_client import Counter, Gauge, Histogram

from models.split_lstm import LSTMConfig, build_split_lstm

# Stub implementations for missing shared modules
def server_checkpoint_path(cluster_id: str, filename: str) -> str:
    """Stub: Get checkpoint path for server."""
    os.makedirs("output", exist_ok=True)
    return os.path.join("output", f"server_{cluster_id}_{filename}")

def save_torch_checkpoint(model, optimizer, path: str, extra: Optional[Dict] = None):
    """Stub: Save PyTorch checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    print(f"[Server] Saved checkpoint to {path}")

def load_torch_checkpoint(model, optimizer, path: str, map_location: str = "cpu"):
    """Stub: Load PyTorch checkpoint."""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[Server] Loaded checkpoint from {path}")
        return {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict']}
    return None

@dataclass
class TrainingRecord:
    """Stub: Training record dataclass."""
    ts: float
    role: str
    mode: str
    cluster_id: str
    client_id: str
    epoch: Optional[int]
    step: int
    loss: float
    extra: Optional[Dict]

def log_training_record(record: TrainingRecord):
    """Stub: Log training record."""
    pass

def setup_flask_metrics(app: Flask):
    """Stub: Setup Flask metrics."""
    from prometheus_client import CollectorRegistry
    return CollectorRegistry()

# Default no-op metrics for non-Flask contexts (e.g., unit tests).
class _NoMetric:
    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        return None

    def observe(self, *args, **kwargs):
        return None

    def set(self, *args, **kwargs):
        return None


SERVER_TRAIN_LOSS = _NoMetric()
SERVER_PRED_MEAN = _NoMetric()
SERVER_GLOBAL_STEPS = _NoMetric()
SERVER_CLUSTERING_CHECKS = _NoMetric()
SERVER_ERROR_EMA = _NoMetric()


# =========================
#   Config & helpers
# =========================

@dataclass
class ServerRuntimeConfig:
    """
    Runtime configuration for the clustered split-learning server.
    """

    cluster_id: str
    learning_rate: float

    clustering_api_url: Optional[str]
    clustering_check_interval_steps: int

    server_service_name: Optional[str]
    server_namespace: Optional[str]

    grad_ema_alpha: float
    error_ema_alpha: float


def parse_bool(env_value: str, default: bool = False) -> bool:
    if env_value is None:
        return default
    return env_value.lower() in ("1", "true", "yes", "y")


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    return val if val is not None else default


def build_lstm_config_from_env() -> (LSTMConfig, int):
    cfg = LSTMConfig(
        input_size=int(_get_env("MODEL_INPUT_SIZE", "11")),
        hidden_size=int(_get_env("MODEL_HIDDEN_SIZE", "64")),
        num_layers=int(_get_env("MODEL_NUM_LAYERS", "4")),
        output_size=int(_get_env("MODEL_OUTPUT_SIZE", "1")),
        dropout=float(_get_env("MODEL_DROPOUT", "0.0")),
        bidirectional=parse_bool(_get_env("MODEL_BIDIRECTIONAL", "false")),
    )
    split_layer = int(_get_env("MODEL_SPLIT_LAYER", "2"))
    return cfg, split_layer


def build_runtime_config_from_env() -> ServerRuntimeConfig:
    return ServerRuntimeConfig(
        cluster_id=_get_env("CLUSTER_ID", "default-cluster"),
        learning_rate=float(_get_env("LEARNING_RATE", "1e-3")),
        clustering_api_url=_get_env("CLUSTERING_API_URL"),
        clustering_check_interval_steps=int(_get_env("CLUSTERING_CHECK_INTERVAL_STEPS", "500")),
        server_service_name=_get_env("SERVER_SERVICE_NAME"),
        server_namespace=_get_env("SERVER_NAMESPACE", _get_env("SERVER_NAMESPACE_DEFAULT", "split-ns")),
        grad_ema_alpha=float(_get_env("GRAD_EMA_ALPHA", "0.1")),
        error_ema_alpha=float(_get_env("ERROR_EMA_ALPHA", "0.1")),
    )


# =========================
#   Server state container
# =========================

class ClusteredServerState:
    """
    Holds model, optimizer, config, and bookkeeping for one server pod (one cluster).

    This server:
      - trains the server-side LSTM layers
      - tracks per-client gradients (features) and errors
      - periodically calls the clustering service API
      - DOES NOT own cluster mapping or server lifecycle
    """

    def __init__(self):
        self.runtime_cfg = build_runtime_config_from_env()
        self.lstm_cfg, self.split_layer = build_lstm_config_from_env()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build server-side part of the split LSTM
        _client_model, server_model = build_split_lstm(self.lstm_cfg, self.split_layer)
        self.model = server_model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.runtime_cfg.learning_rate)
        self.criterion = nn.MSELoss()

        self.global_step: int = 0
        self._last_cluster_check_step: int = -1

        # Lock for thread safety
        self.lock = Lock()

        # Per-client aggregated statistics
        self.grad_features_per_client: Dict[str, np.ndarray] = {}
        self.error_ema_per_client: Dict[str, float] = {}

        # Try to load checkpoint
        ckpt_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
        extra = load_torch_checkpoint(self.model, self.optimizer, ckpt_path, map_location=str(self.device))
        if extra is not None:
            print(f"[Server {self.runtime_cfg.cluster_id}] Loaded checkpoint from {ckpt_path}")
            # Optionally restore global_step from extra
            if "global_step" in extra:
                self.global_step = int(extra["global_step"])

    # -------------
    #  Core methods
    # -------------

    def train_step(
        self,
        client_id: str,
        activations_cpu: torch.Tensor,
        targets_cpu: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Perform a single training step on the server-side layers.
        """
        self.global_step += 1

        activations = activations_cpu.to(self.device)
        targets = targets_cpu.to(self.device)

        # Ensure activations are 3D for LSTM: [batch, seq_len, hidden]
        if activations.dim() == 2:
            activations = activations.unsqueeze(1)
        activations.requires_grad_(True)

        preds = self.model(activations)  # [batch, 1]
        loss = self.criterion(preds, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        grad_activations = activations.grad.detach().cpu()
        if grad_activations.dim() == 3:
            grad_activations = grad_activations.squeeze(1)
        loss_value = loss.item()

        # Aggregation for clustering
        self._update_client_stats(client_id, grad_activations, loss_value)

        # Log training record
        log_training_record(
            TrainingRecord(
                ts=time.time(),
                role="server",
                mode="train",
                cluster_id=self.runtime_cfg.cluster_id,
                client_id=client_id,
                epoch=None,
                step=self.global_step,
                loss=loss_value,
                extra=None,
            )
        )

        # Metrics
        labels = (self.runtime_cfg.cluster_id, client_id)
        SERVER_GLOBAL_STEPS.labels(self.runtime_cfg.cluster_id).inc()
        SERVER_TRAIN_LOSS.labels(*labels).observe(loss_value)
        SERVER_ERROR_EMA.labels(*labels).set(self.error_ema_per_client.get(client_id, loss_value))
        SERVER_PRED_MEAN.labels(*labels).set(float(preds.detach().cpu().mean().item()))

        # Periodically call clustering service
        self._maybe_check_clustering_validity()

        # Periodic checkpoint
        if self.global_step % 500 == 0:
            save_torch_checkpoint(
                self.model,
                self.optimizer,
                server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt"),
                extra={"global_step": self.global_step},
            )

        return {
            "grad_activations": grad_activations,
            "loss": loss_value,
        }

    def inference_step(
        self,
        client_id: str,
        activations_cpu: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Perform inference on server-side layers.
        """
        with torch.no_grad():
            activations = activations_cpu.to(self.device)
            if activations.dim() == 2:
                activations = activations.unsqueeze(1)
            preds = self.model(activations)  # [batch, 1]
            preds_cpu = preds.detach().cpu()

            metrics = {
                "mean_prediction": float(preds_cpu.mean().item()),
            }

            SERVER_PRED_MEAN.labels(self.runtime_cfg.cluster_id, client_id).set(metrics["mean_prediction"])

        return {
            "predictions": preds_cpu,
            "metrics": metrics,
        }

    # --------------------------
    #  Aggregation for clustering
    # --------------------------

    def _update_client_stats(
        self,
        client_id: str,
        grad_activations_cpu: torch.Tensor,
        loss_value: float,
    ) -> None:
        """
        Update per-client gradient feature vector and error EMA.

        grad_activations_cpu: [batch, seq_len, hidden_dim]
        We reduce it to a 1D vector (hidden_dim) by mean pooling.
        """
        cfg = self.runtime_cfg

        # Mean pool over batch and time axes
        grad_feature = grad_activations_cpu.mean(dim=(0, 1)).numpy().astype(float).reshape(-1)

        # Exponential moving average for gradient features
        if client_id not in self.grad_features_per_client:
            self.grad_features_per_client[client_id] = grad_feature
        else:
            alpha = cfg.grad_ema_alpha
            prev = self.grad_features_per_client[client_id]
            self.grad_features_per_client[client_id] = alpha * grad_feature + (1.0 - alpha) * prev

        # Exponential moving average for errors (we use loss as "forecasting error")
        if client_id not in self.error_ema_per_client:
            self.error_ema_per_client[client_id] = float(loss_value)
        else:
            beta = cfg.error_ema_alpha
            prev_err = self.error_ema_per_client[client_id]
            self.error_ema_per_client[client_id] = beta * float(loss_value) + (1.0 - beta) * prev_err

    # --------------------------
    #  Clustering validity hook
    # --------------------------

    def _maybe_check_clustering_validity(self) -> None:
        """
        Periodically send aggregated per-client gradients + errors
        to the clustering service. The clustering service + cluster manager
        handle cluster mapping and server lifecycle.
        """
        cfg = self.runtime_cfg
        interval = cfg.clustering_check_interval_steps

        if not cfg.clustering_api_url or interval <= 0:
            return

        if self._last_cluster_check_step >= 0 and (self.global_step - self._last_cluster_check_step) < interval:
            return

        self._last_cluster_check_step = self.global_step

        if not self.grad_features_per_client:
            print("[Server] No gradient features collected yet; skipping clustering call.")
            return

        timestamp = datetime.utcnow().isoformat() + "Z"

        client_gradients = {
            cid: vec.tolist()
            for cid, vec in self.grad_features_per_client.items()
        }

        forecasting_errors = {
            cid: float(err)
            for cid, err in self.error_ema_per_client.items()
        }

        payload = {
            "timestamp": timestamp,
            "drift_detected": False,        # TODO: wire real drift signal if desired
            "cluster_by": "gradients",
            "client_gradients": client_gradients,
            "forecasting_errors": forecasting_errors,
        }

        try:
            resp = requests.post(
                cfg.clustering_api_url.rstrip("/") + "/cluster",
                json=payload,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            SERVER_CLUSTERING_CHECKS.labels(cfg.cluster_id, "success").inc()
        except Exception as e:
            print(f"[Server cluster={cfg.cluster_id}] Clustering API call failed: {e}")
            SERVER_CLUSTERING_CHECKS.labels(cfg.cluster_id, "error").inc()
            return

        new_clustering = bool(data.get("new_clustering", False))
        print(
            f"[Server cluster={cfg.cluster_id}] Clustering check done. "
            f"new_clustering={new_clustering}, "
            f"n_assignments={len(data.get('cluster_assignments') or {})}"
        )

    # =========================
#   Flask routes
# =========================


def create_app() -> Flask:
    app = Flask(__name__)
    registry = setup_flask_metrics(app)
    state = ClusteredServerState()

    # Prometheus metrics
    cluster_id = state.runtime_cfg.cluster_id
    global SERVER_TRAIN_LOSS, SERVER_PRED_MEAN, SERVER_GLOBAL_STEPS, SERVER_CLUSTERING_CHECKS, SERVER_ERROR_EMA
    SERVER_TRAIN_LOSS = Histogram(
        "split_server_train_loss",
        "Server-side training loss per client",
        ["cluster_id", "client_id"],
        registry=registry,
    )
    SERVER_PRED_MEAN = Gauge(
        "split_server_prediction_mean",
        "Mean server-side prediction per client",
        ["cluster_id", "client_id"],
        registry=registry,
    )
    SERVER_GLOBAL_STEPS = Counter(
        "split_server_global_steps_total",
        "Global training steps processed by server",
        ["cluster_id"],
        registry=registry,
    )
    SERVER_CLUSTERING_CHECKS = Counter(
        "split_server_clustering_checks_total",
        "Calls from server to clustering service",
        ["cluster_id", "result"],
        registry=registry,
    )
    SERVER_ERROR_EMA = Gauge(
        "split_server_error_ema",
        "EMA of forecasting error per client as tracked by server",
        ["cluster_id", "client_id"],
        registry=registry,
    )

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return {"status": "ok", "cluster_id": state.runtime_cfg.cluster_id}

    @app.route("/train_step", methods=["POST"])
    def train_step_route():
        with state.lock:
            payload = pickle.loads(request.data)
            client_id = payload.get("client_id", "unknown-client")

            activations_cpu = payload["activations"]
            targets_cpu = payload["targets"]

            result = state.train_step(client_id, activations_cpu, targets_cpu)

            resp_payload = {
                "grad_activations": result["grad_activations"],
                "loss": result["loss"],
            }
            resp_bytes = pickle.dumps(resp_payload)
            return Response(resp_bytes, content_type="application/octet-stream")

    @app.route("/inference_step", methods=["POST"])
    def inference_step_route():
        with state.lock:
            payload = pickle.loads(request.data)
            client_id = payload.get("client_id", "unknown-client")

            activations_cpu = payload["activations"]

            result = state.inference_step(client_id, activations_cpu)

            resp_payload = {
                "predictions": result["predictions"],
                "metrics": result["metrics"],
            }
            resp_bytes = pickle.dumps(resp_payload)
            return Response(resp_bytes, content_type="application/octet-stream")

    return app


if __name__ == "__main__":
    port = int(_get_env("SERVER_PORT", "5002"))
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=port)
