# client/clustered_client.py

import os
import sys
import pickle
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any


# Add project root to Python path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import requests
import torch
from torch import nn, optim
from prometheus_client import Gauge, Histogram, start_http_server
from server.server import build_lstm_config_from_env, parse_bool
from models.split_lstm import LSTMConfig, build_split_lstm
from data.data_streaming import StreamingIterableDataset


# -------------------------
# Stub implementations for missing shared modules
# -------------------------
def save_run_config(config: dict, name: str):
    """Stub: Save run configuration to file."""
    import json
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", name), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Client] Saved config to output/{name}")


def client_checkpoint_path(client_id: str) -> str:
    """Stub: Get checkpoint path for client."""
    return os.path.join("output", f"client_{client_id}_checkpoint.pt")


def save_torch_checkpoint(model, optimizer, epoch, path: str):
    """Stub: Save PyTorch checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    print(f"[Client] Saved checkpoint to {path}")


def load_torch_checkpoint(path: str):
    """Stub: Load PyTorch checkpoint."""
    if os.path.exists(path):
        return torch.load(path)
    return None


@dataclass
class TrainingRecord:
    """Training record for client-side events."""
    ts: float
    role: str
    mode: str
    cluster_id: str
    client_id: str
    epoch: Optional[int]
    step: int
    loss: float
    extra: Optional[Dict[str, Any]] = None


def log_training_record(record: TrainingRecord) -> None:
    """Append a training record to a JSONL log file."""
    import json
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", "client_training_records.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


# -------------------------
# Runtime config
# -------------------------
@dataclass
class ClientRuntimeConfig:
    """
    Runtime configuration for the clustered split-learning client.
    Includes dynamic cluster and client setup.
    """

    client_id: str
    cluster_id: str
    data_root: str
    num_clusters: int
    num_clients_per_cluster: int

    batch_size: int
    num_epochs: int
    learning_rate: float

    data_stream_api_url: Optional[str]
    metrics_port: int


def build_runtime_config_from_env() -> ClientRuntimeConfig:
    client_id = os.environ.get("CLIENT_ID", "9900002")  # Default to a valid client ID from dataset
    num_clients_per_cluster = int(os.environ.get("NUM_CLIENTS_PER_CLUSTER", "3"))

    # Calculate cluster_id - handle both "client_1" and integer ID formats
    if "_" in client_id:
        # Format: "client_1" -> extract number
        client_num = int(client_id.split("_")[-1])
        cluster_id = f"{(client_num - 1) // num_clients_per_cluster}"
    else:
        # For now, default to cluster 0, but ideally should read from cluster-assignment.csv
        cluster_id = os.environ.get("CLUSTER_ID", "0")

    # Ensure cluster_id is just the number (not "cluster_0")
    if cluster_id.startswith("cluster_"):
        cluster_id = cluster_id.replace("cluster_", "")

    return ClientRuntimeConfig(
        client_id=client_id,
        cluster_id=cluster_id,
        data_root=os.environ.get("DATA_ROOT", "/mnt/shared/data"),
        num_clusters=int(os.environ.get("NUM_CLUSTERS", "3")),
        num_clients_per_cluster=num_clients_per_cluster,
        batch_size=int(os.environ.get("BATCH_SIZE", 32)),
        num_epochs=int(os.environ.get("NUM_EPOCHS", 5)),
        learning_rate=float(os.environ.get("LEARNING_RATE", 1e-3)),
        data_stream_api_url=os.environ.get("DATA_STREAM_API_URL", "http://localhost:5001"),
        metrics_port=int(os.environ.get("CLIENT_METRICS_PORT", 9100)),
    )


# -------------------------
# Main client runner
# -------------------------
class ClusteredClientRunner:
    def __init__(self, device: Optional[torch.device] = None):
        self.runtime_cfg = build_runtime_config_from_env()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_server_url = self._build_server_url()

        config_snapshot = {
            "component": "client",
            "client_id": self.runtime_cfg.client_id,
            "cluster_id": self.runtime_cfg.cluster_id,
            "num_clusters": self.runtime_cfg.num_clusters,
            "num_clients_per_cluster": self.runtime_cfg.num_clients_per_cluster,
            "data": {"data_root": self.runtime_cfg.data_root},
            "training": {
                "batch_size": self.runtime_cfg.batch_size,
                "num_epochs": self.runtime_cfg.num_epochs,
                "learning_rate": self.runtime_cfg.learning_rate,
            },
        }
        save_run_config(config_snapshot, name=f"client_{self.runtime_cfg.client_id}_config.json")

        self.train_loader, self.val_loader, self.inference_loader = self._build_dataloaders()

        self.client_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.client_model.parameters(), lr=self.runtime_cfg.learning_rate)

        self._init_metrics()

        # Track best model
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    # ---------- Build / init helpers ----------
    def _build_dataloaders(
        self,
    ) -> Tuple[StreamingIterableDataset, StreamingIterableDataset, StreamingIterableDataset]:
        cluster_id = self.runtime_cfg.cluster_id
        print(f"[Client {self.runtime_cfg.client_id}] Loading preprocessed data for cluster: {cluster_id}")
        print(f"[Client {self.runtime_cfg.client_id}] Stream API URL: {self.runtime_cfg.data_stream_api_url}")

        if not self.runtime_cfg.data_stream_api_url:
            raise ValueError("DATA_STREAM_API_URL must be set to use streaming dataset")

        def ds(mode: str) -> StreamingIterableDataset:
            return StreamingIterableDataset(
                api_url=self.runtime_cfg.data_stream_api_url,
                client_id=str(self.runtime_cfg.client_id),
                cluster_id=str(cluster_id),
                mode=mode,
                batch_size=self.runtime_cfg.batch_size,
            )

        return ds("train"), ds("val"), ds("inference")

    def _build_server_url(self) -> str:
        # Use localhost for local testing, or Kubernetes service name for cluster deployment
        server_port = os.environ.get("SERVER_PORT", "5002")
        if os.environ.get("USE_LOCALHOST", "true").lower() == "true":
            return f"http://localhost:{server_port}"
        return f"http://server-cluster-{self.runtime_cfg.cluster_id}.svc.cluster.local"

    def parse_bool(env_value: str, default: bool = False) -> bool:
        if env_value is None:
            return default
        return env_value.lower() in ("1", "true", "yes", "y")


    def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
        val = os.environ.get(name)
        return val if val is not None else default


    def build_lstm_config_from_env() -> Tuple[LSTMConfig, int]:
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

    def _build_model(self) -> nn.Module:
        cfg, split_layer = build_lstm_config_from_env()
        client_model, _ = build_split_lstm(cfg, split_layer)
        return client_model


    def _init_metrics(self) -> None:
        try:
            port = self.runtime_cfg.metrics_port
            start_http_server(port, addr="0.0.0.0")
            labels = ["client_id"]
            self.m_train_loss = Histogram("split_client_train_loss", "Client training loss", labels)
            self.m_val_loss = Gauge("split_client_val_loss", "Client validation loss", labels)
            print(f"[Client {self.runtime_cfg.client_id}] Metrics server started on port {port}")
        except Exception as e:
            print(f"[Client {self.runtime_cfg.client_id}] Warning: Could not start metrics server: {e}")
            self.m_train_loss = None
            self.m_val_loss = None

    def _reset_stream(self, mode: str = "train") -> None:
        """Reset the data stream offset (called at start of each epoch)."""
        try:
            response = requests.post(
                f"{self.runtime_cfg.data_stream_api_url}/stream/reset",
                json={
                    "client_id": self.runtime_cfg.client_id,
                    "cluster_id": self.runtime_cfg.cluster_id,
                    "mode": mode,
                },
                timeout=5.0,
            )
            response.raise_for_status()
            print(f"[Client {self.runtime_cfg.client_id}] Stream reset for mode: {mode}")
        except Exception as e:
            print(f"[Client {self.runtime_cfg.client_id}] Warning: Could not reset stream: {e}")

    # ---------- Small utilities (to reduce repetition) ----------
    def _prepare_batch_for_lstm(
        self, X_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape X for LSTM: expect [batch, seq_len, features]
        if X_batch.dim() == 1:
            X_batch = X_batch.unsqueeze(1).unsqueeze(2)  # [batch] -> [batch, 1, 1]
        elif X_batch.dim() == 2:
            X_batch = X_batch.unsqueeze(1)  # [batch, features] -> [batch, 1, features]

        X_batch = X_batch.float().to(self.device)
        y_batch = y_batch.float().to(self.device)

        # Reshape y to [batch, 1] if needed
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(1)

        return X_batch, y_batch

    @staticmethod
    def _pack_payload(client_id: str, activations: torch.Tensor, targets: torch.Tensor) -> bytes:
        return pickle.dumps(
            {
                "client_id": client_id,
                "activations": activations.detach().cpu(),
                "targets": targets.detach().cpu() if targets.requires_grad else targets.cpu(),
            }
        )

    @staticmethod
    def _post_pickle(url: str, payload: bytes, timeout: float = 30.0) -> dict:
        response = requests.post(
            url,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=timeout,
        )
        response.raise_for_status()
        return pickle.loads(response.content)

    # -------------------------
    # Training and evaluation
    # -------------------------
    def train_epochs(self, start_global_step: int = 0):
        """Train the client model using split learning."""
        print(f"[Client {self.runtime_cfg.client_id}] Starting training...")
        print(f"[Client {self.runtime_cfg.client_id}] Server URL: {self.current_server_url}")

        self.client_model.train()
        total_loss = 0.0
        num_batches = 0
        last_epoch = 0

        try:
            for epoch in range(self.runtime_cfg.num_epochs):
                last_epoch = epoch + 1
                epoch_loss = 0.0
                batch_count = 0

                # Reset stream at the start of each epoch
                self._reset_stream(mode="train")

                print(f"\n[Client {self.runtime_cfg.client_id}] Epoch {epoch+1}/{self.runtime_cfg.num_epochs}")

                for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                    X_batch, y_batch = self._prepare_batch_for_lstm(X_batch, y_batch)

                    # Forward pass through client model
                    activations = self.client_model(X_batch)  # possibly [batch, seq_len, hidden_dim]
                    if activations.dim() == 3:
                        activations = activations[:, -1, :]  # [batch, hidden_dim]

                    # Send activations to server for forward + backward pass
                    try:
                        payload = self._pack_payload(
                            client_id=self.runtime_cfg.client_id,
                            activations=activations,
                            targets=y_batch,
                        )
                        result = self._post_pickle(f"{self.current_server_url}/train_step", payload, timeout=30.0)

                        grad_activations = result["grad_activations"]
                        server_loss = result["loss"]

                        # Backpropagate through client model
                        self.optimizer.zero_grad()
                        activations.backward(gradient=grad_activations.to(self.device))
                        self.optimizer.step()

                        epoch_loss += server_loss
                        total_loss += server_loss
                        batch_count += 1
                        num_batches += 1

                        if (batch_idx + 1) % 10 == 0:
                            avg_loss = epoch_loss / batch_count
                            print(f"  Batch {batch_idx+1}: Loss = {avg_loss:.4f}")

                    except Exception as e:
                        print(f"[Client {self.runtime_cfg.client_id}] Error in training step: {e}")
                        continue

                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
                print(
                    f"[Client {self.runtime_cfg.client_id}] Epoch {epoch+1} completed. "
                    f"Avg Loss: {avg_epoch_loss:.4f}, Batches: {batch_count}"
                )

                # Run validation every epoch (or every N epochs for faster training)
                val_frequency = 1
                if (epoch + 1) % val_frequency == 0:
                    val_metrics = self.evaluate(epoch=epoch + 1)

                    # Save best model based on validation loss
                    if val_metrics and val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.best_epoch = epoch + 1
                        best_model_path = client_checkpoint_path(f"{self.runtime_cfg.client_id}_best")
                        save_torch_checkpoint(self.client_model, self.optimizer, epoch + 1, best_model_path)
                        print(
                            f"[Client {self.runtime_cfg.client_id}] ✓ New best model saved! "
                            f"Val Loss: {self.best_val_loss:.4f}"
                        )

            if num_batches > 0:
                print(f"\n[Client {self.runtime_cfg.client_id}] Training completed!")
                print(
                    f"[Client {self.runtime_cfg.client_id}] Total batches: {num_batches}, "
                    f"Avg Loss: {total_loss/num_batches:.4f}"
                )
                print(
                    f"[Client {self.runtime_cfg.client_id}] Best validation loss: "
                    f"{self.best_val_loss:.4f} at epoch {self.best_epoch}"
                )

                # Final validation
                print(f"\n[Client {self.runtime_cfg.client_id}] Running final validation...")
                _ = self.evaluate()

                # Save final checkpoint
                ckpt_path = client_checkpoint_path(self.runtime_cfg.client_id)
                save_torch_checkpoint(self.client_model, self.optimizer, last_epoch, ckpt_path)
            else:
                print(f"[Client {self.runtime_cfg.client_id}] No batches processed!")

        except Exception as e:
            print(f"[Client {self.runtime_cfg.client_id}] Error during training: {e}")
            import traceback
            traceback.print_exc()

    def evaluate(self, epoch: Optional[int] = None):
        """
        Evaluate the client model on validation set.

        Args:
            epoch: Current epoch number for logging

        Returns:
            dict: Evaluation metrics (loss, mae, rmse, mse)
        """
        print(f"[Client {self.runtime_cfg.client_id}] Running validation...")

        self.client_model.eval()

        # Reset validation stream
        self._reset_stream(mode="val")

        val_loss = 0.0
        squared_errors = []
        absolute_errors = []
        num_batches = 0
        num_samples = 0

        try:
            with torch.no_grad():
                for _, (X_batch, y_batch) in enumerate(self.val_loader):
                    X_batch, y_batch = self._prepare_batch_for_lstm(X_batch, y_batch)

                    # Forward pass through client model
                    activations = self.client_model(X_batch)
                    if activations.dim() == 3:
                        activations = activations[:, -1, :]

                    # Send to server for validation
                    try:
                        payload = self._pack_payload(
                            client_id=self.runtime_cfg.client_id,
                            activations=activations,
                            targets=y_batch,
                        )
                        result = self._post_pickle(f"{self.current_server_url}/validate_step", payload, timeout=30.0)

                        batch_loss = result["loss"]
                        predictions = result["predictions"]

                        val_loss += batch_loss
                        errors = (predictions - y_batch.cpu().numpy())
                        squared_errors.extend((errors ** 2).flatten().tolist())
                        absolute_errors.extend(np.abs(errors).flatten().tolist())

                        num_batches += 1
                        num_samples += len(y_batch)

                    except Exception as e:
                        print(f"[Client {self.runtime_cfg.client_id}] Error in validation step: {e}")
                        continue

            if num_batches > 0:
                metrics = {
                    "loss": val_loss / num_batches,
                    "mse": float(np.mean(squared_errors)),
                    "mae": float(np.mean(absolute_errors)),
                    "rmse": float(np.sqrt(np.mean(squared_errors))),
                    "num_samples": int(num_samples),
                    "num_batches": int(num_batches),
                }

                epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
                print(f"[Client {self.runtime_cfg.client_id}] Validation{epoch_str}:")
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MSE: {metrics['mse']:.4f}")
                print(f"  Samples: {metrics['num_samples']}")

                if self.m_val_loss is not None:
                    self.m_val_loss.labels(client_id=self.runtime_cfg.client_id).set(metrics["loss"])

                return metrics

            print(f"[Client {self.runtime_cfg.client_id}] No validation batches processed!")
            return None

        except Exception as e:
            print(f"[Client {self.runtime_cfg.client_id}] Error during validation: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            self.client_model.train()  # Switch back to training mode

    def inference_with_drift_monitoring(self):
        """Run inference with drift monitoring - test implementation."""
        print(f"[Client {self.runtime_cfg.client_id}] Starting inference with drift monitoring...")
        print(f"[Client {self.runtime_cfg.client_id}] Cluster: {self.runtime_cfg.cluster_id}")
        print(f"[Client {self.runtime_cfg.client_id}] Device: {self.device}")

        try:
            print(f"[Client {self.runtime_cfg.client_id}] Testing data stream...")
            for i, (X_batch, y_batch) in enumerate(self.inference_loader):
                print(
                    f"[Client {self.runtime_cfg.client_id}] Received batch {i+1}: "
                    f"X shape={X_batch.shape}, y shape={y_batch.shape}"
                )

                # Reuse same preparation logic
                X_batch, _ = self._prepare_batch_for_lstm(X_batch, y_batch)
                print(
                    f"[Client {self.runtime_cfg.client_id}] Reshaped X to: {X_batch.shape}, dtype: {X_batch.dtype}"
                )

                with torch.no_grad():
                    output = self.client_model(X_batch)
                    print(f"[Client {self.runtime_cfg.client_id}] Model output shape: {output.shape}")

                # Only test first batch for now
                if i >= 0:
                    break

            print(f"[Client {self.runtime_cfg.client_id}] Inference test completed successfully!")

        except Exception as e:
            print(f"[Client {self.runtime_cfg.client_id}] Error during inference: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    runner = ClusteredClientRunner()

    # Check command line argument for mode
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        runner.train_epochs()
    else:
        # Default to inference test
        runner.inference_with_drift_monitoring()
