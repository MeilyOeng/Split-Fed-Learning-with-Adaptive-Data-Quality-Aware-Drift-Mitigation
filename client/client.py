# client/clustered_client.py

import os
import sys
import time
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Add project root to Python path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import requests
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

from models.split_lstm import LSTMConfig, build_split_lstm
from data.data_streaming import StreamingIterableDataset

# Stub implementations for missing shared modules
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, path)
    print(f"[Client] Saved checkpoint to {path}")

def load_torch_checkpoint(path: str):
    """Stub: Load PyTorch checkpoint."""
    if os.path.exists(path):
        return torch.load(path)
    return None

def log_training_record(record):
    """Stub: Log training record."""
    pass

class TrainingRecord:
    """Stub: Training record dataclass."""
    pass

def create_registry():
    """Stub: Create metrics registry."""
    return None


@dataclass
class ClientRuntimeConfig:
    """
    Runtime configuration for the clustered split-learning client.
    Includes dynamic cluster and client setup.
    """
    client_id: str                  # Unique client ID (e.g., client1)
    cluster_id: str                 # Dynamically calculated cluster ID (e.g., cluster_0)
    data_root: str                  # Root directory for dataset
    num_clusters: int               # Number of clusters
    num_clients_per_cluster: int    # Number of clients per cluster

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
        # Format: integer ID (e.g., "9900002") - use cluster from assignment file
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
            "data": {
                "data_root": self.runtime_cfg.data_root,
            },
            "training": {
                "batch_size": self.runtime_cfg.batch_size,
                "num_epochs": self.runtime_cfg.num_epochs,
                "learning_rate": self.runtime_cfg.learning_rate,
            },
        }
        save_run_config(config_snapshot, name=f"client_{self.runtime_cfg.client_id}_config.json")

        self.train_loader, self.val_loader, self.inference_loader = self._build_dataloaders()

        self.client_model = self._build_model()
        self.client_model.to(self.device)

        self.optimizer = optim.Adam(self.client_model.parameters(), lr=self.runtime_cfg.learning_rate)
        self._init_metrics()

    def _build_dataloaders(self) -> Tuple[StreamingIterableDataset, StreamingIterableDataset, StreamingIterableDataset]:
        cluster_id = self.runtime_cfg.cluster_id
        data_path = os.path.join(self.runtime_cfg.data_root, f"{cluster_id}.pkl")
        print(f"[Client {self.runtime_cfg.client_id}] Loading preprocessed data for cluster: {cluster_id}")
        print(f"[Client {self.runtime_cfg.client_id}] Stream API URL: {self.runtime_cfg.data_stream_api_url}")

        if not self.runtime_cfg.data_stream_api_url:
            raise ValueError("DATA_STREAM_API_URL must be set to use streaming dataset")

        return (
            StreamingIterableDataset(
                api_url=self.runtime_cfg.data_stream_api_url,
                client_id=str(self.runtime_cfg.client_id),  # Ensure string format
                cluster_id=str(cluster_id),  # Ensure string format
                mode="train",
                batch_size=self.runtime_cfg.batch_size,
            ),
            StreamingIterableDataset(
                api_url=self.runtime_cfg.data_stream_api_url,
                client_id=str(self.runtime_cfg.client_id),
                cluster_id=str(cluster_id),
                mode="val",
                batch_size=self.runtime_cfg.batch_size,
            ),
            StreamingIterableDataset(
                api_url=self.runtime_cfg.data_stream_api_url,
                client_id=str(self.runtime_cfg.client_id),
                cluster_id=str(cluster_id),
                mode="inference",
                batch_size=self.runtime_cfg.batch_size,
            ),
        )

    def _build_server_url(self) -> str:
        # Use localhost for local testing, or Kubernetes service name for cluster deployment
        server_port = os.environ.get("SERVER_PORT", "5002")
        if os.environ.get("USE_LOCALHOST", "true").lower() == "true":
            return f"http://localhost:{server_port}"
        else:
            return f"http://server-cluster-{self.runtime_cfg.cluster_id}.svc.cluster.local"

    def _build_model(self) -> nn.Module:
        # Note: input_size should match your actual feature dimension
        # The stream service currently sends 1D values, so you may need to reshape
        # For now using input_size=1 (single value per timestep)
        # You may need to adjust this based on your actual data preprocessing
        # Note: dropout=0.0 because individual LSTM layers use num_layers=1,
        # and dropout only applies when num_layers > 1
        client_model, _ = build_split_lstm(LSTMConfig(
            input_size=1,  # Single value per timestep from stream service
            hidden_size=32,
            num_layers=2,
            output_size=1,
            dropout=0.0,  # Set to 0.0 to avoid warning (dropout only works with num_layers > 1)
            bidirectional=False,
        ), split_layer=1)
        return client_model

    def _init_metrics(self):
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

    # Training and inference logic
    def train_epochs(self, start_global_step=0):
        """Train the client model using split learning."""
        print(f"[Client {self.runtime_cfg.client_id}] Starting training...")
        print(f"[Client {self.runtime_cfg.client_id}] Server URL: {self.current_server_url}")
        
        self.client_model.train()
        total_loss = 0.0
        num_batches = 0
        
        try:
            for epoch in range(self.runtime_cfg.num_epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                print(f"\n[Client {self.runtime_cfg.client_id}] Epoch {epoch+1}/{self.runtime_cfg.num_epochs}")
                
                for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                    # Reshape data for LSTM
                    if X_batch.dim() == 1:
                        X_batch = X_batch.unsqueeze(1).unsqueeze(2)  # [batch] -> [batch, 1, 1]
                    elif X_batch.dim() == 2:
                        X_batch = X_batch.unsqueeze(1)  # [batch, features] -> [batch, 1, features]
                    
                    X_batch = X_batch.float().to(self.device)
                    y_batch = y_batch.float().to(self.device)
                    
                    # Reshape targets to match model output [batch, 1]
                    if y_batch.dim() == 1:
                        y_batch = y_batch.unsqueeze(1)
                    
                    # Forward pass through client model
                    activations = self.client_model(X_batch)  # [batch, seq_len, hidden_dim]
                    # Use last timestep for split learning
                    if activations.dim() == 3:
                        activations = activations[:, -1, :]  # [batch, hidden_dim]
                    
                    # Send activations to server for forward + backward pass
                    activations_cpu = activations.detach().cpu()
                    targets_cpu = y_batch.cpu()
                    
                    # Send to server
                    try:
                        import pickle
                        payload = pickle.dumps({
                            "client_id": self.runtime_cfg.client_id,
                            "activations": activations_cpu,
                            "targets": targets_cpu,
                        })
                        
                        response = requests.post(
                            f"{self.current_server_url}/train_step",
                            data=payload,
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=30.0
                        )
                        response.raise_for_status()
                        
                        # Receive gradients from server
                        result = pickle.loads(response.content)
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
                print(f"[Client {self.runtime_cfg.client_id}] Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")
            
            if num_batches > 0:
                print(f"\n[Client {self.runtime_cfg.client_id}] Training completed!")
                print(f"[Client {self.runtime_cfg.client_id}] Total batches: {num_batches}, Avg Loss: {total_loss/num_batches:.4f}")
            else:
                print(f"[Client {self.runtime_cfg.client_id}] No batches processed!")
                
        except Exception as e:
            print(f"[Client {self.runtime_cfg.client_id}] Error during training: {e}")
            import traceback
            traceback.print_exc()

    def evaluate(self):
        """Evaluate the client model."""
        print(f"[Client {self.runtime_cfg.client_id}] Evaluation not yet implemented")
        pass

    def inference_with_drift_monitoring(self):
        """Run inference with drift monitoring - test implementation."""
        print(f"[Client {self.runtime_cfg.client_id}] Starting inference with drift monitoring...")
        print(f"[Client {self.runtime_cfg.client_id}] Cluster: {self.runtime_cfg.cluster_id}")
        print(f"[Client {self.runtime_cfg.client_id}] Device: {self.device}")
        
        # Test: Try to get one batch from the inference loader
        try:
            print(f"[Client {self.runtime_cfg.client_id}] Testing data stream...")
            for i, (X_batch, y_batch) in enumerate(self.inference_loader):
                print(f"[Client {self.runtime_cfg.client_id}] Received batch {i+1}: X shape={X_batch.shape}, y shape={y_batch.shape}")
                
                # Reshape data for LSTM: [batch_size] -> [batch_size, seq_len=1, features=1]
                # The stream service sends 1D tensors, but LSTM expects [batch, seq_len, features]
                if X_batch.dim() == 1:
                    # Reshape to [batch, 1, 1] for single timestep, single feature
                    X_batch = X_batch.unsqueeze(1).unsqueeze(2)  # [batch] -> [batch, 1, 1]
                elif X_batch.dim() == 2:
                    # If it's [batch, features], add seq_len dimension
                    X_batch = X_batch.unsqueeze(1)  # [batch, features] -> [batch, 1, features]
                
                # Convert to float32 (LSTM requires float32, not float64)
                X_batch = X_batch.float()
                X_batch = X_batch.to(self.device)
                print(f"[Client {self.runtime_cfg.client_id}] Reshaped X to: {X_batch.shape}, dtype: {X_batch.dtype}")
                
                # Test forward pass
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
    import sys
    runner = ClusteredClientRunner()
    
    # Check command line argument for mode
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        runner.train_epochs()
    else:
        # Default to inference test
        runner.inference_with_drift_monitoring()