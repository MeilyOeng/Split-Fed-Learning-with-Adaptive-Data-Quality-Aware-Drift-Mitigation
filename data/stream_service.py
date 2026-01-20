"""
Server-side streaming service that serves cluster-specific batches to clients.
"""

import os
import pickle
from threading import Lock

import pandas as pd
import torch
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
# Get the absolute path to the cluster directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTER_DIR = os.path.join(BASE_DIR, "dataset", "output")  # Directory containing preprocessed cluster data

cluster_batches = {}
client_offsets = {}
offset_lock = Lock()

def load_cluster_data():
    """Load preprocessed cluster data into memory."""
    global cluster_batches
    
    # Try loading from pickle files first
    if os.path.exists(CLUSTER_DIR) and os.listdir(CLUSTER_DIR):
        for file in os.listdir(CLUSTER_DIR):
            if file.endswith(".pkl"):
                cluster_id = file.split("_")[1].split(".")[0]  # Extract cluster number
                file_path = os.path.join(CLUSTER_DIR, file)
                try:
                    cluster_batches[cluster_id] = pd.read_pickle(file_path)
                    print(f"Loaded cluster {cluster_id} from {file}")
                except Exception as e:
                    print(f"Error loading pickle {file}: {e}")
        
        if cluster_batches:
            print(f"Successfully loaded {len(cluster_batches)} clusters from pickle files")
            return
    
    # Fallback: Load from CSV files and create clusters on-the-fly
    print("Pickle loading failed, attempting to load from CSV...")
    dataset_path = os.path.join(BASE_DIR, "dataset", "dataset.csv")
    assignment_path = os.path.join(BASE_DIR, "dataset", "cluster-assignment.csv")
    
    if not os.path.exists(dataset_path) or not os.path.exists(assignment_path):
        print(f"Warning: CSV files not found: {dataset_path} or {assignment_path}")
        return
    
    try:
        print("Reading dataset...")
        dataset = pd.read_csv(dataset_path)
        print("Reading cluster assignments...")
        cluster_assignments = pd.read_csv(assignment_path)
        
        # Reshape dataset to transform column IDs into row entries
        print("Processing dataset...")
        merged_data = dataset.melt(id_vars=["Fecha_Hora"], var_name="id", value_name="value")
        merged_data["id"] = merged_data["id"].astype(int)
        merged_data = merged_data.merge(cluster_assignments, left_on="id", right_on="ID")
        
        # Group data by clusters
        for cluster_id, cluster_rows in merged_data.groupby("Cluster"):
            cluster_batches[str(cluster_id)] = cluster_rows
            print(f"Loaded cluster {cluster_id} from CSV ({len(cluster_rows)} rows)")
        
        print(f"Successfully loaded {len(cluster_batches)} clusters from CSV files")
    except Exception as e:
        print(f"Error loading from CSV: {e}")
        import traceback
        traceback.print_exc()

def initialize_service():
    """Initialize the service by loading cluster data."""
    try:
        load_cluster_data()
        print(f"Cluster data loaded into memory. Loaded {len(cluster_batches)} clusters.")
    except Exception as e:
        print(f"Error loading cluster data: {e}")
        print("Service will start but may not function correctly until data is loaded.")

# Initialize the service when the module is loaded
initialize_service()

@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API information."""
    return jsonify({
        "service": "Split-Fed Learning Data Streaming Service",
        "available_endpoints": {
            "/": "GET - This information page",
            "/health": "GET - Health check endpoint",
            "/stream/next": "POST - Stream next batch of data"
        },
        "loaded_clusters": list(cluster_batches.keys())
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "clusters_loaded": len(cluster_batches),
        "cluster_ids": list(cluster_batches.keys())
    })

@app.route("/stream/reset", methods=["POST"])
def stream_reset():
    """Reset the stream offset for a client (used at start of each epoch)."""
    data = request.get_json()
    client_id = int(data["client_id"])
    cluster_id = str(data["cluster_id"])
    mode = data.get("mode", "train")
    
    key = (cluster_id, client_id, mode)
    with offset_lock:
        client_offsets[key] = 0
    
    return jsonify({"status": "reset", "client_id": client_id, "cluster_id": cluster_id, "mode": mode})

@app.route("/stream/next", methods=["POST"])
def stream_next():
    """Stream the next batch of data to the client."""
    data = request.get_json()
    client_id = int(data["client_id"])  # Convert to int to match the id column
    cluster_id = str(data["cluster_id"])  # Ensure it's a string
    mode = data["mode"]
    batch_size = data["batch_size"]
    window_size = int(os.environ.get("WINDOW_SIZE", "24"))

    # Ensure valid cluster ID and data availability
    if cluster_id not in cluster_batches:
        return jsonify({"error": f"Cluster {cluster_id} not found!"}), 404

    # Retrieve batch for the cluster
    cluster_data = cluster_batches[cluster_id]
    clients_data = cluster_data[cluster_data["id"] == client_id]  # Filter by client_id

    # Split into batches
    if len(clients_data) == 0:
        return jsonify({"error": "No data found for this client!"}), 404

    if len(clients_data) < (window_size + 1):
        return jsonify({"error": f"Insufficient data! Need at least {window_size + 1} values, but only {len(clients_data)} available."}), 400

    values = clients_data["value"].to_numpy(dtype=float)

    key = (cluster_id, client_id, mode)
    with offset_lock:
        start_idx = client_offsets.get(key, 0)
        last_start = start_idx + batch_size - 1
        last_target_idx = last_start + window_size
        if last_target_idx >= len(values):
            return Response(status=204)
        client_offsets[key] = start_idx + batch_size

    # Build windowed sequences for time series forecasting
    X_windows = []
    y_targets = []
    for i in range(start_idx, start_idx + batch_size):
        X_windows.append(values[i:i + window_size])
        y_targets.append(values[i + window_size])

    X_batch = torch.tensor(X_windows, dtype=torch.float32).unsqueeze(-1)  # [batch, window, 1]
    y_batch = torch.tensor(y_targets, dtype=torch.float32)  # [batch]

    # Serialize and send batch
    payload = pickle.dumps({"X": X_batch, "y": y_batch})
    return Response(payload, content_type="application/octet-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
