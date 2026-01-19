"""
Server-side streaming service that serves cluster-specific batches to clients.
"""

import os
import pickle
import pandas as pd
import torch
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
# Get the absolute path to the cluster directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTER_DIR = os.path.join(BASE_DIR, "dataset", "output")  # Directory containing preprocessed cluster data

cluster_batches = {}

def load_cluster_data():
    """Load preprocessed cluster data into memory."""
    global cluster_batches
    if not os.path.exists(CLUSTER_DIR):
        print(f"Warning: Cluster directory not found: {CLUSTER_DIR}")
        return
    
    if not os.listdir(CLUSTER_DIR):
        print(f"Warning: Cluster directory is empty: {CLUSTER_DIR}")
        return
    
    for file in os.listdir(CLUSTER_DIR):
        if file.endswith(".pkl"):
            cluster_id = file.split("_")[1].split(".")[0]  # Extract cluster number
            file_path = os.path.join(CLUSTER_DIR, file)
            cluster_batches[cluster_id] = pd.read_pickle(file_path)
            print(f"Loaded cluster {cluster_id} from {file}")

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

@app.route("/stream/next", methods=["POST"])
def stream_next():
    """Stream the next batch of data to the client."""
    data = request.get_json()
    client_id = int(data["client_id"])  # Convert to int to match the id column
    cluster_id = str(data["cluster_id"])  # Ensure it's a string
    mode = data["mode"]
    batch_size = data["batch_size"]

    # Ensure valid cluster ID and data availability
    if cluster_id not in cluster_batches:
        return jsonify({"error": f"Cluster {cluster_id} not found!"}), 404

    # Retrieve batch for the cluster
    cluster_data = cluster_batches[cluster_id]
    clients_data = cluster_data[cluster_data["id"] == client_id]  # Filter by client_id

    # Split into batches
    if len(clients_data) == 0:
        return jsonify({"error": "No data found for this client!"}), 404

    # Send the next batch
    X_batch = torch.tensor(clients_data.iloc[:batch_size]["value"].values)
    y_batch = torch.tensor(X_batch)  # Placeholder for illustrative purposes

    # Serialize and send batch
    payload = pickle.dumps({"X": X_batch, "y": y_batch})
    return Response(payload, content_type="application/octet-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)