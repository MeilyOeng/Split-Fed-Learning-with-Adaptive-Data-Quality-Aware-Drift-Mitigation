"""
Server-side preprocessing script to assign data to clusters.

This script reads:
- `dataset.csv`, which contains time-series data with IDs.
- `cluster-assignment.csv`, which maps each ID to a specific cluster.

It pre-splits the dataset by clusters and prepares it for streaming to clients.
"""

import pandas as pd
import os

def preprocess_clusters(dataset_path: str, assignment_path: str, output_dir: str):
    """
    Preprocesses the dataset into cluster-specific files.

    Args:
        dataset_path: Path to the dataset file (e.g., dataset.csv).
        assignment_path: Path to the cluster assignment file (e.g., cluster-assignment.csv).
        output_dir: Directory to save preprocessed cluster-specific files.
    """
    print("Reading dataset...")
    dataset = pd.read_csv(dataset_path)

    print("Reading cluster assignments...")
    cluster_assignments = pd.read_csv(assignment_path)

    # Reshape dataset to transform column IDs into row entries
    print("Merging dataset with cluster assignments...")
    merged_data = dataset.melt(id_vars=["Fecha_Hora"], var_name="id", value_name="value")
    # Convert id column to integer to match the ID column in cluster_assignments
    merged_data["id"] = merged_data["id"].astype(int)
    merged_data = merged_data.merge(cluster_assignments, left_on="id", right_on="ID")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group data by clusters
    print("Splitting dataset into clusters...")
    cluster_data = {}
    for cluster_id, cluster_rows in merged_data.groupby("Cluster"):
        output_path = os.path.join(output_dir, f"cluster_{cluster_id}.pkl")
        cluster_data[cluster_id] = cluster_rows
        cluster_rows.to_pickle(output_path)
        print(f"Cluster {cluster_id}: Saved to {output_path}")

    print("Preprocessing complete!")

# Call the preprocessing function
if __name__ == "__main__":
    preprocess_clusters(
        dataset_path="./dataset/dataset.csv",
        assignment_path="./dataset/cluster-assignment.csv",
        output_dir="./dataset/output/",
    )