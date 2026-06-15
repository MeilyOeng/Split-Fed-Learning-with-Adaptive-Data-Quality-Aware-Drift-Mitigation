import pandas as pd

# Load files
lstm = pd.read_csv("dataset/lstm_results_updated.csv")  # already mapped to actual_id
test = pd.read_csv("dataset/test_report.csv")

# Rename for consistency
lstm = lstm.rename(columns={"cell_id": "client_id"})

# Keep only relevant metrics
lstm_metrics = lstm[["client_id", "mae", "rmse", "mse", "r2"]]
test_metrics = test[["client_id", "cluster_id", "mae", "rmse", "mse"]]

# Merge on client_id
merged = pd.merge(test_metrics, lstm_metrics, on="client_id", suffixes=("_split", "_lstm"))

# -----------------------------
# 1. Per-client comparison
# -----------------------------
merged["mae_diff"] = merged["mae_split"] - merged["mae_lstm"]
merged["rmse_diff"] = merged["rmse_split"] - merged["rmse_lstm"]
merged["mse_diff"] = merged["mse_split"] - merged["mse_lstm"]

merged.to_csv("comparison_per_client.csv", index=False)

print("Per-client comparison saved.")

# -----------------------------
# 2. Per-cluster comparison
# -----------------------------
cluster_comparison = merged.groupby("cluster_id").agg({
    "mae_split": "mean",
    "mae_lstm": "mean",
    "rmse_split": "mean",
    "rmse_lstm": "mean",
    "mse_split": "mean",
    "mse_lstm": "mean"
}).reset_index()

cluster_comparison["mae_diff"] = cluster_comparison["mae_split"] - cluster_comparison["mae_lstm"]
cluster_comparison["rmse_diff"] = cluster_comparison["rmse_split"] - cluster_comparison["rmse_lstm"]
cluster_comparison["mse_diff"] = cluster_comparison["mse_split"] - cluster_comparison["mse_lstm"]

cluster_comparison.to_csv("comparison_per_cluster.csv", index=False)

print("Per-cluster comparison saved.")

# -----------------------------
# 3. Overall comparison (all clients)
# -----------------------------
overall = merged[[
    "mae_split", "mae_lstm",
    "rmse_split", "rmse_lstm",
    "mse_split", "mse_lstm"
]].mean()

overall_diff = {
    "mae_diff": overall["mae_split"] - overall["mae_lstm"],
    "rmse_diff": overall["rmse_split"] - overall["rmse_lstm"],
    "mse_diff": overall["mse_split"] - overall["mse_lstm"]
}

print("\n=== Overall Comparison (All Clients) ===")
print(overall)
print("\nDifferences:")
print(overall_diff)

# Save overall
pd.DataFrame([overall.to_dict() | overall_diff]).to_csv("comparison_overall.csv", index=False)