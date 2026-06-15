#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_LABELS = ["ep30-noES", "earlystopped", "warmup30es15"]
PALETTE = ["#7F8C8D", "#2E6F95", "#C84C09", "#2A9D8F", "#8E44AD"]
DISPLAY_NAMES = {
    "ep30-noES": "Fixed 30 Epochs",
    "earlystopped": "Early Stopping",
    "warmup30es15": "Warmup 30 + ES15",
}
RUN_COLORS = {
    "ep30-noES": PALETTE[1],
    "earlystopped": PALETTE[2],
    "warmup30es15": PALETTE[3],
    "ep50-noES": PALETTE[4],
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


def log(message: str) -> None:
    print(f"[compare_training_plots] {message}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def display_name(run_label: str) -> str:
    return DISPLAY_NAMES.get(run_label, run_label)


def run_output_dir(run_label: str) -> Path:
    return Path(f"output-{run_label}")


def load_client_alignment(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"cluster_id": str, "client_id": str})
    df["cluster_id"] = df["cluster_id"].map(normalize_id)
    df["client_id"] = df["client_id"].map(normalize_id)
    return df


def train_metrics_path(output_dir: Path, cluster_id: str, client_id: str) -> Path:
    return output_dir / f"cluster_{cluster_id}" / f"client_{client_id}_train_metrics.json"


def load_train_history(output_dir: Path, cluster_id: str, client_id: str) -> List[dict]:
    path = train_metrics_path(output_dir, cluster_id, client_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    history = data.get("history", [])
    return history if isinstance(history, list) else []


def load_train_report(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "results" / "train_report.csv"
    df = pd.read_csv(path, dtype={"cluster_id": str, "client_id": str})
    df["cluster_id"] = df["cluster_id"].map(normalize_id)
    df["client_id"] = df["client_id"].map(normalize_id)
    return df


def build_mean_loss_curve(alignment_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    epoch_values: Dict[int, Dict[str, List[float]]] = {}
    history_files = 0
    missing_histories = 0

    for row in alignment_df.itertuples(index=False):
        history = load_train_history(output_dir, normalize_id(row.cluster_id), normalize_id(row.client_id))
        if not history:
            missing_histories += 1
            continue
        history_files += 1
        for entry in history:
            epoch = int(entry.get("epoch", 0))
            if epoch <= 0:
                continue
            bucket = epoch_values.setdefault(epoch, {"train_loss": [], "val_loss": []})
            train_loss = entry.get("train_loss")
            val_loss = entry.get("val_loss")
            if train_loss is not None:
                bucket["train_loss"].append(float(train_loss))
            if val_loss is not None:
                bucket["val_loss"].append(float(val_loss))

    records = []
    for epoch in sorted(epoch_values):
        bucket = epoch_values[epoch]
        train_values = bucket["train_loss"]
        val_values = bucket["val_loss"]
        records.append(
            {
                "epoch": epoch,
                "mean_train_loss": float(np.mean(train_values)) if train_values else np.nan,
                "mean_val_loss": float(np.mean(val_values)) if val_values else np.nan,
                "num_train_clients": len(train_values),
                "num_val_clients": len(val_values),
            }
        )

    log(
        f"loaded loss histories from {history_files} clients in {output_dir.name}; "
        f"missing histories for {missing_histories} clients"
    )
    return pd.DataFrame(records)


def plot_training_dynamics(alignment_df: pd.DataFrame, run_labels: List[str], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(run_labels), figsize=(6 * len(run_labels), 4.8), sharey=True)
    if len(run_labels) == 1:
        axes = [axes]

    for ax, run_label in zip(axes, run_labels):
        curve_df = build_mean_loss_curve(alignment_df, run_output_dir(run_label))
        if curve_df.empty:
            ax.text(0.5, 0.5, "No history data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(display_name(run_label))
            continue

        ax.plot(
            curve_df["epoch"],
            curve_df["mean_train_loss"],
            color=PALETTE[1],
            linewidth=2.2,
            label="Train loss",
        )
        ax.plot(
            curve_df["epoch"],
            curve_df["mean_val_loss"],
            color=PALETTE[2],
            linewidth=2.2,
            label="Validation loss",
        )
        ax.set_title(display_name(run_label))
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        last_count = int(curve_df["num_train_clients"].iloc[-1]) if not curve_df.empty else 0
        ax.text(
            0.98,
            0.02,
            f"clients at last epoch: {last_count}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )

    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    fig.suptitle("Training Dynamics: Mean Train vs Validation Loss", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "training_dynamics_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {out_path}")


def plot_stopping_epoch_distribution(run_labels: List[str], output_dir: Path) -> None:
    stopping_runs = [label for label in run_labels if label in {"earlystopped", "warmup30es15"}]
    if not stopping_runs:
        log("skipping stopping epoch distribution: no early-stopping runs selected")
        return

    fig, axes = plt.subplots(1, len(stopping_runs), figsize=(5.5 * len(stopping_runs), 4.4), sharey=True)
    if len(stopping_runs) == 1:
        axes = [axes]

    for ax, run_label in zip(axes, stopping_runs):
        report_df = load_train_report(run_output_dir(run_label))
        epochs = pd.to_numeric(report_df["completed_epochs"], errors="coerce").dropna().astype(int)
        if epochs.empty:
            ax.text(0.5, 0.5, "No stopping data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(display_name(run_label))
            continue
        bins = np.arange(epochs.min(), epochs.max() + 2) - 0.5
        ax.hist(epochs, bins=bins, color=RUN_COLORS.get(run_label, PALETTE[1]), edgecolor="white")
        ax.axvline(float(epochs.median()), color="#222222", linestyle="--", linewidth=1.2, label="Median stop")
        ax.set_title(display_name(run_label))
        ax.set_xlabel("Epoch stopped")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Number of clients")
    fig.suptitle("Stopping Epoch Distribution", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "stopping_epoch_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {out_path}")


def plot_metric_variance(alignment_df: pd.DataFrame, run_labels: List[str], metric: str, output_dir: Path) -> None:
    series_list = []
    labels = []
    colors = []
    for run_label in run_labels:
        column = f"{run_label}_{metric}"
        if column not in alignment_df.columns:
            continue
        values = pd.to_numeric(alignment_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        series_list.append(values.to_numpy())
        labels.append(display_name(run_label))
        colors.append(RUN_COLORS.get(run_label, PALETTE[1]))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    box = ax.boxplot(series_list, patch_artist=True, tick_labels=labels)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#222222")
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.4)
    ax.set_title(f"{metric.upper()} Variance Across Clients")
    ax.set_ylabel(metric.upper())
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    out_path = output_dir / f"metric_variance_boxplot_{metric}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {out_path}")


def plot_best_worst_gap(alignment_df: pd.DataFrame, run_labels: List[str], metric: str, output_dir: Path) -> None:
    labels = []
    gaps = []
    colors = []
    for run_label in run_labels:
        column = f"{run_label}_{metric}"
        if column not in alignment_df.columns:
            continue
        values = pd.to_numeric(alignment_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        labels.append(display_name(run_label))
        gaps.append(float(values.max() - values.min()))
        colors.append(RUN_COLORS.get(run_label, PALETTE[1]))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(labels, gaps, color=colors, edgecolor="#222222", linewidth=0.7)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{gap:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Best vs Worst Client Gap ({metric.upper()})")
    ax.set_ylabel(f"{metric.upper()} gap (max - min)")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    out_path = output_dir / f"best_worst_gap_{metric}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {out_path}")


def plot_topk_worst_clients(alignment_df: pd.DataFrame, run_labels: List[str], metric: str, top_k: int, output_dir: Path) -> None:
    metric_columns = [f"{run_label}_{metric}" for run_label in run_labels if f"{run_label}_{metric}" in alignment_df.columns]
    df = alignment_df[["cluster_id", "client_id", *metric_columns]].copy()
    for column in metric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=metric_columns)
    if df.empty:
        log("skipping top-k worst plot: no common metric rows")
        return

    # Use worst-case MAE across methods so the selected clients represent the strongest failures.
    df["selection_score"] = df[metric_columns].max(axis=1)
    worst_df = df.sort_values("selection_score", ascending=False).head(top_k).copy()
    worst_df["client_label"] = worst_df.apply(
        lambda row: f"C{normalize_id(row['cluster_id'])}-{normalize_id(row['client_id'])}", axis=1
    )

    x = np.arange(len(worst_df))
    width = 0.8 / max(len(run_labels), 1)

    fig, ax = plt.subplots(figsize=(max(11.0, top_k * 1.2), 5.2))
    for idx, run_label in enumerate(run_labels):
        column = f"{run_label}_{metric}"
        if column not in worst_df.columns:
            continue
        offset = (idx - (len(run_labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            worst_df[column],
            width=width,
            label=display_name(run_label),
            color=RUN_COLORS.get(run_label, PALETTE[1]),
            edgecolor="#222222",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(worst_df["client_label"], rotation=35, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Top-{top_k} Worst Clients by Worst-Case {metric.upper()}")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path = output_dir / f"top_{top_k}_worst_clients_{metric}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create training-dynamics and robustness plots from run comparison outputs."
    )
    parser.add_argument(
        "--comparison-csv",
        default="comparison_outputs/run_compare/client_alignment.csv",
        help="Path to client_alignment.csv",
    )
    parser.add_argument(
        "--runs",
        default=",".join(DEFAULT_RUN_LABELS),
        help="Comma-separated run labels from client_alignment.csv",
    )
    parser.add_argument(
        "--metric",
        default="mae",
        choices=["mae", "rmse", "smape", "r2"],
        help="Metric used for client-variance boxplot.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of worst-case clients to show in the worst-client comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_outputs/run_compare/training_analysis_plots",
        help="Directory to store generated plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison_csv = Path(args.comparison_csv)
    output_dir = Path(args.output_dir)
    run_labels = parse_csv_list(args.runs)

    ensure_dir(output_dir)

    alignment_df = load_client_alignment(comparison_csv)
    missing_columns = [label for label in run_labels if f"{label}_mae" not in alignment_df.columns]
    if missing_columns:
        raise SystemExit(f"Missing run columns in {comparison_csv}: {missing_columns}")

    plot_training_dynamics(alignment_df, run_labels, output_dir)
    plot_stopping_epoch_distribution(run_labels, output_dir)
    plot_metric_variance(alignment_df, run_labels, args.metric, output_dir)
    plot_best_worst_gap(alignment_df, run_labels, "mae", output_dir)
    plot_topk_worst_clients(alignment_df, run_labels, "mae", args.top_k, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
