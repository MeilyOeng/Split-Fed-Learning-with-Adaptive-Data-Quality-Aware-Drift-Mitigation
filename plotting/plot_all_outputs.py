#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BASELINE = PROJECT_ROOT / "dataset" / "lstm_output" / "lstm_results_updated.csv"
DEFAULT_ASSIGNMENTS = PROJECT_ROOT / "dataset" / "cluster-assignment.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def normalize_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def discover_report_dirs(project_root: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in project_root.iterdir()
            if path.is_dir() and path.name.startswith("report-")
        ],
        key=lambda p: p.name,
    )


def report_label(report_dir: Path) -> str:
    name = report_dir.name
    if name.startswith("report-"):
        return name[len("report-") :]
    return name


def prediction_csv_path(report_dir: Path, cluster_id: str, client_id: str) -> Path:
    return report_dir / "predictions" / f"cluster_{cluster_id}" / f"client_{client_id}_predictions.csv"


def load_prediction_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")
    df = pd.read_csv(path)
    required = {"timestamp", "y_true", "y_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction CSV missing columns {missing}: {path}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["timestamp", "y_true", "y_pred"]).reset_index(drop=True)
    return df


def compute_mae_from_prediction_csv(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "abs_error" in df.columns:
        values = pd.to_numeric(df["abs_error"], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean())
    if {"y_true", "y_pred"}.issubset(df.columns):
        y_true = pd.to_numeric(df["y_true"], errors="coerce")
        y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
        mask = y_true.notna() & y_pred.notna()
        if mask.any():
            return float((y_true[mask] - y_pred[mask]).abs().mean())
    return None


def compute_metrics_from_prediction_csv(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        return None
    y_true = pd.to_numeric(df["y_true"], errors="coerce")
    y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    if not mask.any():
        return None
    yt = y_true[mask].to_numpy(dtype=np.float64)
    yp = y_pred[mask].to_numpy(dtype=np.float64)
    err = yp - yt
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
    }


def load_assignments(assignments_path: Path) -> pd.DataFrame:
    df = pd.read_csv(assignments_path)
    if not {"ID", "Cluster"}.issubset(df.columns):
        raise ValueError(f"Assignments file must contain ID and Cluster columns: {assignments_path}")
    df = df.copy()
    df["client_id"] = df["ID"].map(normalize_id)
    df["cluster_id"] = df["Cluster"].map(normalize_id)
    return df[["client_id", "cluster_id"]]


def load_baseline_mae_map(baseline_path: Path, assignments_path: Path) -> Dict[Tuple[str, str], float]:
    baseline = pd.read_csv(baseline_path)
    if "cell_id" in baseline.columns and "client_id" not in baseline.columns:
        baseline = baseline.rename(columns={"cell_id": "client_id"})
    if "client_id" not in baseline.columns or "mae" not in baseline.columns:
        raise ValueError(f"Baseline file must contain client_id/cell_id and mae columns: {baseline_path}")
    baseline = baseline.copy()
    baseline["client_id"] = baseline["client_id"].map(normalize_id)
    baseline["mae"] = pd.to_numeric(baseline["mae"], errors="coerce")
    assignments = load_assignments(assignments_path)
    merged = assignments.merge(baseline[["client_id", "mae"]], on="client_id", how="left")
    out: Dict[Tuple[str, str], float] = {}
    for _, row in merged.iterrows():
        if pd.notna(row["mae"]):
            out[(str(row["cluster_id"]), str(row["client_id"]))] = float(row["mae"])
    return out


def load_baseline_metric_map(baseline_path: Path, assignments_path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    baseline = pd.read_csv(baseline_path)
    if "cell_id" in baseline.columns and "client_id" not in baseline.columns:
        baseline = baseline.rename(columns={"cell_id": "client_id"})
    required = {"client_id", "mae", "rmse", "mse", "r2"}
    if not required.issubset(baseline.columns):
        raise ValueError(f"Baseline file must contain columns {sorted(required)}: {baseline_path}")
    baseline = baseline.copy()
    baseline["client_id"] = baseline["client_id"].map(normalize_id)
    for col in ("mae", "rmse", "mse", "r2"):
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
    assignments = load_assignments(assignments_path)
    merged = assignments.merge(baseline[["client_id", "mae", "rmse", "mse", "r2"]], on="client_id", how="left")
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, row in merged.iterrows():
        if pd.notna(row["mae"]) and pd.notna(row["rmse"]) and pd.notna(row["mse"]) and pd.notna(row["r2"]):
            out[(str(row["cluster_id"]), str(row["client_id"]))] = {
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "mse": float(row["mse"]),
                "r2": float(row["r2"]),
            }
    return out


def choose_default_client(report_dirs: Sequence[Path]) -> Tuple[str, str]:
    common: Optional[set[Tuple[str, str]]] = None
    for report_dir in report_dirs:
        pairs = set()
        pred_root = report_dir / "predictions"
        for cluster_dir in pred_root.glob("cluster_*"):
            cluster_id = cluster_dir.name.split("cluster_", 1)[1]
            for path in cluster_dir.glob("client_*_predictions.csv"):
                client_id = path.name.split("client_", 1)[1].split("_predictions.csv", 1)[0]
                pairs.add((cluster_id, client_id))
        common = pairs if common is None else (common & pairs)
    if not common:
        raise RuntimeError("Could not find a common cluster/client prediction file across the selected report directories")
    return sorted(common, key=lambda item: (int(item[0]), int(item[1])))[0]


def common_prediction_pairs(report_dirs: Sequence[Path]) -> List[Tuple[str, str]]:
    common: Optional[set[Tuple[str, str]]] = None
    for report_dir in report_dirs:
        pairs = set()
        pred_root = report_dir / "predictions"
        for cluster_dir in pred_root.glob("cluster_*"):
            cluster_id = cluster_dir.name.split("cluster_", 1)[1]
            for path in cluster_dir.glob("client_*_predictions.csv"):
                client_id = path.name.split("client_", 1)[1].split("_predictions.csv", 1)[0]
                pairs.add((cluster_id, client_id))
        common = pairs if common is None else (common & pairs)
    if not common:
        return []
    return sorted(common, key=lambda item: (int(item[0]), int(item[1])))


def choose_best_and_worst_clients(
    report_dirs: Sequence[Path],
    *,
    cluster_id: Optional[str],
) -> Tuple[Tuple[str, str], Tuple[str, str], str]:
    pairs = common_prediction_pairs(report_dirs)
    if cluster_id is not None:
        pairs = [pair for pair in pairs if pair[0] == str(cluster_id)]
    if not pairs:
        raise RuntimeError("No common clients available for best/worst selection")

    selector_report = report_dirs[0]
    selector_label = report_label(selector_report)
    scored: List[Tuple[float, Tuple[str, str]]] = []
    for pair in pairs:
        path = prediction_csv_path(selector_report, pair[0], pair[1])
        mae = compute_mae_from_prediction_csv(path)
        if mae is not None:
            scored.append((mae, pair))
    if not scored:
        raise RuntimeError(f"No MAE values found in selector report: {selector_report}")

    scored.sort(key=lambda item: item[0])
    best_pair = scored[0][1]
    worst_pair = scored[-1][1]
    return best_pair, worst_pair, selector_label


def plot_prediction_comparison(
    report_dirs: Sequence[Path],
    *,
    cluster_id: str,
    client_id: str,
    save_path: Path,
    range_points: Optional[int],
) -> None:
    frames: List[Tuple[str, pd.DataFrame]] = []
    for report_dir in report_dirs:
        label = report_label(report_dir)
        path = prediction_csv_path(report_dir, cluster_id, client_id)
        frames.append((label, load_prediction_frame(path)))

    if not frames:
        raise RuntimeError("No prediction frames loaded")

    # Use the first report's timestamps and actual values as the base.
    _, base_df = frames[0]
    if range_points is not None:
        if range_points <= 0:
            raise ValueError(f"--range-points must be positive, got {range_points}")
        base_df = base_df.tail(range_points).reset_index(drop=True)
    if base_df.empty:
        raise RuntimeError(f"Selected range produced an empty plot for cluster={cluster_id} client={client_id}")
    base_timestamps = base_df["timestamp"]
    actual_values = -base_df["y_true"].to_numpy(dtype=np.float64)

    plt.figure(figsize=(14, 5))
    plt.plot(base_timestamps, actual_values, label="Actual", linewidth=2.6, color="#D62728")

    colors = ["#2E6F95", "#C84C09", "#2A9D8F", "#8E44AD", "#D62728", "#9467BD"]
    for idx, (label, df) in enumerate(frames):
        merged = base_df[["timestamp"]].merge(df[["timestamp", "y_pred"]], on="timestamp", how="left")
        pred_values = -merged["y_pred"].to_numpy(dtype=np.float64)
        plt.plot(base_timestamps, pred_values, label=label, linewidth=1.2, color=colors[idx % len(colors)])

    plt.title(f"Prediction comparison | cluster={cluster_id} client={client_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def build_report_mae_maps(report_dirs: Sequence[Path]) -> Dict[str, Dict[Tuple[str, str], float]]:
    out: Dict[str, Dict[Tuple[str, str], float]] = {}
    for report_dir in report_dirs:
        label = report_label(report_dir)
        mae_map: Dict[Tuple[str, str], float] = {}
        pred_root = report_dir / "predictions"
        for cluster_dir in pred_root.glob("cluster_*"):
            cluster_id = cluster_dir.name.split("cluster_", 1)[1]
            for path in cluster_dir.glob("client_*_predictions.csv"):
                client_id = path.name.split("client_", 1)[1].split("_predictions.csv", 1)[0]
                mae = compute_mae_from_prediction_csv(path)
                if mae is not None:
                    mae_map[(cluster_id, client_id)] = mae
        out[label] = mae_map
    return out


def build_report_metric_maps(report_dirs: Sequence[Path]) -> Dict[str, Dict[Tuple[str, str], Dict[str, float]]]:
    out: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}
    for report_dir in report_dirs:
        label = report_label(report_dir)
        metric_map: Dict[Tuple[str, str], Dict[str, float]] = {}
        pred_root = report_dir / "predictions"
        for cluster_dir in pred_root.glob("cluster_*"):
            cluster_id = cluster_dir.name.split("cluster_", 1)[1]
            for path in cluster_dir.glob("client_*_predictions.csv"):
                client_id = path.name.split("client_", 1)[1].split("_predictions.csv", 1)[0]
                metrics = compute_metrics_from_prediction_csv(path)
                if metrics is not None:
                    metric_map[(cluster_id, client_id)] = metrics
        out[label] = metric_map
    return out


def plot_cluster_mae_bars(
    report_dirs: Sequence[Path],
    *,
    baseline_mae_map: Dict[Tuple[str, str], float],
    assignments_path: Path,
    save_path: Path,
) -> None:
    report_mae_maps = build_report_mae_maps(report_dirs)
    report_labels = [report_label(path) for path in report_dirs]
    assignments = load_assignments(assignments_path)
    cluster_ids = sorted(assignments["cluster_id"].unique().tolist(), key=lambda x: int(x))
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), squeeze=False)
    axes_flat = axes.flatten()

    colors = ["#7F8C8D", "#2E6F95", "#C84C09", "#2A9D8F", "#8E44AD"]

    for ax, cluster_id in zip(axes_flat, cluster_ids):
        cluster_clients = (
            assignments[assignments["cluster_id"] == str(cluster_id)]["client_id"]
            .astype(str)
            .tolist()
        )
        cluster_clients = sorted(cluster_clients, key=lambda x: int(x))
        x = np.arange(len(cluster_clients))
        n_series = 1 + len(report_labels)
        width = min(0.18, 0.9 / max(n_series, 1))
        offset_start = -width * (n_series - 1) / 2.0

        baseline_vals = [baseline_mae_map.get((str(cluster_id), client_id), np.nan) for client_id in cluster_clients]
        ax.bar(
            x + offset_start,
            baseline_vals,
            width=width,
            color=colors[0],
            label="LSTM",
            edgecolor="black",
            linewidth=0.9,
            hatch="//",
            zorder=3,
        )

        for idx, label in enumerate(report_labels, start=1):
            mae_map = report_mae_maps[label]
            vals = [mae_map.get((str(cluster_id), client_id), np.nan) for client_id in cluster_clients]
            ax.bar(
                x + offset_start + idx * width,
                vals,
                width=width,
                color=colors[idx % len(colors)],
                label=label,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )

        ax.set_title(f"Cluster {cluster_id}")
        ax.set_xlabel("Client")
        ax.set_ylabel("MAE")
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_clients, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    for ax in axes_flat[len(cluster_ids):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5))
    fig.suptitle("Client-level MAE comparison by cluster")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overall_metric_bars(
    report_dirs: Sequence[Path],
    *,
    baseline_metric_map: Dict[Tuple[str, str], Dict[str, float]],
    save_path: Path,
) -> None:
    report_metric_maps = build_report_metric_maps(report_dirs)
    report_labels = [report_label(path) for path in report_dirs]
    common_clients = set(baseline_metric_map.keys())
    for label in report_labels:
        common_clients &= set(report_metric_maps[label].keys())
    if not common_clients:
        raise RuntimeError("No common clients across baseline and selected reports for overall metric comparison")

    common_clients = sorted(common_clients, key=lambda item: (int(item[0]), int(item[1])))
    metric_names = ["mae", "rmse", "mse", "r2"]
    series_labels = ["LSTM"] + report_labels
    colors = ["#7F8C8D", "#2E6F95", "#C84C09", "#2A9D8F", "#8E44AD"]

    summary_values: Dict[str, List[float]] = {metric: [] for metric in metric_names}
    for metric in metric_names:
        baseline_values = [baseline_metric_map[key][metric] for key in common_clients]
        summary_values[metric].append(float(np.mean(baseline_values)))
        for label in report_labels:
            model_values = [report_metric_maps[label][key][metric] for key in common_clients]
            summary_values[metric].append(float(np.mean(model_values)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    axes_flat = axes.flatten()
    x = np.arange(len(series_labels))

    for ax, metric in zip(axes_flat, metric_names):
        vals = summary_values[metric]
        bars = []
        for idx, value in enumerate(vals):
            style = {
                "color": colors[idx % len(colors)],
                "zorder": 3,
            }
            if idx == 0:
                style.update({
                    "edgecolor": "black",
                    "linewidth": 0.9,
                    "hatch": "//",
                })
            else:
                style.update({
                    "edgecolor": "white",
                    "linewidth": 0.4,
                })
            bars.append(ax.bar(x[idx], value, width=0.62, **style)[0])
        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(series_labels, rotation=20, ha="right")
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.01 if metric != "r2" else bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(f"Overall metric comparison across {len(common_clients)} common clients")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple plots comparing saved report predictions and MAE by cluster.")
    p.add_argument("--reports", type=str, default=None, help="Comma-separated report directories. Default: all report-* directories.")
    p.add_argument("--cluster", type=str, default=None, help="Restrict best/worst prediction plots to one cluster. If omitted, select best/worst globally.")
    p.add_argument("--range-points", "--range-point", dest="range_points", type=int, default=None, help="Number of final points to display in the prediction line plots.")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="Baseline LSTM CSV with cell_id/client_id and mae.")
    p.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS, help="Cluster assignment CSV with ID and Cluster.")
    p.add_argument("--output-dir", type=Path, default=Path("plots/outputs"), help="Directory where the two plots are saved.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report_dirs = (
        [PROJECT_ROOT / item for item in parse_csv_list(args.reports)]
        if args.reports
        else discover_report_dirs(PROJECT_ROOT)
    )
    if not report_dirs:
        raise RuntimeError("No report directories found")

    for report_dir in report_dirs:
        if not report_dir.exists():
            raise FileNotFoundError(f"Missing report directory: {report_dir}")

    selected_cluster = normalize_id(args.cluster) if args.cluster is not None else None
    best_pair, worst_pair, selector_label = choose_best_and_worst_clients(report_dirs, cluster_id=selected_cluster)

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    ensure_dir(output_dir)

    plot_prediction_comparison(
        report_dirs,
        cluster_id=best_pair[0],
        client_id=best_pair[1],
        save_path=output_dir / f"prediction_compare_best_cluster_{best_pair[0]}_client_{best_pair[1]}.png",
        range_points=args.range_points,
    )
    plot_prediction_comparison(
        report_dirs,
        cluster_id=worst_pair[0],
        client_id=worst_pair[1],
        save_path=output_dir / f"prediction_compare_worst_cluster_{worst_pair[0]}_client_{worst_pair[1]}.png",
        range_points=args.range_points,
    )

    baseline_mae_map = load_baseline_mae_map(
        args.baseline if args.baseline.is_absolute() else (PROJECT_ROOT / args.baseline),
        args.assignments if args.assignments.is_absolute() else (PROJECT_ROOT / args.assignments),
    )
    baseline_metric_map = load_baseline_metric_map(
        args.baseline if args.baseline.is_absolute() else (PROJECT_ROOT / args.baseline),
        args.assignments if args.assignments.is_absolute() else (PROJECT_ROOT / args.assignments),
    )
    plot_cluster_mae_bars(
        report_dirs,
        baseline_mae_map=baseline_mae_map,
        assignments_path=args.assignments if args.assignments.is_absolute() else (PROJECT_ROOT / args.assignments),
        save_path=output_dir / "cluster_mae_bars.png",
    )
    plot_overall_metric_bars(
        report_dirs,
        baseline_metric_map=baseline_metric_map,
        save_path=output_dir / "overall_metric_bars.png",
    )

    scope = f"cluster={selected_cluster}" if selected_cluster is not None else "all clusters"
    log(f"Best/worst selected from first report: {selector_label} ({scope})")
    log(f"Saved BEST prediction plot for cluster={best_pair[0]} client={best_pair[1]}")
    log(f"Saved WORST prediction plot for cluster={worst_pair[0]} client={worst_pair[1]}")
    log(f"Saved MAE cluster bar plot")
    log(f"Saved overall metric bar plot")
    log(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
