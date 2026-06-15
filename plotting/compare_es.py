#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = [
    "output-earlystopped",
    "output-ep30-noES",
    "output-warmup30es15",
    "output-ep50-noES",
]
DEFAULT_METRICS = ["mae", "rmse", "smape", "r2"]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def safe_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def safe_quantile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    s = sorted(vals)
    pos = (len(s) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(s[low])
    frac = pos - low
    return float(s[low] * (1.0 - frac) + s[high] * frac)


def metric_higher_is_better(metric: str) -> bool:
    return metric == "r2"


def write_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_label_from_dir(run_dir: str) -> str:
    name = Path(run_dir).name
    if name.startswith("output-"):
        return name[len("output-") :]
    if name == "output":
        return "output"
    return name


def load_report_csv(path: Path, kind: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind} report: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_runs(run_dirs: Sequence[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    test_frames: Dict[str, pd.DataFrame] = {}
    train_frames: Dict[str, pd.DataFrame] = {}
    for run_dir in run_dirs:
        label = run_label_from_dir(run_dir)
        base = PROJECT_ROOT / run_dir / "results"
        test_df = load_report_csv(base / "test_report.csv", kind=f"test for {label}")
        train_df = load_report_csv(base / "train_report.csv", kind=f"train for {label}")
        test_df["run_label"] = label
        train_df["run_label"] = label
        test_frames[label] = test_df
        train_frames[label] = train_df
    return test_frames, train_frames


def summarize_metric_values(values: List[Optional[float]], metric: str) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {
            f"avg_{metric}": None,
            f"median_{metric}": None,
            f"p90_{metric}": None,
            f"best_{metric}": None,
            f"worst_{metric}": None,
        }
    return {
        f"avg_{metric}": safe_mean(vals),
        f"median_{metric}": safe_quantile(vals, 0.50),
        f"p90_{metric}": safe_quantile(vals, 0.90),
        f"best_{metric}": (max(vals) if metric_higher_is_better(metric) else min(vals)),
        f"worst_{metric}": (min(vals) if metric_higher_is_better(metric) else max(vals)),
    }


def build_overall_summary(test_frames: Dict[str, pd.DataFrame], train_frames: Dict[str, pd.DataFrame], metrics: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    for label in test_frames:
        test_df = test_frames[label]
        train_df = train_frames[label]
        row = {
            "run_label": label,
            "num_clients": int(len(test_df)),
            "num_clusters": int(test_df["cluster_id"].nunique()) if "cluster_id" in test_df.columns else None,
            "num_early_stopped": int(train_df["early_stopped"].map(safe_bool).sum()) if "early_stopped" in train_df.columns else None,
            "early_stop_rate": (
                float(train_df["early_stopped"].map(safe_bool).mean() * 100.0)
                if "early_stopped" in train_df.columns and len(train_df) > 0
                else None
            ),
            "avg_completed_epochs": safe_mean(safe_float(v) for v in train_df.get("completed_epochs", pd.Series(dtype=float)).tolist()),
            "avg_best_val_raw_mae": safe_mean(safe_float(v) for v in train_df.get("best_val_raw_mae", pd.Series(dtype=float)).tolist()),
        }
        for metric in metrics:
            row.update(summarize_metric_values([safe_float(v) for v in test_df.get(metric, pd.Series(dtype=float)).tolist()], metric))
        rows.append(row)
    return rows


def build_cluster_summary(test_frames: Dict[str, pd.DataFrame], train_frames: Dict[str, pd.DataFrame], metrics: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    all_clusters = sorted(
        {str(cid) for df in test_frames.values() for cid in df["cluster_id"].astype(str).tolist()},
        key=lambda x: int(x) if x.isdigit() else x,
    )
    for label in test_frames:
        test_df = test_frames[label]
        train_df = train_frames[label]
        for cluster_id in all_clusters:
            test_sub = test_df[test_df["cluster_id"].astype(str) == str(cluster_id)]
            if test_sub.empty:
                continue
            train_sub = train_df[train_df["cluster_id"].astype(str) == str(cluster_id)]
            row = {
                "run_label": label,
                "cluster_id": cluster_id,
                "num_clients": int(len(test_sub)),
                "num_early_stopped": int(train_sub["early_stopped"].map(safe_bool).sum()) if "early_stopped" in train_sub.columns else None,
                "early_stop_rate": (
                    float(train_sub["early_stopped"].map(safe_bool).mean() * 100.0)
                    if "early_stopped" in train_sub.columns and len(train_sub) > 0
                    else None
                ),
                "avg_completed_epochs": safe_mean(safe_float(v) for v in train_sub.get("completed_epochs", pd.Series(dtype=float)).tolist()),
            }
            for metric in metrics:
                row.update(summarize_metric_values([safe_float(v) for v in test_sub.get(metric, pd.Series(dtype=float)).tolist()], metric))
            rows.append(row)
    return rows


def build_client_alignment(test_frames: Dict[str, pd.DataFrame], train_frames: Dict[str, pd.DataFrame], metrics: Sequence[str]) -> List[Dict]:
    clients = sorted(
        {
            (str(row["cluster_id"]), str(row["client_id"]))
            for df in test_frames.values()
            for _, row in df.iterrows()
        },
        key=lambda x: ((int(x[0]) if x[0].isdigit() else x[0]), (int(x[1]) if x[1].isdigit() else x[1])),
    )
    rows: List[Dict] = []
    labels = list(test_frames.keys())
    for cluster_id, client_id in clients:
        row: Dict[str, object] = {"cluster_id": cluster_id, "client_id": client_id}
        for metric in metrics:
            row[f"best_run_by_{metric}"] = None
            row[f"worst_run_by_{metric}"] = None
            row[f"spread_{metric}"] = None

        for label in labels:
            test_match = test_frames[label][
                (test_frames[label]["cluster_id"].astype(str) == cluster_id)
                & (test_frames[label]["client_id"].astype(str) == client_id)
            ]
            train_match = train_frames[label][
                (train_frames[label]["cluster_id"].astype(str) == cluster_id)
                & (train_frames[label]["client_id"].astype(str) == client_id)
            ]
            if test_match.empty:
                continue
            test_rec = test_match.iloc[0]
            train_rec = train_match.iloc[0] if not train_match.empty else None
            for metric in metrics:
                row[f"{label}_{metric}"] = safe_float(test_rec.get(metric))
            row[f"{label}_early_stopped"] = safe_bool(train_rec.get("early_stopped")) if train_rec is not None else None
            row[f"{label}_completed_epochs"] = safe_float(train_rec.get("completed_epochs")) if train_rec is not None else None
            row[f"{label}_best_epoch"] = safe_float(train_rec.get("best_epoch")) if train_rec is not None else None
            row[f"{label}_best_val_raw_mae"] = safe_float(train_rec.get("best_val_raw_mae")) if train_rec is not None else None
            row[f"{label}_stop_reason"] = None if train_rec is None else train_rec.get("stop_reason")

        for metric in metrics:
            metric_map = {
                label: safe_float(row.get(f"{label}_{metric}"))
                for label in labels
                if safe_float(row.get(f"{label}_{metric}")) is not None
            }
            if not metric_map:
                continue
            if metric_higher_is_better(metric):
                best_label = max(metric_map, key=lambda k: metric_map[k])
                worst_label = min(metric_map, key=lambda k: metric_map[k])
                spread = float(metric_map[best_label] - metric_map[worst_label])
            else:
                best_label = min(metric_map, key=lambda k: metric_map[k])
                worst_label = max(metric_map, key=lambda k: metric_map[k])
                spread = float(metric_map[worst_label] - metric_map[best_label])
            row[f"best_run_by_{metric}"] = best_label
            row[f"worst_run_by_{metric}"] = worst_label
            row[f"spread_{metric}"] = spread
        rows.append(row)
    return rows


def build_winners(client_rows: List[Dict], metrics: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    for row in client_rows:
        for metric in metrics:
            winner = row.get(f"best_run_by_{metric}")
            loser = row.get(f"worst_run_by_{metric}")
            if winner is None:
                continue
            rows.append(
                {
                    "cluster_id": row["cluster_id"],
                    "client_id": row["client_id"],
                    "metric": metric,
                    "winner_run": winner,
                    "winner_value": row.get(f"{winner}_{metric}"),
                    "loser_run": loser,
                    "loser_value": (None if loser is None else row.get(f"{loser}_{metric}")),
                    "spread": row.get(f"spread_{metric}"),
                }
            )
    return rows


def build_markdown_report(
    output_dir: Path,
    run_dirs: Sequence[str],
    overall_rows: List[Dict],
    cluster_rows: List[Dict],
    client_rows: List[Dict],
    metrics: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# Run Comparison Report")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir}`")
    lines.append("")

    lines.append("## Overall Summary")
    lines.append("")
    headers = ["run_label", "num_clients", "num_early_stopped", "early_stop_rate", "avg_completed_epochs"] + [f"avg_{metric}" for metric in metrics]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in overall_rows:
        lines.append("| " + " | ".join(str(row.get(h)) for h in headers) + " |")
    lines.append("")

    lines.append("## Per-Cluster Summary")
    lines.append("")
    cluster_headers = ["run_label", "cluster_id", "num_clients", "num_early_stopped", "early_stop_rate"] + [f"avg_{metric}" for metric in metrics]
    lines.append("| " + " | ".join(cluster_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(cluster_headers)) + " |")
    for row in cluster_rows:
        lines.append("| " + " | ".join(str(row.get(h)) for h in cluster_headers) + " |")
    lines.append("")

    lines.append("## Early-Stopped Clients")
    lines.append("")
    run_labels = sorted({row["run_label"] for row in overall_rows})
    for label in run_labels:
        stopped = [row for row in client_rows if safe_bool(row.get(f"{label}_early_stopped"))]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Count: `{len(stopped)}`")
        if stopped:
            ids = [f"cluster {row['cluster_id']} client {row['client_id']}" for row in stopped]
            lines.append(f"- Clients: {', '.join(ids)}")
        else:
            lines.append("- Clients: none")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `overall_summary.csv` compares runs at the full-run level.")
    lines.append("- `client_alignment.csv` compares the same client across runs and includes early-stop columns.")
    lines.append("- `winners.csv` lists which run wins for each client and metric.")
    lines.append("")
    (output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")

def build_cluster_best_worst_clients(
    test_frames: Dict[str, pd.DataFrame],
) -> List[Dict]:
    rows: List[Dict] = []
    metric = "mae"

    for label, test_df in test_frames.items():
        if (
            "cluster_id" not in test_df.columns
            or "client_id" not in test_df.columns
            or metric not in test_df.columns
        ):
            continue

        for cluster_id, sub_df in test_df.groupby(test_df["cluster_id"].astype(str)):
            metric_rows = []

            for _, rec in sub_df.iterrows():
                value = safe_float(rec.get(metric))
                if value is None:
                    continue

                metric_rows.append(
                    {
                        "client_id": str(rec.get("client_id")),
                        "mae": value,
                    }
                )

            if not metric_rows:
                continue

            # MAE: lower is better
            best = min(metric_rows, key=lambda x: x["mae"])
            worst = max(metric_rows, key=lambda x: x["mae"])

            rows.append(
                {
                    "run_label": label,
                    "cluster_id": cluster_id,
                    "best_client_id": best["client_id"],
                    "best_mae": best["mae"],
                    "worst_client_id": worst["client_id"],
                    "worst_mae": worst["mae"],
                    "mae_spread": worst["mae"] - best["mae"],
                }
            )

    return rows
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare multiple output-* run directories.")
    p.add_argument("--runs", type=str, default=",".join(DEFAULT_RUNS), help="Comma-separated output directories to compare")
    p.add_argument("--output-dir", type=Path, default=Path("comparison_outputs") / "run_compare", help="Directory for comparison outputs")
    p.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS), help="Comma-separated test metrics to compare")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = parse_csv_list(args.runs)
    metrics = parse_csv_list(args.metrics)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    log(f"Comparing runs: {run_dirs}")
    test_frames, train_frames = load_runs(run_dirs)
    overall_rows = build_overall_summary(test_frames, train_frames, metrics)
    cluster_rows = build_cluster_summary(test_frames, train_frames, metrics)
    client_rows = build_client_alignment(test_frames, train_frames, metrics)
    winner_rows = build_winners(client_rows, metrics)
    cluster_best_worst_rows = build_cluster_best_worst_clients(test_frames)
    if overall_rows:
        write_csv(output_dir / "overall_summary.csv", overall_rows, overall_rows[0].keys())
    if client_rows:
        write_csv(output_dir / "client_alignment.csv", client_rows, client_rows[0].keys())
    if winner_rows:
        write_csv(output_dir / "winners.csv", winner_rows, winner_rows[0].keys())
    if cluster_best_worst_rows:
        write_csv(
        output_dir / "cluster_best_worst_clients.csv",
        cluster_best_worst_rows,
        cluster_best_worst_rows[0].keys(),
    )
    build_markdown_report(output_dir, run_dirs, overall_rows, cluster_rows, client_rows, metrics)
    log(f"Saved comparison outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
