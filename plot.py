"""
SplitFed Evaluation — Analysis & Visualization
================================================
Reads 4 CSV result files and produces:
  1. Console summary table (per-method & per-cluster stats)
  2. Bar charts comparing MAE, RMSE, MSE across methods and clusters
  3. Box plots showing per-client error distributions
  4. Heatmap of per-client MAE across methods
  5. Cluster-level breakdown plots

Usage:
    python splitfed_analysis.py

Expected CSV files (place in the same directory as this script,
or edit FILE_PATHS below):
    results_random_seed1.csv
    results_random_seed2.csv
    results_ordered.csv
    results_baseline.csv

Each CSV must contain at least these columns:
    cluster_id, client_id, mae, rmse, mse, mape
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit file paths if needed
# ─────────────────────────────────────────────

FILE_PATHS = {
    "Random Epoch":  "output-4/results/test_report.csv",
    "Random Batch":  "output/results/test_report.csv",
    "Ordered Epoch": "output-2/results/test_report.csv",
    "Baseline":      "output-1/results/test_report.csv",
}

OUTPUT_DIR = "plots"          # folder where figures are saved
METRICS    = ["mae", "rmse", "mse"]
CLUSTERS   = [1, 2, 3, 4]

# Palette — one color per method.
# Keys must match the keys in FILE_PATHS above.
# Any method not listed here gets a color auto-assigned from the fallback pool.
COLORS = {
    "Random Epoch":  "#378ADD",
    "Random Batch":  "#1D9E75",
    "Ordered Epoch": "#BA7517",
    "Baseline":      "#E24B4A",
}

_FALLBACK_COLORS = [
    "#7F77DD", "#639922", "#D4537E", "#888780",
    "#EF9F27", "#5DCAA5", "#D85A30", "#185FA5",
]

def get_color(method: str) -> str:
    """Return the configured color for a method, or auto-assign one."""
    if method in COLORS:
        return COLORS[method]
    idx = hash(method) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_data(file_paths: dict) -> dict[str, pd.DataFrame]:
    dfs = {}
    for label, path in file_paths.items():
        if not os.path.exists(path):
            print(f"  [WARNING] File not found: {path}  — skipping '{label}'")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        dfs[label] = df
        print(f"  Loaded '{label}': {len(df)} rows from {path}")
    if not dfs:
        sys.exit("No CSV files could be loaded. Check FILE_PATHS.")
    return dfs


def summary_stats(series: pd.Series) -> dict:
    return {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "p90":    series.quantile(0.90),
        "p10":    series.quantile(0.10),
        "min":    series.min(),
        "max":    series.max(),
    }


def print_table(title: str, rows: list[dict], col_order: list[str]):
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")
    # header
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in col_order}
    header = "  " + "  ".join(c.ljust(widths[c]) for c in col_order)
    print(header)
    print("  " + "  ".join("─" * widths[c] for c in col_order))
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in col_order))


def set_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F8F8F8",
        "axes.grid":         True,
        "grid.color":        "white",
        "grid.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.spines.bottom":False,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.framealpha": 0.9,
    })


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

def compute_global_stats(dfs: dict) -> pd.DataFrame:
    rows = []
    for label, df in dfs.items():
        for metric in METRICS:
            if metric not in df.columns:
                continue
            s = summary_stats(df[metric])
            rows.append({"method": label, "metric": metric, **s})
    return pd.DataFrame(rows)


def compute_cluster_stats(dfs: dict) -> pd.DataFrame:
    rows = []
    for label, df in dfs.items():
        for cl in CLUSTERS:
            sub = df[df["cluster_id"] == cl]
            if sub.empty:
                continue
            for metric in METRICS:
                if metric not in sub.columns:
                    continue
                s = summary_stats(sub[metric])
                rows.append({"method": label, "cluster": cl, "metric": metric, **s})
    return pd.DataFrame(rows)


def print_console_report(dfs: dict, global_stats: pd.DataFrame, cluster_stats: pd.DataFrame):
    print("\n" + "═" * 80)
    print("  SPLITFED EVALUATION REPORT")
    print("═" * 80)

    # Global summary
    mae_g = global_stats[global_stats["metric"] == "mae"].copy()
    rows = []
    baseline_mean = None
    for _, r in mae_g.iterrows():
        rows.append({
            "Method":      r["method"],
            "Mean MAE":    f"{r['mean']:.4f}",
            "Median MAE":  f"{r['median']:.4f}",
            "Std MAE":     f"{r['std']:.4f}",
            "P90 MAE":     f"{r['p90']:.4f}",
            "Mean RMSE":   f"{global_stats[(global_stats.method==r['method'])&(global_stats.metric=='rmse')]['mean'].values[0]:.4f}",
            "Clients":     str(len(dfs[r["method"]])),
        })
        if r["method"] == "Baseline":
            baseline_mean = r["mean"]

    print_table("GLOBAL SUMMARY — ALL CLUSTERS", rows,
                ["Method","Mean MAE","Median MAE","Std MAE","P90 MAE","Mean RMSE","Clients"])

    # Improvement vs baseline
    if baseline_mean:
        print(f"\n  Improvement vs Baseline (Mean MAE):")
        for _, r in mae_g.iterrows():
            if r["method"] == "Baseline":
                continue
            delta = (baseline_mean - r["mean"]) / baseline_mean * 100
            sign = "+" if delta > 0 else ""
            print(f"    {r['method']:<22}: {sign}{delta:.1f}%")

    # Per-cluster
    for cl in CLUSTERS:
        cl_rows = cluster_stats[(cluster_stats["cluster"] == cl) & (cluster_stats["metric"] == "mae")]
        rows = []
        for _, r in cl_rows.iterrows():
            rows.append({
                "Method":     r["method"],
                "Mean MAE":   f"{r['mean']:.4f}",
                "Median MAE": f"{r['median']:.4f}",
                "P90 MAE":    f"{r['p90']:.4f}",
                "Mean RMSE":  f"{cluster_stats[(cluster_stats.method==r['method'])&(cluster_stats.cluster==cl)&(cluster_stats.metric=='rmse')]['mean'].values[0]:.4f}",
            })
        print_table(f"CLUSTER {cl}", rows,
                    ["Method","Mean MAE","Median MAE","P90 MAE","Mean RMSE"])

    # MAPE note
    print("\n  NOTE: MAPE is excluded from primary analysis — several clients with near-zero")
    print("  consumption produce astronomically inflated values (e.g. >1,000,000%). Use MAE/RMSE.")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def bar_global_metrics(dfs: dict, global_stats: pd.DataFrame, out_dir: str):
    """Figure 1 — Grouped bar chart: mean MAE, RMSE, MSE across all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Figure 1 — Global Mean Metrics by Method (all clusters)", fontsize=12, fontweight="bold", y=1.02)

    methods = list(dfs.keys())
    x = np.arange(len(methods))
    colors = [get_color(m) for m in methods]

    for ax, metric in zip(axes, METRICS):
        vals = [global_stats[(global_stats.method == m) & (global_stats.metric == metric)]["mean"].values[0]
                for m in methods]
        bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
        ax.set_title(f"Mean {metric.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(metric.upper())
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_global_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def bar_cluster_mae_rmse(dfs: dict, cluster_stats: pd.DataFrame, out_dir: str):
    """Figure 2 — Grouped bar chart: MAE and RMSE per cluster, grouped by method."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Figure 2 — Per-Cluster Mean MAE and RMSE by Method", fontsize=12, fontweight="bold")

    methods = list(dfs.keys())
    n = len(methods)
    x = np.arange(n)
    colors = [get_color(m) for m in methods]

    for col, cl in enumerate(CLUSTERS):
        for row, metric in enumerate(["mae", "rmse"]):
            ax = axes[row][col]
            vals = []
            for m in methods:
                v = cluster_stats[(cluster_stats.method == m) &
                                   (cluster_stats.cluster == cl) &
                                   (cluster_stats.metric == metric)]["mean"]
                vals.append(v.values[0] if len(v) else 0)
            bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
            ax.set_title(f"Cluster {cl} — {metric.upper()}")
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=7)
            ax.set_ylabel(metric.upper())
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # Legend
    patches = [mpatches.Patch(color=get_color(m), label=m) for m in methods]
    fig.legend(handles=patches, loc="lower center", ncol=n, bbox_to_anchor=(0.5, -0.03), fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_cluster_mae_rmse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def bar_cluster_medians(dfs: dict, cluster_stats: pd.DataFrame, out_dir: str):
    """Figure 3 — Median MAE per cluster (grouped bar, methods side-by-side)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Figure 3 — Median MAE by Cluster and Method", fontsize=12, fontweight="bold")

    methods = list(dfs.keys())
    n = len(methods)
    width = 0.18
    x = np.arange(len(CLUSTERS))

    for i, m in enumerate(methods):
        vals = []
        for cl in CLUSTERS:
            v = cluster_stats[(cluster_stats.method == m) &
                               (cluster_stats.cluster == cl) &
                               (cluster_stats.metric == "mae")]["median"]
            vals.append(v.values[0] if len(v) else 0)
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, color=get_color(m),
                      label=m, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in CLUSTERS])
    ax.set_ylabel("Median MAE")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_median_mae_by_cluster.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def boxplots_mae(dfs: dict, out_dir: str):
    """Figure 4 — Box plots of per-client MAE distribution by method."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    fig.suptitle("Figure 4 — Per-Client MAE Distribution by Cluster (box plots)", fontsize=12, fontweight="bold")

    methods = list(dfs.keys())
    colors = [get_color(m) for m in methods]

    for col, cl in enumerate(CLUSTERS):
        ax = axes[col]
        data = []
        for m in methods:
            df = dfs[m]
            sub = df[df["cluster_id"] == cl]["mae"].dropna()
            data.append(sub.values)

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_title(f"Cluster {cl}")
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=7)
        ax.set_ylabel("MAE" if col == 0 else "")

    patches = [mpatches.Patch(color=get_color(m), label=m) for m in methods]
    fig.legend(handles=patches, loc="lower center", ncol=len(methods),
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_boxplots_mae.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def heatmap_client_mae(dfs: dict, out_dir: str):
    """Figure 5 — Heatmap: per-client MAE for each method."""
    methods = list(dfs.keys())
    # Use the first method's client list as index
    ref_df = list(dfs.values())[0][["cluster_id", "client_id"]].copy()
    ref_df = ref_df.sort_values(["cluster_id", "client_id"]).reset_index(drop=True)
    ref_df["label"] = ref_df["cluster_id"].astype(str) + "/" + ref_df["client_id"].astype(str)

    matrix = pd.DataFrame(index=ref_df["label"])
    for m, df in dfs.items():
        df2 = df.copy()
        df2["label"] = df2["cluster_id"].astype(str) + "/" + df2["client_id"].astype(str)
        df2 = df2.set_index("label")["mae"]
        matrix[m] = ref_df["label"].map(df2).values

    # Cap at 95th percentile for visibility
    cap = matrix.stack().quantile(0.95)
    matrix_capped = matrix.clip(upper=cap)

    fig, ax = plt.subplots(figsize=(10, max(6, len(ref_df) * 0.22)))
    fig.suptitle("Figure 5 — Per-Client MAE Heatmap (capped at 95th pct for visibility)",
                 fontsize=11, fontweight="bold")

    im = ax.imshow(matrix_capped.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(ref_df)))
    ax.set_yticklabels(ref_df["label"], fontsize=6)

    # Cluster separators
    cluster_ids = ref_df["cluster_id"].values
    for i in range(1, len(cluster_ids)):
        if cluster_ids[i] != cluster_ids[i-1]:
            ax.axhline(i - 0.5, color="white", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="MAE", pad=0.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_heatmap_client_mae.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def bar_improvement_vs_baseline(dfs: dict, global_stats: pd.DataFrame,
                                 cluster_stats: pd.DataFrame, out_dir: str):
    """Figure 6 — % improvement in mean MAE vs Baseline, globally and per cluster."""
    if "Baseline" not in dfs:
        print("  [SKIP] Figure 6 requires 'Baseline' in FILE_PATHS.")
        return

    methods_no_base = [m for m in dfs.keys() if m != "Baseline"]
    colors_no_base = [get_color(m) for m in methods_no_base]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    fig.suptitle("Figure 6 — % Mean MAE Improvement vs Baseline", fontsize=12, fontweight="bold")

    scopes = [("All clusters", None)] + [(f"Cluster {cl}", cl) for cl in CLUSTERS]
    for ax, (title, cl) in zip(axes, scopes):
        vals = []
        for m in methods_no_base:
            if cl is None:
                base = global_stats[(global_stats.method=="Baseline")&(global_stats.metric=="mae")]["mean"].values[0]
                mv   = global_stats[(global_stats.method==m)&(global_stats.metric=="mae")]["mean"].values[0]
            else:
                base = cluster_stats[(cluster_stats.method=="Baseline")&(cluster_stats.cluster==cl)&(cluster_stats.metric=="mae")]["mean"].values[0]
                mv   = cluster_stats[(cluster_stats.method==m)&(cluster_stats.cluster==cl)&(cluster_stats.metric=="mae")]["mean"].values[0]
            vals.append((base - mv) / base * 100)

        x = np.arange(len(methods_no_base))
        bars = ax.bar(x, vals, color=colors_no_base, width=0.5, edgecolor="white")
        ax.axhline(0, color="#333", linewidth=0.7, linestyle="--")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods_no_base], fontsize=7)
        ax.set_ylabel("% improvement" if ax == axes[0] else "")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.5 if val >= 0 else -2),
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_improvement_vs_baseline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def bar_p90_comparison(dfs: dict, cluster_stats: pd.DataFrame, out_dir: str):
    """Figure 7 — P90 MAE by method and cluster (tail-error comparison)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle("Figure 7 — P90 MAE by Cluster and Method (tail error)", fontsize=12, fontweight="bold")

    methods = list(dfs.keys())
    n = len(methods)
    width = 0.18
    x = np.arange(len(CLUSTERS))

    for i, m in enumerate(methods):
        vals = []
        for cl in CLUSTERS:
            v = cluster_stats[(cluster_stats.method == m) &
                               (cluster_stats.cluster == cl) &
                               (cluster_stats.metric == "mae")]["p90"]
            vals.append(v.values[0] if len(v) else 0)
        offset = (i - n/2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, color=get_color(m),
               label=m, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in CLUSTERS])
    ax.set_ylabel("P90 MAE")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig7_p90_mae_by_cluster.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# CATASTROPHIC FORGETTING ANALYSIS
# ─────────────────────────────────────────────

def assign_training_position(df: pd.DataFrame, cluster_col: str = "cluster_id",
                              client_col: str = "client_id") -> pd.DataFrame:
    """
    Within each cluster, rank clients by client_id to approximate training order
    (first client seen = rank 1).  Then label the bottom half 'Early' and the
    top half 'Late'.  Returns a copy of df with two new columns:
        train_rank   – integer rank within cluster (1-based)
        train_group  – 'Early' | 'Late'
    """
    df = df.copy()
    df["train_rank"] = (
        df.groupby(cluster_col)[client_col]
          .rank(method="first", ascending=True)
          .astype(int)
    )
    half = df.groupby(cluster_col)["train_rank"].transform("max") / 2
    df["train_group"] = np.where(df["train_rank"] <= half, "Early", "Late")
    return df


def print_forgetting_report(dfs: dict):
    """Console table: mean MAE for Early vs Late clients, and the gap."""
    print(f"\n{'═'*80}")
    print("  CATASTROPHIC FORGETTING ANALYSIS")
    print(f"{'═'*80}")
    print("  Logic: within each cluster clients are sorted by client_id to proxy")
    print("  training order.  The first 50% are 'Early', the last 50% are 'Late'.")
    print("  Forgetting signal: Late MAE − Early MAE  (positive = forgetting present)")

    for cl in CLUSTERS:
        rows = []
        for label, df in dfs.items():
            sub = assign_training_position(df[df["cluster_id"] == cl].copy())
            early_mae = sub[sub["train_group"] == "Early"]["mae"].mean()
            late_mae  = sub[sub["train_group"] == "Late"]["mae"].mean()
            gap       = late_mae - early_mae
            pct       = (gap / early_mae * 100) if early_mae else float("nan")
            rows.append({
                "Method":    label,
                "Early MAE": f"{early_mae:.4f}",
                "Late MAE":  f"{late_mae:.4f}",
                "Gap (L-E)": f"{gap:+.4f}",
                "Gap %":     f"{pct:+.1f}%",
                "Signal":    "⚠ forgetting" if gap > 0.1 else ("✓ stable" if gap <= 0 else "~ mild"),
            })
        print_table(f"CLUSTER {cl}", rows,
                    ["Method","Early MAE","Late MAE","Gap (L-E)","Gap %","Signal"])


def fig_forgetting_bars(dfs: dict, out_dir: str):
    """
    Figure 8 — Grouped bar: Early vs Late mean MAE per cluster per method.
    Side-by-side bars; Early = solid fill, Late = hatched fill of same color.
    """
    methods = list(dfs.keys())
    n_methods = len(methods)
    n_clusters = len(CLUSTERS)

    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4.5), sharey=False)
    if n_clusters == 1:
        axes = [axes]
    fig.suptitle("Figure 8 — Early vs Late Client MAE (catastrophic forgetting proxy)",
                 fontsize=12, fontweight="bold")

    width = 0.35
    x = np.arange(n_methods)

    for ax, cl in zip(axes, CLUSTERS):
        early_vals, late_vals = [], []
        for label, df in dfs.items():
            sub = assign_training_position(df[df["cluster_id"] == cl].copy())
            early_vals.append(sub[sub["train_group"] == "Early"]["mae"].mean())
            late_vals.append(sub[sub["train_group"] == "Late"]["mae"].mean())

        colors = [get_color(m) for m in methods]
        bars_e = ax.bar(x - width/2, early_vals, width, color=colors,
                        alpha=0.9, label="Early", edgecolor="white", linewidth=0.5)
        bars_l = ax.bar(x + width/2, late_vals,  width, color=colors,
                        alpha=0.45, hatch="///", label="Late",
                        edgecolor=[get_color(m) for m in methods], linewidth=0.8)

        # Gap annotation above each pair
        for i, (ev, lv) in enumerate(zip(early_vals, late_vals)):
            gap = lv - ev
            top = max(ev, lv) * 1.04
            color = "#A32D2D" if gap > 0.1 else ("#3B6D11" if gap <= 0 else "#BA7517")
            ax.text(x[i], top, f"{gap:+.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold")

        ax.set_title(f"Cluster {cl}")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=7.5)
        ax.set_ylabel("Mean MAE" if cl == CLUSTERS[0] else "")

    # Shared legend
    solid_patch  = mpatches.Patch(facecolor="#888", alpha=0.9, label="Early clients (first 50%)")
    hatch_patch  = mpatches.Patch(facecolor="#888", alpha=0.45, hatch="///",
                                  edgecolor="#888", label="Late clients (last 50%)")
    fig.legend(handles=[solid_patch, hatch_patch], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig8_forgetting_early_vs_late.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_forgetting_gap_heatmap(dfs: dict, out_dir: str):
    """
    Figure 9 — Heatmap: forgetting gap (Late MAE − Early MAE) for every
    method × cluster combination.  Red = forgetting, green = improvement.
    """
    methods  = list(dfs.keys())
    gap_data = np.zeros((len(methods), len(CLUSTERS)))

    for j, cl in enumerate(CLUSTERS):
        for i, (label, df) in enumerate(dfs.items()):
            sub = assign_training_position(df[df["cluster_id"] == cl].copy())
            early = sub[sub["train_group"] == "Early"]["mae"].mean()
            late  = sub[sub["train_group"] == "Late"]["mae"].mean()
            gap_data[i, j] = late - early

    fig, ax = plt.subplots(figsize=(6, max(3, len(methods) * 0.7 + 1.5)))
    fig.suptitle("Figure 9 — Forgetting Gap Heatmap  (Late MAE − Early MAE)\n"
                 "Red = forgetting  |  Green = late clients improved",
                 fontsize=11, fontweight="bold")

    vmax = max(abs(gap_data.min()), abs(gap_data.max()), 0.5)
    im = ax.imshow(gap_data, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(CLUSTERS)))
    ax.set_xticklabels([f"Cluster {c}" for c in CLUSTERS])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(CLUSTERS)):
            val = gap_data[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Late − Early MAE", pad=0.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig9_forgetting_gap_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_forgetting_boxplots(dfs: dict, out_dir: str):
    """
    Figure 10 — Box plots: per-client MAE distribution split by Early/Late,
    one subplot per cluster, one pair of boxes per method.
    """
    methods   = list(dfs.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, len(CLUSTERS), figsize=(4 * len(CLUSTERS), 4.5), sharey=False)
    if len(CLUSTERS) == 1:
        axes = [axes]
    fig.suptitle("Figure 10 — Per-Client MAE Distribution: Early vs Late (box plots)",
                 fontsize=12, fontweight="bold")

    for ax, cl in zip(axes, CLUSTERS):
        positions, box_data, box_colors, labels_x = [], [], [], []
        tick_centers = []

        for i, (label, df) in enumerate(dfs.items()):
            sub  = assign_training_position(df[df["cluster_id"] == cl].copy())
            early = sub[sub["train_group"] == "Early"]["mae"].dropna().values
            late  = sub[sub["train_group"] == "Late"]["mae"].dropna().values

            base = i * 3
            positions += [base + 0.7, base + 1.3]
            box_data  += [early, late]
            col = get_color(label)
            box_colors += [col, col]
            tick_centers.append(base + 1.0)

        bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                        patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=3, alpha=0.4))

        for patch, col, idx in zip(bp["boxes"], box_colors, range(len(box_colors))):
            patch.set_facecolor(col)
            patch.set_alpha(0.9 if idx % 2 == 0 else 0.45)

        ax.set_xticks(tick_centers)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=7.5)
        ax.set_title(f"Cluster {cl}")
        ax.set_ylabel("MAE" if cl == CLUSTERS[0] else "")

    solid_patch = mpatches.Patch(facecolor="#888", alpha=0.9,  label="Early")
    light_patch = mpatches.Patch(facecolor="#888", alpha=0.45, label="Late")
    fig.legend(handles=[solid_patch, light_patch], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig10_forgetting_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_forgetting_rank_scatter(dfs: dict, out_dir: str):
    """
    Figure 11 — Scatter: training rank (x) vs MAE (y) per client, one subplot
    per method.  A positive trend line indicates forgetting.
    """
    methods = list(dfs.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Figure 11 — Training Rank vs MAE  (positive slope = forgetting)",
                 fontsize=12, fontweight="bold")

    cluster_markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    cluster_palette = {1: "#378ADD", 2: "#1D9E75", 3: "#BA7517", 4: "#E24B4A"}

    for ax, label in zip(axes, methods):
        df = dfs[label]
        all_sub = assign_training_position(df.copy())

        for cl in CLUSTERS:
            sub = all_sub[all_sub["cluster_id"] == cl]
            ax.scatter(sub["train_rank"], sub["mae"],
                       color=cluster_palette[cl], marker=cluster_markers[cl],
                       alpha=0.65, s=28, label=f"Cluster {cl}")

        # Overall trend line across all clusters
        ranks = all_sub["train_rank"].values
        maes  = all_sub["mae"].values
        if len(ranks) > 1:
            z = np.polyfit(ranks, maes, 1)
            p = np.poly1d(z)
            xs = np.linspace(ranks.min(), ranks.max(), 100)
            ax.plot(xs, p(xs), color="#333", linewidth=1.2, linestyle="--",
                    label=f"trend  slope={z[0]:+.3f}")

        ax.set_title(label)
        ax.set_xlabel("Training rank (within cluster)")
        ax.set_ylabel("MAE" if label == methods[0] else "")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig11_forgetting_rank_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    set_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "═"*80)
    print("  Loading CSVs …")
    print("═"*80)
    dfs = load_data(FILE_PATHS)

    print("\n" + "═"*80)
    print("  Computing statistics …")
    print("═"*80)
    global_stats  = compute_global_stats(dfs)
    cluster_stats = compute_cluster_stats(dfs)

    print_console_report(dfs, global_stats, cluster_stats)

    # ── Catastrophic forgetting ──────────────────
    print_forgetting_report(dfs)

    print("\n" + "═"*80)
    print("  Generating plots …")
    print("═"*80)
    bar_global_metrics(dfs, global_stats, OUTPUT_DIR)
    bar_cluster_mae_rmse(dfs, cluster_stats, OUTPUT_DIR)
    bar_cluster_medians(dfs, cluster_stats, OUTPUT_DIR)
    boxplots_mae(dfs, OUTPUT_DIR)
    heatmap_client_mae(dfs, OUTPUT_DIR)
    bar_improvement_vs_baseline(dfs, global_stats, cluster_stats, OUTPUT_DIR)
    bar_p90_comparison(dfs, cluster_stats, OUTPUT_DIR)

    # ── Catastrophic forgetting plots ───────────
    fig_forgetting_bars(dfs, OUTPUT_DIR)
    fig_forgetting_gap_heatmap(dfs, OUTPUT_DIR)
    fig_forgetting_boxplots(dfs, OUTPUT_DIR)
    fig_forgetting_rank_scatter(dfs, OUTPUT_DIR)

    print(f"\n  Done. All figures saved to → ./{OUTPUT_DIR}/")
    print("═"*80 + "\n")


if __name__ == "__main__":
    main()