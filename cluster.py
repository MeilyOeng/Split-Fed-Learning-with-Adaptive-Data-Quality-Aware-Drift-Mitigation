"""
Cluster Profiling & Analysis
=============================
Explains WHY clusters behave differently in the forgetting analysis
by profiling the raw consumption time-series data of each client.

Input files:
  - consumption CSV  : wide format — rows = timestamps, cols = client IDs
                       first column must be the timestamp column (any name, e.g. Fecha_Hora)
                       e.g.:  Fecha_Hora, 9906537, 9906758, ...
                              2023-01-01 00:00, 12.3, 0.4, ...

  - cluster mapping  : two columns — client_id, cluster_id
                       e.g.:  client_id, cluster_id
                              9906537, 1
                              9906758, 2

Outputs (saved to OUTPUT_DIR):
  Figure 1  — Client count per cluster
  Figure 2  — Mean vs median consumption per cluster (outlier dominance)
  Figure 3  — Consumption distribution per cluster (violin + box)
  Figure 4  — Coefficient of variation per client, grouped by cluster
  Figure 5  — Mean daily load profile per cluster (avg consumption by hour)
  Figure 6  — Rolling variability over time per cluster
  Figure 7  — Near-zero consumption / sparsity per client per cluster
  Figure 8  — Intra-cluster client correlation heatmap (heterogeneity)
  Figure 9  — Per-client consumption range (min / mean / max)
  Figure 10 — Temporal trend per cluster (linear fit over time)

Usage:
    python cluster_analysis.py

Edit the CONFIG section below to match your file paths.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit these paths
# ─────────────────────────────────────────────

CONSUMPTION_CSV  = "dataset/dataset.csv"   # wide-format: rows=timestamps, cols=client_ids
CLUSTER_MAP_CSV  = "dataset/cluster-assignment.csv"   # columns: client_id, cluster_id
TIMESTAMP_COL    = "Fecha_Hora"             # name of the timestamp column in consumption CSV
OUTPUT_DIR       = "plots/cluster_analysis"

# If your timestamp has a non-standard format supply a strftime string,
# e.g. "%d/%m/%Y %H:%M"  — set None to let pandas infer it automatically
TIMESTAMP_FORMAT = None

# Readings <= this value are treated as "near-zero" (same unit as your data)
NEAR_ZERO_THRESHOLD = 0.5

# Rolling window for variability-over-time plot (number of time steps)
# e.g. 24*7 = 1 week of hourly data
ROLLING_WINDOW = 24 * 7

# Cluster colour palette
CLUSTER_COLORS  = {1: "#378ADD", 2: "#1D9E75", 3: "#BA7517", 4: "#E24B4A"}
_FALLBACK_COLORS = ["#7F77DD", "#639922", "#D4537E", "#888780"]


def cluster_color(cl):
    if cl in CLUSTER_COLORS:
        return CLUSTER_COLORS[cl]
    return _FALLBACK_COLORS[int(cl) % len(_FALLBACK_COLORS)]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_inputs():
    """Load and validate both CSVs. Returns (long_df, cmap_df, wide_df)."""

    # ── cluster map ──────────────────────────────────────────
    if not os.path.exists(CLUSTER_MAP_CSV):
        sys.exit(f"Cluster map not found: {CLUSTER_MAP_CSV}")
    cmap = pd.read_csv(CLUSTER_MAP_CSV)
    cmap.columns = [c.strip().lower() for c in cmap.columns]
    rename = {}
    for c in cmap.columns:
        if "client" in c or c == "id":
            rename[c] = "client_id"
        elif "cluster" in c:
            rename[c] = "cluster_id"

    cmap = cmap.rename(columns=rename)

    cmap["client_id"] = cmap["client_id"].astype(str).str.strip()
    cmap["cluster_id"] = cmap["cluster_id"].astype(int)
    print(f"  Cluster map : {len(cmap)} clients across "
          f"{cmap['cluster_id'].nunique()} clusters {sorted(cmap['cluster_id'].unique())}")

    # ── consumption ──────────────────────────────────────────
    if not os.path.exists(CONSUMPTION_CSV):
        sys.exit(f"Consumption file not found: {CONSUMPTION_CSV}")
    raw = pd.read_csv(CONSUMPTION_CSV)
    raw.columns = [c.strip() for c in raw.columns]
    # Convert all client columns to numeric safely
    raw.iloc[:, 1:] = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    # Now invert sign
    raw.iloc[:, 1:] = -raw.iloc[:, 1:]

    ts_col = TIMESTAMP_COL
    if ts_col not in raw.columns:
        ts_col = raw.columns[0]
        print(f"  [WARNING] '{TIMESTAMP_COL}' not found — using '{ts_col}' as timestamp")

    raw[ts_col] = pd.to_datetime(raw[ts_col], format=TIMESTAMP_FORMAT,
                                 infer_datetime_format=True)
    raw = raw.set_index(ts_col).sort_index()

    valid_clients = set(cmap["client_id"].values)
    keep = [c for c in raw.columns if str(c).strip() in valid_clients]
    missing = valid_clients - set(str(c).strip() for c in raw.columns)
    if missing:
        print(f"  [WARNING] {len(missing)} cluster-map clients absent from consumption CSV")
    raw = raw[keep]
    print(f"  Consumption : {len(raw)} timesteps × {raw.shape[1]} clients  "
          f"({raw.index[0]} → {raw.index[-1]})")

    # Melt to long format
    long = (raw.reset_index()
               .melt(id_vars=ts_col, var_name="client_id", value_name="consumption"))
    long.columns = ["timestamp", "client_id", "consumption"]
    long["client_id"]   = long["client_id"].astype(str).str.strip()
    long["consumption"] = pd.to_numeric(long["consumption"], errors="coerce")
    long = long.dropna(subset=["consumption"])
    long = long.merge(cmap[["client_id", "cluster_id"]], on="client_id", how="left")

    return long, cmap, raw


# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "#F8F8F8",
        "axes.grid":          True,
        "grid.color":         "white",
        "grid.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": False,
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
    })


# ─────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────

def print_cluster_report(long: pd.DataFrame, cmap: pd.DataFrame):
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))

    print(f"\n{'═'*80}")
    print("  CLUSTER PROFILING REPORT")
    print(f"{'═'*80}")
    print(f"  {'Cluster':<10} {'Clients':<10} {'Mean':<10} {'Median':<10} "
          f"{'Std':<10} {'CV':<8} {'NearZero%':<12} {'Skew':<8}")
    print(f"  {'─'*78}")
    for cl in clusters:
        sub = long[long["cluster_id"] == cl]["consumption"]
        n   = cmap[cmap["cluster_id"] == cl].shape[0]
        m   = sub.mean()
        med = sub.median()
        s   = sub.std()
        cv  = s / m if m > 0 else float("nan")
        nz  = (sub <= NEAR_ZERO_THRESHOLD).mean() * 100
        sk  = sub.skew()
        print(f"  {cl:<10} {n:<10} {m:<10.3f} {med:<10.3f} "
              f"{s:<10.3f} {cv:<8.3f} {nz:<12.1f} {sk:<8.3f}")

    print(f"\n  CV        = std / mean  (>1 means highly variable — hard to predict)")
    print(f"  NearZero% = % readings ≤ {NEAR_ZERO_THRESHOLD}  (inflates MAPE, destabilises gradients)")
    print(f"  Skew > 1  = heavy right tail (a few very large consumers dominate the cluster)")

    print(f"\n{'─'*80}")
    print("  PER-CLIENT SUMMARY (sorted by mean consumption, within each cluster)")
    print(f"{'─'*80}")
    for cl in clusters:
        sub = long[long["cluster_id"] == cl]
        cs  = (sub.groupby("client_id")["consumption"]
                  .agg(mean="mean", std="std", min="min", max="max",
                       near_zero=lambda x: (x <= NEAR_ZERO_THRESHOLD).mean() * 100)
                  .sort_values("mean", ascending=False))
        cs["cv"] = cs["std"] / cs["mean"]
        print(f"\n  Cluster {cl}:")
        print(f"  {'Client':<14} {'Mean':>8} {'Std':>8} {'Min':>8} "
              f"{'Max':>8} {'CV':>8} {'NearZero%':>10}")
        print(f"  {'─'*66}")
        for cid, row in cs.iterrows():
            print(f"  {str(cid):<14} {row['mean']:>8.3f} {row['std']:>8.3f} "
                  f"{row['min']:>8.3f} {row['max']:>8.3f} "
                  f"{row['cv']:>8.3f} {row['near_zero']:>9.1f}%")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def _save(fig, out_dir: str, fname: str):
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig1_cluster_sizes(cmap: pd.DataFrame, out_dir: str):
    clusters = sorted(cmap["cluster_id"].unique())
    counts   = [cmap[cmap["cluster_id"] == cl].shape[0] for cl in clusters]
    colors   = [cluster_color(cl) for cl in clusters]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.suptitle("Figure 1 — Client Count per Cluster", fontsize=12, fontweight="bold")
    bars = ax.bar([f"Cluster {c}" for c in clusters], counts,
                  color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of clients")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_cluster_sizes.png")


def fig2_mean_consumption(long: pd.DataFrame, out_dir: str):
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))
    means    = [long[long["cluster_id"]==cl]["consumption"].mean()   for cl in clusters]
    medians  = [long[long["cluster_id"]==cl]["consumption"].median() for cl in clusters]
    stds     = [long[long["cluster_id"]==cl]["consumption"].std()    for cl in clusters]
    x        = np.arange(len(clusters))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Figure 2 — Mean vs Median Consumption per Cluster\n"
                 "(large gap = right-skewed distribution / outlier clients driving the mean up)",
                 fontsize=11, fontweight="bold")
    ax.bar(x - width/2, means,   width, color=[cluster_color(c) for c in clusters],
           alpha=0.9, label="Mean", edgecolor="white",
           yerr=stds, capsize=4, error_kw={"elinewidth": 1, "alpha": 0.5})
    ax.bar(x + width/2, medians, width, color=[cluster_color(c) for c in clusters],
           alpha=0.45, hatch="///", label="Median",
           edgecolor=[cluster_color(c) for c in clusters], linewidth=0.8)
    for i, (m, med, s) in enumerate(zip(means, medians, stds)):
        ax.text(x[i], max(m, med) + s * 1.1 + 0.1,
                f"gap={m-med:.2f}", ha="center", fontsize=8, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_ylabel("Consumption")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig2_mean_consumption.png")


def fig3_violin(long: pd.DataFrame, out_dir: str):
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))
    data     = [long[long["cluster_id"]==cl]["consumption"].dropna().values
                for cl in clusters]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Figure 3 — Consumption Distribution per Cluster (violin + box)\n"
                 "(wide violin = high variance;  long upper tail = dominant outlier clients)",
                 fontsize=11, fontweight="bold")
    parts = ax.violinplot(data, positions=range(len(clusters)),
                          showmedians=True, showextrema=False)
    for pc, cl in zip(parts["bodies"], clusters):
        pc.set_facecolor(cluster_color(cl))
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    bp = ax.boxplot(data, positions=range(len(clusters)), widths=0.08,
                    patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8, color="#555"),
                    capprops=dict(linewidth=0.8, color="#555"),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3, color="#999"))
    for patch, cl in zip(bp["boxes"], clusters):
        patch.set_facecolor(cluster_color(cl))
        patch.set_alpha(0.85)

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_ylabel("Consumption")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_violin_distribution.png")


def fig4_cv_per_client(long: pd.DataFrame, out_dir: str):
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))
    all_cvs, all_colors, all_labels = [], [], []
    for cl in clusters:
        sub = long[long["cluster_id"] == cl]
        for cid, grp in sub.groupby("client_id"):
            m  = grp["consumption"].mean()
            s  = grp["consumption"].std()
            cv = s / m if m > 0 else 0
            all_cvs.append(cv)
            all_colors.append(cluster_color(cl))
            all_labels.append(str(cid))

    fig, ax = plt.subplots(figsize=(max(10, len(all_cvs) * 0.25), 4.5))
    fig.suptitle("Figure 4 — Coefficient of Variation per Client (std / mean)\n"
                 "(CV > 1 = consumption is more variable than its own average — hard to predict)",
                 fontsize=11, fontweight="bold")
    ax.bar(range(len(all_cvs)), all_cvs, color=all_colors,
           width=0.7, edgecolor="white", linewidth=0.3)
    ax.axhline(1.0, color="#E24B4A", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xticks(range(len(all_cvs)))
    ax.set_xticklabels(all_labels, rotation=90, fontsize=6)
    ax.set_ylabel("CV (std / mean)")

    patches = [mpatches.Patch(color=cluster_color(cl), label=f"Cluster {cl}")
               for cl in clusters]
    patches.append(mpatches.Patch(color="#E24B4A", label="CV = 1 threshold"))
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_cv_per_client.png")


def fig5_daily_load_profile(long: pd.DataFrame, out_dir: str):
    if not pd.api.types.is_datetime64_any_dtype(long["timestamp"]):
        print("  [SKIP Fig 5] timestamp column is not datetime — cannot extract hour")
        return
    long = long.copy()
    long["hour"] = long["timestamp"].dt.hour
    clusters     = sorted(long["cluster_id"].dropna().unique().astype(int))

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Figure 5 — Mean Daily Load Profile by Cluster\n"
                 "(diverging shapes = different usage patterns driving inter-cluster heterogeneity)",
                 fontsize=11, fontweight="bold")
    for cl in clusters:
        sub     = long[long["cluster_id"] == cl]
        profile = sub.groupby("hour")["consumption"].mean()
        std_b   = sub.groupby("hour")["consumption"].std()
        ax.plot(profile.index, profile.values, color=cluster_color(cl),
                linewidth=2, marker="o", markersize=3, label=f"Cluster {cl}")
        ax.fill_between(profile.index,
                        profile.values - std_b.values,
                        profile.values + std_b.values,
                        alpha=0.12, color=cluster_color(cl))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean consumption")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig5_daily_load_profile.png")


def fig6_rolling_variability(long: pd.DataFrame, out_dir: str):
    if not pd.api.types.is_datetime64_any_dtype(long["timestamp"]):
        print("  [SKIP Fig 6] timestamp not datetime")
        return
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))
    fig, axes = plt.subplots(len(clusters), 1,
                             figsize=(12, 2.8 * len(clusters)), sharex=True)
    if len(clusters) == 1:
        axes = [axes]
    fig.suptitle(f"Figure 6 — Rolling Variability Over Time per Cluster  "
                 f"(window = {ROLLING_WINDOW} steps)\n"
                 "(spikes = high-variance periods that increase prediction difficulty)",
                 fontsize=11, fontweight="bold")
    for ax, cl in zip(axes, clusters):
        sub      = long[long["cluster_id"] == cl]
        ts       = sub.groupby("timestamp")["consumption"].mean().sort_index()
        roll_std = ts.rolling(window=ROLLING_WINDOW, min_periods=1).std()
        ax.fill_between(ts.index, 0, roll_std.values,
                        color=cluster_color(cl), alpha=0.45)
        ax.plot(ts.index, roll_std.values, color=cluster_color(cl), linewidth=0.8)
        ax.set_ylabel(f"Cluster {cl}\nroll. std", fontsize=9)
        ax.set_yticks([])
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    _save(fig, out_dir, "fig6_rolling_variability.png")


def fig7_sparsity(long: pd.DataFrame, out_dir: str):
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))
    fig, axes = plt.subplots(1, len(clusters),
                             figsize=(4.5 * len(clusters), 4.5), sharey=False)
    if len(clusters) == 1:
        axes = [axes]
    fig.suptitle(f"Figure 7 — Near-Zero Consumption per Client  (threshold ≤ {NEAR_ZERO_THRESHOLD})\n"
                 "(sparse clients inflate MAPE and produce unstable gradients during training)",
                 fontsize=11, fontweight="bold")
    for ax, cl in zip(axes, clusters):
        sub = long[long["cluster_id"] == cl]
        sparsity = (sub.groupby("client_id")["consumption"]
                       .apply(lambda x: (x <= NEAR_ZERO_THRESHOLD).mean() * 100)
                       .sort_values(ascending=False))
        ax.bar(range(len(sparsity)), sparsity.values,
               color=cluster_color(cl), width=0.6, edgecolor="white")
        ax.axhline(10, color="#E24B4A", linewidth=1, linestyle="--", alpha=0.7,
                   label=">10% threshold")
        ax.set_title(f"Cluster {cl}")
        ax.set_xlabel("Clients (sorted by sparsity)")
        ax.set_ylabel("% near-zero readings" if cl == clusters[0] else "")
        ax.set_xticks(range(len(sparsity)))
        ax.set_xticklabels([str(c)[-6:] for c in sparsity.index],
                           rotation=90, fontsize=6)
        n_sparse = (sparsity > 10).sum()
        ax.text(0.97, 0.97, f"{n_sparse}/{len(sparsity)} clients >10%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#E24B4A")
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_sparsity.png")


def fig8_intra_cluster_correlation(long: pd.DataFrame, out_dir: str):
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))
    fig, axes = plt.subplots(1, len(clusters),
                             figsize=(4.8 * len(clusters), 4.8))
    if len(clusters) == 1:
        axes = [axes]
    fig.suptitle("Figure 8 — Intra-Cluster Pairwise Correlation Heatmap\n"
                 "(mean r close to 0 = clients behave differently → shared model struggles)",
                 fontsize=11, fontweight="bold")
    for ax, cl in zip(axes, clusters):
        sub  = long[long["cluster_id"] == cl].copy()
        wide = sub.pivot_table(index="timestamp", columns="client_id",
                               values="consumption", aggfunc="mean")
        wide = wide.dropna(axis=1, how="all").ffill().fillna(0)

        if wide.shape[1] < 2:
            ax.text(0.5, 0.5, "Not enough clients",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Cluster {cl}")
            continue

        corr   = wide.corr()
        labels = [str(c)[-5:] for c in corr.columns]
        upper  = corr.values[np.triu_indices_from(corr.values, k=1)]
        mean_r = upper.mean()

        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_title(f"Cluster {cl}  —  mean r = {mean_r:.2f}")
        plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
    fig.tight_layout()
    _save(fig, out_dir, "fig8_correlation_heatmap.png")


def fig9_client_range(long: pd.DataFrame, out_dir: str):
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))
    fig, axes = plt.subplots(1, len(clusters),
                             figsize=(4 * len(clusters), 4.5), sharey=False)
    if len(clusters) == 1:
        axes = [axes]
    fig.suptitle("Figure 9 — Per-Client Consumption Range  (min / mean / max)\n"
                 "(outlier clients with huge max values drag up cluster-level error)",
                 fontsize=11, fontweight="bold")
    for ax, cl in zip(axes, clusters):
        sub   = long[long["cluster_id"] == cl]
        cs    = sub.groupby("client_id")["consumption"].agg(
                    min="min", mean="mean", max="max").sort_values("mean")
        x     = np.arange(len(cs))
        col   = cluster_color(cl)

        ax.fill_between(x, cs["min"], cs["max"],
                        alpha=0.18, color=col, label="min–max range")
        ax.plot(x, cs["mean"], color=col, linewidth=2,   label="mean")
        ax.plot(x, cs["min"],  color=col, linewidth=0.7, linestyle=":")
        ax.plot(x, cs["max"],  color=col, linewidth=0.7, linestyle=":")
        ax.set_title(f"Cluster {cl}")
        ax.set_xlabel("Clients (sorted by mean)")
        ax.set_ylabel("Consumption" if cl == clusters[0] else "")
        ax.set_xticks([])
        ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out_dir, "fig9_client_range.png")


def fig10_temporal_trend(long: pd.DataFrame, out_dir: str):
    if not pd.api.types.is_datetime64_any_dtype(long["timestamp"]):
        print("  [SKIP Fig 10] timestamp not datetime")
        return
    clusters = sorted(long["cluster_id"].dropna().unique().astype(int))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle("Figure 10 — Temporal Consumption Trend per Cluster\n"
                 "(positive slope = rising demand; increases model staleness over rounds)",
                 fontsize=11, fontweight="bold")
    for cl in clusters:
        sub  = long[long["cluster_id"] == cl]
        ts   = sub.groupby("timestamp")["consumption"].mean().sort_index()
        ax.plot(ts.index, ts.values, color=cluster_color(cl),
                linewidth=0.7, alpha=0.55, label=f"Cluster {cl}")

        x_num = np.arange(len(ts))
        slope, intercept, r, p, _ = scipy_stats.linregress(x_num, ts.values)
        trend = intercept + slope * x_num
        ax.plot(ts.index, trend, color=cluster_color(cl), linewidth=2,
                linestyle="--",
                label=f"C{cl} trend  slope={slope:+.4f}  p={p:.3f}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Mean consumption")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "fig10_temporal_trend.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    set_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "═"*80)
    print("  Loading data …")
    print("═"*80)
    long, cmap, wide = load_inputs()

    print_cluster_report(long, cmap)

    print("\n" + "═"*80)
    print("  Generating cluster analysis plots …")
    print("═"*80)
    fig1_cluster_sizes(cmap, OUTPUT_DIR)
    fig2_mean_consumption(long, OUTPUT_DIR)
    fig3_violin(long, OUTPUT_DIR)
    fig4_cv_per_client(long, OUTPUT_DIR)
    fig5_daily_load_profile(long, OUTPUT_DIR)
    fig6_rolling_variability(long, OUTPUT_DIR)
    fig7_sparsity(long, OUTPUT_DIR)
    fig8_intra_cluster_correlation(long, OUTPUT_DIR)
    fig9_client_range(long, OUTPUT_DIR)
    fig10_temporal_trend(long, OUTPUT_DIR)

    print(f"\n  Done. All figures saved to → ./{OUTPUT_DIR}/")
    print("═"*80 + "\n")


if __name__ == "__main__":
    main()