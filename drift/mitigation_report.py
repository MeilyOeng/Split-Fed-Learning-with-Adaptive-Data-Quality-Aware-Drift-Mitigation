#!/usr/bin/env python3
"""Build a Markdown report (tables + plots) from a completed multi-seed mitigation suite.

Reads only from <suite-root>/runseed<N>/{comparison/*, */pre_drift_predictions,
*/adaptation_predictions, */persistent_drift_final_holdout, selected_drift_clients.csv,
suite_resolved_config.yaml} across every completed runseed<N> directory. Training
outputs and checkpoints remain untouched; generated material is written only below
<suite-root>/report/ unless an explicit --output is supplied.

Produces:
  report/report.md                              cross-seed tables + summary plots
  report/tables/*.csv                            the same tables, as raw CSV
  report/plots/*.{png,pdf}                       publication-quality summary plots
  report/plots/mae/seed<N>/client_<id>_mae_plot.{png,pdf}
                                                  per-drifted-client MAE-over-time
                                                  line plot for every seed, segmented
                                                  into pre_drift / adaptation / final_holdout

Usage:
  venv/bin/python drift/mitigation_report.py --suite-root outputs/drift_mitigation/<suite_id>
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------
# Fixed thesis figure configuration. Identity never relies on hue alone: each
# method also has a stable marker and line style for grayscale reproduction.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ThesisStyle:
    """Single source of truth for the thesis figure appearance."""

    font_family: str = "serif"
    font_serif: tuple[str, ...] = ("STIX Two Text", "STIXGeneral", "DejaVu Serif")
    base_font_size: float = 10.0
    axes_label_size: float = 10.0
    tick_label_size: float = 9.0
    legend_font_size: float = 8.5
    annotation_font_size: float = 8.5
    title_font_size: float = 11.0
    figure_width_in: float = 6.5
    figure_height_curve_in: float = 3.9
    figure_height_scatter_in: float = 5.0
    line_width: float = 1.8
    marker_size: float = 42.0
    grid_line_width: float = 0.6
    spine_line_width: float = 0.8
    diagonal_line_width: float = 1.1
    output_dpi: int = 300


STYLE = ThesisStyle()

M4_AGGREGATION = "M4_quality_aware_splitfed"
M4_WINDOW = "M4_window"
METHOD_ORDER = [
    "R0_clean_frozen",
    "M0_drift_frozen",
    "M1_continued_splitfed",
    "M2_oracle_client_only",
    "M3_oracle_client_ewc",
    M4_AGGREGATION,
    M4_WINDOW,
]
MITIGATION_METHODS = METHOD_ORDER[2:]
CONTROL_METHOD = "R1_clean_continued_splitfed"

METHOD_LABELS = {
    "R0_clean_frozen": "R0 clean (baseline)",
    "M0_drift_frozen": "M0 drift-frozen (no mitigation)",
    "M1_continued_splitfed": "M1 continued SplitFed",
    "M2_oracle_client_only": "M2 client-only oracle",
    "M3_oracle_client_ewc": "M3 client-only + EWC",
    M4_AGGREGATION: "M4-A quality-aware SplitFed (aggregation)",
    M4_WINDOW: "M4-W quality-aware SplitFed (window)",
    "R1_clean_continued_splitfed": "R1 clean-continued (control)",
}
METHOD_KEY_TEXT = {
    "R0_clean_frozen": "R0 = clean frozen counterfactual",
    "M0_drift_frozen": "M0 = drifted frozen model (no mitigation)",
    "M1_continued_splitfed": "M1 = continued SplitFed",
    "M2_oracle_client_only": "M2 = oracle client-only adaptation",
    "M3_oracle_client_ewc": "M3 = oracle client-only adaptation with EWC",
    M4_AGGREGATION: "M4-A = quality-aware SplitFed with aggregation weighting",
    M4_WINDOW: "M4-W = quality-aware SplitFed with per-window weighting",
    "R1_clean_continued_splitfed": "R1 = clean continued-training control",
}
METHOD_SHORT = {
    "R0_clean_frozen": "R0",
    "M0_drift_frozen": "M0",
    "M1_continued_splitfed": "M1",
    "M2_oracle_client_only": "M2",
    "M3_oracle_client_ewc": "M3",
    M4_AGGREGATION: "M4-A",
    M4_WINDOW: "M4-W",
    "R1_clean_continued_splitfed": "R1",
}
METHOD_COLORS = {
    "R0_clean_frozen": "#5B6268",        # neutral grey
    "M0_drift_frozen": "#C23B3F",        # red
    "M1_continued_splitfed": "#6A3D9A",  # purple
    "M2_oracle_client_only": "#2F6F96",  # blue
    "M3_oracle_client_ewc": "#17877B",   # teal
    "R1_clean_continued_splitfed": "#92979C",  # secondary grey
    M4_AGGREGATION: "#B83E73",  # pink; variants differ by marker/line style
    M4_WINDOW: "#B83E73",
}
METHOD_LINESTYLES = {
    "R0_clean_frozen": "solid",
    "M0_drift_frozen": (0, (5, 2)),
    "M1_continued_splitfed": (0, (2, 1)),
    "M2_oracle_client_only": (0, (4, 1, 1, 1)),
    "M3_oracle_client_ewc": (0, (1, 1)),
    "R1_clean_continued_splitfed": (0, (6, 2)),
    M4_AGGREGATION: (0, (3, 1, 1, 1)),
    M4_WINDOW: (0, (6, 1.5)),
}
METHOD_MARKERS = {
    "R0_clean_frozen": "o",
    "M0_drift_frozen": "s",
    "M1_continued_splitfed": "^",
    "M2_oracle_client_only": "D",
    "M3_oracle_client_ewc": "v",
    "R1_clean_continued_splitfed": "P",
    M4_AGGREGATION: "X",
    M4_WINDOW: "*",
}
INK_PRIMARY = "#111111"
INK_SECONDARY = "#404040"
INK_MUTED = "#6B6B6B"
GRIDLINE = "#D9D9D9"
PHASE_COLORS = ("#F4F7FA", "#FFF5E8", "#F2F8F4")

PLOT_RCPARAMS = {
    "font.family": STYLE.font_family,
    "font.serif": list(STYLE.font_serif),
    "font.size": STYLE.base_font_size,
    "mathtext.fontset": "stix",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": STYLE.spine_line_width,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelsize": STYLE.axes_label_size,
    "axes.titlesize": STYLE.title_font_size,
    "axes.titleweight": "normal",
    "xtick.labelsize": STYLE.tick_label_size,
    "ytick.labelsize": STYLE.tick_label_size,
    "legend.fontsize": STYLE.legend_font_size,
    "figure.titlesize": STYLE.title_font_size,
    "figure.titleweight": "normal",
    "lines.linewidth": STYLE.line_width,
    "lines.markersize": float(np.sqrt(STYLE.marker_size)),
    "legend.frameon": False,
    "savefig.dpi": STYLE.output_dpi,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(PLOT_RCPARAMS)


def _available_methods(
    frame: pd.DataFrame,
    candidates: list[str] | tuple[str, ...] = tuple(METHOD_ORDER),
) -> list[str]:
    """Return known methods in thesis order, allowing reports on older suites."""
    if "method" not in frame.columns:
        return []
    present = set(frame["method"].dropna().astype(str))
    return [method for method in candidates if method in present]


def _style_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(STYLE.spine_line_width)
    ax.spines["bottom"].set_linewidth(STYLE.spine_line_width)
    ax.grid(axis=grid_axis, color=GRIDLINE, linewidth=STYLE.grid_line_width, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def _save_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=STYLE.output_dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=STYLE.output_dpi, bbox_inches="tight")


def _bootstrap_ci95(values: np.ndarray, *, n_resamples: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean.

    Unlike mean +/- 1.96*SEM (the normal approximation), this respects the
    sample's own bounds -- it can never produce a lower bound below the
    smallest observed value, so a non-negative quantity like MAE never gets
    an impossible negative CI bound. Matters most exactly where the normal
    approximation is weakest: small n (single-seed suites, n=4 for
    drifted_per_cluster=1) and heavy-tailed per-client error distributions
    (a handful of high-error clients dominate the spread).
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        point = float(values.mean()) if len(values) else float("nan")
        return point, point
    rng = np.random.default_rng(seed)
    resampled_means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    lower, upper = np.percentile(resampled_means, [2.5, 97.5])
    return float(lower), float(upper)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover_seed_runs(suite_root: Path) -> dict[int, Path]:
    runs = {}
    for path in sorted(suite_root.glob("runseed*")):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"runseed(\d+)", path.name)
        if match and (path / "comparison").is_dir():
            runs[int(match.group(1))] = path
    if not runs:
        raise SystemExit(f"no completed runseed<N>/ directories with a comparison/ folder found under {suite_root}")
    return runs


def _segment_sizes(run_dir: Path) -> dict[str, int]:
    config = yaml.safe_load((run_dir / "suite_resolved_config.yaml").read_text(encoding="utf-8"))
    return {k: int(v) for k, v in config["segments"].items()}


# --------------------------------------------------------------------------
# Cross-seed table loaders
# --------------------------------------------------------------------------

def load_recovery(seed: int, run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "comparison" / "recovery_summary.csv")
    df.insert(0, "seed", seed)
    return df


def load_retention(seed: int, run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "comparison" / "retention_summary.csv")
    df.insert(0, "seed", seed)
    return df


def load_continued_training_control(seed: int, run_dir: Path) -> pd.DataFrame:
    path = run_dir / "comparison" / "continued_training_control.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "seed", seed)
    return df


def load_cluster_r0(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "comparison" / "cluster_comparison.csv")
    return df[df.method == "R0_clean_frozen"][["cluster_id", "n", "mae", "rmse"]].reset_index(drop=True)


def load_affected_clients(seed_runs: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for seed, run_dir in seed_runs.items():
        frame = pd.read_csv(
            run_dir / "comparison" / "affected_client_comparison.csv",
            dtype={"cluster_id": str, "client_id": str},
        )
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_retention_clients(seed_runs: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for seed, run_dir in seed_runs.items():
        for method, filename in (
            ("M2_oracle_client_only", "m2_retention_per_client.csv"),
            ("M3_oracle_client_ewc", "m3_retention_per_client.csv"),
        ):
            frame = pd.read_csv(
                run_dir / "comparison" / filename,
                dtype={"cluster_id": str, "client_id": str},
            )
            frame.insert(0, "method", method)
            frame.insert(0, "seed", seed)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_control_clients(seed_runs: dict[int, Path]) -> pd.DataFrame:
    """Per-client final-holdout metrics for control (never drift-selected) clients.

    comparison/clean_control_comparison.csv is the control-cohort mirror of
    affected_client_comparison.csv -- same columns, pre-filtered to
    selected_for_drift == False (52 = 13 controls_per_cluster x 4 clusters,
    constant across seeds since only cluster *membership* among controls
    varies with client_selection_seed, not the count).
    """
    frames = []
    for seed, run_dir in seed_runs.items():
        frame = pd.read_csv(
            run_dir / "comparison" / "clean_control_comparison.csv",
            dtype={"cluster_id": str, "client_id": str},
        )
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_client_scales(seed_runs: dict[int, Path]) -> pd.DataFrame:
    """Per-(seed, client) z-score scale (std) from the offline preparation manifest.

    comparison/*.csv never carries this -- it lives in the sibling
    run2_dataseed<seed>/manifests/scaler_manifest.csv (or the older
    exp2_dataseed<seed> name, for suites prepared before that directory was
    renamed). Used to build NMAE = MAE / scale, putting clients with very
    different natural magnitudes (see cohort heterogeneity) on a comparable
    footing for cross-seed and cross-client comparison.
    """
    frames = []
    for seed, run_dir in seed_runs.items():
        suite_root = run_dir.parent
        dataseed_dir = next(
            (suite_root / candidate for candidate in (f"run2_dataseed{seed}", f"exp2_dataseed{seed}")
             if (suite_root / candidate).is_dir()),
            None,
        )
        if dataseed_dir is None:
            raise SystemExit(f"seed {seed}: no run2_dataseed{seed}/ or exp2_dataseed{seed}/ found beside {run_dir}")
        frame = pd.read_csv(
            dataseed_dir / "manifests" / "scaler_manifest.csv",
            usecols=["cluster_id", "client_id", "scale"],
            dtype={"cluster_id": str, "client_id": str},
        )
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def attach_nmae(clients: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    """NMAE = MAE / each client's own z-score scale (std of its clean training target).

    A client with MAE=2 on a scale-10 signal and a client with MAE=0.4 on a
    scale-2 signal are both "0.2 standard deviations of error" under NMAE,
    rather than looking 5x apart under raw MAE -- the fair-comparison metric
    the cohort-heterogeneity section (clients differ hugely in natural
    magnitude) shows raw MAE cannot provide across seeds or clients.
    """
    merged = clients.merge(scales, on=["seed", "cluster_id", "client_id"], how="left", validate="many_to_one")
    if merged.scale.isna().any():
        missing = merged.loc[merged.scale.isna(), ["seed", "cluster_id", "client_id"]].drop_duplicates()
        raise SystemExit(f"missing scaler for {len(missing)} client-seed rows, e.g.:\n{missing.head()}")
    merged["nmae"] = merged.mae / merged.scale
    return merged


def load_affected_client_nmae_stats(affected_clients_scaled: pd.DataFrame) -> pd.DataFrame:
    """Per-(seed, method) mean/median/std/bootstrap-95%-CI of affected-client
    NMAE. Mirrors load_affected_client_stats exactly, substituting nmae for mae."""
    rows = []
    for (seed, method), frame in affected_clients_scaled.groupby(["seed", "method"]):
        values = frame.nmae.to_numpy()
        ci_low, ci_high = _bootstrap_ci95(values)
        rows.append({
            "seed": seed, "method": method, "n": len(values),
            "mean_nmae": values.mean(), "median_nmae": float(np.median(values)), "std_nmae": values.std(ddof=1) if len(values) > 1 else 0.0,
            "ci_low_nmae": ci_low, "ci_high_nmae": ci_high,
        })
    return pd.DataFrame(rows)


def summarise_nmae_aggregate(nmae_stats: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of the per-seed mean NMAE across available seeds -- mirrors
    recovery_aggregate_mae's mean-of-seed-means semantics exactly (equal
    weight per seed, not pooled across all client-seed rows)."""
    aggregate = nmae_stats.groupby("method")[["mean_nmae"]].agg(["mean", "std"]).round(4)
    aggregate.columns = [f"{a}_{b}" for a, b in aggregate.columns]
    return aggregate.reset_index().rename(columns={"mean_nmae_mean": "nmae_mean", "mean_nmae_std": "nmae_std"})


def load_affected_client_stats(affected_clients: pd.DataFrame) -> pd.DataFrame:
    """Per-(seed, method) mean/median/std/bootstrap-95%-CI of affected-client MAE.

    Computed from the actual per-client population (n=32 affected clients per
    standard seed, fewer for the drifted_per_cluster=1/21 suites) in
    comparison/affected_client_comparison.csv -- not fabricated, and the mean
    is consistent by construction with recovery_summary.csv's E_clean/
    E_frozen/E_mitigated (verified to match to the last digit). The CI is a
    percentile bootstrap, not mean +/- 1.96*SEM: at small n (e.g. n=4 for
    drifted_per_cluster=1) or with heavy-tailed per-client errors, the normal
    approximation can produce a CI whose lower bound is negative, which is
    impossible for a non-negative quantity like MAE.
    """
    rows = []
    for (seed, method), frame in affected_clients.groupby(["seed", "method"]):
        values = frame.mae.to_numpy()
        ci_low, ci_high = _bootstrap_ci95(values)
        rows.append({
            "seed": seed, "method": method, "n": len(values),
            "mean_mae": values.mean(), "median_mae": float(np.median(values)), "std_mae": values.std(ddof=1) if len(values) > 1 else 0.0,
            "ci_low_mae": ci_low, "ci_high_mae": ci_high,
        })
    return pd.DataFrame(rows)


def build_client_benefits(affected_clients: pd.DataFrame) -> pd.DataFrame:
    """Paired final-holdout MAE reduction relative to drift-frozen M0."""
    keys = ["seed", "cluster_id", "client_id"]
    pivot = affected_clients.pivot(index=keys, columns="method", values="mae")
    frames = []
    for method in (candidate for candidate in MITIGATION_METHODS if candidate in pivot.columns):
        frame = pivot[["M0_drift_frozen", method]].reset_index()
        frame = frame.rename(columns={"M0_drift_frozen": "mae_m0", method: "mae_mitigated"})
        frame["method"] = method
        frame["mae_reduction"] = frame.mae_m0 - frame.mae_mitigated
        frame["improved"] = frame.mae_reduction > 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_control_impact(control_clients: pd.DataFrame) -> pd.DataFrame:
    """Paired final-holdout MAE changes relative to R0 and M0 for controls.

    ``mae_harm`` records total error relative to the untouched clean R0 model;
    ``mae_change_vs_m0`` isolates the change caused by adaptation. M2/M3 update
    only oracle-selected affected clients, so their control-client MAE equals
    M0 exactly. It need not equal R0 because non-drift-selected controls can
    still be fault-injected. M1 and both M4 variants FedAvg-aggregate across
    the whole cluster and can move a control client's error in either direction.
    """
    empty_columns = [
        "seed", "cluster_id", "client_id", "method", "mae_r0", "mae_m0",
        "mae_method", "mae_harm", "mae_change_vs_m0", "harmed", "worse_than_m0",
    ]
    if control_clients.empty:
        # drifted_per_cluster == cluster size (e.g. 21/21): every client is
        # affected, so there are no control clients at all -- not an error,
        # just nothing to compare here for this suite.
        return pd.DataFrame(columns=empty_columns)
    keys = ["seed", "cluster_id", "client_id"]
    pivot = control_clients.pivot(index=keys, columns="method", values="mae")
    frames = []
    for method in (candidate for candidate in MITIGATION_METHODS if candidate in pivot.columns):
        frame = pivot[["R0_clean_frozen", "M0_drift_frozen", method]].reset_index()
        frame = frame.rename(columns={"R0_clean_frozen": "mae_r0",
                                      "M0_drift_frozen": "mae_m0", method: "mae_method"})
        frame["method"] = method
        frame["mae_harm"] = frame.mae_method - frame.mae_r0
        frame["mae_change_vs_m0"] = frame.mae_method - frame.mae_m0
        frame["harmed"] = frame.mae_harm > 0
        frames.append(frame)
        frame["worse_than_m0"] = frame.mae_change_vs_m0 > 0
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=empty_columns)


def summarise_control_impact(control_impact: pd.DataFrame) -> pd.DataFrame:
    empty_columns = [
        "seed", "method", "n_clients", "mean_mae_harm", "median_mae_harm",
        "q25_mae_harm", "q75_mae_harm", "n_harmed", "pct_harmed",
        "mean_change_vs_m0", "median_change_vs_m0", "n_worse_than_m0", "pct_worse_than_m0",
    ]
    if control_impact.empty:
        return pd.DataFrame(columns=empty_columns)
    rows = []
    grouped = list(control_impact.groupby(["seed", "method"]))
    grouped += [(('all', method), frame) for method, frame in control_impact.groupby("method")]
    for (seed, method), frame in grouped:
        values = frame.mae_harm
        adaptation_values = frame.mae_change_vs_m0
        rows.append({
            "seed": seed,
            "method": method,
            "n_clients": len(frame),
            "mean_mae_harm": values.mean(),
            "median_mae_harm": values.median(),
            "q25_mae_harm": values.quantile(0.25),
            "q75_mae_harm": values.quantile(0.75),
            "n_harmed": int(frame.harmed.sum()),
            "pct_harmed": 100.0 * frame.harmed.mean(),
            "mean_change_vs_m0": adaptation_values.mean(),
            "median_change_vs_m0": adaptation_values.median(),
            "n_worse_than_m0": int(frame.worse_than_m0.sum()),
            "pct_worse_than_m0": 100.0 * frame.worse_than_m0.mean(),
        })
    return pd.DataFrame(rows)


def summarise_client_benefits(client_benefits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = list(client_benefits.groupby(["seed", "method"]))
    grouped += [(('all', method), frame) for method, frame in client_benefits.groupby("method")]
    for (seed, method), frame in grouped:
        values = frame.mae_reduction
        rows.append({
            "seed": seed,
            "method": method,
            "n_clients": len(frame),
            "mean_mae_reduction": values.mean(),
            "median_mae_reduction": values.median(),
            "q25_mae_reduction": values.quantile(0.25),
            "q75_mae_reduction": values.quantile(0.75),
            "n_improved": int(frame.improved.sum()),
            "pct_improved": 100.0 * frame.improved.mean(),
        })
    return pd.DataFrame(rows)


def summarise_retention_clients(retention_clients: pd.DataFrame) -> pd.DataFrame:
    mae = retention_clients[retention_clients.metric == "mae"]
    rows = []
    grouped = list(mae.groupby(["seed", "method"]))
    grouped += [(('all', method), frame) for method, frame in mae.groupby("method")]
    for (seed, method), frame in grouped:
        values = frame.retention_delta
        rows.append({
            "seed": seed,
            "method": method,
            "n_clients": len(frame),
            "mean_delta": values.mean(),
            "median_delta": values.median(),
            "q25_delta": values.quantile(0.25),
            "q75_delta": values.quantile(0.75),
            "n_worsened": int((values > 0).sum()),
            "pct_worsened": 100.0 * (values > 0).mean(),
        })
    return pd.DataFrame(rows)


def summarise_affected_baseline(affected_clients: pd.DataFrame) -> pd.DataFrame:
    r0 = affected_clients[affected_clients.method == "R0_clean_frozen"]
    return (
        r0.groupby(["seed", "cluster_id"]).mae
        .agg(n_clients="count", mean_mae="mean", median_mae="median", std_mae="std")
        .reset_index()
    )


def load_integrity(seed: int, run_dir: Path) -> dict:
    acceptance = json.loads((run_dir / "comparison" / "acceptance_report.json").read_text(encoding="utf-8"))
    repro_path = run_dir / "comparison" / "reproducibility_report.json"
    repro = json.loads(repro_path.read_text(encoding="utf-8")) if repro_path.is_file() else {}
    return {
        "seed": seed,
        "accepted": acceptance.get("accepted"),
        "identical_holdout_keys": acceptance.get("identical_holdout_keys"),
        "primary_metric": acceptance.get("primary_metric"),
        "prediction_keys_identical": repro.get("prediction_keys_identical"),
    }


def build_tables(seed_runs: dict[int, Path], affected_clients: pd.DataFrame,
                 retention_clients: pd.DataFrame, client_benefits: pd.DataFrame,
                 control_impact: pd.DataFrame) -> dict[str, pd.DataFrame]:
    recovery_all = pd.concat([load_recovery(s, d) for s, d in seed_runs.items()], ignore_index=True)
    retention_all = pd.concat([load_retention(s, d) for s, d in seed_runs.items()], ignore_index=True)
    confound_all = pd.concat([load_continued_training_control(s, d) for s, d in seed_runs.items()], ignore_index=True)
    integrity = pd.DataFrame([load_integrity(s, d) for s, d in seed_runs.items()])
    cluster_r0 = load_cluster_r0(next(iter(seed_runs.values())))

    recovery_mae = recovery_all[recovery_all.metric == "mae"].copy()
    recovery_aggregate = (
        recovery_mae.groupby("method")[["E_clean", "E_frozen", "E_mitigated", "drift_damage", "recovered_error", "recovery_rate"]]
        .agg(["mean", "std"]).round(4)
    )
    recovery_aggregate.columns = [f"{a}_{b}" for a, b in recovery_aggregate.columns]
    recovery_aggregate = recovery_aggregate.reset_index()

    retention_mae = retention_all[retention_all.metric == "mae"].copy()
    retention_aggregate = (
        retention_mae.groupby("method")[["retention_delta", "retention_pct_change", "pct_worsened"]]
        .agg(["mean", "std"]).round(4)
    )
    retention_aggregate.columns = [f"{a}_{b}" for a, b in retention_aggregate.columns]
    retention_aggregate = retention_aggregate.reset_index()

    return {
        "recovery_by_seed": recovery_all.round(4),
        "recovery_aggregate_mae": recovery_aggregate,
        "retention_by_seed": retention_all.round(4),
        "retention_aggregate_mae": retention_aggregate,
        "continued_training_control_by_seed": confound_all.round(4),
        "cluster_r0_heterogeneity": cluster_r0.round(4),
        "affected_r0_by_seed_cluster": summarise_affected_baseline(affected_clients).round(4),
        "client_benefit_summary_mae": summarise_client_benefits(client_benefits).round(4),
        "control_impact_summary_mae": summarise_control_impact(control_impact).round(4),
        "retention_client_summary_mae": summarise_retention_clients(retention_clients).round(4),
        "data_integrity": integrity,
    }


# --------------------------------------------------------------------------
# Cross-seed plots
# --------------------------------------------------------------------------

def plot_recovery_by_seed(client_stats: pd.DataFrame, out_dir: Path,
                          stem: str = "affected_mae_by_seed") -> str:
    """Affected-cohort MAE with descriptive client-level bootstrap 95% intervals."""
    seeds = sorted(client_stats.seed.unique())
    methods = _available_methods(client_stats)
    fig, ax = plt.subplots(figsize=(STYLE.figure_width_in, STYLE.figure_height_curve_in), layout="constrained")
    x = np.arange(len(seeds), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods))
    for offset, method in zip(offsets, methods):
        rows = client_stats[client_stats.method == method].set_index("seed").reindex(seeds)
        yerr = np.vstack([
            (rows.mean_mae - rows.ci_low_mae).clip(lower=0),
            (rows.ci_high_mae - rows.mean_mae).clip(lower=0),
        ])
        ax.errorbar(
            x + offset,
            rows.mean_mae,
            yerr=yerr,
            fmt=METHOD_MARKERS[method],
            markersize=np.sqrt(STYLE.marker_size),
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.45,
            ecolor=METHOD_COLORS[method],
            elinewidth=1.0,
            capsize=2.5,
            linestyle="none",
            label=METHOD_SHORT[method],
            zorder=3,
        )
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Client-selection seed")
    ax.set_ylabel("Affected-cohort MAE")
    ax.set_title("Final-holdout error across client-selection seeds")
    ax.set_ylim(bottom=0)
    _style_axis(ax)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19), handletextpad=0.4,
              columnspacing=1.0)
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    n_clients = int(client_stats[client_stats.method == "R0_clean_frozen"].n.iloc[0])
    adapted_text = ", ".join(
        METHOD_SHORT[method] for method in methods if method in MITIGATION_METHODS
    )
    return (
        "Affected-cohort final-holdout MAE by client-selection seed and method. Points are client "
        "means; error bars are a percentile bootstrap 95% CI of the mean (5000 resamples), not the "
        "normal approximation -- it cannot produce a negative lower bound for this non-negative metric "
        f"even at small n (n={n_clients} selected clients per seed). R0 is the clean counterfactual, M0 "
        f"is the unmitigated drift condition; the adapted methods shown are {adapted_text}. See the companion "
        "distribution figure for median and std alongside the mean."
    )


def plot_mae_distribution_by_seed(client_stats: pd.DataFrame, out_dir: Path,
                                  stem: str = "affected_mae_distribution_by_seed") -> str:
    """Mean +/- 1 std (not a CI of the mean) plus median, per seed and method.

    Companion to plot_recovery_by_seed: that figure's error bars describe
    uncertainty in *estimating* the mean; this one describes the actual
    spread of the client population and where the median sits relative to
    the mean, which is the more direct read when a handful of high-error
    clients pull the mean away from what most clients experience.
    """
    seeds = sorted(client_stats.seed.unique())
    methods = _available_methods(client_stats)
    fig, ax = plt.subplots(figsize=(STYLE.figure_width_in, STYLE.figure_height_curve_in), layout="constrained")
    x = np.arange(len(seeds), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods))
    for offset, method in zip(offsets, methods):
        rows = client_stats[client_stats.method == method].set_index("seed").reindex(seeds)
        ax.errorbar(
            x + offset,
            rows.mean_mae,
            yerr=rows.std_mae,
            fmt=METHOD_MARKERS[method],
            markersize=np.sqrt(STYLE.marker_size),
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.45,
            ecolor=METHOD_COLORS[method],
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.85,
            linestyle="none",
            label=METHOD_SHORT[method],
            zorder=3,
        )
        ax.scatter(x + offset, rows.median_mae, marker="_", s=130, color="black",
                   linewidths=1.6, zorder=4)
    median_handle = plt.Line2D([0], [0], marker="_", color="black", linestyle="none",
                               markersize=11, markeredgewidth=1.6, label="median")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Client-selection seed")
    ax.set_ylabel("Affected-cohort MAE")
    ax.set_title("Mean +/- 1 std and median, by seed and method")
    ax.set_ylim(bottom=0)
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [median_handle], labels + ["median"], ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.19), handletextpad=0.4, columnspacing=1.0)
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    return (
        "Affected-cohort final-holdout MAE by client-selection seed and method: colored markers are "
        "the client mean with error bars at +/-1 standard deviation (the actual spread of the client "
        "population, not an uncertainty interval on the mean); black ticks are the median. Where the "
        "median sits well below the mean, a minority of high-error clients are pulling the mean up -- "
        "most clients are doing better than the mean alone suggests."
    )


def plot_nmae_by_seed(nmae_stats: pd.DataFrame, out_dir: Path,
                      stem: str = "affected_nmae_by_seed") -> str:
    """Affected-cohort NMAE (MAE / per-client scale) -- same layout as
    plot_recovery_by_seed, so the two are directly visually comparable."""
    seeds = sorted(nmae_stats.seed.unique())
    methods = _available_methods(nmae_stats)
    fig, ax = plt.subplots(figsize=(STYLE.figure_width_in, STYLE.figure_height_curve_in), layout="constrained")
    x = np.arange(len(seeds), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods))
    for offset, method in zip(offsets, methods):
        rows = nmae_stats[nmae_stats.method == method].set_index("seed").reindex(seeds)
        yerr = np.vstack([
            (rows.mean_nmae - rows.ci_low_nmae).clip(lower=0),
            (rows.ci_high_nmae - rows.mean_nmae).clip(lower=0),
        ])
        ax.errorbar(
            x + offset,
            rows.mean_nmae,
            yerr=yerr,
            fmt=METHOD_MARKERS[method],
            markersize=np.sqrt(STYLE.marker_size),
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.45,
            ecolor=METHOD_COLORS[method],
            elinewidth=1.0,
            capsize=2.5,
            linestyle="none",
            label=METHOD_SHORT[method],
            zorder=3,
        )
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Client-selection seed")
    ax.set_ylabel("Affected-cohort NMAE (MAE / client scale)")
    ax.set_title("Normalized error is far more stable across seeds than raw MAE")
    ax.set_ylim(bottom=0)
    _style_axis(ax)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19), handletextpad=0.4,
              columnspacing=1.0)
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    n_clients = int(nmae_stats[nmae_stats.method == "R0_clean_frozen"].n.iloc[0])
    return (
        "Affected-cohort NMAE by client-selection seed and method: each client's final-holdout MAE "
        "divided by that same client's own z-score scale (the std of its clean-training target), "
        "before aggregating. This removes the confound in the raw-MAE plot above, where seeds with "
        "harder cohorts (see cohort heterogeneity) simply have larger clients on average -- NMAE "
        "expresses error in client-relative standard-deviation units, so seeds are now comparable on "
        "a like-for-like basis. Points are client means; error bars are a percentile bootstrap 95% CI "
        f"of the mean (5000 resamples; n={n_clients} selected clients per seed), which -- unlike the "
        "normal approximation -- cannot produce an impossible negative lower bound at small n."
    )


def plot_recovery_rate_by_seed(recovery_mae: pd.DataFrame, out_dir: Path,
                               stem: str = "recovery_rate_by_seed") -> str:
    methods = _available_methods(recovery_mae, tuple(MITIGATION_METHODS))
    seeds = sorted(recovery_mae.seed.unique())
    fig, ax = plt.subplots(figsize=(STYLE.figure_width_in, STYLE.figure_height_curve_in), layout="constrained")
    for method in methods:
        rows = recovery_mae[recovery_mae.method == method].set_index("seed").reindex(seeds)
        ax.plot(
            seeds,
            100.0 * rows.recovery_rate,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=np.sqrt(STYLE.marker_size),
            linewidth=STYLE.line_width,
            label=METHOD_SHORT[method],
            zorder=3,
        )
    ax.axhline(100, color=INK_MUTED, linestyle=(0, (3, 2)), linewidth=STYLE.diagonal_line_width, zorder=1)
    ax.text(seeds[0], 103, "100% = return to the clean R0 error", color=INK_MUTED,
            fontsize=STYLE.annotation_font_size, ha="left", va="bottom")
    ax.set_xticks(seeds)
    ax.set_xlabel("Client-selection seed")
    ax.set_ylabel("Recovered drift damage (%)")
    ax.set_title("Recovery rate is strongly cohort-dependent")
    ax.set_ylim(0, max(210, 1.08 * 100.0 * recovery_mae.recovery_rate.max()))
    _style_axis(ax)
    ax.legend(ncol=3, loc="upper left")
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    return (
        "MAE recovery rate by seed. A value of 100% returns to the clean R0 baseline; values above "
        "100% indicate over-recovery, where the adapted model has lower MAE than R0 on the affected "
        "cohort. Lines connect seeds only to aid method tracking; seed is not a temporal variable."
    )


def plot_client_benefit_distribution(client_benefits: pd.DataFrame, out_dir: Path,
                                     stem: str = "client_mae_reduction") -> str:
    """Distribution of paired per-client MAE reduction relative to M0."""
    methods = _available_methods(client_benefits, tuple(MITIGATION_METHODS))
    positions = np.arange(len(methods), 0, -1)
    data = [client_benefits[client_benefits.method == method].mae_reduction.to_numpy()
            for method in methods]
    fig, ax = plt.subplots(figsize=(STYLE.figure_width_in, STYLE.figure_height_curve_in), layout="constrained")
    bp = ax.boxplot(
        data,
        positions=positions,
        vert=False,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": INK_SECONDARY, "linewidth": 0.8},
        capprops={"color": INK_SECONDARY, "linewidth": 0.8},
    )
    rng = np.random.default_rng(42)
    for position, method, values, box in zip(positions, methods, data, bp["boxes"]):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_edgecolor("black")
        box.set_alpha(0.35)
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(values, position + jitter, s=STYLE.marker_size, color=METHOD_COLORS[method], alpha=0.25,
                   edgecolor="none", zorder=2)
        pct = 100.0 * np.mean(values > 0)
        ax.text(0.99, position, f"{pct:.1f}% improved", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=STYLE.annotation_font_size, color=INK_SECONDARY)
    ax.axvline(0, color=INK_PRIMARY, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    ax.set_xscale("symlog", linthresh=0.5, linscale=1.0)
    ax.set_yticks(positions, [METHOD_SHORT[method] for method in methods])
    ax.set_xlabel("Per-client MAE reduction relative to M0 (positive = improvement)")
    ax.set_title("Mitigation benefit across all affected client-seed evaluations")
    _style_axis(ax, grid_axis="x")
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    n_evaluations = len(client_benefits[client_benefits.method == methods[0]])
    n_seeds = client_benefits.seed.nunique()
    n_clients = n_evaluations // n_seeds if n_seeds else 0
    return (
        f"Paired final-holdout MAE reduction, M0 minus each mitigated method, over all {n_evaluations} "
        f"affected client-seed evaluations ({n_clients} clients x {n_seeds} seed(s)). Positive values "
        "indicate improvement. "
        "The symmetric-log x-axis is linear from -0.5 to 0.5 MAE and logarithmic outside that "
        "range so both near-zero effects and high-error client outliers remain visible."
    )


def plot_control_impact_distribution(control_impact: pd.DataFrame, out_dir: Path,
                                     stem: str = "control_client_harm") -> str | None:
    """Distribution of paired per-client MAE change relative to R0, for control clients.

    Returns None (no figure written) when there are no control clients at
    all -- e.g. drifted_per_cluster equals the full cluster size, so every
    client is affected and this comparison is not applicable to this suite.
    """
    if control_impact.empty:
        return None
    methods = _available_methods(control_impact, tuple(MITIGATION_METHODS))
    positions = np.arange(len(methods), 0, -1)
    data = [control_impact[control_impact.method == method].mae_change_vs_m0.to_numpy()
            for method in methods]
    fig, ax = plt.subplots(figsize=(STYLE.figure_width_in, STYLE.figure_height_curve_in), layout="constrained")
    bp = ax.boxplot(
        data,
        positions=positions,
        vert=False,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": INK_SECONDARY, "linewidth": 0.8},
        capprops={"color": INK_SECONDARY, "linewidth": 0.8},
    )
    rng = np.random.default_rng(42)
    for position, method, values, box in zip(positions, methods, data, bp["boxes"]):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_edgecolor("black")
        box.set_alpha(0.35)
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(values, position + jitter, s=STYLE.marker_size, color=METHOD_COLORS[method], alpha=0.25,
                   edgecolor="none", zorder=2)
        pct = 100.0 * np.mean(values > 0)
        ax.text(0.99, position, f"{pct:.1f}% worse", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=STYLE.annotation_font_size, color=INK_SECONDARY)
    ax.axvline(0, color=INK_PRIMARY, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    ax.set_xscale("symlog", linthresh=0.5, linscale=1.0)
    ax.set_yticks(positions, [METHOD_SHORT[method] for method in methods])
    ax.set_xlabel("Per-client MAE change relative to M0 (positive = worse)")
    ax.set_title("Adaptation impact on non-drift-selected clients")
    _style_axis(ax, grid_axis="x")
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    n_evaluations = len(control_impact[control_impact.method == methods[0]])
    n_seeds = control_impact.seed.nunique()
    n_controls = n_evaluations // n_seeds if n_seeds else 0
    full_update_text = ", ".join(
        METHOD_SHORT[method] for method in ("M1_continued_splitfed", M4_AGGREGATION, M4_WINDOW)
        if method in methods
    )
    return (
        f"Paired final-holdout MAE change on control (never drift-selected) clients, method minus M0, "
        f"over all {n_evaluations} control client-seed evaluations ({n_controls} controls x {n_seeds} "
        "seed(s)). Positive values mean adaptation made a client worse than the drift/fault-frozen "
        "baseline. M2 and M3 update only oracle-selected affected clients, so their control rows equal "
        f"M0 exactly. {full_update_text} aggregate across the whole cluster and can move controls in "
        "either direction. Some controls are fault-injected, so M0 can differ from clean R0; the table "
        "reports both comparisons."
    )


def plot_cohort_heterogeneity(cluster_r0: pd.DataFrame, affected_baseline: pd.DataFrame,
                              out_dir: Path, stem: str = "cohort_heterogeneity") -> str:
    """Population cluster difficulty and selected-cohort difficulty by seed."""
    clusters = sorted(affected_baseline.cluster_id.unique(), key=int)
    seeds = sorted(affected_baseline.seed.unique())
    matrix = (
        affected_baseline.pivot(index="seed", columns="cluster_id", values="mean_mae")
        .reindex(index=seeds, columns=clusters)
    )
    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(STYLE.figure_width_in, STYLE.figure_height_curve_in), gridspec_kw={"width_ratios": [0.9, 1.35]}, layout="constrained"
    )
    full = cluster_r0.assign(cluster_id=cluster_r0.cluster_id.astype(str)).set_index("cluster_id").reindex(clusters)
    bars = ax_bar.bar(clusters, full.mae, color=METHOD_COLORS["R0_clean_frozen"], alpha=0.82,
                      edgecolor="black", linewidth=0.5, zorder=3)
    for bar, value in zip(bars, full.mae):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center",
                    va="bottom", fontsize=STYLE.annotation_font_size)
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Clean R0 MAE")
    ax_bar.set_title("a  Full population")
    ax_bar.set_ylim(0, 1.18 * full.mae.max())
    _style_axis(ax_bar)

    image = ax_heat.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
    for row in range(len(seeds)):
        for col in range(len(clusters)):
            value = matrix.iloc[row, col]
            text_color = "white" if value > 0.65 * np.nanmax(matrix.to_numpy()) else INK_PRIMARY
            ax_heat.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color,
                         fontsize=STYLE.annotation_font_size)
    per_cluster_n = int(affected_baseline.n_clients.iloc[0])
    ax_heat.set_xticks(range(len(clusters)), clusters)
    ax_heat.set_yticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax_heat.set_xlabel(f"Cluster ({per_cluster_n} selected client{'s' if per_cluster_n != 1 else ''} each)")
    ax_heat.set_ylabel("Client-selection seed")
    ax_heat.set_title("b  Selected affected cohort")
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.03)
    cbar.set_label("Mean clean R0 MAE", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    return (
        "Clean R0 baseline heterogeneity. Panel a shows population-level cluster MAE (identical "
        f"across seeds). Panel b shows mean R0 MAE among the {per_cluster_n} affected client"
        f"{'s' if per_cluster_n != 1 else ''} selected from each cluster at each seed. Every seed has "
        f"the same {per_cluster_n}-per-cluster composition; variation therefore comes from which "
        "clients are selected within a cluster."
    )


def plot_retention_distribution(retention_clients: pd.DataFrame, out_dir: Path,
                                stem: str = "retention_distribution") -> str:
    """Per-client retention and the paired incremental effect of EWC."""
    mae = retention_clients[retention_clients.metric == "mae"]
    m2 = mae[mae.method == "M2_oracle_client_only"]
    paired = (
        mae.pivot(index=["seed", "cluster_id", "client_id"], columns="method", values="retention_delta")
        .reset_index()
    )
    paired["ewc_effect"] = paired["M3_oracle_client_ewc"] - paired["M2_oracle_client_only"]
    seeds = sorted(mae.seed.unique())
    positions = np.arange(len(seeds))
    rng = np.random.default_rng(42)

    fig, (ax_delta, ax_ewc) = plt.subplots(1, 2, figsize=(STYLE.figure_width_in, STYLE.figure_height_curve_in), layout="constrained")
    delta_data = [m2[m2.seed == seed].retention_delta.to_numpy() for seed in seeds]
    boxes = ax_delta.boxplot(delta_data, positions=positions, widths=0.5, patch_artist=True,
                             showfliers=False, medianprops={"color": "black", "linewidth": 1.1})
    for position, values, box in zip(positions, delta_data, boxes["boxes"]):
        box.set_facecolor(METHOD_COLORS["M2_oracle_client_only"])
        box.set_alpha(0.32)
        box.set_edgecolor("black")
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax_delta.scatter(position + jitter, values, s=10, color=METHOD_COLORS["M2_oracle_client_only"],
                         alpha=0.45, edgecolor="none", zorder=2)
        pct = 100.0 * np.mean(values > 0)
        ax_delta.text(position, 0.98, f"{pct:.1f}%", transform=ax_delta.get_xaxis_transform(),
                      ha="center", va="top", fontsize=STYLE.annotation_font_size, color=INK_SECONDARY)
    ax_delta.axhline(0, color=INK_PRIMARY, linestyle=(0, (3, 2)), linewidth=0.9)
    ax_delta.set_yscale("symlog", linthresh=0.25, linscale=1.0)
    ax_delta.set_xticks(positions, [str(seed) for seed in seeds])
    ax_delta.set_xlabel("Client-selection seed")
    ax_delta.set_ylabel("Clean retention delta (MAE)")
    ax_delta.set_title("a  M2 client-level retention")
    _style_axis(ax_delta)

    effect_data = [paired[paired.seed == seed].ewc_effect.to_numpy() for seed in seeds]
    boxes = ax_ewc.boxplot(effect_data, positions=positions, widths=0.5, patch_artist=True,
                           showfliers=True, flierprops={"markersize": 3},
                           medianprops={"color": "black", "linewidth": 1.1})
    for box in boxes["boxes"]:
        box.set_facecolor(METHOD_COLORS["M3_oracle_client_ewc"])
        box.set_alpha(0.32)
        box.set_edgecolor("black")
    ax_ewc.axhline(0, color=INK_PRIMARY, linestyle=(0, (3, 2)), linewidth=0.9)
    ax_ewc.set_xticks(positions, [str(seed) for seed in seeds])
    ax_ewc.set_xlabel("Client-selection seed")
    ax_ewc.set_ylabel("M3 minus M2 retention delta (MAE)")
    ax_ewc.set_title("b  Incremental EWC effect")
    _style_axis(ax_ewc)
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    return (
        "Client-level clean-task retention. Delta is counterfactual clean MAE after adaptation "
        "minus R0 clean MAE: positive values mean forgetting and negative values mean improvement. "
        "Panel a shows M2 (percentages are clients with positive delta; the symmetric-log axis keeps "
        "large improvements visible). Panel b shows the paired M3-minus-M2 delta, isolating the "
        "incremental numerical effect of EWC; negative values favour M3."
    )


def plot_continued_training_confound(control: pd.DataFrame, out_dir: Path,
                                     stem: str = "continued_training_confound") -> str:
    rows = control[(control.metric == "mae") & (control.group == "affected")].sort_values("seed")
    clean = 100.0 * rows.clean_continued_pct_change.to_numpy()
    drift = 100.0 * rows.drift_continued_pct_change.to_numpy()
    seeds = rows.seed.astype(int).to_numpy()
    low = min(clean.min(), drift.min()) - 2
    high = max(clean.max(), drift.max()) + 2
    fig, ax = plt.subplots(figsize=(STYLE.figure_width_in, STYLE.figure_height_scatter_in), layout="constrained")
    ax.plot([low, high], [low, high], color=INK_MUTED, linestyle=(0, (3, 2)), linewidth=STYLE.diagonal_line_width,
            label="equal gain", zorder=1)
    ax.scatter(clean, drift, s=STYLE.marker_size, color=METHOD_COLORS["M1_continued_splitfed"], edgecolor="black",
               linewidth=0.5, zorder=3)
    for x_value, y_value, seed in zip(clean, drift, seeds):
        ax.annotate(str(seed), (x_value, y_value), xytext=(5, 4), textcoords="offset points",
                    fontsize=STYLE.annotation_font_size)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Clean-stream gain from continued training (%)")
    ax.set_ylabel("Drift-stream M1 gain over M0 (%)")
    ax.set_title("Continued-training confound on affected clients")
    _style_axis(ax)
    ax.legend(loc="lower right")
    _save_figure(fig, out_dir, stem)
    plt.close(fig)
    return (
        "Affected-cohort MAE gain from continued training on the clean counterfactual (R0-R1) "
        "versus the gain on the drifted stream (M0-M1), by seed. Points above the diagonal have a "
        "larger drift-stream gain than can be attributed to clean continued training alone."
    )


# --------------------------------------------------------------------------
# Per-client MAE-over-time timelines
# --------------------------------------------------------------------------

def _load_cohort(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "selected_drift_clients.csv", dtype={"cluster_id": str, "client_id": str})
    df["selected_for_drift"] = df.selected_for_drift.astype(str).str.lower().isin(["true", "1"])
    return df


def _pre_drift_anchor(run_dir: Path, cluster_id: str, client_id: str) -> int:
    path = run_dir / "R0_clean_frozen" / "pre_drift_predictions" / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
    return int(pd.read_csv(path, usecols=["window_start_index"]).window_start_index.min())


def _load_phase_series(method_dir: Path, cluster_id: str, client_id: str, phase: str) -> pd.DataFrame | None:
    if phase == "pre_drift":
        path = method_dir / "pre_drift_predictions" / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
    elif phase == "adaptation":
        path = method_dir / "adaptation_predictions" / f"cluster_{cluster_id}" / f"client_{client_id}_adaptation_predictions.csv"
    elif phase == "final_holdout":
        preferred = method_dir / "persistent_drift_final_holdout" / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
        path = preferred if preferred.exists() else method_dir / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
    else:
        raise ValueError(phase)
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=lambda c: c in {"window_start_index", "abs_error", "adaptation_epoch"})
    if "adaptation_epoch" in df.columns:
        df = df.sort_values(["window_start_index", "adaptation_epoch"]).drop_duplicates("window_start_index", keep="last")
    return df[["window_start_index", "abs_error"]]


def build_client_series(run_dir: Path, cluster_id: str, client_id: str, anchor: int, total_windows: int) -> dict[str, pd.Series]:
    series = {}
    for method in METHOD_ORDER:
        method_dir = run_dir / method
        parts = []
        for phase in ("pre_drift", "adaptation", "final_holdout"):
            part = _load_phase_series(method_dir, cluster_id, client_id, phase)
            if part is not None:
                parts.append(part)
        if not parts:
            continue
        combined = pd.concat(parts, ignore_index=True)
        combined["deployment_window"] = combined.window_start_index.astype(int) - anchor
        combined = combined.drop_duplicates("deployment_window").set_index("deployment_window").sort_index()
        series[method] = combined.abs_error.reindex(range(total_windows))
    return series


ROLLING_WINDOW = 24  # hours/windows per day, matches windowing.window_size


def plot_client_mae(client_id: str, cluster_id: str, seed: int, series: dict[str, pd.Series],
                     boundaries: tuple[int, int], total_windows: int, out_path: Path) -> None:
    fig, (ax, ax_summary) = plt.subplots(
        2, 1, figsize=(STYLE.figure_width_in, 6.0), gridspec_kw={"height_ratios": [3.4, 1.2]}, layout="constrained"
    )
    phase_ranges = ((0, boundaries[0]), (boundaries[0], boundaries[1]),
                    (boundaries[1], total_windows))
    for (start, end), color in zip(phase_ranges, PHASE_COLORS):
        ax.axvspan(start - 0.5, end - 0.5, color=color, zorder=0)

    for method in METHOD_ORDER:
        s = series.get(method)
        if s is None or s.isna().all():
            continue
        smoothed = s.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2, center=True).mean()
        ax.plot(
            smoothed.index,
            smoothed.values,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=STYLE.line_width,
            label=METHOD_SHORT[method],
            zorder=3,
        )
    for boundary in boundaries:
        ax.axvline(boundary - 0.5, color=INK_MUTED, linestyle=(0, (3, 2)), linewidth=0.9, zorder=2)
    mid_points = tuple((start + end - 1) / 2 for start, end in phase_ranges)
    for label, mid in zip(("pre-drift", "adaptation", "final holdout"), mid_points):
        ax.text(mid, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=STYLE.annotation_font_size, color=INK_SECONDARY)
    ax.set_xlim(0, total_windows - 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Deployment window")
    ax.set_ylabel(f"{ROLLING_WINDOW}-window rolling MAE")
    ax.set_title("a  Error trajectory")
    _style_axis(ax)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17), handlelength=2.4,
              columnspacing=1.0, handletextpad=0.35)

    final_values = []
    for method in METHOD_ORDER:
        s = series.get(method)
        if s is None:
            continue
        value = s.iloc[boundaries[1]:total_windows].mean()
        if pd.notna(value):
            final_values.append((method, float(value)))
    y_positions = np.arange(len(final_values))[::-1]
    for y_position, (method, value) in zip(y_positions, final_values):
        ax_summary.hlines(y_position, 0, value, color=METHOD_COLORS[method], linewidth=STYLE.line_width, alpha=0.8)
        ax_summary.scatter(value, y_position, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                           edgecolor="black", linewidth=0.45, s=STYLE.marker_size, zorder=3)
        ax_summary.annotate(f"{value:.2f}", (value, y_position), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=STYLE.annotation_font_size)
    ax_summary.set_yticks(y_positions, [METHOD_SHORT[method] for method, _ in final_values])
    ax_summary.set_xlim(left=0)
    ax_summary.margins(x=0.18, y=0.15)
    ax_summary.set_xlabel("MAE")
    ax_summary.set_title("b  Final holdout")
    _style_axis(ax_summary, grid_axis="x")

    fig.suptitle(f"Client {client_id} | cluster {cluster_id} | seed {seed}", y=1.03)
    fig.savefig(out_path, dpi=STYLE.output_dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=STYLE.output_dpi, bbox_inches="tight")
    plt.close(fig)


def build_all_client_plots(seed_runs: dict[int, Path], plots_root: Path) -> dict[int, list[tuple[str, str, str]]]:
    index: dict[int, list[tuple[str, str, str]]] = {}
    for seed, run_dir in seed_runs.items():
        cohort = _load_cohort(run_dir)
        drifted = cohort[cohort.selected_for_drift].sort_values(["cluster_id", "client_id"])
        if drifted.empty:
            continue
        sizes = _segment_sizes(run_dir)
        boundaries = (sizes["pre_drift"], sizes["pre_drift"] + sizes["adaptation"])
        total_windows = sizes["pre_drift"] + sizes["adaptation"] + sizes["final_holdout"]
        first_cluster, first_client = drifted.iloc[0].cluster_id, drifted.iloc[0].client_id
        anchor = _pre_drift_anchor(run_dir, first_cluster, first_client)
        seed_dir = plots_root / "mae" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for _, row in drifted.iterrows():
            cluster_id, client_id = row.cluster_id, row.client_id
            series = build_client_series(run_dir, cluster_id, client_id, anchor, total_windows)
            if not series:
                continue
            out_path = seed_dir / f"client_{client_id}_mae_plot.png"
            plot_client_mae(client_id, cluster_id, seed, series, boundaries, total_windows, out_path)
            entries.append((client_id, cluster_id, str(out_path.relative_to(plots_root.parent))))
        index[seed] = entries
        print(f"[seed {seed}] wrote {len(entries)} per-client MAE plots -> {seed_dir}")
    return index


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def render_report(suite_root: Path, seed_runs: dict[int, Path], tables: dict[str, pd.DataFrame],
                  retention_clients: pd.DataFrame,
                  client_index: dict[int, list[tuple[str, str, str]]], report_dir: Path,
                  plots_dir: Path, captions: dict[str, str]) -> None:
    control_impact_summary = tables["control_impact_summary_mae"]
    has_controls = not control_impact_summary.empty
    if has_controls:
        control_impact_by_seed = control_impact_summary[control_impact_summary.seed.astype(str) != "all"]
        control_impact_all = control_impact_summary[control_impact_summary.seed.astype(str) == "all"]
        control_n = int(control_impact_by_seed[control_impact_by_seed.method == "M2_oracle_client_only"].n_clients.iloc[0])
        m2_m3_control_all = control_impact_all[control_impact_all.method.isin(
            ["M2_oracle_client_only", "M3_oracle_client_ewc"])]
        full_update_methods = [method for method in ("M1_continued_splitfed", M4_AGGREGATION, M4_WINDOW)
                               if method in set(control_impact_all.method)]
        full_update_control = []
        full_update_method_text = ", ".join(METHOD_SHORT[method] for method in full_update_methods)
        for method in full_update_methods:
            row = control_impact_all[control_impact_all.method == method].iloc[0]
            pct = control_impact_by_seed[control_impact_by_seed.method == method].pct_worse_than_m0
            full_update_control.append(
                f"{METHOD_SHORT[method]}: mean {row.mean_change_vs_m0:+.3f} MAE; "
                f"{pct.min():.3g}-{pct.max():.3g}% worse than M0 across seeds"
            )
        full_update_control_text = "; ".join(full_update_control)
    seeds = sorted(seed_runs)
    plots_rel = os.path.relpath(plots_dir, report_dir)
    recovery_all = tables["recovery_by_seed"]
    recovery_mae = recovery_all[recovery_all.metric == "mae"].copy()
    available_mitigation_methods = _available_methods(
        recovery_mae, tuple(MITIGATION_METHODS)
    )
    method_key_methods = [
        "R0_clean_frozen", "M0_drift_frozen", *available_mitigation_methods, CONTROL_METHOD,
    ]
    method_key_text = "; ".join(METHOD_KEY_TEXT[method] for method in method_key_methods)
    timeline_method_text = ", ".join(METHOD_SHORT[method] for method in method_key_methods[:-1])
    retention_mae = tables["retention_by_seed"]
    retention_mae = retention_mae[retention_mae.metric == "mae"].copy()

    nmae_by_seed = tables["nmae_by_seed"]
    nmae_aggregate = tables["nmae_aggregate"]
    r0_mae_by_seed = recovery_mae.groupby("seed").E_clean.first()
    r0_nmae_by_seed = nmae_by_seed[nmae_by_seed.method == "R0_clean_frozen"].set_index("seed").mean_nmae
    mae_cv = 100.0 * r0_mae_by_seed.std() / r0_mae_by_seed.mean()
    nmae_cv = 100.0 * r0_nmae_by_seed.std() / r0_nmae_by_seed.mean()

    final_mae = recovery_mae.pivot(index="seed", columns="method", values="E_mitigated")
    winners = final_mae.idxmin(axis=1)
    winner_counts = winners.value_counts()
    winner_text = ", ".join(
        f"{METHOD_SHORT[method]} at {count}/{len(seeds)} seeds"
        for method, count in winner_counts.items()
    )
    winner_seeds = "; ".join(
        f"{METHOD_SHORT[method]}: {', '.join(str(int(seed)) for seed in winners[winners == method].index)}"
        for method in winner_counts.index
    )
    all_over_recovered = [
        int(seed) for seed, frame in recovery_mae.groupby("seed")
        if (frame.recovery_rate > 1).all()
    ]
    benefit_all = tables["client_benefit_summary_mae"]
    benefit_all = benefit_all[benefit_all.seed.astype(str) == "all"]
    benefit_text = "; ".join(
        f"{METHOD_SHORT[row.method]} {int(row.n_improved)}/{int(row.n_clients)} "
        f"({row.pct_improved:.1f}%)"
        for row in benefit_all.itertuples()
    )

    retention_pivot = retention_mae.pivot(index="seed", columns="method", values="retention_delta")
    max_seed_mean_ewc_gap = (
        retention_pivot["M3_oracle_client_ewc"] - retention_pivot["M2_oracle_client_only"]
    ).abs().max()
    paired_retention = retention_clients[retention_clients.metric == "mae"].pivot(
        index=["seed", "cluster_id", "client_id"], columns="method", values="retention_delta"
    )
    max_client_ewc_gap = (
        paired_retention["M3_oracle_client_ewc"] - paired_retention["M2_oracle_client_only"]
    ).abs().max()
    worsened_min = retention_mae.pct_worsened.min()
    worsened_max = retention_mae.pct_worsened.max()

    control_mae = tables["continued_training_control_by_seed"]
    control_mae = control_mae[(control_mae.metric == "mae") & (control_mae.group == "affected")].copy()
    confound_above = int((control_mae.drift_continued_pct_change >
                          control_mae.clean_continued_pct_change).sum())
    closest_confound_seed = int(control_mae.loc[
        (control_mae.drift_continued_pct_change - control_mae.clean_continued_pct_change).abs().idxmin(),
        "seed",
    ])
    affected_baseline = tables["affected_r0_by_seed_cluster"]
    selected_seed_mean = affected_baseline.groupby("seed").mean_mae.mean()

    valid_seed_counts = (
        recovery_all.assign(valid=recovery_all.denominator_status == "valid")
        .groupby(["metric", "seed"]).valid.all().groupby("metric").sum().to_dict()
    )

    compact_recovery = recovery_mae[
        ["seed", "method", "E_clean", "E_frozen", "E_mitigated", "recovery_rate"]
    ].copy()
    compact_recovery["recovery_rate_pct"] = 100.0 * compact_recovery.pop("recovery_rate")
    compact_recovery = compact_recovery.round(3)
    compact_retention = retention_mae[
        ["seed", "method", "retention_delta", "pct_worsened", "n_improved", "n_worsened"]
    ].round(3)
    compact_control = control_mae[
        ["seed", "clean_continued_gain", "clean_continued_pct_change",
         "drift_continued_gain", "drift_continued_pct_change"]
    ].copy()
    compact_control[["clean_continued_pct_change", "drift_continued_pct_change"]] *= 100.0
    compact_control = compact_control.rename(columns={
        "clean_continued_pct_change": "clean_gain_pct",
        "drift_continued_pct_change": "drift_gain_pct",
    }).round(3)

    lines: list[str] = []
    lines.append("# Drift and fault mitigation report\n")
    lines.append(
        f"Suite: `{suite_root.name}`  \nClient-selection seeds: {', '.join(str(s) for s in seeds)}. "
        "The experiment, data-order, drift, checkpoint-training, and adaptation seeds are fixed; "
        "one shared checkpoint H is reused across the suite.\n"
    )
    lines.append(f"> **Method key.** {method_key_text}.\n")

    lines.append("## 1. Integrity and metric scope\n")
    lines.append(f"All {len(seeds)} runs passed the recorded acceptance and prediction-key reproducibility checks.\n")
    lines.append(_md(tables["data_integrity"]) + "\n")
    lines.append(
        f"Recovery-rate denominators are valid at {valid_seed_counts.get('mae', 0)}/{len(seeds)} seeds "
        f"for MAE, {valid_seed_counts.get('rmse', 0)}/{len(seeds)} for RMSE, and "
        f"{valid_seed_counts.get('smape', 0)}/{len(seeds)} for SMAPE. Cross-seed recovery conclusions "
        "therefore use MAE. The complete metric rows, including denominator status, are available in "
        "[recovery_by_seed.csv](tables/recovery_by_seed.csv).\n"
    )

    lines.append("## 2. Primary result: final-holdout MAE and recovery\n")
    lines.append(
        f"The lowest final affected-cohort MAE is achieved by {winner_text}. The seed-specific winners "
        f"are {winner_seeds}; the ordering is not uniform across cohorts. "
        + (f"All mitigation candidates over-recover at seeds {', '.join(map(str, all_over_recovered))}. "
           if all_over_recovered else "")
        + "Recovery above 100% means final MAE is lower than the clean R0 counterfactual, not an error in the plot.\n"
    )
    lines.append(f"![Affected-cohort MAE by seed and method]({plots_rel}/affected_mae_by_seed.png)\n")
    lines.append(f"*{captions['affected_mae']}*\n")
    lines.append(f"![Mean, std, and median MAE by seed and method]({plots_rel}/affected_mae_distribution_by_seed.png)\n")
    lines.append(f"*{captions['affected_mae_distribution']}*\n")
    lines.append(f"![MAE recovery rate by seed]({plots_rel}/recovery_rate_by_seed.png)\n")
    lines.append(f"*{captions['recovery_rate']}*\n")
    lines.append("### Aggregate across seeds (MAE; mean and standard deviation)\n")
    lines.append(_md(tables["recovery_aggregate_mae"]) + "\n")
    lines.append("### Per-seed MAE\n")
    lines.append(_md(compact_recovery) + "\n")

    lines.append("## 3. Normalized MAE (NMAE): fair comparison across seeds and clients\n")
    lines.append(
        "Raw MAE is not directly comparable across seeds or clients: which clients a cohort happens to "
        "contain (see cohort heterogeneity, section 8) changes the *natural magnitude* of the signal being "
        "forecast, independent of how well any method performs. NMAE divides each client's final-holdout "
        "MAE by that same client's own z-score scale (the std of its clean-training target, from the "
        "offline preparation manifest) before aggregating, expressing error in client-relative "
        f"standard-deviation units. For R0, the raw-MAE coefficient of variation across the {len(seeds)} seeds is "
        f"{mae_cv:.1f}%, versus {nmae_cv:.1f}% for NMAE -- "
        + ("a substantially tighter spread once cohort-scale differences are removed.\n"
           if nmae_cv < mae_cv else
           "not a tighter spread here, so cohort-scale differences are not the dominant source of "
           "this suite's cross-seed variation -- read this alongside the cohort-heterogeneity section.\n")
    )
    lines.append(f"![Affected-cohort NMAE by seed and method]({plots_rel}/affected_nmae_by_seed.png)\n")
    lines.append(f"*{captions['affected_nmae']}*\n")
    lines.append(f"### Aggregate across seeds (NMAE; mean of the {len(seeds)} per-seed means, and their std)\n")
    lines.append(_md(nmae_aggregate) + "\n")
    lines.append("### Per-seed NMAE\n")
    lines.append(_md(nmae_by_seed) + "\n")
    lines.append("[Full per-client NMAE](tables/nmae_by_seed.csv) and "
                 "[aggregate](tables/nmae_aggregate.csv).\n")

    lines.append("## 4. Client-level mitigation benefit\n")
    lines.append(
        f"The paired M0 comparison shows improvement for {benefit_text}. This view prevents a small "
        "number of high-error clients from hiding how consistently the methods help individual clients.\n"
    )
    lines.append(f"![Per-client MAE reduction]({plots_rel}/client_mae_reduction.png)\n")
    lines.append(f"*{captions['client_benefit']}*\n")
    lines.append(_md(benefit_all.round(3)) + "\n")

    lines.append("## 5. Collateral impact on non-drifted (control) clients\n")
    if has_controls:
        lines.append(
            f"M2 and M3 update only the oracle-selected affected clients: control-client MAE is identical "
            f"to M0 for all {control_n} controls at every seed -- {int(m2_m3_control_all.n_worse_than_m0.sum())} "
            f"worse than M0 out of {int(m2_m3_control_all.n_clients.sum())} client-seed evaluations, exactly "
            "0% by construction. Some controls are fault-injected, so equality to M0 does not imply equality "
            f"to clean R0. {full_update_method_text} instead FedAvg-aggregate across the whole cluster and can "
            "move control clients in either direction. Their M0-relative adaptation effects are: "
            f"{full_update_control_text}. The per-client distribution remains important because a favorable "
            "population mean can coexist with collateral harm to individual controls.\n"
        )
        lines.append(f"![Collateral impact on control clients]({plots_rel}/control_client_harm.png)\n")
        lines.append(f"*{captions['control_impact']}*\n")
        lines.append(_md(control_impact_all.round(3)) + "\n")
        lines.append("[Full per-seed control-impact table](tables/control_impact_summary_mae.csv).\n")
    else:
        lines.append(
            "Not applicable to this suite: `population.drifted_per_cluster` equals the full cluster size, "
            "so every client is drift-affected and there are no control clients to compare against.\n"
        )

    lines.append("## 6. Clean-task retention and EWC\n")
    lines.append(
        "Retention delta is defined as post-adaptation clean/counterfactual MAE minus R0 clean MAE. "
        "Thus positive is forgetting and negative is improvement. The seed-level mean is negative for "
        f"both M2 and M3 at every seed, yet {worsened_min:.3g}-{worsened_max:.3g}% of individual clients "
        "worsen, so a mean alone is insufficient. "
        f"M3 and M2 differ by at most {max_seed_mean_ewc_gap:.4g} MAE in a seed mean and "
        f"{max_client_ewc_gap:.4g} MAE for any paired client evaluation; this suite shows a numerically "
        "negligible EWC effect at lambda=1.0, without constituting a significance test.\n"
    )
    lines.append(f"![Client-level retention and EWC effect]({plots_rel}/retention_distribution.png)\n")
    lines.append(f"*{captions['retention']}*\n")
    lines.append("### Per-seed MAE retention\n")
    lines.append(_md(compact_retention) + "\n")
    lines.append("[Full retention table](tables/retention_by_seed.csv) and "
                 "[client-level summary](tables/retention_client_summary_mae.csv).\n")

    lines.append("## 7. Continued-training confound\n")
    lines.append(
        "R1 estimates improvement from additional training on the clean stream, while M1-M0 combines "
        "continued training with the drifted stream. The drift-stream percentage gain is larger at "
        f"{confound_above}/{len(seeds)} seeds, with seed {closest_confound_seed} closest to the equality line. This supports "
        "a drift-specific contribution but does not identify it causally.\n"
    )
    lines.append(f"![Continued-training confound]({plots_rel}/continued_training_confound.png)\n")
    lines.append(f"*{captions['confound']}*\n")
    lines.append(_md(compact_control) + "\n")
    lines.append("[All metrics and cohorts](tables/continued_training_control_by_seed.csv).\n")

    per_cluster_n = int(affected_baseline.n_clients.iloc[0])
    lines.append("## 8. Cohort and cluster heterogeneity\n")
    lines.append(
        f"Affected-cohort clean R0 MAE ranges from {selected_seed_mean.min():.3f} to "
        f"{selected_seed_mean.max():.3f} across seeds. Every cohort contains exactly {per_cluster_n} selected "
        "client" + ("s" if per_cluster_n != 1 else "") + " from each of the four clusters, so this variation is caused by which clients are selected "
        "within clusters—not by changing cluster counts. Cluster 4 is the most difficult at the population "
        "level, but the seed-by-cluster panel shows that within-cluster selection also matters.\n"
    )
    lines.append(f"![Cohort and cluster heterogeneity]({plots_rel}/cohort_heterogeneity.png)\n")
    lines.append(f"*{captions['heterogeneity']}*\n")
    lines.append(_md(tables["cluster_r0_heterogeneity"]) + "\n")
    lines.append("[Seed-by-cluster values](tables/affected_r0_by_seed_cluster.csv).\n")

    lines.append("## 9. Limitations\n")
    lines.append(
        f"- The {len(seeds)} seeds vary client selection only; they are not independent end-to-end training runs.\n"
        "- Client-seed observations are descriptive and not all independent because a client may be selected "
        "under more than one seed. The plotted 95% intervals quantify within-cohort client dispersion, not "
        "uncertainty over retraining the complete system.\n"
        "- Recovery rate is a ratio and becomes sensitive when drift damage is small; always read it beside "
        "the absolute MAE figure.\n"
        "- RMSE and SMAPE recovery denominators are invalid for part or all of this suite, so only MAE supports "
        "the cross-seed recovery comparison.\n"
        "- R0 and M0 were not evaluated during adaptation; their timeline gaps denote missing evaluation, not zero error.\n"
    )

    lines.append("## 10. Per-client MAE timelines\n")
    lines.append(
        f"Each affected client has a two-panel figure. The upper panel shows the {ROLLING_WINDOW}-window "
        "centered rolling MAE with shaded pre-drift, adaptation, and final-holdout phases. The lower panel "
        f"ranks exact final-holdout MAE for {timeline_method_text}. R0/M0 gaps during adaptation reflect that the "
        "frozen methods were not evaluated in that phase.\n"
    )
    for seed in seeds:
        entries = client_index.get(seed, [])
        lines.append(f"\n**Seed {seed}** ({len(entries)} drifted clients):\n")
        for client_id, cluster_id, rel_path in entries:
            lines.append(f"- [client {client_id} (cluster {cluster_id})]({rel_path})")
    lines.append("")

    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True, help="top-level suite folder containing runseed<N>/ subdirectories")
    parser.add_argument("--output", type=Path, default=None, help="defaults to <suite-root>/report")
    args = parser.parse_args()

    suite_root = args.suite_root.resolve()
    report_dir = (args.output or (suite_root / "report")).resolve()
    tables_dir = report_dir / "tables"
    plots_dir = report_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    seed_runs = discover_seed_runs(suite_root)
    print(f"Found {len(seed_runs)} completed seed runs: {sorted(seed_runs)}")

    affected_clients = load_affected_clients(seed_runs)
    retention_clients = load_retention_clients(seed_runs)
    client_benefits = build_client_benefits(affected_clients)
    control_clients = load_control_clients(seed_runs)
    control_impact = build_control_impact(control_clients)
    tables = build_tables(seed_runs, affected_clients, retention_clients, client_benefits, control_impact)
    for name, df in tables.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)
    print(f"Wrote {len(tables)} tables -> {tables_dir}")

    recovery_mae = tables["recovery_by_seed"]
    recovery_mae = recovery_mae[recovery_mae.metric == "mae"]
    client_stats = load_affected_client_stats(affected_clients)

    client_scales = load_client_scales(seed_runs)
    affected_clients_scaled = attach_nmae(affected_clients, client_scales)
    nmae_stats = load_affected_client_nmae_stats(affected_clients_scaled)
    tables["nmae_by_seed"] = nmae_stats.round(4)
    tables["nmae_aggregate"] = summarise_nmae_aggregate(nmae_stats)
    tables["nmae_by_seed"].to_csv(tables_dir / "nmae_by_seed.csv", index=False)
    tables["nmae_aggregate"].to_csv(tables_dir / "nmae_aggregate.csv", index=False)

    captions = {
        "affected_mae": plot_recovery_by_seed(client_stats, plots_dir),
        "affected_mae_distribution": plot_mae_distribution_by_seed(client_stats, plots_dir),
        "affected_nmae": plot_nmae_by_seed(nmae_stats, plots_dir),
        "recovery_rate": plot_recovery_rate_by_seed(recovery_mae, plots_dir),
        "client_benefit": plot_client_benefit_distribution(client_benefits, plots_dir),
        "control_impact": plot_control_impact_distribution(control_impact, plots_dir),
        "retention": plot_retention_distribution(retention_clients, plots_dir),
        "confound": plot_continued_training_confound(tables["continued_training_control_by_seed"], plots_dir),
        "heterogeneity": plot_cohort_heterogeneity(
            tables["cluster_r0_heterogeneity"], tables["affected_r0_by_seed_cluster"], plots_dir
        ),
    }
    print(f"Wrote {sum(1 for c in captions.values() if c is not None)} summary figures (.pdf + .png) -> {plots_dir}")

    # These two names were produced by the pre-publication version and are no
    # longer referenced. Remove only these known generated artifacts so a
    # regenerated report does not mix legacy and current visual styles.
    for legacy_stem in ("recovery_rate_boxplot", "cluster_heterogeneity"):
        for suffix in (".png", ".pdf"):
            (plots_dir / f"{legacy_stem}{suffix}").unlink(missing_ok=True)

    client_index = build_all_client_plots(seed_runs, plots_dir)

    render_report(suite_root, seed_runs, tables, retention_clients, client_index, report_dir,
                  plots_dir, captions)
    print(f"Wrote report -> {report_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
