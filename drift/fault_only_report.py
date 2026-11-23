#!/usr/bin/env python3
"""Slim cross-seed report for the fault-only (R0/M0/M1/M4) ablation.

drift/mitigation_report.py cannot run against this suite family: its
retention and control-impact sections are hard-wired to the M2/M3 oracle
methods (e.g. literal "M2_oracle_client_only"/"M3_oracle_client_ewc"
lookups), which drift/run_fault_only_sweep.sh deliberately excludes to keep
this ablation scoped to M1/M4 plus the two baselines
evaluate_completed_suite requires. Rather than partially patch a tool built
around comparing all four candidate methods together, this script covers
just the part that generalizes cleanly: per-seed comparison/recovery_summary.csv
and comparison/run_comparison.csv, both already keyed only on whichever
methods actually ran, aggregated across seeds into cross-seed mean/std
tables, one plot, and a short markdown summary.

Usage: python -m drift.fault_only_report --suite-root <dir> [--output <dir>]
  <dir> must contain runseed<N>/comparison/ for each completed seed
  (see drift/run_fault_only_sweep.sh's epoch<E>/ directories).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def discover_seed_runs(suite_root: Path) -> dict[int, Path]:
    runs = {}
    for path in sorted(suite_root.glob("runseed*")):
        match = re.fullmatch(r"runseed(\d+)", path.name)
        if match and (path / "comparison").is_dir():
            runs[int(match.group(1))] = path
    if not runs:
        raise SystemExit(f"no completed runseed<N>/comparison/ found under {suite_root}")
    return runs


def load_recovery(seed_runs: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for seed, run_dir in seed_runs.items():
        frame = pd.read_csv(run_dir / "comparison" / "recovery_summary.csv")
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_run_comparison(seed_runs: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for seed, run_dir in seed_runs.items():
        frame = pd.read_csv(run_dir / "comparison" / "run_comparison.csv")
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    return (
        recovery.groupby(["metric", "method"])
        .agg(
            n_seeds=("recovery_rate", "size"),
            mean_recovery_rate=("recovery_rate", "mean"),
            std_recovery_rate=("recovery_rate", "std"),
            mean_E_clean=("E_clean", "mean"),
            mean_E_frozen=("E_frozen", "mean"),
            mean_E_mitigated=("E_mitigated", "mean"),
            mean_recovered_error=("recovered_error", "mean"),
        )
        .reset_index()
    )


def aggregate_run_comparison(run_comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        run_comparison.groupby(["method", "group"])
        .agg(
            n_seeds=("mae", "size"),
            mean_mae=("mae", "mean"), std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
            mean_smape=("smape", "mean"), std_smape=("smape", "std"),
        )
        .reset_index()
    )


def plot_recovery_rate(recovery: pd.DataFrame, out_dir: Path, stem: str = "fault_only_recovery_rate") -> str:
    mae = recovery[recovery.metric == "mae"].copy()
    methods = sorted(mae.method.unique())
    seeds = sorted(mae.seed.unique())
    positions = np.arange(len(seeds))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    for index, method in enumerate(methods):
        by_seed = mae[mae.method == method].set_index("seed").recovery_rate
        values = [by_seed.get(seed, np.nan) for seed in seeds]
        ax.bar(positions + index * width, values, width=width, label=method)
    ax.axhline(1.0, color="black", linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_xticks(positions + width * (len(methods) - 1) / 2, [str(seed) for seed in seeds])
    ax.set_xlabel("Client-selection seed")
    ax.set_ylabel("Recovery rate (MAE)")
    ax.set_title("Fault-only ablation: recovery rate by seed")
    ax.legend(fontsize=8)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_dir / f"{stem}{suffix}", dpi=150)
    plt.close(fig)
    return stem


def render_markdown(
    report_dir: Path, recovery_agg: pd.DataFrame, run_agg: pd.DataFrame, seeds: list[int], plot_stem: str,
) -> None:
    lines = [
        "# Fault-only mitigation ablation",
        "",
        (
            "Real process drift is disabled (`population.drifted_per_cluster: 0`); the only "
            "perturbation is controlled-window fault injection, targeting the same client cohort "
            "the drift matrix would select for `drift_count=8` at each seed. `selected_for_drift` "
            "in the underlying comparison CSVs has been repurposed to mark that fault cohort -- see "
            "each seed's `manifests/fault_only_relabel_note.json`. Methods: R0_clean_frozen and "
            "M0_drift_frozen (baselines evaluate_completed_suite requires) plus the two mitigation "
            "candidates under test, M1_continued_splitfed and M4_quality_aware_splitfed.",
        ),
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Recovery rate (mean +/- std across seeds, MAE)",
        "",
        recovery_agg.loc[
            recovery_agg.metric == "mae",
            ["method", "n_seeds", "mean_recovery_rate", "std_recovery_rate", "mean_E_mitigated"],
        ].round(4).to_markdown(index=False),
        "",
        f"![recovery rate](plots/{plot_stem}.png)",
        "",
        "## Fault-affected vs control MAE/RMSE/SMAPE (mean +/- std across seeds)",
        "",
        run_agg[["method", "group", "n_seeds", "mean_mae", "std_mae", "mean_rmse", "mean_smape"]]
        .round(4).to_markdown(index=False),
        "",
    ]
    (report_dir / "fault_only_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(suite_root: str | Path, output: str | Path | None = None) -> Path:
    suite_root = Path(suite_root).resolve()
    report_dir = (Path(output) if output else (suite_root / "report")).resolve()
    tables_dir, plots_dir = report_dir / "tables", report_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    seed_runs = discover_seed_runs(suite_root)
    recovery = load_recovery(seed_runs)
    run_comparison = load_run_comparison(seed_runs)
    recovery_agg = aggregate_recovery(recovery)
    run_agg = aggregate_run_comparison(run_comparison)

    recovery.to_csv(tables_dir / "recovery_by_seed.csv", index=False)
    recovery_agg.to_csv(tables_dir / "recovery_aggregate.csv", index=False)
    run_comparison.to_csv(tables_dir / "run_comparison_by_seed.csv", index=False)
    run_agg.to_csv(tables_dir / "run_comparison_aggregate.csv", index=False)

    plot_stem = plot_recovery_rate(recovery, plots_dir)
    render_markdown(report_dir, recovery_agg, run_agg, sorted(seed_runs), plot_stem)
    return report_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report_dir = build_report(args.suite_root, args.output)
    print(f"Wrote fault-only report -> {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
