"""Part H/I: cross-lambda summary tables and the stability-plasticity view.

Reads one or more lambda_sweep_summary.csv files (one row per lambda,
optionally one file per client-selection seed from repeated sweeps) and
produces an aggregate table and the stability-plasticity trade-off table
Part I asks for: adaptation/drift performance on one axis, clean retention
degradation on the other, for every lambda.

Never selects an "optimal lambda" from held-out final-holdout test data
automatically -- this module only ever reports the trade-off. If a
descriptive reference point is computed (see ``descriptive_lambda_note``),
it is explicitly labeled descriptive and states which metric it used;
callers must not treat it as a tuned hyperparameter recommendation.

    python -m drift.mitigation.diagnostics.summarize_diagnostics \
        --lambda-sweep-summary diagnostics/ewc_lambda_sweep/<suite_id>/comparison/lambda_sweep_summary.csv [...] \
        --output diagnostics/ewc_lambda_sweep/<suite_id>/comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from . import common


def load_summaries(paths: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(Path(path)) for path in paths]
    return pd.concat(frames, ignore_index=True)


def aggregate_across_seeds(combined: pd.DataFrame) -> pd.DataFrame:
    return combined.groupby("lambda").agg(
        n_runs=("suite_id", "nunique"),
        mean_E_M3_affected=("E_M3_affected", "mean"), median_E_M3_affected=("E_M3_affected", "median"),
        mean_recovery_rate=("recovery_rate", "mean"), median_recovery_rate=("recovery_rate", "median"),
        mean_retention_delta=("retention_delta_vs_R0", "mean"), median_retention_delta=("retention_delta_vs_R0", "median"),
    ).reset_index()


def stability_plasticity_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    """One row per lambda: adaptation/drift performance vs clean retention
    degradation. This is the primary EWC trade-off view (Part I) -- not a
    recommendation, purely the measured trade-off curve.
    """
    table = aggregate[["lambda", "mean_E_M3_affected", "mean_recovery_rate", "mean_retention_delta"]].rename(columns={
        "mean_E_M3_affected": "adaptation_error_mae",
        "mean_recovery_rate": "adaptation_recovery_rate",
        "mean_retention_delta": "clean_retention_degradation",
    })
    return table.sort_values("lambda").reset_index(drop=True)


def descriptive_lambda_note(table: pd.DataFrame) -> dict[str, Any]:
    """A purely descriptive reference point -- NOT a tuned/optimal-lambda
    selection, and NOT computed via any held-out test-set hyperparameter
    search. States exactly which metric it is based on so callers cannot
    mistake it for a recommendation.
    """
    if table.empty:
        return {"descriptive_only": True, "note": "no lambda rows available"}
    valid = table.dropna(subset=["clean_retention_degradation"])
    if valid.empty:
        return {"descriptive_only": True, "note": "no retention_delta data available to describe the trade-off"}
    smallest_measured_degradation = valid.loc[valid.clean_retention_degradation.idxmin()]
    return {
        "descriptive_only": True,
        "label": "gradient-scale/trade-off reference point, NOT an optimal-lambda recommendation",
        "metric_used": "clean_retention_degradation (mean_retention_delta_vs_R0)",
        "lambda_with_smallest_measured_clean_retention_degradation": float(smallest_measured_degradation["lambda"]),
        "note": (
            "this is the lambda with the smallest measured clean-retention degradation among the "
            "lambdas actually swept -- not a search over held-out test data, and not necessarily best "
            "on adaptation performance; read the full stability_plasticity_table.csv for the trade-off"
        ),
    }


def maybe_plot_stability_plasticity(table: pd.DataFrame, output_path: Path) -> bool:
    """Optional. Returns True if a plot was written, False if matplotlib is
    unavailable -- the diagnostic command must succeed either way.
    """
    if not common.matplotlib_available() or table.empty:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(table["adaptation_recovery_rate"], table["clean_retention_degradation"], marker="o")
    for _, row in table.iterrows():
        ax.annotate(f"λ={row['lambda']:g}", (row["adaptation_recovery_rate"], row["clean_retention_degradation"]))
    ax.set_xlabel("adaptation recovery rate")
    ax.set_ylabel("clean retention degradation (MAE delta vs R0)")
    ax.set_title("EWC stability-plasticity trade-off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return True


def run_summarize_diagnostics(*, lambda_sweep_summaries: list[str | Path], output: str | Path) -> dict[str, Any]:
    output = Path(output)
    combined = load_summaries(lambda_sweep_summaries)
    aggregate = aggregate_across_seeds(combined)
    table = stability_plasticity_table(aggregate)
    note = descriptive_lambda_note(table)

    common.write_csv(output / "lambda_sweep_aggregate.csv", aggregate)
    common.write_csv(output / "stability_plasticity_table.csv", table)
    common.write_json(output / "descriptive_lambda_note.json", note)
    plotted = maybe_plot_stability_plasticity(table, output / "plots" / "stability_plasticity.png")

    return {"output": str(output), "lambdas_summarized": len(table), "plot_written": plotted}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate lambda-sweep summaries into the stability-plasticity trade-off view (never selects an optimal lambda).")
    parser.add_argument("--lambda-sweep-summary", dest="summaries", action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/ewc_lambda_sweep/comparison"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_summarize_diagnostics(lambda_sweep_summaries=args.summaries, output=args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())