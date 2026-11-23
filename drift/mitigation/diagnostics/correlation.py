"""Part C: descriptive association between representation shift and
downstream performance.

Reads representation.py's own already-written representation_per_client.csv
and the canonical evaluator's already-written comparison/*.csv outputs
(affected_client_comparison.csv, <method>_retention_per_client.csv) --
both read-only, neither is recomputed here. Reports Pearson and Spearman
correlations with sample counts; never claims causality or statistical
significance from correlation alone (see the descriptive language in
metadata.json and the CSV's own column names).

    python -m drift.mitigation.diagnostics.correlation \
        --representation-per-client diagnostics/representation_mismatch/<suite_id>/representation_per_client.csv \
        --comparison-root <suite_root>/comparison --method M2 --output diagnostics/representation_correlation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from . import common


def _short_method_name(method: str) -> str:
    return method.split("_", 1)[0].lower()


def run_correlation_diagnostic(
    *, representation_per_client_csv: str | Path, comparison_root: str | Path, method: str, output: str | Path,
    segment: str = "drifted_final_holdout",
) -> dict[str, Any]:
    representation_per_client_csv, comparison_root, output = (
        Path(representation_per_client_csv), Path(comparison_root), Path(output)
    )
    rep = pd.read_csv(representation_per_client_csv, dtype={"cluster_id": str, "client_id": str})
    rep = rep.loc[(rep.method == method) & (rep.segment == segment)].copy()

    affected_path = comparison_root / "affected_client_comparison.csv"
    affected = pd.read_csv(affected_path, dtype={"cluster_id": str, "client_id": str})
    method_mae = affected.loc[affected.method == method, ["cluster_id", "client_id", "mae"]].rename(columns={"mae": "method_mae"})
    merged = rep.merge(method_mae, on=["cluster_id", "client_id"], how="inner")

    m1_mae = affected.loc[affected.method == "M1_continued_splitfed", ["cluster_id", "client_id", "mae"]].rename(columns={"mae": "m1_mae"})
    if not m1_mae.empty:
        merged = merged.merge(m1_mae, on=["cluster_id", "client_id"], how="left")
        merged["method_minus_m1_mae"] = merged["method_mae"] - merged["m1_mae"]

    recovery_path = comparison_root / "recovery_summary.csv"
    if recovery_path.is_file():
        recovery = pd.read_csv(recovery_path)
        method_recovery = recovery.loc[(recovery.method == method) & (recovery.metric == "mae"), "recovery_rate"]
        merged["method_recovery_rate_suite_level"] = float(method_recovery.iloc[0]) if not method_recovery.empty else None

    retention_path = comparison_root / f"{_short_method_name(method)}_retention_per_client.csv"
    if retention_path.is_file():
        retention = pd.read_csv(retention_path, dtype={"cluster_id": str, "client_id": str})
        retention_mae = retention.loc[retention.metric == "mae", ["cluster_id", "client_id", "retention_delta"]]
        merged = merged.merge(retention_mae, on=["cluster_id", "client_id"], how="left")

    representation_metrics = ["representation_l2", "representation_l2_normalized", "cosine_similarity", "representation_mean_shift"]
    performance_metrics = [col for col in ("method_mae", "method_minus_m1_mae", "retention_delta") if col in merged.columns]

    rows: list[dict[str, Any]] = []
    for rep_metric in representation_metrics:
        if rep_metric not in merged.columns:
            continue
        for perf_metric in performance_metrics:
            assoc = common.pearson_spearman(merged[rep_metric], merged[perf_metric])
            rows.append({
                "representation_metric": rep_metric, "performance_metric": perf_metric,
                "method": method, "segment": segment, **assoc,
            })

    result = pd.DataFrame(rows)
    common.write_csv(output / "representation_performance_correlations.csv", result)
    common.write_json(output / "metadata.json", {
        "note": "Pearson/Spearman are descriptive associations only. No causal or "
                "statistical-significance claim is made from correlation alone.",
        "method": method, "segment": segment, "n_clients_matched": int(len(merged)),
    })
    return {"output": str(output), "rows": len(result), "n_clients_matched": int(len(merged))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Correlate representation shift with performance (descriptive only, read-only).")
    parser.add_argument("--representation-per-client", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--method", default="M2_oracle_client_only")
    parser.add_argument("--segment", default="drifted_final_holdout")
    parser.add_argument("--output", type=Path, default=Path("diagnostics/representation_correlation"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_correlation_diagnostic(
        representation_per_client_csv=args.representation_per_client, comparison_root=args.comparison_root,
        method=args.method, segment=args.segment, output=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())