"""Part K: cross-suite representation comparison.

Accepts multiple ``--suite-root`` arguments (each an already-completed
suite) and builds one ``cross_suite_representation_summary.csv`` comparing
drift parameters, method, adaptation epochs, representation shift, and
downstream MAE/gap/retention across suites -- read-only throughout.

This module does NOT recompute representations itself: it reads each
suite's own already-produced ``representation_summary.csv`` (i.e. you run
representation.py against each suite root first) plus each suite's
``suite_resolved_config.yaml`` and ``comparison/*.csv``. If two suites have
incompatible model architectures, they are still included in the output
but explicitly flagged (``architecture_compatible=False``,
representation-shift columns left blank) rather than silently compared as
if commensurate -- see ``_architecture_signature``.

    python -m drift.mitigation.diagnostics.cross_suite \
        --suite-root <mild_ep1> --representation-summary <mild_ep1_repr.csv> \
        --suite-root <mild_ep3> --representation-summary <mild_ep3_repr.csv> \
        --suite-root <strong_ep1> --representation-summary <strong_ep1_repr.csv> \
        --output diagnostics/cross_suite
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import common

M1_METHOD = "M1_continued_splitfed"
M2_METHOD = "M2_oracle_client_only"


def _architecture_signature(config: dict[str, Any]) -> tuple:
    model = config.get("model", {}) or {}
    windowing = config.get("windowing", {}) or {}
    return (
        model.get("input_size"), model.get("hidden_size"), model.get("num_layers"),
        model.get("split_layer"), model.get("output_size"), model.get("bidirectional"),
        windowing.get("window_size"), windowing.get("forecast_horizon"),
    )


def _read_group_mae(comparison_root: Path, method: str, group: str = "affected") -> float | None:
    path = comparison_root / "run_comparison.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    row = frame.loc[(frame.method == method) & (frame.group == group), "mae"]
    return float(row.iloc[0]) if not row.empty else None


def _read_retention_delta(comparison_root: Path, method_short: str, metric: str = "mae") -> float | None:
    path = comparison_root / "retention_summary.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "method" not in frame.columns:
        return None
    row = frame.loc[(frame.method.str.lower().str.startswith(method_short)) & (frame.get("metric", pd.Series(dtype=str)) == metric), "retention_delta"]
    return float(row.iloc[0]) if not row.empty else None


def build_cross_suite_row(
    *, suite_root: Path, representation_summary_path: Path | None, method: str, reference_architecture: tuple | None,
) -> dict[str, Any]:
    config = yaml.safe_load((suite_root / "suite_resolved_config.yaml").read_text(encoding="utf-8"))
    comparison_root = suite_root / "comparison"
    architecture = _architecture_signature(config)
    architecture_compatible = reference_architecture is None or architecture == reference_architecture

    row: dict[str, Any] = {
        "suite_id": config.get("suite_id"),
        "drift_intercept_delta": config.get("process", {}).get("intercept_delta"),
        "drift_ar_coefficient_delta": str(config.get("process", {}).get("ar_coefficient_delta")),
        "method": method,
        "adaptation_epochs": (config.get("adaptation", {}) or {}).get(
            "client_only_adaptation_epochs" if method == M2_METHOD else "adaptation_epochs"
        ),
        "architecture_compatible": architecture_compatible,
        "final_method_MAE": _read_group_mae(comparison_root, method),
        "M0_MAE": _read_group_mae(comparison_root, "M0_drift_frozen"),
        "M1_minus_M2_MAE_gap": None,
        "clean_retention_delta": _read_retention_delta(comparison_root, method.split("_", 1)[0].lower()) if method == M2_METHOD else None,
        "mean_representation_shift": None, "median_representation_shift": None, "mean_cosine_similarity": None,
        "H_server_MAE_delta_from_representation_shift": None,
    }

    m1_mae = _read_group_mae(comparison_root, M1_METHOD)
    m2_mae = _read_group_mae(comparison_root, M2_METHOD)
    if m1_mae is not None and m2_mae is not None:
        row["M1_minus_M2_MAE_gap"] = m1_mae - m2_mae

    if representation_summary_path is not None and representation_summary_path.is_file() and architecture_compatible:
        rep = pd.read_csv(representation_summary_path)
        rep = rep.loc[(rep.method == method) & (rep.segment == "drifted_final_holdout")]
        if not rep.empty:
            row["mean_representation_shift"] = float(rep["mean_representation_l2"].iloc[0])
            row["median_representation_shift"] = float(rep["median_representation_l2"].iloc[0])
            row["mean_cosine_similarity"] = float(rep["mean_cosine_similarity"].iloc[0])

    server_compat_path = None
    if representation_summary_path is not None:
        server_compat_path = representation_summary_path.parent / "server_compatibility_summary.csv"
    if server_compat_path is not None and server_compat_path.is_file() and architecture_compatible:
        server = pd.read_csv(server_compat_path)
        server = server.loc[(server.method == method) & (server.segment == "drifted_final_holdout")]
        if not server.empty:
            row["H_server_MAE_delta_from_representation_shift"] = float(server["mean_representation_induced_MAE_delta"].iloc[0])

    return row


def run_cross_suite_comparison(
    *, suite_roots: list[str | Path], representation_summaries: list[str | Path | None], method: str, output: str | Path,
) -> dict[str, Any]:
    if len(suite_roots) != len(representation_summaries):
        raise ValueError("suite_roots and representation_summaries must be the same length (pass None entries where absent)")
    output = Path(output)
    suite_roots = [Path(item) for item in suite_roots]
    representation_summaries = [Path(item) if item else None for item in representation_summaries]

    reference_architecture = None
    rows: list[dict[str, Any]] = []
    for index, (suite_root, rep_summary) in enumerate(zip(suite_roots, representation_summaries)):
        config = yaml.safe_load((suite_root / "suite_resolved_config.yaml").read_text(encoding="utf-8"))
        architecture = _architecture_signature(config)
        if reference_architecture is None:
            reference_architecture = architecture
        row = build_cross_suite_row(suite_root=suite_root, representation_summary_path=rep_summary, method=method, reference_architecture=reference_architecture)
        rows.append(row)

    result_frame = pd.DataFrame(rows)
    common.write_csv(output / "cross_suite_representation_summary.csv", result_frame)
    incompatible = result_frame.loc[~result_frame.architecture_compatible, "suite_id"].tolist() if not result_frame.empty else []
    metadata = common.diagnostic_provenance(
        diagnostic_name="cross_suite", repo_root=Path(__file__).resolve().parents[3],
        parameters={"method": method, "suite_roots": [str(item) for item in suite_roots]},
    )
    metadata["architecture_incompatible_suites"] = incompatible
    common.write_json(output / "metadata.json", metadata)
    return {"output": str(output), "rows": len(result_frame), "architecture_incompatible_suites": incompatible}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-suite representation comparison across multiple already-completed suites (read-only).")
    parser.add_argument("--suite-root", dest="suite_roots", action="append", required=True)
    parser.add_argument("--representation-summary", dest="representation_summaries", action="append", default=[], help="one per --suite-root, in order; omit an entry with an empty string if unavailable for that suite")
    parser.add_argument("--method", default="M2_oracle_client_only")
    parser.add_argument("--output", type=Path, default=Path("diagnostics/cross_suite"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = list(args.representation_summaries) + [None] * (len(args.suite_roots) - len(args.representation_summaries))
    summaries = [item if item else None for item in summaries]
    result = run_cross_suite_comparison(suite_roots=args.suite_roots, representation_summaries=summaries, method=args.method, output=args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())