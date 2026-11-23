"""Matrix-wide rollup: concatenate every condition's own report tables
(built by condition_report.py) into one cross-condition CSV per metric,
tagged with the (drift_type, severity, drift_count, epochs) that
distinguishes each condition. Reads only the condition-level
``tables/*.csv`` that :mod:`drift.mitigation_report` already computed and
validated per condition -- this module recomputes no statistic itself, it
only reshapes and concatenates.

Including a condition here is also its archival trigger: once a
condition's results are folded into the matrix-wide summary, its
condition_status.csv row advances report_complete -> archived.
"""
from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import Any

from . import condition_report, paths, status as status_mod

# name -> whether to keep only the seed=="all" aggregate rows (the
# per-condition cross-seed summary) rather than every per-seed row, which
# would make the matrix-wide table 5x larger for no analytical benefit at
# this rollup level (per-seed detail remains in each condition's own
# report/tables/ directory).
_METRIC_TABLES: dict[str, bool] = {
    "recovery_aggregate_mae": False,
    "retention_aggregate_mae": False,
    "nmae_aggregate": False,
    "client_benefit_summary_mae": True,
    "control_impact_summary_mae": True,
    "retention_client_summary_mae": True,
}

_CONDITION_TAGS = ("drift_type", "severity", "drift_count", "epochs", "condition_id")


def _tag(frame: pd.DataFrame, *, drift_type: str, severity: str, drift_count: int, epochs: int, condition_id: str) -> pd.DataFrame:
    tagged = frame.copy()
    tagged.insert(0, "condition_id", condition_id)
    tagged.insert(0, "epochs", epochs)
    tagged.insert(0, "drift_count", drift_count)
    tagged.insert(0, "severity", severity)
    tagged.insert(0, "drift_type", drift_type)
    return tagged


def aggregate_matrix(output_root: str | Path, *, archive: bool = True, rebuild_conditions: bool = False) -> dict[str, Any]:
    """Fold every ``report_complete`` (or already ``archived``) condition's
    tables into ``summaries/<metric>.csv``, and (unless ``archive=False``)
    advance freshly-included conditions to ``archived``.

    ``rebuild_conditions=True`` forces every condition's own
    ``tables/*.csv`` to be regenerated (via ``condition_report.
    build_condition_report``) before it's folded in, even if that
    condition is already ``archived``. Use this after adding a method
    variant (e.g. a side-channel script like ``run_m4_window_full_matrix.py``
    that appends a method to suites' ``selected_methods`` after their
    condition report was already built) whose data mitigation_report.py's
    METHOD_ORDER already knows how to read, but which predates the cached
    condition tables -- the normal ``build_condition`` state machine treats
    ``archived`` as done and will never regenerate them on its own.
    """
    output_root = Path(output_root)
    conditions = status_mod.condition_store(paths.status_dir(output_root))
    rows = [row for row in conditions.all_rows() if row["state"] in ("report_complete", "archived")]
    if not rows:
        return {"conditions_included": 0, "written": {}}

    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in _METRIC_TABLES}
    newly_archived: list[str] = []
    for row in rows:
        drift_type, severity = row["drift_type"], row["severity"]
        drift_count, epochs = int(row["drift_count"]), int(row["epochs"])
        if rebuild_conditions:
            seeds = tuple(int(seed) for seed in str(row["seeds"]).split(",") if seed)
            condition_report.build_condition_report(
                {}, output_root, drift_type=drift_type, severity=severity,
                drift_count=drift_count, epochs=epochs, seeds=seeds,
            )
        report_dir = paths.condition_reports_dir(output_root, drift_count, drift_type, severity, epochs)
        for name, all_seeds_only in _METRIC_TABLES.items():
            table_path = report_dir / "tables" / f"{name}.csv"
            if not table_path.is_file():
                raise FileNotFoundError(f"condition {row['condition_id']} is {row['state']} but missing {table_path}")
            frame = pd.read_csv(table_path)
            if all_seeds_only and "seed" in frame.columns:
                frame = frame[frame.seed.astype(str) == "all"].drop(columns=["seed"])
            frames[name].append(_tag(
                frame, drift_type=drift_type, severity=severity, drift_count=drift_count,
                epochs=epochs, condition_id=row["condition_id"],
            ))
        if archive and row["state"] == "report_complete":
            conditions.advance_to(row["condition_id"], "archived")
            newly_archived.append(row["condition_id"])

    summaries_dir = paths.summaries_dir(output_root)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, parts in frames.items():
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_CONDITION_TAGS)
        target = summaries_dir / f"{name}.csv"
        combined.to_csv(target, index=False)
        written[name] = str(target)

    return {"conditions_included": len(rows), "newly_archived": newly_archived, "written": written}
