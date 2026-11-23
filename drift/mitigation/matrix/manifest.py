"""matrix_manifest.csv: one row per suite (180 rows), rebuildable at any time
from the protocol, generated configs, status stores, and each method's own
authoritative run_manifest.json. Never a source of truth itself.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from ..core import canonical_hash
from . import paths, status as status_mod
from .protocol import SEVERITY_PARAMETERS

FIELDNAMES = [
    "scenario_id", "suite_id", "seed", "drift_type", "severity", "intercept_delta", "ar_coefficient_delta",
    "drifted_per_cluster", "controls_per_cluster", "faulty_per_cluster", "adaptation_epochs",
    "quality_weighting_mode", "config_path", "config_sha256", "preparation_path", "epoch_suite_path",
    "checkpoint_h_path", "fault_config_path", "dq_config_path", "scenario_status", "suite_status",
    "preparation_status", "evaluation_status", "mitigation_report_status", "archive_status",
    "started_at", "completed_at", "runtime_seconds", "failure_reason",
]


def _method_run_times(suite_root: Path) -> tuple[str, str, float | None, str]:
    """Start/end/runtime/failure summary for a suite.

    drift.mitigation.prepared's own run_manifest.json records no
    started_at/completed_at (verified against its source) -- wall-clock
    timing is purely an orchestration-layer concern, so the matrix runner
    records it itself in a sidecar ``matrix_runtime.json`` written next to
    (never inside) each method's run_manifest.json. Empty/None when the
    suite hasn't run yet -- never fabricated. Failures are read from the
    authoritative run_manifest.json status field.
    """
    runtime_path = suite_root / "matrix_runtime.json"
    runtime: dict[str, Any] = {}
    if runtime_path.is_file():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            runtime = {}
    failures = []
    for method in paths.METHOD_NAMES:
        manifest_path = suite_root / method / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "complete":
            failures.append(f"{method}: status={manifest.get('status')!r}")
    methods_timing = runtime.get("methods", {})
    starts = [row["started_at"] for row in methods_timing.values() if row.get("started_at")]
    ends = [row["completed_at"] for row in methods_timing.values() if row.get("completed_at")]
    started_at = min(starts) if starts else ""
    completed_at = max(ends) if ends and len(ends) == len(paths.METHOD_NAMES) else ""
    runtime_seconds = runtime.get("suite_wall_clock_seconds")
    for method, row in methods_timing.items():
        if row.get("failed"):
            failures.append(f"{method}: {row.get('failure_reason', 'failed')}")
    return started_at, completed_at, runtime_seconds, "; ".join(failures)


def build_matrix_manifest(protocol: dict[str, Any], output_root: str | Path) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    scenarios = status_mod.scenario_store(output_root / "status")
    suites = status_mod.suite_store(output_root / "status")
    conditions = status_mod.condition_store(output_root / "status")
    seeds = tuple(protocol["seeds"])
    rows: list[dict[str, Any]] = []
    for seed, drift_type, severity, drift_count in paths.iter_scenarios(seeds):
        scen_id = paths.scenario_id(seed, drift_type, severity, drift_count)
        scen_row = scenarios.get(scen_id)
        severity_params = SEVERITY_PARAMETERS[severity]
        for epochs in paths.EPOCH_SETTINGS:
            sid = paths.suite_id(scen_id, epochs)
            config_p = paths.config_path(output_root, drift_count, drift_type, scen_id, epochs)
            config_sha256 = ""
            if config_p.is_file():
                config_sha256 = canonical_hash(yaml.safe_load(config_p.read_text(encoding="utf-8")))
            suite_row = suites.get(sid)
            cond_id = paths.condition_id(drift_type, severity, drift_count, epochs)
            cond_row = conditions.get(cond_id)
            suite_r = paths.suite_root(output_root, drift_count, drift_type, scen_id, epochs)
            started_at, completed_at, runtime_seconds, failure_reason = (
                _method_run_times(suite_r) if suite_r.is_dir() else ("", "", None, "")
            )
            rows.append({
                "scenario_id": scen_id,
                "suite_id": sid,
                "seed": seed,
                "drift_type": drift_type,
                "severity": severity,
                "intercept_delta": severity_params["intercept_delta"],
                "ar_coefficient_delta": severity_params["ar_coefficient_delta"][0],
                "drifted_per_cluster": drift_count,
                "controls_per_cluster": protocol["population"]["clients_per_cluster"] - drift_count,
                "faulty_per_cluster": protocol["population"]["faulty_clients_per_cluster"],
                "adaptation_epochs": epochs,
                "quality_weighting_mode": protocol["m4_primary_mode"],
                "config_path": str(config_p),
                "config_sha256": config_sha256,
                "preparation_path": str(paths.prepared_root(output_root, drift_count, drift_type, scen_id)),
                "epoch_suite_path": str(suite_r),
                "checkpoint_h_path": protocol["checkpoint_h"]["path"],
                "fault_config_path": protocol["fault_config"]["path"],
                "dq_config_path": protocol["dq_config"]["path"],
                "scenario_status": (scen_row or {}).get("state", "planned"),
                "suite_status": (suite_row or {}).get("state", "planned"),
                "preparation_status": (scen_row or {}).get("state") == "preparation_accepted",
                "evaluation_status": (suite_row or {}).get("state") == "evaluated",
                "mitigation_report_status": (cond_row or {}).get("state", "waiting_for_seeds"),
                "archive_status": (cond_row or {}).get("state") == "archived",
                "started_at": started_at,
                "completed_at": completed_at,
                "runtime_seconds": runtime_seconds if runtime_seconds is not None else "",
                "failure_reason": failure_reason,
            })
    return rows


def write_matrix_manifest(protocol: dict[str, Any], output_root: str | Path) -> Path:
    output_root = Path(output_root)
    rows = build_matrix_manifest(protocol, output_root)
    target = output_root / "matrix_manifest.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return target
