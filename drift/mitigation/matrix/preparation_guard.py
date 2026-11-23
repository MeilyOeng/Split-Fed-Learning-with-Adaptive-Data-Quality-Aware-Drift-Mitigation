"""Fail-closed verification that an existing prepared/ directory really
matches the config a suite is about to reuse it for (§13).

Reuses drift.mitigation.prepared.preparation.validate_offline_preparation
plus the existing core.py primitives (stable_select_per_cluster,
selected_cohort_hash, file_sha256, ProcessSpec) that
drift.mitigation.prepared.preparation itself is built from -- this module
never reimplements their hashing logic, it only recomputes the same
quantities from the config under test and compares them against what
prepare_offline_data already recorded in the preparation's own manifests.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..core import ProcessSpec, canonical_hash, file_sha256, selected_cohort_hash, stable_select_per_cluster
from ..prepared import suite_identity as _suite_identity
from ..prepared.preparation import validate_offline_preparation
from ..prepared.training import baseline_h_contract_from_config


class PreparationMismatch(ValueError):
    pass


class CheckpointHIncompatible(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreparationMismatch(f"expected manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checkpoint_h_compatible(
    checkpoint_h: str | Path,
    config: dict[str, Any],
    *,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed (CheckpointHIncompatible) unless ``checkpoint_h``'s
    stamped-or-derived BaselineHContract is compatible with this config's
    data/architecture.

    Reuses drift.mitigation.prepared.suite_identity's own
    ``_resolve_h_baseline_contract`` / ``_assert_baseline_h_contract_compatible``
    directly (the same private helpers ``initialize_prepared_suite_root``
    calls at run time) rather than reimplementing the derivation-vs-stamped
    fallback logic a second time -- so this pre-flight check can never
    silently diverge from what actually gets enforced when a method runs.
    """
    checkpoint_h = Path(checkpoint_h).resolve()
    manifest_path = checkpoint_h / "checkpoint_H_epoch130_manifest.json"
    acceptance_path = checkpoint_h / "checkpoint_H_acceptance.json"
    if not manifest_path.is_file() or not acceptance_path.is_file():
        raise CheckpointHIncompatible(
            f"checkpoint H at {checkpoint_h} is missing checkpoint_H_epoch130_manifest.json or "
            f"checkpoint_H_acceptance.json -- run validate-checkpoint-h before using it"
        )
    h_manifest = _read_json(manifest_path)
    h_acceptance = _read_json(acceptance_path)
    if not h_acceptance.get("accepted") or int(h_manifest.get("epoch", -1)) != 130:
        raise CheckpointHIncompatible(f"checkpoint H at {checkpoint_h} acceptance is false or not epoch 130")
    h_baseline = _suite_identity._resolve_h_baseline_contract(checkpoint_h, config, h_manifest)
    scenario_contract = asdict(baseline_h_contract_from_config(config, prepared))
    try:
        _suite_identity._assert_baseline_h_contract_compatible(h_baseline["contract"], scenario_contract)
    except ValueError as exc:
        raise CheckpointHIncompatible(f"{checkpoint_h}: {exc}") from exc
    return {
        "checkpoint_h": str(checkpoint_h),
        "compatible": True,
        "baseline_h_contract_source": h_baseline["source"],
        "baseline_h_contract_hash": h_baseline["contract_hash"],
    }


def verify_existing_preparation(
    prepared_root: str | Path,
    config: dict[str, Any],
    *,
    source_dir: str | Path,
    assignments_path: str | Path,
    checkpoint_h: str | Path | None = None,
) -> dict[str, Any]:
    """Raise PreparationMismatch (fail closed) unless ``prepared_root`` was
    genuinely built from a preparation-compatible config. Returns a summary
    of every check performed on success.
    """
    prepared_root = Path(prepared_root).resolve()
    validation = validate_offline_preparation(prepared_root)  # accepted + schema_version==2 + M0/M1/M2 binding
    manifests = prepared_root / "manifests"
    checks: dict[str, bool] = {"acceptance_report_accepted": True}

    prepared = _read_json(manifests / "prepared_dataset_hashes.json")

    # --- process / drift identity ---
    process = config["process"]
    process_spec = ProcessSpec(
        drift_type=str(process["drift_type"]), drift_source=str(process["drift_source"]),
        old_intercept=float(process["old_intercept"]), old_ar_coefficients=tuple(process["old_ar_coefficients"]),
        intercept_delta=float(process["intercept_delta"]), ar_coefficient_delta=tuple(process["ar_coefficient_delta"]),
        innovation_sigma=float(process["innovation_sigma"]), innovation_scale=float(process["innovation_scale"]),
        innovation_source=str(process.get("innovation_source", "observed_process_residual")),
        gradual_transition_windows=(int(process["gradual_transition_windows"]) if process.get("gradual_transition_windows") is not None else None),
        seasonal_period=(float(process["seasonal_period"]) if process.get("seasonal_period") is not None else None),
        seasonal_phase=float(process.get("seasonal_phase", 0.0)),
    )
    process_spec.validate()
    expected_process_manifest = process_spec.equation_manifest() | {
        "benchmark_interpretation": "controlled counterfactual benchmark generated offline using observed future residuals; not online-generatable without future information",
        "residual_visibility": "offline_preparation_audit_only",
    }
    expected_process_hash = canonical_hash(expected_process_manifest)
    checks["drift_process_identity"] = expected_process_hash == prepared.get("process_parameters_hash")
    if not checks["drift_process_identity"]:
        raise PreparationMismatch(
            f"{prepared_root}: drift process identity mismatch -- config's process.* does not match "
            f"what this preparation was built from (expected {expected_process_hash}, recorded "
            f"{prepared.get('process_parameters_hash')})"
        )

    # --- windowing / segments (partial: config-derivable fields only) ---
    split_manifest = _read_json(manifests / "split_manifest.json")
    windowing, segments, splits = config["windowing"], config["segments"], config.get("splits", {})
    expected_deployment_count = int(segments["pre_drift"]) + int(segments["adaptation"]) + int(segments["final_holdout"])
    checks["windowing_identity"] = (
        int(split_manifest["window_size"]) == int(windowing["window_size"])
        and int(split_manifest["forecast_horizon"]) == int(windowing["forecast_horizon"])
        and int(split_manifest["deployment"]["count"]) == expected_deployment_count
    )
    if "train_windows" in splits:
        checks["windowing_identity"] = checks["windowing_identity"] and int(split_manifest["clean_train"]["count"]) == int(splits["train_windows"])
    if "validation_windows" in splits:
        checks["windowing_identity"] = checks["windowing_identity"] and int(split_manifest["clean_validation"]["count"]) == int(splits["validation_windows"])
    if not checks["windowing_identity"]:
        raise PreparationMismatch(f"{prepared_root}: windowing/segments/splits identity mismatch against split_manifest.json")

    # --- drift cohort identity (recomputed from cluster assignment + config, not just read back) ---
    assignment = pd.read_csv(assignments_path, dtype=str)
    columns = {str(column).lower(): column for column in assignment.columns}
    clusters: dict[str, list[str]] = {}
    for _, row in assignment.iterrows():
        clusters.setdefault(str(row[columns["cluster"]]), []).append(str(row[columns["id"]]))
    cohort = stable_select_per_cluster(
        clusters, per_cluster=int(config["population"]["drifted_per_cluster"]), seed=int(config["seeds"]["client_selection_seed"]),
    )
    expected_cohort_hash = selected_cohort_hash(cohort)
    recorded_cohort = _read_json(manifests / "selected_drift_clients_hash.json")
    checks["cohort_identity"] = expected_cohort_hash == recorded_cohort.get("sha256")
    if not checks["cohort_identity"]:
        raise PreparationMismatch(
            f"{prepared_root}: drift cohort identity mismatch -- config's population.drifted_per_cluster "
            f"/ seeds.client_selection_seed do not reproduce the recorded selected_drift_clients_hash.json"
        )

    # --- expected 84-client population ---
    total_clients = int(config["population"]["total_clients"])
    checks["population_size"] = sum(len(members) for members in clusters.values()) == total_clients
    if not checks["population_size"]:
        raise PreparationMismatch(f"{prepared_root}: cluster assignment does not contain {total_clients} clients")

    # --- fault config identity ---
    faults = config.get("faults", {})
    if faults.get("enabled"):
        fault_identity_path = manifests / "fault_configuration_identity.json"
        recorded_fault = _read_json(fault_identity_path)
        expected_fault_sha = file_sha256(faults["config_path"])
        checks["fault_config_identity"] = recorded_fault.get("fault_config_sha256") == expected_fault_sha
        if not checks["fault_config_identity"]:
            raise PreparationMismatch(f"{prepared_root}: faults.config_path content has changed since this preparation was built")

    # --- DQ config identity ---
    quality_weighting = config.get("quality_weighting", {})
    if quality_weighting.get("enabled"):
        dq_identity_path = manifests / "quality_weighting_identity.json"
        if dq_identity_path.is_file():
            recorded_dq = _read_json(dq_identity_path)
            expected_dq_sha = file_sha256(quality_weighting["config_path"])
            dq_recorded_sha = recorded_dq.get("dq_config_sha256") or recorded_dq.get("config_sha256")
            checks["dq_config_identity"] = dq_recorded_sha == expected_dq_sha
            if not checks["dq_config_identity"]:
                raise PreparationMismatch(f"{prepared_root}: quality_weighting.config_path content has changed since this preparation was built")

    # --- source dataset / assignment identity ---
    source_manifest = _read_json(manifests / "source_manifest.json")
    expected_assignment_sha = file_sha256(assignments_path)
    checks["assignment_identity"] = source_manifest.get("assignment", {}).get("sha256") == expected_assignment_sha
    if not checks["assignment_identity"]:
        raise PreparationMismatch(f"{prepared_root}: cluster-assignment.csv content has changed since this preparation was built")
    source_dir = Path(source_dir)
    for cluster_row in source_manifest.get("clusters", []):
        cluster_path = source_dir / f"cluster_{cluster_row['cluster_id']}.pkl"
        if file_sha256(cluster_path) != cluster_row["sha256"]:
            raise PreparationMismatch(f"{prepared_root}: source file {cluster_path} content has changed since this preparation was built")
    checks["source_identity"] = True

    checkpoint_h_result: dict[str, Any] | None = None
    if checkpoint_h is not None:
        checkpoint_h_result = verify_checkpoint_h_compatible(checkpoint_h, config, prepared=prepared)
        checks["checkpoint_h_compatible"] = True

    return {
        "prepared_root": str(prepared_root), "valid": True, "checks": checks, "validation": validation,
        "checkpoint_h": checkpoint_h_result,
    }
