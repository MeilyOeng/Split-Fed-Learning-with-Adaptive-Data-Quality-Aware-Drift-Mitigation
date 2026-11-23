"""Top-level orchestration: prepare each scenario once, run every suite's
methods + evaluation, and advance the three status.py state machines.

This module owns every status-store mutation in the matrix package -- every
other module here (config_gen, preparation_guard, identity, manifest,
condition_report) is read-only with respect to scenario/suite/condition
state. It never reimplements training, evaluation, or preparation
semantics -- it only calls the already-validated
:mod:`drift.mitigation.prepared` entry points (``prepare_offline_data``,
``run_prepared_method``, ``evaluate_completed_suite``) in the right order,
with the right identity checks ahead of each expensive step, and records
wall-clock timing that those entry points don't track themselves.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Iterable

from ..core import load_suite_yaml, write_json
from ..evaluation import evaluate_completed_suite
from ..prepared.preparation import prepare_offline_data
from ..prepared.runner import run_prepared_method
from . import identity as identity_mod
from . import paths
from . import status as status_mod
from .condition_report import build_condition_report
from .preparation_guard import verify_checkpoint_h_compatible, verify_existing_preparation

_ALL_EPOCHS = paths.EPOCH_SETTINGS


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _resolve(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


class MatrixRunError(RuntimeError):
    pass


def _advance_to_at_least(
    store: status_mod.StatusStore, key: str, target_state: str, *, extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Like ``StatusStore.advance_to``, but a safe no-op when the row has
    already moved past ``target_state`` -- e.g. re-running prepare_scenario
    or run_suite against an already-completed scenario/suite must not trip
    ``advance_to``'s backward-jump guard on an *earlier* intermediate
    milestone it passed through on a prior call. Only ever used for
    intermediate milestones within a function; each function's own final
    target state is always requested directly via ``advance_to`` itself,
    since request-the-current-state-again is already a safe no-op there.
    """
    row = store.get(key)
    if row is not None and store.states.index(row["state"]) > store.states.index(target_state):
        return row
    return store.advance_to(key, target_state, extra_fields=extra_fields)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def prepare_scenario(
    protocol: dict[str, Any],
    output_root: str | Path,
    *,
    seed: int,
    drift_type: str,
    severity: str,
    drift_count: int,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Idempotently generate-or-verify the shared ``prepared/`` for one
    scenario. Safe to call once per suite (epoch_01 and epoch_03 both call
    this and land on the same result) since preparation only depends on the
    preparation-identity subset of the config (see identity.py).
    """
    output_root, repo_root = Path(output_root), Path(repo_root)
    scen_id = paths.scenario_id(seed, drift_type, severity, drift_count)
    scenarios = status_mod.scenario_store(paths.status_dir(output_root))
    scenarios.initialize(scen_id, {
        "scenario_id": scen_id, "seed": seed, "drift_type": drift_type,
        "severity": severity, "drift_count": drift_count,
    })

    canonical_epochs = _ALL_EPOCHS[0]
    config_p = paths.config_path(output_root, drift_count, drift_type, scen_id, canonical_epochs)
    if not config_p.is_file():
        raise MatrixRunError(f"scenario config missing (run config generation first): {config_p}")
    config = load_suite_yaml(config_p)
    source_dir = _resolve(repo_root, protocol["source_dataset_path"])
    assignments_path = _resolve(repo_root, protocol["cluster_assignment_path"]["path"])
    for other_epochs in _ALL_EPOCHS[1:]:
        other_config_p = paths.config_path(output_root, drift_count, drift_type, scen_id, other_epochs)
        if not other_config_p.is_file():
            raise MatrixRunError(f"scenario config missing (run config generation first): {other_config_p}")
        identity_mod.assert_preparation_compatible(
            config, load_suite_yaml(other_config_p), source_dir=source_dir, assignments_path=assignments_path,
            label_a=paths.epoch_dirname(canonical_epochs), label_b=paths.epoch_dirname(other_epochs),
        )
    _advance_to_at_least(scenarios, scen_id, "config_generated")

    prepared_root = paths.prepared_root(output_root, drift_count, drift_type, scen_id)
    if prepared_root.is_dir():
        verify_existing_preparation(prepared_root, config, source_dir=source_dir, assignments_path=assignments_path)
    else:
        result = prepare_offline_data(config_p, source_dir, assignments_path, prepared_root)
        if not result["acceptance"]["accepted"]:
            raise MatrixRunError(f"preparation for {scen_id} was not accepted: {result['acceptance']}")
    scenarios.advance_to(scen_id, "preparation_accepted", extra_fields={"prepared_path": str(prepared_root)})
    return {"scenario_id": scen_id, "prepared_root": str(prepared_root)}


def _suite_already_complete(suite_root: Path, methods: list[str]) -> bool:
    if not suite_root.is_dir():
        return False
    for method in methods:
        manifest = _read_json_if_exists(suite_root / method / "run_manifest.json")
        if manifest is None or manifest.get("status") != "complete":
            return False
    acceptance = _read_json_if_exists(suite_root / "comparison" / "acceptance_report.json")
    return bool(acceptance and acceptance.get("accepted"))


def run_suite(
    protocol: dict[str, Any],
    output_root: str | Path,
    *,
    seed: int,
    drift_type: str,
    severity: str,
    drift_count: int,
    epochs: int,
    repo_root: str | Path,
    checkpoint_h: str | Path | None = None,
    checkpoint_storage_mode: str = "auto",
    restart_incomplete: bool = False,
) -> dict[str, Any]:
    """Idempotently run every method a suite's config selects, then evaluate.

    Requires ``prepare_scenario`` to have already reached
    ``preparation_accepted`` for this scenario -- callers iterating the
    whole matrix should call it first; this function does not call it
    implicitly so a partial-matrix run can prepare scenarios and run suites
    as two clearly separated, independently resumable phases.
    """
    output_root, repo_root = Path(output_root), Path(repo_root)
    scen_id = paths.scenario_id(seed, drift_type, severity, drift_count)
    sid = paths.suite_id(scen_id, epochs)
    scenarios = status_mod.scenario_store(paths.status_dir(output_root))
    scenario_row = scenarios.get(scen_id)
    if scenario_row is None or scenario_row["state"] != "preparation_accepted":
        raise MatrixRunError(f"scenario {scen_id} is not preparation_accepted yet; call prepare_scenario first")

    suites = status_mod.suite_store(paths.status_dir(output_root))
    suites.initialize(sid, {
        "suite_id": sid, "scenario_id": scen_id, "seed": seed, "drift_type": drift_type,
        "severity": severity, "drift_count": drift_count, "epochs": epochs,
    })

    config_p = paths.config_path(output_root, drift_count, drift_type, scen_id, epochs)
    if not config_p.is_file():
        raise MatrixRunError(f"suite config missing (run config generation first): {config_p}")
    config = load_suite_yaml(config_p)
    _advance_to_at_least(suites, sid, "config_generated", extra_fields={"config_path": str(config_p)})

    prepared_root = paths.prepared_root(output_root, drift_count, drift_type, scen_id)
    prepared = json.loads((prepared_root / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))
    checkpoint_h_path = _resolve(repo_root, checkpoint_h or protocol["checkpoint_h"]["path"])
    verify_checkpoint_h_compatible(checkpoint_h_path, config, prepared=prepared)
    _advance_to_at_least(suites, sid, "preparation_accepted")

    suite_root = paths.suite_root(output_root, drift_count, drift_type, scen_id, epochs)
    selected_methods = [str(name) for name in config["methods"]]
    runtime_path = suite_root / "matrix_runtime.json"

    if not _suite_already_complete(suite_root, selected_methods):
        suite_root.mkdir(parents=True, exist_ok=True)
        runtime = _read_json_if_exists(runtime_path) or {"methods": {}}
        suite_started_at = runtime.get("suite_started_at") or _now()
        runtime["suite_started_at"] = suite_started_at
        wall_start = datetime.datetime.now(datetime.timezone.utc)
        for method in selected_methods:
            method_row = runtime["methods"].setdefault(method, {})
            if method_row.get("completed_at") and not method_row.get("failed"):
                manifest = _read_json_if_exists(suite_root / method / "run_manifest.json")
                if manifest is not None and manifest.get("status") == "complete":
                    continue
            method_row["started_at"] = _now()
            write_json(runtime_path, runtime)
            try:
                run_prepared_method(
                    method, config_path=config_p, suite_root=suite_root, preparation_root=prepared_root,
                    checkpoint_h=checkpoint_h_path, checkpoint_storage_mode=checkpoint_storage_mode,
                    restart_incomplete=restart_incomplete,
                )
            except Exception as exc:
                method_row["failed"] = True
                method_row["failure_reason"] = str(exc)
                write_json(runtime_path, runtime)
                raise MatrixRunError(f"suite {sid} method {method} failed: {exc}") from exc
            method_row["completed_at"] = _now()
            method_row["failed"] = False
            write_json(runtime_path, runtime)
        wall_end = datetime.datetime.now(datetime.timezone.utc)
        runtime["suite_completed_at"] = _now()
        runtime["suite_wall_clock_seconds"] = (wall_end - wall_start).total_seconds()
        write_json(runtime_path, runtime)
    _advance_to_at_least(suites, sid, "methods_complete", extra_fields={"methods_complete_count": len(selected_methods)})

    evaluation = evaluate_completed_suite(suite_root)
    if not evaluation["acceptance"]["accepted"]:
        raise MatrixRunError(f"suite {sid} evaluation was not accepted: {evaluation['acceptance']}")
    suites.advance_to(sid, "evaluated")
    return {"suite_id": sid, "suite_root": str(suite_root), "evaluation": evaluation["acceptance"]}


def refresh_condition_readiness(
    protocol: dict[str, Any],
    output_root: str | Path,
    *,
    drift_type: str,
    severity: str,
    drift_count: int,
    epochs: int,
) -> dict[str, Any]:
    """Advance a condition from ``waiting_for_seeds`` to ``ready`` once every
    seed's suite for it has reached ``evaluated``. A pure read-then-advance
    over suite_status.csv -- never itself runs anything.
    """
    output_root = Path(output_root)
    seeds = tuple(protocol["seeds"])
    cond_id = paths.condition_id(drift_type, severity, drift_count, epochs)
    conditions = status_mod.condition_store(paths.status_dir(output_root))
    conditions.initialize(cond_id, {
        "condition_id": cond_id, "drift_type": drift_type, "severity": severity,
        "drift_count": drift_count, "epochs": epochs,
    })
    suites = status_mod.suite_store(paths.status_dir(output_root))
    evaluated_seeds = []
    for seed in seeds:
        scen_id = paths.scenario_id(seed, drift_type, severity, drift_count)
        row = suites.get(paths.suite_id(scen_id, epochs))
        if row is not None and row["state"] == "evaluated":
            evaluated_seeds.append(seed)
    row = conditions.get(cond_id)
    if row is not None and row["state"] == "waiting_for_seeds" and len(evaluated_seeds) == len(seeds):
        conditions.advance_to(cond_id, "ready", extra_fields={"seeds": ",".join(str(s) for s in sorted(evaluated_seeds))})
    return {"condition_id": cond_id, "evaluated_seeds": evaluated_seeds, "expected_seeds": list(seeds)}


def build_condition(
    protocol: dict[str, Any],
    output_root: str | Path,
    *,
    drift_type: str,
    severity: str,
    drift_count: int,
    epochs: int,
) -> dict[str, Any]:
    """Build the 5-seed mitigation report for one condition. Requires the
    condition to already be ``ready``. A condition already at
    ``report_complete`` or ``archived`` is a no-op that returns the
    existing report location without rebuilding anything -- rebuilding
    (and re-advancing a state machine already past ``report_complete``)
    is aggregate.py's job, not this function's.
    """
    output_root = Path(output_root)
    cond_id = paths.condition_id(drift_type, severity, drift_count, epochs)
    conditions = status_mod.condition_store(paths.status_dir(output_root))
    row = conditions.get(cond_id)
    if row is None or row["state"] not in ("ready", "report_running", "report_complete", "archived"):
        raise MatrixRunError(f"condition {cond_id} is not ready yet; call refresh_condition_readiness first")
    if row["state"] in ("report_complete", "archived"):
        return {
            "condition_id": cond_id, "report_dir": row.get("report_dir") or str(paths.condition_reports_dir(
                output_root, drift_count, drift_type, severity, epochs,
            )),
            "seeds": [int(seed) for seed in str(row.get("seeds") or "").split(",") if seed],
        }
    if row["state"] == "ready":
        conditions.advance_to(cond_id, "report_running")
    result = build_condition_report(
        protocol, output_root, drift_type=drift_type, severity=severity, drift_count=drift_count, epochs=epochs,
    )
    conditions.advance_to(cond_id, "report_complete", extra_fields={"report_dir": result["report_dir"]})
    return result


def run_matrix(
    protocol: dict[str, Any],
    output_root: str | Path,
    *,
    repo_root: str | Path,
    checkpoint_h: str | Path | None = None,
    checkpoint_storage_mode: str = "auto",
    restart_incomplete: bool = False,
    build_reports: bool = True,
    seeds: Iterable[int] | None = None,
    drift_types: Iterable[str] | None = None,
    severities: Iterable[str] | None = None,
    drift_counts: Iterable[int] | None = None,
    epoch_settings: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Drive the full (or a filtered subset of the) matrix end to end:
    prepare every scenario, run every suite, and build every condition
    report whose seeds have all completed. Every step is individually
    idempotent, so re-running this after a partial failure resumes rather
    than repeats completed work.
    """
    output_root, repo_root = Path(output_root), Path(repo_root)
    wanted_seeds = set(seeds) if seeds is not None else set(protocol["seeds"])
    wanted_drift_types = set(drift_types) if drift_types is not None else set(paths.DRIFT_TYPES)
    wanted_severities = set(severities) if severities is not None else set(paths.SEVERITIES)
    wanted_drift_counts = set(drift_counts) if drift_counts is not None else set(paths.DRIFT_COUNTS)
    wanted_epochs = set(epoch_settings) if epoch_settings is not None else set(paths.EPOCH_SETTINGS)

    scenarios_run: list[dict[str, Any]] = []
    suites_run: list[dict[str, Any]] = []
    touched_conditions: set[tuple[str, str, int, int]] = set()
    for seed, drift_type, severity, drift_count in paths.iter_scenarios(tuple(protocol["seeds"])):
        if seed not in wanted_seeds or drift_type not in wanted_drift_types:
            continue
        if severity not in wanted_severities or drift_count not in wanted_drift_counts:
            continue
        scenarios_run.append(prepare_scenario(
            protocol, output_root, seed=seed, drift_type=drift_type, severity=severity,
            drift_count=drift_count, repo_root=repo_root,
        ))
        for epochs in paths.EPOCH_SETTINGS:
            if epochs not in wanted_epochs:
                continue
            suites_run.append(run_suite(
                protocol, output_root, seed=seed, drift_type=drift_type, severity=severity,
                drift_count=drift_count, epochs=epochs, repo_root=repo_root, checkpoint_h=checkpoint_h,
                checkpoint_storage_mode=checkpoint_storage_mode, restart_incomplete=restart_incomplete,
            ))
            touched_conditions.add((drift_type, severity, drift_count, epochs))

    conditions_built: list[dict[str, Any]] = []
    for drift_type, severity, drift_count, epochs in sorted(touched_conditions):
        readiness = refresh_condition_readiness(
            protocol, output_root, drift_type=drift_type, severity=severity, drift_count=drift_count, epochs=epochs,
        )
        if build_reports and len(readiness["evaluated_seeds"]) == len(readiness["expected_seeds"]):
            conditions_built.append(build_condition(
                protocol, output_root, drift_type=drift_type, severity=severity, drift_count=drift_count, epochs=epochs,
            ))

    from .manifest import write_matrix_manifest
    manifest_path = write_matrix_manifest(protocol, output_root)
    return {
        "scenarios_run": len(scenarios_run), "suites_run": len(suites_run),
        "conditions_built": len(conditions_built), "matrix_manifest": str(manifest_path),
    }
