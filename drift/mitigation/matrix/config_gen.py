"""Deterministic, idempotent generator for the 90 scenario x 2 epoch configs.

Renders every config from ONE canonical template
(``protocol["canonical_template"]["path"]``, the already-validated
sudden/fault/DQ config with the currently-approved M3/EWC lambda) by
overriding only the fields the matrix protocol actually varies:
seeds, population.drifted_per_cluster/controls_per_cluster,
process.{drift_type,intercept_delta,ar_coefficient_delta,
gradual_transition_windows,seasonal_period,seasonal_phase},
faults.config_path, adaptation.*, and suite_id.

Everything else -- model architecture, windowing, segments, splits,
normalization, ewc.lambda, quality_weighting, methods -- passes through the
template byte-for-byte, so the "currently validated M3/EWC configuration"
is never retuned by this generator.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ..core import canonical_hash, load_suite_yaml
from . import paths
from .protocol import SEVERITY_PARAMETERS


def _load_template(protocol: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    template_path = Path(protocol["canonical_template"]["path"])
    if not template_path.is_absolute():
        template_path = repo_root / template_path
    text = template_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    actual_sha = protocol["canonical_template"]["sha256"]
    from ..core import file_sha256
    if file_sha256(template_path) != actual_sha:
        raise ValueError(
            f"canonical template {template_path} has changed since the protocol was generated "
            f"(sha256 mismatch); the matrix protocol is immutable -- regenerate the protocol "
            f"deliberately if the template was meant to change."
        )
    return data


def render_scenario_config(
    protocol: dict[str, Any],
    template: dict[str, Any],
    *,
    seed: int,
    drift_type: str,
    severity: str,
    drift_count: int,
    epochs: int,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    scen_id = paths.scenario_id(seed, drift_type, severity, drift_count)
    sid = paths.suite_id(scen_id, epochs)
    severity_params = SEVERITY_PARAMETERS[severity]

    config["suite_id"] = sid.replace("__", "_")  # load_suite_yaml requires [a-z0-9][a-z0-9_]*, no leading/trailing/double underscores forbidden explicitly but keep it conservative-clean
    config["seeds"] = {
        "experiment_seed": seed,
        "client_selection_seed": seed,
        "data_order_seed": seed,
        "drift_seed": seed,
        "fault_selection_seed": seed,
        "h_training_seed": 42,
        "adaptation_seed": seed,
    }
    total_clients = protocol["population"]["total_clients"]
    clusters = protocol["population"]["clusters"]
    clients_per_cluster = protocol["population"]["clients_per_cluster"]
    config["population"] = {
        "total_clients": total_clients,
        "clusters": clusters,
        "drifted_per_cluster": drift_count,
        "controls_per_cluster": clients_per_cluster - drift_count,
    }
    process = dict(config["process"])
    process["drift_type"] = drift_type
    process["intercept_delta"] = severity_params["intercept_delta"]
    process["ar_coefficient_delta"] = list(severity_params["ar_coefficient_delta"])
    process.pop("gradual_transition_windows", None)
    process.pop("seasonal_period", None)
    process.pop("seasonal_phase", None)
    if drift_type == "gradual":
        process["gradual_transition_windows"] = protocol["gradual_transition_windows"]
    elif drift_type == "seasonal":
        process["seasonal_period"] = protocol["seasonal"]["period"]
        process["seasonal_phase"] = protocol["seasonal"]["phase"]
    config["process"] = process

    faults = dict(config.get("faults") or {})
    faults["config_path"] = protocol["fault_config"]["path"]
    config["faults"] = faults

    quality_weighting = dict(config.get("quality_weighting") or {})
    quality_weighting["config_path"] = protocol["dq_config"]["path"]
    quality_weighting["mode"] = protocol["m4_primary_mode"]
    config["quality_weighting"] = quality_weighting

    config["adaptation"] = {
        "adaptation_epochs": epochs,
        "client_only_adaptation_epochs": epochs,
        "ewc_adaptation_epochs": epochs,
    }
    expected_methods = list(paths.METHOD_NAMES)
    if list(config.get("methods") or []) != expected_methods:
        raise ValueError(
            f"canonical template's methods list {config.get('methods')!r} does not match the "
            f"required matrix method set {expected_methods!r}"
        )
    return config


def _diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for key in sorted(set(a) | set(b)):
        path_ = f"{prefix}.{key}" if prefix else str(key)
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(_diff(va, vb, path_))
        elif va != vb:
            out.append(f"{path_}: {va!r} != {vb!r}")
    return out


def write_scenario_config(path: Path, config: dict[str, Any]) -> str:
    """Write, or verify byte-for-byte-equivalent reuse of, one config file.

    Returns "written", "unchanged", or raises ValueError (fail closed) if
    an existing file at ``path`` has different semantic content.
    """
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if canonical_hash(existing) == canonical_hash(config):
            return "unchanged"
        raise ValueError(
            f"config already exists at {path} with different content; refusing to silently "
            f"overwrite. Differing fields: {_diff(existing, config)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        load_suite_yaml(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return "written"


def generate_all_configs(protocol: dict[str, Any], *, repo_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    template = _load_template(protocol, repo_root)
    seeds = tuple(protocol["seeds"])
    written, unchanged, generated_paths = 0, 0, []
    seen_suite_ids: set[str] = set()
    for seed, drift_type, severity, drift_count in paths.iter_scenarios(seeds):
        scen_id = paths.scenario_id(seed, drift_type, severity, drift_count)
        for epochs in paths.EPOCH_SETTINGS:
            sid = paths.suite_id(scen_id, epochs)
            if sid in seen_suite_ids:
                raise ValueError(f"duplicate suite id generated: {sid}")
            seen_suite_ids.add(sid)
            config = render_scenario_config(
                protocol, template, seed=seed, drift_type=drift_type, severity=severity,
                drift_count=drift_count, epochs=epochs,
            )
            target = paths.config_path(output_root, drift_count, drift_type, scen_id, epochs)
            outcome = write_scenario_config(target, config)
            if outcome == "written":
                written += 1
            else:
                unchanged += 1
            generated_paths.append(str(target))
    expected = protocol["matrix_counts"]["suites"]
    if len(generated_paths) != expected:
        raise ValueError(f"generated {len(generated_paths)} configs, expected {expected}")
    return {"written": written, "unchanged": unchanged, "total": len(generated_paths), "paths": generated_paths}
