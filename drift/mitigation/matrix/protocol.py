"""Frozen experiment-matrix protocol: construction, (de)serialization, and
immutability enforcement for protocol.yaml.

protocol.yaml is the single source of truth every other matrix module reads
from -- it is written once per output root and never silently changed
afterward (see ``write_protocol``).
"""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ..core import canonical_hash, file_sha256
from . import paths

MATRIX_SCHEMA_VERSION = "drift_fault_mitigation_v2_protocol_schema_v1"
GENERATOR_VERSION = "matrix_generator_v1"

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_CHECKPOINT_H = "outputs/drift_mitigation/checkpoint_H_clean_offline_prepared_seed42_epochs130"
DEFAULT_CANONICAL_TEMPLATE = "drift/configs/mitigation_sudden_process_seed42_fault_dq.yaml"
DEFAULT_FAULT_CONFIG = "data quality/configs/fault_injection_controlled_window_8clients.yaml"
DEFAULT_DQ_CONFIG = "data quality/configs/data_quality_scoring.yaml"
DEFAULT_SOURCE_DIR = "dataset/output_clean"
DEFAULT_ASSIGNMENTS = "dataset/cluster-assignment.csv"

SEVERITY_PARAMETERS = {
    "mild": {"intercept_delta": 0.40, "ar_coefficient_delta": [0.10]},
    "strong": {"intercept_delta": 0.80, "ar_coefficient_delta": [0.20]},
}


def _git_identity(repo_root: Path) -> dict[str, Any]:
    """Lightweight git provenance: commit + dirty flag only. Not
    core.git_record's full diff/untracked-file hashing, which is
    expensive and meant for a single training run's audit trail, not a
    one-time protocol stamp covering the whole repository.
    """
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain=v1"))
        return {"commit": commit, "working_tree_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "working_tree_dirty": None}


def _checkpoint_h_identity(checkpoint_h: Path) -> dict[str, Any]:
    manifest_path = checkpoint_h / "checkpoint_H_epoch130_manifest.json"
    acceptance_path = checkpoint_h / "checkpoint_H_acceptance.json"
    if not manifest_path.is_file() or not acceptance_path.is_file():
        raise FileNotFoundError(
            f"checkpoint H at {checkpoint_h} is missing checkpoint_H_epoch130_manifest.json or "
            f"checkpoint_H_acceptance.json -- run validate-checkpoint-h before building the protocol"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not acceptance.get("accepted"):
        raise ValueError(f"checkpoint H at {checkpoint_h} is not accepted: {acceptance}")
    return {
        "path": str(checkpoint_h),
        "accepted": True,
        "combined_tensor_hash": manifest["combined_tensor_hash"],
        "combined_optimizer_hash": manifest["combined_optimizer_hash"],
        "registry_combined_artifact_hash": manifest["registry_combined_artifact_hash"],
        "epoch": manifest["epoch"],
    }


def build_protocol(
    *,
    output_root: str | Path,
    repo_root: str | Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    checkpoint_h: str | Path = DEFAULT_CHECKPOINT_H,
    canonical_template: str | Path = DEFAULT_CANONICAL_TEMPLATE,
    fault_config: str | Path = DEFAULT_FAULT_CONFIG,
    dq_config: str | Path = DEFAULT_DQ_CONFIG,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    assignments_path: str | Path = DEFAULT_ASSIGNMENTS,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    checkpoint_h = Path(checkpoint_h)
    checkpoint_h_abs = checkpoint_h if checkpoint_h.is_absolute() else repo_root / checkpoint_h
    canonical_template = Path(canonical_template)
    canonical_template_abs = canonical_template if canonical_template.is_absolute() else repo_root / canonical_template
    fault_config = Path(fault_config)
    fault_config_abs = fault_config if fault_config.is_absolute() else repo_root / fault_config
    dq_config = Path(dq_config)
    dq_config_abs = dq_config if dq_config.is_absolute() else repo_root / dq_config
    assignments_path = Path(assignments_path)
    assignments_abs = assignments_path if assignments_path.is_absolute() else repo_root / assignments_path

    for label, path in (
        ("canonical_template", canonical_template_abs),
        ("fault_config", fault_config_abs),
        ("dq_config", dq_config_abs),
        ("assignments", assignments_abs),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    source_dir_abs = source_dir if Path(source_dir).is_absolute() else repo_root / source_dir
    if not Path(source_dir_abs).is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir_abs}")

    protocol: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": _git_identity(repo_root),
        "output_root": str(output_root),
        "population": {
            "clusters": 4,
            "clients_per_cluster": 21,
            "total_clients": 84,
            "faulty_clients_per_cluster": paths.FAULTY_PER_CLUSTER,
            "drift_fault_selection": "independent",
        },
        "seeds": list(seeds),
        "drift_types": list(paths.DRIFT_TYPES),
        "drift_counts": list(paths.DRIFT_COUNTS),
        "severities": SEVERITY_PARAMETERS,
        "gradual_transition_windows": 700,
        "seasonal": {"period": 168, "phase": 0},
        "segments": {"pre_drift": 524, "adaptation": 700, "final_holdout": 524},
        "epoch_settings": list(paths.EPOCH_SETTINGS),
        "methods": list(paths.METHOD_NAMES),
        "m4_primary_mode": "aggregation",
        "checkpoint_h": _checkpoint_h_identity(checkpoint_h_abs),
        "fault_config": {"path": str(fault_config), "sha256": file_sha256(fault_config_abs)},
        "dq_config": {"path": str(dq_config), "sha256": file_sha256(dq_config_abs)},
        "canonical_template": {"path": str(canonical_template), "sha256": file_sha256(canonical_template_abs)},
        "source_dataset_path": str(source_dir),
        "cluster_assignment_path": {"path": str(assignments_path), "sha256": file_sha256(assignments_abs)},
        "matrix_counts": {
            "scenarios": len(seeds) * len(paths.DRIFT_TYPES) * len(SEVERITY_PARAMETERS) * len(paths.DRIFT_COUNTS),
            "suites": len(seeds) * len(paths.DRIFT_TYPES) * len(SEVERITY_PARAMETERS) * len(paths.DRIFT_COUNTS) * len(paths.EPOCH_SETTINGS),
            "method_executions": (
                len(seeds) * len(paths.DRIFT_TYPES) * len(SEVERITY_PARAMETERS) * len(paths.DRIFT_COUNTS)
                * len(paths.EPOCH_SETTINGS) * len(paths.METHOD_NAMES)
            ),
            "condition_reports": len(paths.DRIFT_TYPES) * len(SEVERITY_PARAMETERS) * len(paths.DRIFT_COUNTS) * len(paths.EPOCH_SETTINGS),
        },
    }
    protocol["protocol_hash"] = canonical_hash({key: value for key, value in protocol.items() if key not in ("generated_at", "git")})
    return protocol


def write_protocol(path: str | Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Write protocol.yaml, or verify an existing one is identical.

    Immutability is enforced on ``protocol_hash`` (which excludes the
    inherently-per-run ``generated_at``/``git`` fields) -- rerunning the
    generator with the same inputs must never silently change the frozen
    matrix definition. A real change requires deleting protocol.yaml
    explicitly first.
    """
    path = Path(path)
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError(
                f"protocol.yaml already exists at {path} with a different protocol_hash "
                f"({existing.get('protocol_hash')} != {protocol['protocol_hash']}); refusing to "
                f"overwrite an immutable matrix protocol. Delete it explicitly to regenerate."
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    return protocol


def load_protocol(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"protocol.yaml not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))
