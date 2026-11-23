"""Preparation identity vs. runtime/adaptation identity.

``drift.mitigation.prepared.preparation.prepare_offline_data`` reads a
specific, narrow subset of a resolved suite config -- confirmed by reading
its implementation, not assumed. ``adaptation.*``, ``adaptation_seed``,
``data_order_seed``, ``h_training_seed``, ``model.*``, ``ewc.*``,
``runtime.*``, and ``quality_weighting.mode`` are never read by
preparation (they only affect checkpoint H reuse via BaselineHContract, or
the R0..M4 adaptation/evaluation phases via prepared.runner) -- so two
configs that differ only in those fields are always preparation-compatible.

This module makes that boundary explicit and machine-checked, since no
such check exists anywhere in drift.mitigation.prepared today: preparation
identity is never hashed or compared against a config at reuse time there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..core import canonical_hash, file_sha256

# Exactly the config fields drift.mitigation.prepared.preparation.prepare_offline_data
# reads, verified against its source. Anything not listed here (adaptation.*,
# adaptation_seed, data_order_seed, h_training_seed, model.*, ewc.*, runtime.*,
# quality_weighting.mode, methods, suite_id) is runtime/H/adaptation identity
# and is explicitly allowed to differ between epoch_01 and epoch_03.
_POPULATION_FIELDS = ("total_clients", "clusters", "drifted_per_cluster")
_PROCESS_FIELDS = (
    "drift_type", "drift_source", "old_intercept", "old_ar_coefficients",
    "intercept_delta", "ar_coefficient_delta", "innovation_sigma", "innovation_scale",
    "innovation_source", "gradual_transition_windows", "seasonal_period", "seasonal_phase",
)
_SEGMENT_FIELDS = ("pre_drift", "adaptation", "final_holdout")
_SPLIT_FIELDS = ("train_windows", "validation_windows")
_WINDOW_FIELDS = ("window_size", "forecast_horizon")


def preparation_identity(
    config: Mapping[str, Any],
    *,
    source_dir: str | Path,
    assignments_path: str | Path,
) -> dict[str, Any]:
    """The exact set of fields that determine prepared-dataset content.

    ``config`` must already be resolved (i.e. the return value of
    ``drift.mitigation.core.load_suite_yaml``, which fills in every
    optional field's default), so two configs that omit a field
    identically still compare equal here.
    """
    population = config.get("population", {})
    source = config.get("source", {})
    process = config.get("process", {})
    segments = config.get("segments", {})
    splits = config.get("splits", {})
    windowing = config.get("windowing", {})
    normalization = config.get("normalization", {})
    seeds = config.get("seeds", {})
    faults = config.get("faults", {})
    quality_weighting = config.get("quality_weighting", {})

    material: dict[str, Any] = {
        "population": {key: population.get(key) for key in _POPULATION_FIELDS},
        "source_observations_per_client": source.get("observations_per_client"),
        "source_dir_sha256": _directory_manifest_hash(Path(source_dir)),
        "assignments_sha256": file_sha256(assignments_path),
        "process": {key: process.get(key) for key in _PROCESS_FIELDS},
        "segments": {key: segments.get(key) for key in _SEGMENT_FIELDS},
        "splits": {key: splits.get(key) for key in _SPLIT_FIELDS},
        "windowing": {key: windowing.get(key) for key in _WINDOW_FIELDS},
        "normalization_method": normalization.get("method"),
        "seeds": {
            "client_selection_seed": seeds.get("client_selection_seed"),
            "drift_seed": seeds.get("drift_seed"),
            "fault_selection_seed": seeds.get("fault_selection_seed"),
        },
        "faults": {
            "enabled": bool(faults.get("enabled", False)),
            "config_path": faults.get("config_path"),
            "config_sha256": file_sha256(faults["config_path"]) if faults.get("enabled") and faults.get("config_path") else None,
        },
        "quality_weighting": {
            "enabled": bool(quality_weighting.get("enabled", False)),
            "config_path": quality_weighting.get("config_path"),
            "config_sha256": (
                file_sha256(quality_weighting["config_path"])
                if quality_weighting.get("enabled") and quality_weighting.get("config_path")
                else None
            ),
        },
    }
    return {"fields": material, "hash": canonical_hash(material)}


def _directory_manifest_hash(directory: Path) -> str:
    """Cheap, deterministic content fingerprint for a source directory: the
    sorted list of (relative path, size, mtime_ns) for every regular file.

    Not a full content hash (hashing the whole clean dataset on every
    preparation-identity check would be prohibitively slow across 90
    scenarios) -- this is enough to fail closed on a swapped or edited
    source directory while staying fast. prepare_offline_data itself still
    computes and records full per-file SHA-256 in source_manifest.json;
    that remains the authoritative content hash.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {directory}")
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            entries.append((str(path.relative_to(directory)), stat.st_size, stat.st_mtime_ns))
    return canonical_hash(entries)


def assert_preparation_compatible(
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
    *,
    source_dir: str | Path,
    assignments_path: str | Path,
    label_a: str = "config_a",
    label_b: str = "config_b",
) -> dict[str, Any]:
    """Fail closed unless two resolved configs are preparation-identical.

    Returns the shared identity dict on success. Raises ValueError with the
    exact differing field paths on mismatch -- never silently proceeds and
    never disables this check.
    """
    identity_a = preparation_identity(config_a, source_dir=source_dir, assignments_path=assignments_path)
    identity_b = preparation_identity(config_b, source_dir=source_dir, assignments_path=assignments_path)
    if identity_a["hash"] != identity_b["hash"]:
        diffs = _diff(identity_a["fields"], identity_b["fields"])
        raise ValueError(
            f"{label_a} and {label_b} are not preparation-compatible; a shared preparation "
            f"cannot be reused between them. Differing fields: {diffs}"
        )
    return identity_a


def _diff(a: Mapping[str, Any], b: Mapping[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for key in sorted(set(a) | set(b)):
        path = f"{prefix}.{key}" if prefix else key
        va, vb = a.get(key), b.get(key)
        if isinstance(va, Mapping) and isinstance(vb, Mapping):
            out.extend(_diff(va, vb, path))
        elif va != vb:
            out.append(f"{path}: {va!r} != {vb!r}")
    return out
