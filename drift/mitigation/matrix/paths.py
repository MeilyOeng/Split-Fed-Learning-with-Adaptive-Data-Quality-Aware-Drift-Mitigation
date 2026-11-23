"""Deterministic id/path construction for the experiment matrix.

Pure functions only -- no filesystem access, no validation. Every other
matrix module builds paths through this module so the on-disk layout is
defined in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

DRIFT_TYPES = ("sudden", "gradual", "seasonal")
SEVERITIES = ("mild", "strong")
DRIFT_COUNTS = (1, 8, 21)
EPOCH_SETTINGS = (1, 3)
FAULTY_PER_CLUSTER = 8


def drifted_client_dirname(drift_count: int) -> str:
    return f"drifted_client_{int(drift_count):02d}"


def scenario_id(seed: int, drift_type: str, severity: str, drift_count: int) -> str:
    return f"s{int(seed)}__{drift_type}__{severity}__d{int(drift_count):02d}__f{FAULTY_PER_CLUSTER:02d}"


def epoch_dirname(epochs: int) -> str:
    return f"epoch_{int(epochs):02d}"


def suite_id(scen_id: str, epochs: int) -> str:
    return f"{scen_id}__{epoch_dirname(epochs)}"


def condition_id(drift_type: str, severity: str, drift_count: int, epochs: int) -> str:
    return f"{drift_type}__{severity}__d{int(drift_count):02d}__f{FAULTY_PER_CLUSTER:02d}__e{int(epochs):02d}"


def config_dir(root: Path, drift_count: int, drift_type: str, scen_id: str) -> Path:
    return root / "configs" / drifted_client_dirname(drift_count) / drift_type / scen_id


def config_path(root: Path, drift_count: int, drift_type: str, scen_id: str, epochs: int) -> Path:
    return config_dir(root, drift_count, drift_type, scen_id) / f"{epoch_dirname(epochs)}.yaml"


def scenario_root(root: Path, drift_count: int, drift_type: str, scen_id: str) -> Path:
    return root / "scenarios" / drifted_client_dirname(drift_count) / drift_type / scen_id


def prepared_root(root: Path, drift_count: int, drift_type: str, scen_id: str) -> Path:
    return scenario_root(root, drift_count, drift_type, scen_id) / "prepared"


def suite_root(root: Path, drift_count: int, drift_type: str, scen_id: str, epochs: int) -> Path:
    return scenario_root(root, drift_count, drift_type, scen_id) / epoch_dirname(epochs)


def scenario_log_dir(root: Path, drift_count: int, drift_type: str, scen_id: str) -> Path:
    return root / "logs" / drifted_client_dirname(drift_count) / drift_type / scen_id


def epoch_log_dir(root: Path, drift_count: int, drift_type: str, scen_id: str, epochs: int) -> Path:
    return scenario_log_dir(root, drift_count, drift_type, scen_id) / epoch_dirname(epochs)


def status_dir(root: Path) -> Path:
    return root / "status"


def summaries_dir(root: Path) -> Path:
    return root / "summaries"


def condition_reports_dir(root: Path, drift_count: int, drift_type: str, severity: str, epochs: int) -> Path:
    return (
        summaries_dir(root)
        / "condition_reports"
        / drifted_client_dirname(drift_count)
        / drift_type
        / severity
        / epoch_dirname(epochs)
    )


def condition_staging_dir(root: Path, drift_count: int, drift_type: str, severity: str, epochs: int) -> Path:
    return condition_reports_dir(root, drift_count, drift_type, severity, epochs) / "_staging"


METHOD_NAMES = (
    "R0_clean_frozen",
    "R1_clean_continued_splitfed",
    "M0_drift_frozen",
    "M1_continued_splitfed",
    "M2_oracle_client_only",
    "M3_oracle_client_ewc",
    "M4_quality_aware_splitfed",
)


def iter_scenarios(seeds: tuple[int, ...] = (42, 43, 44, 45, 46)):
    """Yield every (seed, drift_type, severity, drift_count) combination in a
    stable, deterministic order (drift_count outer, drift_type, severity,
    seed inner) -- matches the configs/scenarios/logs directory nesting.
    """
    for drift_count in DRIFT_COUNTS:
        for drift_type in DRIFT_TYPES:
            for severity in SEVERITIES:
                for seed in seeds:
                    yield seed, drift_type, severity, drift_count


def iter_conditions():
    for drift_count in DRIFT_COUNTS:
        for drift_type in DRIFT_TYPES:
            for severity in SEVERITIES:
                for epochs in EPOCH_SETTINGS:
                    yield drift_type, severity, drift_count, epochs
