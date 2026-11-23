"""CLI for the offline-oracle controlled 2x2 diagnostic experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import pandas as pd

from .diagnostic_2x2 import (
    CONDITIONS,
    PROJECT_ROOT,
    build_run_environment,
    evaluate_experiment,
    load_experiment_config,
    prepare_experiment,
    sha256_file,
    stage_required_checkpoints,
    validate_prepared_experiment,
    write_json,
)


DEFAULT_CONFIG = PROJECT_ROOT / "drift" / "configs" / "diagnostic_2x2.yaml"

PROTECTED_RUNTIME_PATHS = (
    "data/stream_service.py",
    "run_split.py",
    "server/server.py",
    "client/client.py",
    "data/data_streaming.py",
)
PROTECTED_RUNTIME_TREES = ("models", "fed_server")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hashes() -> dict[str, str]:
    paths: list[Path] = [PROJECT_ROOT / relative for relative in PROTECTED_RUNTIME_PATHS]
    for relative in PROTECTED_RUNTIME_TREES:
        paths.extend(sorted((PROJECT_ROOT / relative).rglob("*.py")))
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in paths
        if path.exists() and path.is_file()
    }


def _write_run_environment(path: Path, env: Mapping[str, str]) -> None:
    write_json(path.with_suffix(".json"), env)
    lines = [f"{key}={value}" for key, value in sorted(env.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configured_python(config: Mapping[str, object]) -> Path:
    """Return an absolute interpreter path without dereferencing venv symlinks."""
    python_bin = Path(
        str(dict(config.get("runner", {}) or {}).get("python", PROJECT_ROOT / "venv" / "bin" / "python"))
    )
    if not python_bin.is_absolute():
        python_bin = PROJECT_ROOT / python_bin
    return python_bin.absolute()


def _run_subprocess(command: Sequence[str], *, env: Mapping[str, str], log_path: Path) -> int:
    process_env = os.environ.copy()
    process_env.update({str(key): str(value) for key, value in env.items()})
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


def run_prepared_experiment(
    config: Mapping[str, object],
    output_root: str | Path,
    *,
    conditions: Sequence[str] = ("A", "B", "C", "D"),
) -> dict[str, object]:
    root = Path(output_root).resolve()
    validation = validate_prepared_experiment(root)
    experiment = _load_json(root / "experiment_manifest.json")
    cohort = pd.read_csv(
        root / "selected_clients.csv",
        dtype={"client_id": str, "cluster_id": str},
    )
    run_root = root / "runs"
    run_root.mkdir(exist_ok=True)
    python_bin = _configured_python(config)
    if not python_bin.exists():
        raise FileNotFoundError(f"configured Python executable does not exist: {python_bin}")

    invalid_conditions = sorted(set(conditions) - set(CONDITIONS))
    if invalid_conditions:
        raise ValueError(f"unknown conditions: {invalid_conditions}")

    protected_before = _source_hashes()
    condition_results: dict[str, object] = {}
    for condition_id in conditions:
        prepared_dir = root / "prepared" / condition_id
        condition_manifest = _load_json(prepared_dir / "condition_manifest.json")
        run_dir = run_root / condition_id
        frozen_path = run_dir / "results" / "frozen_validation_report.json"
        if run_dir.exists():
            if not frozen_path.exists():
                raise FileExistsError(
                    f"incomplete condition run already exists: {run_dir}; preserve or remove it before retrying"
                )
            frozen = _load_json(frozen_path)
            if not frozen.get("frozen_validation_passed") or not frozen.get("parameters_unchanged"):
                raise AssertionError(f"condition {condition_id} existing run is not a valid frozen result")
            condition_results[condition_id] = {
                "run_dir": str(run_dir),
                "run_log": str(run_dir / "run.log"),
                "frozen_validation_report": str(frozen_path),
                "staged_checkpoint_count": len(frozen.get("checkpoint_hashes_before", {})),
                "parameters_unchanged": True,
                "frozen_validation_passed": True,
                "reused_completed_run": True,
            }
            print(f"[diagnostic_2x2] reusing completed frozen condition {condition_id}: {run_dir}", flush=True)
            continue
        staged = stage_required_checkpoints(config["baseline_checkpoint_dir"], run_dir, cohort)
        env = build_run_environment(config, condition_manifest, run_dir)
        _write_run_environment(run_dir / "diagnostic_run_environment.env", env)
        write_json(
            run_dir / "diagnostic_roles.json",
            {
                "condition_id": condition_id,
                "oracle": {
                    "role": "primary_ground_truth",
                    "execution": "offline_only",
                    "controls_runtime": False,
                },
                "page_hinkley": {
                    "role": "primary_statistical_comparison",
                    "execution": "runtime_logging_only",
                    "controls_runtime": False,
                },
                "adwin": {
                    "role": "secondary_diagnostic",
                    "execution": "runtime_logging_only",
                    "controls_runtime": False,
                },
                "data_quality": {
                    "execution": "offline_only",
                    "controls_runtime": False,
                },
                "runtime_drift_injection_enabled": False,
                "staged_checkpoint_file_hashes": staged,
            },
        )
        print(
            f"[diagnostic_2x2] running condition {condition_id}: "
            f"{len(cohort)} clients, output={run_dir}",
            flush=True,
        )
        returncode = _run_subprocess(
            [str(python_bin), str(PROJECT_ROOT / "run_split.py")],
            env=env,
            log_path=run_dir / "run.log",
        )
        if returncode != 0:
            raise RuntimeError(f"condition {condition_id} run_split.py exited with status {returncode}")
        if not frozen_path.exists():
            raise FileNotFoundError(frozen_path)
        frozen = _load_json(frozen_path)
        if not frozen.get("frozen_validation_passed"):
            raise AssertionError(f"condition {condition_id} failed frozen checkpoint validation")
        condition_results[condition_id] = {
            "run_dir": str(run_dir),
            "run_log": str(run_dir / "run.log"),
            "frozen_validation_report": str(frozen_path),
            "staged_checkpoint_count": len(staged),
            "parameters_unchanged": bool(frozen.get("parameters_unchanged")),
            "frozen_validation_passed": bool(frozen.get("frozen_validation_passed")),
            "reused_completed_run": False,
        }

    protected_after = _source_hashes()
    protected_changed = sorted(
        key
        for key in set(protected_before) | set(protected_after)
        if protected_before.get(key) != protected_after.get(key)
    )
    if protected_changed:
        raise AssertionError(
            "protected runtime source files changed during external orchestration: "
            + ", ".join(protected_changed)
        )
    run_manifest = {
        "execution_mode": experiment["execution_mode"],
        "prepared_validation": validation,
        "conditions": condition_results,
        "protected_runtime_source_hashes_before": protected_before,
        "protected_runtime_source_hashes_after": protected_after,
        "protected_runtime_sources_unchanged": True,
        "run_split_modified_or_wrapped": False,
        "stream_service_modified_or_wrapped": False,
    }
    write_json(root / "run_manifest.json", run_manifest)
    return run_manifest


def _common_arguments(parser: argparse.ArgumentParser, *, smoke: bool = False) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    if smoke:
        parser.add_argument(
            "--smoke",
            action="store_true",
            help="Use the 32 affected clients only. The primary experiment remains the 84-client mode.",
        )
        parser.add_argument(
            "--dq-progress-every",
            type=int,
            default=5000,
            help="Print DQ progress every N rows; use 0 to disable.",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled 2x2 frozen SplitFed diagnostic with offline oracle and DQ analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Materialize A-D inputs and offline labels")
    _common_arguments(prepare_parser, smoke=True)
    validate_parser = subparsers.add_parser("validate", help="Validate prepared A-D artifacts")
    _common_arguments(validate_parser)
    run_parser = subparsers.add_parser("run", help="Run frozen inference and logging-only detectors")
    _common_arguments(run_parser)
    run_parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
    )
    evaluate_parser = subparsers.add_parser("evaluate", help="Perform authoritative offline joins and contrasts")
    _common_arguments(evaluate_parser)
    all_parser = subparsers.add_parser("all", help="Prepare, run, and evaluate")
    _common_arguments(all_parser, smoke=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    if args.command == "prepare":
        result = prepare_experiment(
            config,
            args.output_root,
            smoke=args.smoke,
            dq_progress_every=args.dq_progress_every,
        )
    elif args.command == "validate":
        result = validate_prepared_experiment(args.output_root)
    elif args.command == "run":
        result = run_prepared_experiment(
            config,
            args.output_root,
            conditions=args.conditions,
        )
    elif args.command == "evaluate":
        result = evaluate_experiment(args.output_root)
    else:
        prepare_experiment(
            config,
            args.output_root,
            smoke=args.smoke,
            dq_progress_every=args.dq_progress_every,
        )
        run_prepared_experiment(config, args.output_root)
        result = evaluate_experiment(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
