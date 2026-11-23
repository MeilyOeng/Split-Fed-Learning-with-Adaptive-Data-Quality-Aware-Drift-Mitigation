"""CLI for the drift+fault+DQ+M4 experiment matrix.

Every subcommand is a thin wrapper over one function in this package --
protocol.build_protocol/write_protocol, config_gen.generate_all_configs,
runner.{prepare_scenario,run_suite,build_condition,run_matrix},
aggregate.aggregate_matrix, manifest.write_matrix_manifest, and
status.build_scenario_status_report. No orchestration logic lives here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import aggregate as aggregate_mod
from . import config_gen, manifest as manifest_mod, paths, runner, status as status_mod
from .protocol import DEFAULT_ASSIGNMENTS, DEFAULT_CANONICAL_TEMPLATE, DEFAULT_CHECKPOINT_H, DEFAULT_DQ_CONFIG, DEFAULT_FAULT_CONFIG, DEFAULT_SEEDS, DEFAULT_SOURCE_DIR, build_protocol, load_protocol, write_protocol

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def _str_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _protocol_path(output_root: Path) -> Path:
    return output_root / "protocol.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drift+fault+DQ+M4 experiment matrix orchestrator")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init-protocol", help="build and freeze protocol.yaml for a new matrix output root")
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    init.add_argument("--seeds", type=_int_list, default=DEFAULT_SEEDS)
    init.add_argument("--checkpoint-h", type=Path, default=DEFAULT_CHECKPOINT_H)
    init.add_argument("--canonical-template", type=Path, default=DEFAULT_CANONICAL_TEMPLATE)
    init.add_argument("--fault-config", type=Path, default=DEFAULT_FAULT_CONFIG)
    init.add_argument("--dq-config", type=Path, default=DEFAULT_DQ_CONFIG)
    init.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    init.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)

    generate = commands.add_parser("generate-configs", help="render every scenario/epoch config from the frozen protocol")
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)

    prepare = commands.add_parser("prepare-scenario", help="generate-or-verify one scenario's shared prepared/ data")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--drift-type", choices=paths.DRIFT_TYPES, required=True)
    prepare.add_argument("--severity", choices=paths.SEVERITIES, required=True)
    prepare.add_argument("--drift-count", type=int, choices=paths.DRIFT_COUNTS, required=True)

    run_suite = commands.add_parser("run-suite", help="run every method for one scenario+epoch suite, then evaluate")
    run_suite.add_argument("--output-root", type=Path, required=True)
    run_suite.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    run_suite.add_argument("--seed", type=int, required=True)
    run_suite.add_argument("--drift-type", choices=paths.DRIFT_TYPES, required=True)
    run_suite.add_argument("--severity", choices=paths.SEVERITIES, required=True)
    run_suite.add_argument("--drift-count", type=int, choices=paths.DRIFT_COUNTS, required=True)
    run_suite.add_argument("--epochs", type=int, choices=paths.EPOCH_SETTINGS, required=True)
    run_suite.add_argument("--checkpoint-h", type=Path, default=None)
    run_suite.add_argument("--checkpoint-storage-mode", choices=["auto", "legacy_full"], default="auto")
    run_suite.add_argument("--restart-incomplete", action="store_true")

    build_condition = commands.add_parser("build-condition", help="build the 5-seed mitigation report for one condition")
    build_condition.add_argument("--output-root", type=Path, required=True)
    build_condition.add_argument("--drift-type", choices=paths.DRIFT_TYPES, required=True)
    build_condition.add_argument("--severity", choices=paths.SEVERITIES, required=True)
    build_condition.add_argument("--drift-count", type=int, choices=paths.DRIFT_COUNTS, required=True)
    build_condition.add_argument("--epochs", type=int, choices=paths.EPOCH_SETTINGS, required=True)

    run = commands.add_parser("run", help="drive the full (or a filtered subset of the) matrix end to end")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    run.add_argument("--checkpoint-h", type=Path, default=None)
    run.add_argument("--checkpoint-storage-mode", choices=["auto", "legacy_full"], default="auto")
    run.add_argument("--restart-incomplete", action="store_true")
    run.add_argument("--no-reports", dest="build_reports", action="store_false")
    run.add_argument("--seeds", type=_int_list, default=None)
    run.add_argument("--drift-types", type=_str_list, default=None)
    run.add_argument("--severities", type=_str_list, default=None)
    run.add_argument("--drift-counts", type=_int_list, default=None)
    run.add_argument("--epochs", dest="epoch_settings", type=_int_list, default=None)

    aggregate = commands.add_parser("aggregate", help="fold every report_complete condition into matrix-wide summaries")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--no-archive", dest="archive", action="store_false")
    aggregate.add_argument(
        "--rebuild-conditions", action="store_true",
        help="force-regenerate every condition's own report tables before folding them in, "
             "even if already archived (use after adding a method variant post-hoc)",
    )

    manifest = commands.add_parser("manifest", help="rebuild matrix_manifest.csv from current status/config state")
    manifest.add_argument("--output-root", type=Path, required=True)

    status = commands.add_parser("status", help="print per-scenario overall progress")
    status.add_argument("--output-root", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "init-protocol":
        protocol = build_protocol(
            output_root=args.output_root, repo_root=args.repo_root, seeds=args.seeds,
            checkpoint_h=args.checkpoint_h, canonical_template=args.canonical_template,
            fault_config=args.fault_config, dq_config=args.dq_config, source_dir=args.source_dir,
            assignments_path=args.assignments,
        )
        _print(write_protocol(_protocol_path(args.output_root), protocol))
        return 0

    if args.command == "generate-configs":
        protocol = load_protocol(_protocol_path(args.output_root))
        _print(config_gen.generate_all_configs(protocol, repo_root=args.repo_root, output_root=args.output_root))
        return 0

    if args.command == "prepare-scenario":
        protocol = load_protocol(_protocol_path(args.output_root))
        _print(runner.prepare_scenario(
            protocol, args.output_root, seed=args.seed, drift_type=args.drift_type, severity=args.severity,
            drift_count=args.drift_count, repo_root=args.repo_root,
        ))
        return 0

    if args.command == "run-suite":
        protocol = load_protocol(_protocol_path(args.output_root))
        _print(runner.run_suite(
            protocol, args.output_root, seed=args.seed, drift_type=args.drift_type, severity=args.severity,
            drift_count=args.drift_count, epochs=args.epochs, repo_root=args.repo_root,
            checkpoint_h=args.checkpoint_h, checkpoint_storage_mode=args.checkpoint_storage_mode,
            restart_incomplete=args.restart_incomplete,
        ))
        return 0

    if args.command == "build-condition":
        protocol = load_protocol(_protocol_path(args.output_root))
        _print(runner.build_condition(
            protocol, args.output_root, drift_type=args.drift_type, severity=args.severity,
            drift_count=args.drift_count, epochs=args.epochs,
        ))
        return 0

    if args.command == "run":
        protocol = load_protocol(_protocol_path(args.output_root))
        _print(runner.run_matrix(
            protocol, args.output_root, repo_root=args.repo_root, checkpoint_h=args.checkpoint_h,
            checkpoint_storage_mode=args.checkpoint_storage_mode, restart_incomplete=args.restart_incomplete,
            build_reports=args.build_reports, seeds=args.seeds, drift_types=args.drift_types,
            severities=args.severities, drift_counts=args.drift_counts, epoch_settings=args.epoch_settings,
        ))
        return 0

    if args.command == "aggregate":
        _print(aggregate_mod.aggregate_matrix(
            args.output_root, archive=args.archive, rebuild_conditions=args.rebuild_conditions,
        ))
        return 0

    if args.command == "manifest":
        protocol = load_protocol(_protocol_path(args.output_root))
        _print(str(manifest_mod.write_matrix_manifest(protocol, args.output_root)))
        return 0

    if args.command == "status":
        output_root = args.output_root
        status_dir = paths.status_dir(output_root)
        scenarios = status_mod.scenario_store(status_dir)
        suites = status_mod.suite_store(status_dir)
        conditions = status_mod.condition_store(status_dir)
        report = status_mod.build_scenario_status_report(
            scenarios.all_rows(), suites, conditions,
            scenario_id_fn=paths.scenario_id, suite_id_fn=paths.suite_id, condition_id_fn=paths.condition_id,
        )
        _print(report)
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
