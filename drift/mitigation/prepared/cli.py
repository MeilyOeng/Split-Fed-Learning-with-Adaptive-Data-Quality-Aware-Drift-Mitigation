"""CLI for the offline-prepared mitigation pipeline.

Execution safety comes from validation, not a manual authorization flag:
preparation and checkpoint-H acceptance checks, BaselineHContract
compatibility, suite identity/hash locking, the fail-closed prepared
backend, output/resume protection, and M2's frozen-state checks all still
apply unconditionally. Data preparation and validation never open
checkpoint H.  H training sees an isolated clean-only view created by
:mod:`drift.mitigation.prepared.training`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core import Method, registered_method_names
from ..evaluation import evaluate_completed_suite
from .preparation import prepare_offline_data, validate_offline_preparation
from .runner import run_prepared_method
from .training import train_checkpoint_h, validate_fixed_epoch_h


DEFAULT_CONFIG = Path("drift/configs/mitigation_sudden_process_seed42.yaml")
DEFAULT_SOURCE = Path("dataset/output_clean")
DEFAULT_ASSIGNMENTS = Path("dataset/cluster-assignment.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline-prepared sudden-process mitigation suite")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data", help="materialize immutable raw and normalized datasets")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser("validate-preparation", help="validate an existing preparation manifest")
    validate.add_argument("--preparation", type=Path, required=True)

    train = commands.add_parser("train-checkpoint-h", help="train coherent fixed-epoch H from clean train/validation only")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--preparation", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=130)

    validate_h = commands.add_parser("validate-checkpoint-h", help="validate one coherent fixed-epoch H")
    validate_h.add_argument("--preparation", type=Path, required=True)
    validate_h.add_argument("--checkpoint-h", type=Path, required=True)
    validate_h.add_argument("--epoch", type=int, default=130)

    run = commands.add_parser("run-method", help="run one production method")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--preparation", type=Path, required=True)
    run.add_argument("--checkpoint-h", type=Path, required=True)
    run.add_argument("--suite-root", type=Path, required=True)
    run.add_argument(
        "--method",
        choices=[item.name for item in Method] + [n for n in registered_method_names() if n not in {i.value for i in Method}],
        required=True,
    )
    run.add_argument(
        "--restart-incomplete", action="store_true",
        help="if this method's output directory already exists and is NOT complete, delete "
             "and recreate exactly that one directory before running. A completed method is "
             "always refused regardless of this flag. Never touches the runseed or outer "
             "suite directory.",
    )
    run.add_argument(
        "--legacy-full-checkpoints", action="store_true",
        help="debug/A-B-comparison override: reproduce the original copy-everything-up-front "
             "checkpoint staging behavior instead of the sparse auto mode.",
    )
    run.add_argument(
        "--selection-only", action="store_true",
        help="M3-only hyperparameter-selection mode: adapt and evaluate only "
             "pre-holdout data; never execute either final-holdout phase.",
    )

    evaluate = commands.add_parser("evaluate", help="evaluate completed prepared-method runs")
    evaluate.add_argument("--suite-root", type=Path, required=True)

    full = commands.add_parser("run-full-suite", help="run R0/M0/M1/M2 and evaluate")
    full.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    full.add_argument("--preparation", type=Path, required=True)
    full.add_argument("--checkpoint-h", type=Path, required=True)
    full.add_argument("--suite-root", type=Path, required=True)
    full.add_argument("--legacy-full-checkpoints", action="store_true", help="see run-method --legacy-full-checkpoints")

    compact = commands.add_parser(
        "compact-checkpoints",
        help="optional: convert completed methods' local checkpoints that are provably "
             "identical to checkpoint H into a base-reference declaration and delete the "
             "duplicates. Off by default (dry-run); never runs implicitly.",
    )
    compact.add_argument(
        "--suite-root", type=Path, required=True,
        help="a single completed runseed<N>/ directory, or an outer experiment directory "
             "containing multiple runseed*/ children (auto-detected).",
    )
    compact.add_argument(
        "--dry-run", action="store_true", default=True,
        help="report only, delete nothing (default).",
    )
    compact.add_argument(
        "--apply", dest="dry_run", action="store_false",
        help="actually delete redundant duplicates and write checkpoints/base_reference.json. "
             "Without this flag, compact-checkpoints always dry-runs.",
    )
    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        result = prepare_offline_data(
            args.config, args.source, args.assignments, args.output, overwrite=args.overwrite
        )
        _print(result["acceptance"])
        return 0 if result["acceptance"]["accepted"] else 2
    if args.command == "validate-preparation":
        _print(validate_offline_preparation(args.preparation))
        return 0
    if args.command == "train-checkpoint-h":
        _print(train_checkpoint_h(args.config, args.preparation, args.output, epochs=args.epochs))
        return 0
    if args.command == "validate-checkpoint-h":
        result = validate_fixed_epoch_h(args.checkpoint_h, args.preparation, epoch=args.epoch)
        _print(result)
        return 0 if result.get("historical_access_isolation_verified") else 2
    if args.command == "run-method":
        _print(run_prepared_method(
            args.method,
            config_path=args.config,
            suite_root=args.suite_root,
            preparation_root=args.preparation,
            checkpoint_h=args.checkpoint_h,
            checkpoint_storage_mode="legacy_full" if args.legacy_full_checkpoints else "auto",
            restart_incomplete=args.restart_incomplete,
            selection_only=args.selection_only,
        ))
        return 0
    if args.command == "evaluate":
        result = evaluate_completed_suite(args.suite_root)
        _print(result)
        return 0 if result["acceptance"]["accepted"] else 2
    if args.command == "run-full-suite":
        from .runner import run_full_prepared_suite
        _print(run_full_prepared_suite(
            config_path=args.config,
            suite_root=args.suite_root,
            preparation_root=args.preparation,
            checkpoint_h=args.checkpoint_h,
            checkpoint_storage_mode="legacy_full" if args.legacy_full_checkpoints else "auto",
        ))
        return 0
    if args.command == "compact-checkpoints":
        from .compaction import compact_checkpoints
        _print(compact_checkpoints(args.suite_root, dry_run=args.dry_run))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
