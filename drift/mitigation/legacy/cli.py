"""Command-line entry point for mitigation preparation and smoke acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core import SUITE_ID, load_suite_yaml, write_json
from ..evaluation import evaluate_completed_suite
from .checkpoint_integration import run_tiny_checkpoint_H_fixture
from .service_adapter import run_full_adapter_dry_run, run_two_cluster_fedavg_fixture
from .smoke import run_smoke_suite
from .production_prepare import prepare_production_suite
from .production_runner import run_full_suite, run_production_method


DEFAULT_CONFIG = Path("drift/configs/mitigation_sudden_process_seed42.yaml")
DEFAULT_SMOKE_OUTPUT = Path("outputs/drift_mitigation/smoke") / f"{SUITE_ID}_smoke4"

LEGACY_RUNTIME_AUTHORIZATION_FLAG = "--allow-legacy-runtime-pipeline"
_EXECUTING_COMMANDS = (
    "smoke", "integration-smoke", "prepare", "run-method", "evaluate", "run-full-suite", "train-checkpoint-h",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legacy runtime-processing mitigation pipeline (generation 1). "
        "Does not use the fail-closed prepared dataset backend; see drift.mitigation.prepared for the canonical path."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="strictly validate the approved suite configuration")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    smoke = subparsers.add_parser("smoke", help="run the deterministic four-client R0/M0/M1/M2 acceptance suite")
    smoke.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke.add_argument("--output", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    smoke.add_argument("--overwrite", action="store_true")
    smoke.add_argument(LEGACY_RUNTIME_AUTHORIZATION_FLAG, action="store_true")

    integration = subparsers.add_parser("integration-smoke", help="run smoke plus production scheduler, real FedAvg, and tiny checkpoint-H fixtures")
    integration.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    integration.add_argument("--output", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    integration.add_argument("--overwrite", action="store_true")
    integration.add_argument(LEGACY_RUNTIME_AUTHORIZATION_FLAG, action="store_true")

    prepare = subparsers.add_parser("prepare", help="register H, materialize common data, and perform load-only preflight")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--checkpoint-h", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--preflight-only", action="store_true", required=True)
    prepare.add_argument(LEGACY_RUNTIME_AUTHORIZATION_FLAG, action="store_true")

    run_method_parser = subparsers.add_parser("run-method", help="run one authorized production method")
    run_method_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_method_parser.add_argument("--output", type=Path, required=True)
    run_method_parser.add_argument("--method", required=True, choices=["R0", "M0", "M1", "M2"])
    run_method_parser.add_argument("--authorization", default="")
    run_method_parser.add_argument(LEGACY_RUNTIME_AUTHORIZATION_FLAG, action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate completed method outputs without training")
    evaluate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(LEGACY_RUNTIME_AUTHORIZATION_FLAG, action="store_true")

    full = subparsers.add_parser("run-full-suite", help="run all four methods only after explicit authorization")
    full.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--authorization", default="")
    full.add_argument(LEGACY_RUNTIME_AUTHORIZATION_FLAG, action="store_true")

    train_h = subparsers.add_parser("train-checkpoint-h", help="intentionally unavailable: H is externally trained")
    train_h.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_h.add_argument(LEGACY_RUNTIME_AUTHORIZATION_FLAG, action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in _EXECUTING_COMMANDS and not getattr(args, "allow_legacy_runtime_pipeline", False):
        raise SystemExit(
            f"refusing to run legacy command {args.command!r} without {LEGACY_RUNTIME_AUTHORIZATION_FLAG}: "
            "the legacy pipeline bypasses the fail-closed prepared dataset backend "
            "(drift/mitigation/prepared) and must be explicitly authorized before it starts "
            "any service, subprocess, training, preparation, or evaluation."
        )
    config = load_suite_yaml(args.config)
    if args.command == "validate-config":
        print(json.dumps({"valid": True, "suite_id": config["suite_id"]}, indent=2))
        return 0
    if args.command == "smoke":
        result = run_smoke_suite(args.output, overwrite=args.overwrite)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["acceptance"]["accepted"] else 2
    if args.command == "integration-smoke":
        result = run_smoke_suite(args.output, overwrite=args.overwrite)
        integration_root = Path(args.output) / "integration"
        integration_root.mkdir()
        adapter = run_full_adapter_dry_run()
        fedavg = run_two_cluster_fedavg_fixture()
        checkpoint_h = run_tiny_checkpoint_H_fixture(integration_root / "tiny_checkpoint_H")
        write_json(integration_root / "production_adapter_dry_run.json", adapter)
        write_json(integration_root / "two_cluster_fedavg.json", fedavg)
        checks = dict(result["acceptance"]["checks"])
        checks.update({
            "production_scheduler_adapter_dry_run": bool(adapter["accepted"]),
            "two_cluster_real_fedavg_parity": bool(fedavg["accepted"]),
            "tiny_real_run_split_checkpoint_H": bool(checkpoint_h["accepted"]),
            "tiny_H_loaded_independently_into_R0_M0_M1_M2": bool(checkpoint_h["all_methods_loaded_identical_H"]),
            "production_checkpoint_H_one_client_step_per_batch": bool(checkpoint_h["checkpoint_H_one_optimizer_step_per_client_batch"]),
            "tiny_M1_client_optimizer_steps_exact": all(
                steps == 3 for steps in checkpoint_h["method_checks"]["M1_continued_splitfed"]["client_optimizer_steps"].values()
            ),
            "tiny_M2_client_optimizer_permission_matrix": all(
                steps == (3 if client in checkpoint_h["selected_clients"] else 0)
                for client, steps in checkpoint_h["method_checks"]["M2_oracle_client_only"]["client_optimizer_steps"].items()
            ),
            "tiny_M2_server_optimizer_steps_zero": (
                checkpoint_h["method_checks"]["M2_oracle_client_only"]["server_optimizer_steps"] == 0
            ),
        })
        acceptance = {
            "suite_id": SUITE_ID,
            "smoke": True,
            "integration_smoke": True,
            "accepted": all(checks.values()),
            "complete": True,
            "checks": checks,
        }
        write_json(Path(args.output) / "acceptance_report.json", acceptance)
        write_json(integration_root / "integration_acceptance.json", acceptance)
        payload = {"output": str(args.output), "acceptance": acceptance, "adapter": adapter, "fedavg": fedavg, "checkpoint_H": checkpoint_h}
        print(json.dumps(payload, indent=2, default=str))
        return 0 if acceptance["accepted"] else 2
    if args.command == "prepare":
        result = prepare_production_suite(args.config, args.checkpoint_h, args.output, resume=args.resume, overwrite=args.overwrite, preflight_only=args.preflight_only)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["acceptance"]["accepted"] else 2
    if args.command == "run-method":
        result = run_production_method(args.method, args.output, authorization=args.authorization)
        print(json.dumps(result, indent=2, default=str)); return 0
    if args.command == "evaluate":
        result = evaluate_completed_suite(args.output)
        print(json.dumps(result, indent=2, default=str)); return 0 if result["acceptance"]["accepted"] else 2
    if args.command == "run-full-suite":
        result = run_full_suite(args.output, authorization=args.authorization)
        print(json.dumps(result, indent=2, default=str)); return 0
    raise SystemExit("checkpoint H training is not launched by this mitigation CLI; the registered 130-epoch H is used as input")


if __name__ == "__main__":
    raise SystemExit(main())
