"""R0/M0/M1/M2 orchestration over immutable prepared samples."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

from ..checkpoint_registry import classify_storage_mode, hash_checkpoint_payload, register_checkpoint_h, stage_method_checkpoint
from ..core import (
    AdaptationConfig, M4_QUALITY_AWARE_SPLITFED, Method, MethodRuntimeConfig, ModelArchitecture, RuntimeConfig,
    SCENARIO_DATASET_BINDINGS, SegmentSizes, canonical_hash, ewc_config_from_mapping, load_suite_yaml,
    quality_weighting_config_from_mapping, registered_method_names, resolve_method_policy, write_json,
)
from .preparation import validate_offline_preparation
from .suite_identity import initialize_prepared_suite_root


def _coerce_method(value: Method | str) -> Method | str:
    """Accept enum members, serialized directory values, and CLI names.

    Returns the Method enum member for R0/M0/M1/M2 (unchanged, for full
    backward compatibility), or the bare string for any method registered
    via core.register_method_policy but not one of the four built-ins --
    the open-registry seam is a plain string identity, not a closed enum.
    """
    if isinstance(value, Method):
        return value
    text = str(value).strip()
    for item in Method:
        if text == item.value:
            return item
    name = text.rsplit(".", 1)[-1].upper()
    try:
        return Method[name]
    except KeyError:
        pass
    if text in registered_method_names():
        return text
    raise ValueError(f"unknown mitigation method: {value!r}")


def _method_value(method: Method | str) -> str:
    return method.value if isinstance(method, Method) else str(method)


def _prepared(preparation_root: Path) -> dict[str, Any]:
    validate_offline_preparation(preparation_root)
    return json.loads((preparation_root / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))


def _cohort(preparation_root: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Read the prepared dataset's own cohort manifest rather than asserting
    a fixed 21-clients/32-selected shape, so a differently sized population
    or drifted-per-cluster count remains valid so long as it is internally
    consistent with what the same preparation run recorded elsewhere.
    """
    clusters: dict[str, list[str]] = {}
    selected: list[str] = []
    with (preparation_root / "manifests" / "selected_drift_clients.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clusters.setdefault(str(row["cluster_id"]), []).append(str(row["client_id"]))
            if str(row["selected_for_drift"]).lower() in {"1", "true"}:
                selected.append(str(row["client_id"]))
    for clients in clusters.values():
        clients.sort()
    cluster_sizes = {len(clients) for clients in clusters.values()}
    if len(cluster_sizes) > 1:
        raise ValueError(f"prepared cohort has inconsistent per-cluster client counts: {sorted(cluster_sizes)}")
    recorded = json.loads((preparation_root / "manifests" / "selected_drift_clients_hash.json").read_text(encoding="utf-8"))
    if len(selected) != int(recorded["selected_count"]):
        raise ValueError(
            f"parsed selected-client count {len(selected)} does not match the prepared "
            f"dataset's recorded selected_count {recorded['selected_count']}"
        )
    return clusters, sorted(selected)


def _activate_h(checkpoint_h: Path, method_root: Path, *, mode: str = "auto") -> dict[str, Any]:
    registry = register_checkpoint_h(
        checkpoint_h, expected_epochs=130, expected_num_test_windows=0,
    )
    method_root.mkdir(parents=True, exist_ok=False)
    stage = stage_method_checkpoint(registry, method_root, mode=mode)
    if mode == "auto":
        # Nothing physically duplicated -- every client/server/global
        # artifact resolves to checkpoint H itself (via
        # checkpoints/base_reference.json, written by stage_method_checkpoint
        # above) until this method's own training/adaptation actually saves
        # something locally. "Active" state right now is exactly H,
        # unmodified.
        return {"stage": stage, "active_checkpoints": [], "active_tensor_hash": stage["combined_tensor_state_hash"]}
    initial = method_root / "checkpoints" / "initial"
    active: list[dict[str, Any]] = []
    for cluster in range(1, 5):
        source_cluster = initial / f"cluster_{cluster}"
        target_cluster = method_root / f"cluster_{cluster}"
        target_cluster.mkdir()
        for source in sorted(source_cluster.glob("client_*_final.pt")):
            client_id = source.name.removeprefix("client_").removesuffix("_final.pt")
            for name in (f"client_{client_id}_checkpoint.pt", f"client_{client_id}_final.pt"):
                destination = target_cluster / name
                shutil.copy2(source, destination)
                active.append(hash_checkpoint_payload(destination))
            write_json(target_cluster / f"client_{client_id}_progress.json", {
                "completed_epochs": 0, "current_epoch_loss_sum": 0.0, "current_epoch_batch_count": 0,
                "stream_window_offset": 0, "history": [], "best_epoch": 0, "best_val_score": None,
                "epochs_without_improvement": 0, "early_stopped": False, "stop_reason": None,
                "train_stream_active": False, "deployment_source_epoch": 130,
            })
        for name in ("server_latest.pt", "server_final.pt"):
            destination = target_cluster / name
            shutil.copy2(source_cluster / "server_final.pt", destination)
            active.append(hash_checkpoint_payload(destination))
    global_target = method_root / "global" / "global_server_latest.pt"
    global_target.parent.mkdir()
    shutil.copy2(initial / "global" / "global_server_latest.pt", global_target)
    active.append(hash_checkpoint_payload(global_target))
    active_hash = canonical_hash([row["tensor_state"] for row in active])
    if active_hash != stage["combined_tensor_state_hash"]:
        # Active layout duplicates client/server files for run_split, while the
        # staged set contains one canonical copy. Verify per-copy hashes instead.
        canonical_states = {row["tensor_state"] for row in stage["checkpoints"]}
        if any(row["tensor_state"] not in canonical_states for row in active):
            raise ValueError("active method checkpoints differ from staged H")
    return {"stage": stage, "active_checkpoints": active, "active_tensor_hash": active_hash}


def build_method_environment(
    method: Method | str,
    *,
    suite_root: str | Path,
    preparation_root: str | Path,
    phase: str,
    config: Mapping[str, Any] | None = None,
    checkpoint_storage_mode: str = "auto",
) -> dict[str, str]:
    method = _coerce_method(method)
    policy = resolve_method_policy(method)
    config = dict(config or {})
    seeds = config.get("seeds", {})
    sizes = SegmentSizes(**config["segments"]) if "segments" in config else SegmentSizes()
    model = ModelArchitecture(**dict(config.get("model") or {}))
    model.validate()
    runtime_cfg = RuntimeConfig(**dict(config.get("runtime") or {}))
    runtime_cfg.validate()
    adaptation_cfg = AdaptationConfig(**dict(config.get("adaptation") or {}))
    adaptation_cfg.validate()
    ewc_cfg = ewc_config_from_mapping(config.get("ewc"))
    ewc_cfg.validate()
    quality_weighting_cfg = quality_weighting_config_from_mapping(config.get("quality_weighting"))
    quality_weighting_cfg.validate()
    # M4_quality_aware_splitfed is the only method that can ever activate DQ
    # weighting -- M1/M2/M3/R0/R1/M0 always resolve _quality_active=False,
    # regardless of what quality_weighting says in the config, so their
    # training semantics are provably unchanged even in a DQ-enabled suite.
    _quality_active = quality_weighting_cfg.enabled and _method_value(method) == M4_QUALITY_AWARE_SPLITFED
    adaptation_enabled = phase in {"adaptation_and_holdout", "adaptation_only"} and policy.selected_clients_update
    if method == Method.M2:
        adaptation_epochs = adaptation_cfg.client_only_adaptation_epochs
    elif method == Method.M3:
        adaptation_epochs = adaptation_cfg.ewc_adaptation_epochs
    else:
        adaptation_epochs = adaptation_cfg.adaptation_epochs
    suite_root, preparation_root = Path(suite_root).resolve(), Path(preparation_root).resolve()
    prepared = _prepared(preparation_root)
    _, selected = _cohort(preparation_root)
    runtime = MethodRuntimeConfig(
        adaptation_seed=int(seeds.get("adaptation_seed", 42)),
        data_order_seed=int(seeds.get("data_order_seed", 42)),
        pre_drift_windows=sizes.pre_drift, adaptation_windows=sizes.adaptation,
        final_holdout_windows=sizes.final_holdout, expected_drift_selected_count=len(selected),
    )
    role = SCENARIO_DATASET_BINDINGS["scenario_observed" if policy.drift_enabled else "clean_reference"]
    mode = {
        "pre_drift": "deployment_pre_drift",
        "adaptation_and_holdout": "deployment_final_holdout",
        "adaptation_only": "deployment_pre_drift",
        "final_holdout": "deployment_final_holdout",
        "counterfactual": "deployment_counterfactual_clean_holdout",
    }[phase]
    if phase == "counterfactual":
        role = SCENARIO_DATASET_BINDINGS["clean_counterfactual"]
    role_map = {mode: role, "test": role}
    if phase in {"adaptation_and_holdout", "adaptation_only"}:
        role_map["deployment_adaptation"] = role
    expected_hashes = {role: prepared["roles"][role]["combined_payload_hash"]}
    method_root = suite_root / _method_value(method)
    env = os.environ.copy()
    env.update({
        "OUTPUT_DIR": str(method_root),
        "ASSIGNMENTS_PATH": str(Path("dataset/cluster-assignment.csv").resolve()),
        "PREPARED_DATASET_BACKEND": "true", "PREPARATION_ROOT": str(preparation_root),
        "PREPARED_ALLOWED_ROLES": role,
        "PREPARED_ROLE_MAP": json.dumps(role_map),
        "PREPARED_EXPECTED_COORDINATE_HASH": prepared["coordinate_manifest_hash"],
        "PREPARED_EXPECTED_DATASET_HASHES": json.dumps(expected_hashes),
        "NORMALIZE": "false", "SHARED_X_Y_SCALER": "false",
        "PREPARED_EXPECTED_SCALER_HASH": prepared["scaler_manifest_hash"],
        "DRIFT_INJECTION_ENABLED": "false", "DRIFT_CONFIG_PATH": "",
        "MODEL_INPUT_SIZE": str(model.input_size), "MODEL_HIDDEN_SIZE": str(model.hidden_size),
        "MODEL_NUM_LAYERS": str(model.num_layers), "MODEL_SPLIT_LAYER": str(model.split_layer),
        "MODEL_OUTPUT_SIZE": str(model.output_size), "MODEL_DROPOUT": str(model.dropout),
        "MODEL_BIDIRECTIONAL": "true" if model.bidirectional else "false",
        "WINDOW_SIZE": "24", "FORECAST_HORIZON": "1", "BATCH_SIZE": "128",
        "NUM_EPOCHS": str(adaptation_epochs) if adaptation_enabled else "0", "LOCAL_EPOCHS_PER_GLOBAL_ROUND": "1",
        "TRAINING_MODE": "client_batch_random", "CLIENT_ORDER_STRATEGY": "random_per_round",
        "RANDOM_SEED": str(runtime.adaptation_seed), "RANDOM_SEED_SCHEDULE": str(runtime.data_order_seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "EARLY_STOPPING": "false", "MITIGATION_DISABLE_VALIDATION_REPLAY": "true",
        "CLIENT_LR": "0.001", "SERVER_LR": "0.001",
        "ADAPTATION_WINDOW_OFFSET": str(runtime.pre_drift_windows), "ADAPTATION_WINDOW_LIMIT": str(runtime.adaptation_windows),
        "EXPECTED_DRIFT_SELECTED_COUNT": str(runtime.expected_drift_selected_count),
        "RUN_TEST_AFTER_TRAINING": "true", "DRIFT_SELECTED_CLIENT_IDS": ",".join(selected),
        "USE_QUALITY_WEIGHTS": "true" if (_quality_active and quality_weighting_cfg.mode in ("window", "both")) else "false",
        "USE_QUALITY_AS_INPUT": "false",
        "USE_QUALITY_AGGREGATION_WEIGHTS": "true" if (_quality_active and quality_weighting_cfg.mode in ("aggregation", "both")) else "false",
        "USE_QUALITY_ANCHOR_AGGREGATION": "false",
        "USE_GRADIENT_ALIGNMENT_FILTER": "false", "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL": "false",
        # Pinned to server.py's own existing hardcoded defaults (never
        # changed by this task) so run_manifest.json's auditability record
        # below can't drift from what the server subprocess actually used.
        "QUALITY_SCORE_MIN": "0.05", "QUALITY_SCORE_MAX": "1.0", "QUALITY_SCORE_EPS": "1e-8",
        "QUALITY_AGGREGATION_Q_MIN": "0.5", "QUALITY_AGGREGATION_EPS": "1e-8",
        "TRAIN_STREAM_MODE": "deployment_adaptation",
        "TEST_STREAM_MODE": mode,
        "TEST_WINDOW_OFFSET": "0", "TEST_WINDOW_LIMIT": "",
        "MITIGATION_METHOD": (method.name if isinstance(method, Method) else _method_value(method)) if adaptation_enabled else "",
        "SKIP_TRAINING": "false" if adaptation_enabled else "true",
        "FROZEN_VALIDATION_MODE": "false" if adaptation_enabled else "true",
        "EWC_ENABLED": "true" if (method == Method.M3 and ewc_cfg.enabled) else "false",
        "EWC_LAMBDA": str(ewc_cfg.lambda_),
        "EWC_REFERENCE_ROOT": str(suite_root / "ewc_reference"),
    })
    if checkpoint_storage_mode == "auto":
        # Exact declared path only -- the live subprocess never searches for
        # this, it reads precisely this file (or nothing, in legacy_full
        # mode / any run that predates this feature).
        env["CHECKPOINT_BASE_REFERENCE_PATH"] = str(method_root / "checkpoints" / "base_reference.json")
    if phase in {"pre_drift", "adaptation_only"}:
        env.update({"TEST_WINDOW_OFFSET": "0", "TEST_WINDOW_LIMIT": str(runtime.pre_drift_windows)})
    elif phase in {"final_holdout", "adaptation_and_holdout"}:
        env.update({
            "TEST_WINDOW_OFFSET": str(runtime.pre_drift_windows + runtime.adaptation_windows),
            "TEST_WINDOW_LIMIT": str(runtime.final_holdout_windows),
        })
    elif phase == "counterfactual":
        env.update({"TEST_WINDOW_OFFSET": "0", "TEST_WINDOW_LIMIT": str(runtime.final_holdout_windows)})
    return env


def _run_phase(method_root: Path, env: dict[str, str], phase: str) -> None:
    logs = method_root / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{phase}_stdout.log").open("w", encoding="utf-8") as stdout, (logs / f"{phase}_stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run([sys.executable, "-m", "drift.mitigation.prepared.splitfed_entry"], env=env, stdout=stdout, stderr=stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"prepared method phase failed: {phase}; inspect {logs}")


def _copy_predictions(method_root: Path, destination: Path, pattern: str = "client_*_test_*") -> None:
    for cluster in range(1, 5):
        target = destination / f"cluster_{cluster}"
        target.mkdir(parents=True, exist_ok=True)
        for source in (method_root / f"cluster_{cluster}").glob(pattern):
            shutil.copy2(source, target / source.name)


def _restart_incomplete_method_root(
    method_root: Path, suite_root: Path, method: Method | str, *, restart_incomplete: bool,
) -> None:
    """--restart-incomplete guardrails: delete-and-retry only the exact
    method_root about to be (re)activated, only when every one of six
    explicit checks holds. A completed method can never be overwritten
    (checked first, unconditionally). When the flag is not passed at all,
    this is a complete no-op -- an existing incomplete directory is left for
    _activate_h's own ``mkdir(..., exist_ok=False)`` to raise on, exactly as
    it does today.
    """
    if not restart_incomplete:
        return
    if not method_root.exists():
        return
    manifest_path = method_root / "run_manifest.json"
    status = None
    if manifest_path.is_file():
        try:
            status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = None
    if status == "complete":
        raise FileExistsError(f"method already complete, refusing to restart a completed method: {method_root}")
    resolved_method_root = method_root.resolve()
    resolved_suite_root = suite_root.resolve()
    resolved_outer_suite_dir = suite_root.parent.resolve()
    canonical_name = _method_value(_coerce_method(method))
    checks = {
        "restart_incomplete_flag_set": True,
        "status_not_complete": status != "complete",
        "is_direct_child_of_runseed_dir": resolved_method_root.parent == resolved_suite_root,
        "has_canonical_method_directory_name": resolved_method_root.name == canonical_name,
        "is_not_the_runseed_directory_itself": resolved_method_root != resolved_suite_root,
        "is_not_the_outer_suite_directory": resolved_method_root != resolved_outer_suite_dir,
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise ValueError(f"refusing to restart {method_root}: failed safety checks {failed}")
    shutil.rmtree(resolved_method_root)


def run_prepared_method(
    method: Method | str,
    *,
    config_path: str | Path,
    suite_root: str | Path,
    preparation_root: str | Path,
    checkpoint_h: str | Path,
    checkpoint_storage_mode: str = "auto",
    restart_incomplete: bool = False,
    selection_only: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_suite_yaml(config_path)
    method = _coerce_method(method)
    policy = resolve_method_policy(method)
    if selection_only and method != Method.M3:
        raise ValueError("selection_only is restricted to M3 EWC hyperparameter selection")
    suite_root, preparation_root = Path(suite_root).resolve(), Path(preparation_root).resolve()
    method_root = suite_root / _method_value(method)
    initialize_prepared_suite_root(suite_root, preparation_root, checkpoint_h, config_path)
    _restart_incomplete_method_root(method_root, suite_root, method, restart_incomplete=restart_incomplete)
    activation = _activate_h(Path(checkpoint_h).resolve(), method_root, mode=checkpoint_storage_mode)
    manifest = {
        "method": _method_value(method), "status": "running", "checkpoint_H": activation, "prepared": _prepared(preparation_root),
        "checkpoint_storage_mode": classify_storage_mode(policy), "checkpoint_storage_schema_version": 2 if checkpoint_storage_mode == "auto" else 1,
    }
    write_json(method_root / "run_manifest.json", manifest)
    pre_env = build_method_environment(method, suite_root=suite_root, preparation_root=preparation_root, phase="pre_drift", config=config, checkpoint_storage_mode=checkpoint_storage_mode)
    _run_phase(method_root, pre_env, "pre_drift")
    _copy_predictions(method_root, method_root / "pre_drift_predictions")
    main_phase = "adaptation_and_holdout" if policy.selected_clients_update else "final_holdout"
    ewc_reference_manifest = None
    if method == Method.M3:
        # Built (or reused, or fail-closed on identity mismatch) before the
        # subprocess that will read it starts -- never inside the training
        # subprocess itself.
        from .ewc import build_or_load_ewc_reference
        ewc_reference_manifest = build_or_load_ewc_reference(checkpoint_h, preparation_root, config, suite_root)
    main_environment_phase = "adaptation_only" if selection_only else main_phase
    main_env = build_method_environment(method, suite_root=suite_root, preparation_root=preparation_root, phase=main_environment_phase, config=config, checkpoint_storage_mode=checkpoint_storage_mode)
    if selection_only:
        # Hyperparameter selection must stop before FINAL_HOLDOUT. Keep the
        # canonical adaptation environment and scheduler, but disable the
        # post-training test entry point. The adaptation_only environment maps
        # even an accidental test call to PRE_DRIFT, never FINAL_HOLDOUT.
        main_env["RUN_TEST_AFTER_TRAINING"] = "false"
        _run_phase(method_root, main_env, "adaptation_only")
        _copy_predictions(
            method_root,
            method_root / "adaptation_predictions",
            pattern="client_*_adaptation_predictions.csv",
        )

        # Measure retention only on the earlier PRE_DRIFT segment. This is a
        # frozen, prediction-only replay after adaptation; it never opens the
        # final or counterfactual-final holdout modes.
        post_adaptation_env = build_method_environment(
            method,
            suite_root=suite_root,
            preparation_root=preparation_root,
            phase="pre_drift",
            config=config,
            checkpoint_storage_mode=checkpoint_storage_mode,
        )
        _run_phase(method_root, post_adaptation_env, "post_adaptation_pre_drift")
        _copy_predictions(
            method_root,
            method_root / "post_adaptation_pre_drift_predictions",
        )

        forbidden = (
            method_root / "persistent_drift_final_holdout",
            method_root / "clean_final_holdout",
            method_root / "counterfactual_clean_holdout",
        )
        leaked = [str(path) for path in forbidden if path.exists()]
        if leaked:
            raise RuntimeError(f"selection-only run created forbidden final-holdout artifacts: {leaked}")
    else:
        _run_phase(method_root, main_env, main_phase)
        # R1's FINAL_HOLDOUT data is clean, not drifted -- "persistent_drift_..."
        # would misdescribe it, so it gets its own clearly-named copy instead.
        # This is purely a labeled duplicate for humans/audit: evaluation.py
        # reads R1 (not oracle_enabled) straight from method_root, unaffected.
        if method == Method.R1:
            _copy_predictions(method_root, method_root / "clean_final_holdout")
        else:
            _copy_predictions(method_root, method_root / "persistent_drift_final_holdout")
        if policy.selected_clients_update:
            _copy_predictions(method_root, method_root / "adaptation_predictions", pattern="client_*_adaptation_predictions.csv")
        if policy.oracle_enabled:
            counter_env = build_method_environment(method, suite_root=suite_root, preparation_root=preparation_root, phase="counterfactual", config=config, checkpoint_storage_mode=checkpoint_storage_mode)
            _run_phase(method_root, counter_env, "counterfactual")
            _copy_predictions(method_root, method_root / "counterfactual_clean_holdout")
    adaptation_cfg = AdaptationConfig(**dict(config.get("adaptation") or {}))
    if method == Method.M2:
        method_adaptation_epochs = adaptation_cfg.client_only_adaptation_epochs
    elif method == Method.M3:
        method_adaptation_epochs = adaptation_cfg.ewc_adaptation_epochs
    else:
        method_adaptation_epochs = adaptation_cfg.adaptation_epochs
    ewc_cfg = ewc_config_from_mapping(config.get("ewc"))
    quality_weighting_cfg = quality_weighting_config_from_mapping(config.get("quality_weighting"))
    # main_env is the adaptation-phase environment -- the only phase where
    # these flags can affect actual training (pre_drift/counterfactual are
    # SKIP_TRAINING regardless). Read back from main_env, not recomputed,
    # so this can never drift from what the training subprocess actually
    # received.
    quality_active_for_this_method = (
        main_env["USE_QUALITY_WEIGHTS"] == "true" or main_env["USE_QUALITY_AGGREGATION_WEIGHTS"] == "true"
    )
    manifest.update({
        "status": "complete", "one_adaptation_pass": policy.selected_clients_update,
        "selection_only": selection_only,
        "selection_data_policy": ({
            "adaptation_predictions": "pre_update",
            "retention_evaluation": "post_adaptation_pre_drift_replay",
            "final_holdout_evaluated": False,
            "counterfactual_final_holdout_evaluated": False,
            "run_test_after_adaptation": main_env["RUN_TEST_AFTER_TRAINING"],
        } if selection_only else None),
        "adaptation_epochs": method_adaptation_epochs if policy.selected_clients_update else 0,
        "adaptation_config": {
            "adaptation_epochs": adaptation_cfg.adaptation_epochs,
            "client_only_adaptation_epochs": adaptation_cfg.client_only_adaptation_epochs,
            "ewc_adaptation_epochs": adaptation_cfg.ewc_adaptation_epochs,
        },
        "ewc_config": {
            "enabled": ewc_cfg.enabled, "lambda": ewc_cfg.lambda_, "fisher_source": ewc_cfg.fisher_source,
            "fisher_type": ewc_cfg.fisher_type, "fisher_max_batches": ewc_cfg.fisher_max_batches,
        },
        "ewc_reference": (
            {"identity_hash": ewc_reference_manifest["identity_hash"], "client_count": len(ewc_reference_manifest["clients"])}
            if ewc_reference_manifest is not None else None
        ),
        "quality_weighting": {
            "requested_enabled": quality_weighting_cfg.enabled,
            "requested_mode": quality_weighting_cfg.mode if quality_weighting_cfg.enabled else None,
            "active_for_this_method": quality_active_for_this_method,
            "resolved_env": {
                "USE_QUALITY_WEIGHTS": main_env["USE_QUALITY_WEIGHTS"],
                "USE_QUALITY_AGGREGATION_WEIGHTS": main_env["USE_QUALITY_AGGREGATION_WEIGHTS"],
                "QUALITY_SCORE_MIN": main_env["QUALITY_SCORE_MIN"],
                "QUALITY_SCORE_MAX": main_env["QUALITY_SCORE_MAX"],
                "QUALITY_SCORE_EPS": main_env["QUALITY_SCORE_EPS"],
                "QUALITY_AGGREGATION_Q_MIN": main_env["QUALITY_AGGREGATION_Q_MIN"],
                "QUALITY_AGGREGATION_EPS": main_env["QUALITY_AGGREGATION_EPS"],
            },
        },
    })
    write_json(method_root / "run_manifest.json", manifest)
    return manifest


def run_full_prepared_suite(
    *,
    config_path: str | Path,
    suite_root: str | Path,
    preparation_root: str | Path,
    checkpoint_h: str | Path,
    checkpoint_storage_mode: str = "auto",
) -> dict[str, Any]:
    """Run every method the suite config selects, from independent H copies,
    then evaluate.

    The implementation and smoke stage never calls this automatically --
    it must be invoked explicitly (via the CLI or directly from Python).

    No restart_incomplete parameter here, deliberately: a partially-run
    suite's recovery path is always to drop to run-method
    --restart-incomplete for the one broken method, never to blow away this
    suite-root-level guard.
    """
    suite_root = Path(suite_root).resolve()
    if suite_root.exists():
        raise FileExistsError(f"full prepared suite output already exists: {suite_root}")
    config = load_suite_yaml(config_path)
    selected_methods = [_coerce_method(name) for name in config["methods"]]
    results = {
        _method_value(method): run_prepared_method(
            method,
            config_path=config_path,
            suite_root=suite_root,
            preparation_root=preparation_root,
            checkpoint_h=checkpoint_h,
            checkpoint_storage_mode=checkpoint_storage_mode,
        )
        for method in selected_methods
    }
    from ..evaluation import evaluate_completed_suite
    evaluation = evaluate_completed_suite(suite_root)
    return {"methods": results, "evaluation": evaluation, "complete": True}
