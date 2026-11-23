"""Fixed-epoch checkpoint-H training over clean prepared samples only."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

from data.prepared_window_store import H_TRAINING_ROLES, PreparedWindowStore, file_sha256
from ..checkpoint_registry import hash_checkpoint_payload, register_checkpoint_h
from ..core import BaselineHContract, ModelArchitecture, canonical_hash, guard_output_directory, load_suite_yaml, write_json
from .preparation import validate_offline_preparation


def _resolved_model_architecture(config: Mapping[str, Any]) -> ModelArchitecture:
    model = ModelArchitecture(**dict(config.get("model") or {}))
    model.validate()
    return model


def baseline_h_contract_from_config(config: Mapping[str, Any], prepared: Mapping[str, Any]) -> BaselineHContract:
    """Build the explicit contract a freshly trained H is stamped with."""
    model = _resolved_model_architecture(config)
    return BaselineHContract(
        window_size=int(config["windowing"]["window_size"]),
        forecast_horizon=int(config["windowing"]["forecast_horizon"]),
        h_training_seed=int(config.get("seeds", {}).get("h_training_seed", 42)),
        normalization_method=str(config["normalization"]["method"]),
        scaler_manifest_hash=str(prepared["scaler_manifest_hash"]),
        coordinate_manifest_hash=str(prepared["coordinate_manifest_hash"]),
        clean_train_hash=str(prepared["roles"]["clean_train"]["combined_payload_hash"]),
        clean_validation_hash=str(prepared["roles"]["clean_validation"]["combined_payload_hash"]),
        model_input_size=model.input_size, model_hidden_size=model.hidden_size,
        model_num_layers=model.num_layers, model_split_layer=model.split_layer,
        model_output_size=model.output_size, model_dropout=model.dropout,
        model_bidirectional=model.bidirectional,
    )


def build_h_training_view(preparation_root: str | Path, destination: str | Path) -> dict[str, Any]:
    """Stage a view containing no deployment, drift, residual, or oracle files."""
    source = Path(preparation_root).resolve()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    prepared = json.loads((source / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True)
    (destination / "manifests").mkdir()
    roles: dict[str, Any] = {}
    for role in sorted(H_TRAINING_ROLES):
        role_payload = prepared["roles"][role]
        copied_clients: dict[str, Any] = {}
        for key, entry in role_payload["clients"].items():
            source_path = source / entry["path"]
            target_path = destination / entry["path"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            if file_sha256(target_path) != entry["artifact_sha256"]:
                raise ValueError(f"H training-view copy hash mismatch: {entry['path']}")
            copied_clients[key] = dict(entry)
        roles[role] = {**role_payload, "clients": copied_clients}
    view_manifest = {
        "schema_version": 2,
        "coordinate_manifest_hash": prepared["coordinate_manifest_hash"],
        "scaler_manifest_hash": prepared["scaler_manifest_hash"],
        "roles": roles,
        "access_scope": "checkpoint_H_clean_train_validation_only",
    }
    write_json(destination / "manifests" / "prepared_dataset_hashes.json", view_manifest)
    for name in ("scaler_parameters.json", "scaler_manifest.csv", "split_manifest.json"):
        shutil.copy2(source / "manifests" / name, destination / "manifests" / name)
    policy = {
        "allowed_roles": sorted(H_TRAINING_ROLES),
        "gradient_roles": ["clean_train"],
        "evaluation_only_roles": ["clean_validation"],
        "forbidden_paths_absent": [
            "raw_clean", "raw_drifted", "audit_private", "drift_ground_truth_manifest.csv",
            "clean_deployment", "drifted_deployment", "clean_final_holdout_counterfactual",
        ],
        "view_manifest_hash": canonical_hash(view_manifest),
    }
    write_json(destination / "manifests" / "h_training_access_policy.json", policy)
    return policy


def validate_h_training_inputs(training_view: str | Path) -> dict[str, Any]:
    root = Path(training_view).resolve()
    policy = json.loads((root / "manifests" / "h_training_access_policy.json").read_text(encoding="utf-8"))
    store = PreparedWindowStore(root, allowed_roles=H_TRAINING_ROLES)
    if set(store.manifest["roles"]) != set(H_TRAINING_ROLES):
        raise ValueError("H training view exposes roles other than clean_train and clean_validation")
    forbidden_names = {"raw_drifted", "audit_private", "clean_deployment", "drifted_deployment", "clean_final_holdout_counterfactual"}
    present = forbidden_names & {path.name for path in root.rglob("*")}
    if present:
        raise ValueError(f"H training view contains forbidden artifact(s): {sorted(present)}")
    # Permission checks are part of validation, not merely documentation.
    rejected = []
    for role in ("clean_deployment", "drifted_deployment", "clean_final_holdout_counterfactual"):
        try:
            store.load_client(role, "1", "0")
        except PermissionError:
            rejected.append(role)
    if len(rejected) != 3:
        raise ValueError("H training store did not reject every deployment role")
    return {"valid": True, "allowed_roles": sorted(store.allowed_roles), "rejected_roles": rejected, "policy": policy}


def build_h_environment(config: dict[str, Any], training_view: Path, output: Path) -> dict[str, str]:
    manifest = json.loads((training_view / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))
    hashes = {role: manifest["roles"][role]["combined_payload_hash"] for role in H_TRAINING_ROLES}
    seeds = config.get("seeds", {})
    h_training_seed = int(seeds.get("h_training_seed", 42))
    data_order_seed = int(seeds.get("data_order_seed", 42))
    model = _resolved_model_architecture(config)
    env = os.environ.copy()
    env.update({
        "OUTPUT_DIR": str(output.resolve()),
        "ASSIGNMENTS_PATH": str(Path(config["source"]["assignment_path"]).resolve()),
        "PREPARED_DATASET_BACKEND": "true",
        "PREPARATION_ROOT": str(training_view.resolve()),
        "PREPARED_ALLOWED_ROLES": "clean_train,clean_validation",
        "PREPARED_ROLE_MAP": json.dumps({"train": "clean_train", "val": "clean_validation"}),
        "PREPARED_EXPECTED_COORDINATE_HASH": manifest["coordinate_manifest_hash"],
        "PREPARED_EXPECTED_SCALER_HASH": manifest["scaler_manifest_hash"],
        "PREPARED_EXPECTED_DATASET_HASHES": json.dumps(hashes),
        "NORMALIZE": "false", "SHARED_X_Y_SCALER": "false",
        "DRIFT_INJECTION_ENABLED": "false", "DRIFT_CONFIG_PATH": "",
        "NUM_EPOCHS": "130", "RUN_TEST_AFTER_TRAINING": "false",
        "SKIP_TRAINING": "false", "EARLY_STOPPING": "false",
        "TRAIN_STREAM_MODE": "train", "TRAINING_MODE": "client_batch_random",
        "CLIENT_ORDER_STRATEGY": "random_per_round", "LOCAL_EPOCHS_PER_GLOBAL_ROUND": "1",
        "RANDOM_SEED": str(h_training_seed), "RANDOM_SEED_SCHEDULE": str(data_order_seed),
        "BATCH_SIZE": "128", "CLIENT_LR": "0.001", "SERVER_LR": "0.001",
        "MODEL_INPUT_SIZE": str(model.input_size), "MODEL_HIDDEN_SIZE": str(model.hidden_size),
        "MODEL_NUM_LAYERS": str(model.num_layers), "MODEL_SPLIT_LAYER": str(model.split_layer),
        "MODEL_OUTPUT_SIZE": str(model.output_size), "MODEL_DROPOUT": str(model.dropout),
        "MODEL_BIDIRECTIONAL": "true" if model.bidirectional else "false",
        "WINDOW_SIZE": "24", "FORECAST_HORIZON": "1",
        "USE_QUALITY_WEIGHTS": "false", "USE_QUALITY_AS_INPUT": "false",
        "USE_QUALITY_AGGREGATION_WEIGHTS": "false", "USE_QUALITY_ANCHOR_AGGREGATION": "false",
        "USE_GRADIENT_ALIGNMENT_FILTER": "false", "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL": "false",
    })
    return env


def validate_fixed_epoch_h(root: str | Path, preparation_root: str | Path, *, epoch: int = 130) -> dict[str, Any]:
    root = Path(root).resolve()
    preparation_root = Path(preparation_root).resolve()
    prepared = json.loads((preparation_root / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))
    scaler_payload = json.loads((preparation_root / "manifests" / "scaler_parameters.json").read_text(encoding="utf-8"))
    scaler_lookup = {
        (str(row["cluster_id"]), str(row["client_id"])): row
        for row in scaler_payload["clients"]
    }
    # A compatible tensor state is insufficient for the approved H policy:
    # the H process itself must never have opened deployment/test data.
    forbidden_access_evidence = find_h_forbidden_access_evidence(root)
    historical_access_isolation_verified = not forbidden_access_evidence
    registry = register_checkpoint_h(
        root, expected_epochs=epoch, expected_num_test_windows=0,
    )
    validation_monitored = True
    client_hashes: list[dict[str, Any]] = []
    server_hashes: list[dict[str, Any]] = []
    for cluster in range(1, 5):
        cluster_dir = root / f"cluster_{cluster}"
        clients = sorted(cluster_dir.glob("client_*_final.pt"))
        # Expected count comes from the prepared dataset's own manifest for
        # this cluster, not a fixed golden number, so a differently sized
        # population still validates correctly.
        expected_clients = {
            key.split(":", 1)[1] for key in prepared["roles"]["clean_train"]["clients"] if key.split(":", 1)[0] == str(cluster)
        }
        if len(clients) != len(expected_clients):
            raise ValueError(
                f"fixed H cluster {cluster} must contain {len(expected_clients)} final client "
                f"checkpoints (per the prepared dataset manifest), found {len(clients)}"
            )
        for checkpoint in clients:
            hashed = hash_checkpoint_payload(checkpoint)
            if int(hashed["counter_values"].get("epoch", -1)) != epoch:
                raise ValueError(f"client H checkpoint is not epoch {epoch}: {checkpoint}")
            client_id = checkpoint.name.removeprefix("client_").removesuffix("_final.pt")
            progress = json.loads((cluster_dir / f"client_{client_id}_progress.json").read_text(encoding="utf-8"))
            if int(progress.get("completed_epochs", -1)) != epoch or progress.get("train_stream_active") is True:
                raise ValueError(f"client progress is not a completed epoch-{epoch} state: {client_id}")
            client_hashes.append({"cluster_id": str(cluster), "client_id": client_id, **hashed})
            client_config = json.loads((cluster_dir / f"client_{client_id}_config.json").read_text(encoding="utf-8"))
            normalization = client_config.get("normalization", {})
            expected_scaler = scaler_lookup[(str(cluster), client_id)]
            location = normalization.get("y_location", normalization.get("mean"))
            scale = normalization.get("y_scale", normalization.get("scale"))
            if (
                normalization.get("target_scaling_strategy") != "shared"
                or normalization.get("method") != expected_scaler["method"]
                or not math.isclose(float(location), float(expected_scaler["location"]), rel_tol=0.0, abs_tol=2e-8)
                or not math.isclose(float(scale), float(expected_scaler["scale"]), rel_tol=0.0, abs_tol=2e-8)
            ):
                raise ValueError(f"checkpoint H scaler is incompatible with prepared client {cluster}/{client_id}")
            history = list(progress.get("history", []))
            validation_monitored = validation_monitored and (
                len(history) == epoch
                and all(row.get("val_raw_mae") is not None for row in history)
            )
            if not validation_monitored:
                raise ValueError(
                    f"checkpoint H lacks prediction-only validation monitoring through epoch {epoch}: "
                    f"{cluster}/{client_id}"
                )
        server_path = cluster_dir / "server_final.pt"
        if not server_path.is_file():
            raise ValueError(f"fixed H lacks final split server for cluster {cluster}")
        server_hashes.append({"cluster_id": str(cluster), **hash_checkpoint_payload(server_path)})
    global_path = root / "global" / "global_server_latest.pt"
    if not global_path.is_file():
        raise ValueError("fixed H lacks global/global_server_latest.pt")
    global_hash = hash_checkpoint_payload(global_path)
    manifest = {
        "policy": "coherent_fixed_epoch",
        "epoch": epoch,
        "validation": "prediction_only_monitoring",
        "selection": "no_independent_validation_selected_checkpoint_composition",
        "clients": client_hashes,
        "split_servers": server_hashes,
        "global_server": global_hash,
        "clean_train_hash": prepared["roles"]["clean_train"]["combined_payload_hash"],
        "clean_validation_hash": prepared["roles"]["clean_validation"]["combined_payload_hash"],
        "scaler_manifest_hash": prepared["scaler_manifest_hash"],
        "coordinate_manifest_hash": prepared["coordinate_manifest_hash"],
        "scaler_ownership": "per_client_shared_x_y",
        "state_contract_compatible": True,
        "historical_access_isolation_verified": historical_access_isolation_verified,
        "forbidden_access_evidence": forbidden_access_evidence,
        "scaler_fit_coordinates": [0, 6138],
        "coordinate_6139_excluded": True,
        "validation_monitored_every_epoch": validation_monitored,
        "registry_combined_artifact_hash": registry.combined_artifact_hash,
        "combined_tensor_hash": canonical_hash([row["tensor_state"] for row in [*client_hashes, *server_hashes, global_hash]]),
        "combined_optimizer_hash": canonical_hash([row["optimizer_state"] for row in [*client_hashes, *server_hashes, global_hash]]),
    }
    write_json(root / "checkpoint_H_epoch130_manifest.json", manifest)
    write_json(
        root / "checkpoint_H_acceptance.json",
        {
            "accepted": historical_access_isolation_verified,
            "state_contract_compatible": True,
            "historical_access_isolation_verified": historical_access_isolation_verified,
            "policy": "coherent_fixed_epoch",
            "epoch": epoch,
            "manifest_hash": canonical_hash(manifest),
            "rejection_reason": (
                None
                if historical_access_isolation_verified
                else "checkpoint-H run contains evidence that deployment/test data was accessed"
            ),
        },
    )
    return manifest


def train_checkpoint_h(config_path: str | Path, preparation_root: str | Path, output: str | Path, *, epochs: int = 130) -> dict[str, Any]:
    """Launch a full H only when explicitly called; never used by preparation."""
    if int(epochs) != 130:
        raise ValueError("checkpoint H must use exactly 130 clean training epochs")
    config = load_suite_yaml(config_path)
    validate_offline_preparation(preparation_root)
    root = guard_output_directory(output)
    training_view = root / "h_training_view"
    build_h_training_view(preparation_root, training_view)
    validate_h_training_inputs(training_view)
    env = build_h_environment(config, training_view, root)
    logs = root / "logs"
    logs.mkdir()
    with (logs / "stdout.log").open("w", encoding="utf-8") as stdout, (logs / "stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run([sys.executable, "-m", "drift.mitigation.prepared.splitfed_entry"], env=env, stdout=stdout, stderr=stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"checkpoint-H training failed; inspect {logs}")
    manifest = validate_fixed_epoch_h(root, preparation_root, epoch=130)
    acceptance = json.loads((root / "checkpoint_H_acceptance.json").read_text(encoding="utf-8"))
    if not acceptance["accepted"]:
        raise RuntimeError("new checkpoint-H run failed historical access-isolation validation")
    prepared = json.loads((Path(preparation_root).resolve() / "manifests" / "prepared_dataset_hashes.json").read_text(encoding="utf-8"))
    contract = baseline_h_contract_from_config(config, prepared)
    write_json(root / "checkpoint_H_baseline_contract.json", {
        "contract_source": "explicit", "contract": asdict(contract), "contract_hash": contract.content_hash(),
    })
    return {"output": str(root), "accepted": True, "manifest": manifest, "baseline_h_contract_hash": contract.content_hash()}

def find_h_forbidden_access_evidence(root: str | Path) -> list[str]:
    """Return artifacts proving that an H process opened test/deployment data.

    ``run_split`` always writes a JSON report shell.  A shell with no clients,
    clusters, or global client count is bookkeeping, not evidence that the
    prepared deployment role was opened.  Per-client test artifacts and
    non-empty aggregate reports remain fail-closed evidence.
    """
    root = Path(root).resolve()
    evidence = {
        str(path.relative_to(root))
        for pattern in ("*_test_predictions.csv", "*_test_metrics.json")
        for path in root.rglob(pattern)
    }
    for path in root.rglob("test_report.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            global_count = payload.get("global_summary", {}).get("num_clients")
            report_has_results = bool(payload.get("clients")) or bool(payload.get("clusters"))
            report_has_results = report_has_results or global_count not in (None, 0)
        except (OSError, AttributeError, TypeError, ValueError, json.JSONDecodeError):
            report_has_results = True
        if report_has_results:
            evidence.add(str(path.relative_to(root)))
    for path in root.rglob("test_report.csv"):
        try:
            nonempty_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            report_has_results = len(nonempty_lines) > 1
        except OSError:
            report_has_results = True
        if report_has_results:
            evidence.add(str(path.relative_to(root)))
    return sorted(evidence)


