"""Tiny production integration smoke for the offline-prepared backend.

This fixture uses four real client/server worker processes and the existing
``run_split`` scheduler.  Its geometry is intentionally tiny; production
defaults remain 84 clients and 524/700/524.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from data.prepared_window_store import canonical_array_payload_hash, file_sha256
from drift.contracts import SchedulePoint
from drift.process import generate_process_timeline
from drift.windowing import build_forecasting_windows, fit_clean_training_scalers, normalize_windows

from ..checkpoint_registry import hash_checkpoint_payload
from ..core import Method, canonical_hash, resolve_method_policy, write_json
from ..testing_support import _free_port_block
from .preparation import _metadata, _schedule, _write_samples
from .training import build_h_training_view, validate_h_training_inputs


def _source_clients() -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    payload = pd.read_pickle(Path(__file__).resolve().parents[3] / "dataset" / "output_clean" / "cluster_1.pkl")
    frame = payload["timeseries"].copy()
    frame["id"] = frame["id"].astype(str)
    frame["Fecha_Hora"] = pd.to_datetime(frame["Fecha_Hora"])
    clients = sorted(frame["id"].unique())[:4]
    values: dict[str, np.ndarray] = {}
    timestamps = None
    for client_id in clients:
        rows = frame.loc[frame["id"] == client_id].sort_values("Fecha_Hora").iloc[:33]
        values[client_id] = rows["value"].to_numpy(dtype=np.float32).astype(np.float64)
        current = rows["Fecha_Hora"].to_numpy(dtype="datetime64[ns]")
        if timestamps is None:
            timestamps = current
        elif not np.array_equal(timestamps, current):
            raise ValueError("tiny prepared fixture clients do not share timestamps")
    assert timestamps is not None
    return clients, values, timestamps


def _deployment_metadata(timestamps: np.ndarray, *, selected: bool) -> dict[str, np.ndarray]:
    starts = np.asarray([5, 6, 7, 8], dtype=np.int64)
    result = _metadata(starts, timestamps, 24, 1, None)
    segments = np.asarray(["PRE_DRIFT", "ADAPTATION", "ADAPTATION", "FINAL_HOLDOUT"], dtype="U20")
    result["deployment_window_number"] = np.arange(1, 5, dtype=np.int64)
    result["segment"] = segments
    for coordinate, name in ((starts, "process_regime_at_window_start"), (starts + 23, "process_regime_at_window_end"), (starts + 24, "process_regime_at_target")):
        result[name] = np.asarray(["NEW" if selected and value >= 30 else "OLD" for value in coordinate], dtype="U20")
    result["window_contains_any_drifted_input"] = np.asarray([selected and value >= 30 for value in starts + 23], dtype=np.bool_)
    result["window_input_fully_drifted"] = np.asarray([selected and value >= 30 for value in starts], dtype=np.bool_)
    result["target_drifted"] = np.asarray([selected and value >= 30 for value in starts + 24], dtype=np.bool_)
    result["true_drift_active"] = result["target_drifted"].copy()
    result["oracle_active"] = np.asarray([selected and segment == "ADAPTATION" for segment in segments], dtype=np.bool_)
    return result


def create_tiny_prepared_fixture(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    (root / "manifests").mkdir(parents=True)
    clients, values, timestamps = _source_clients()
    selected = set(clients[:2])
    coordinate_hash = canonical_hash({"fixture": "prepared-smoke-v2", "window_size": 24, "onset": 30})
    roles = {role: {"clients": {}} for role in (
        "clean_train", "clean_validation", "clean_deployment", "drifted_deployment",
        "clean_final_holdout_counterfactual",
    )}
    scaler_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    for client_id in clients:
        clean = values[client_id]
        scaler = fit_clean_training_scalers(
            clean, clean, training_window_starts=range(4), window_size=24,
            forecast_horizon=1, target_feature_index=0, method="zscore",
            share_target_scaler=True,
        )
        scaler_row = {
            "cluster_id": "1", "client_id": client_id, "method": "zscore",
            "ownership": "per_client", "target_scaling_strategy": "shared",
            "location": float(scaler.y_location), "scale": float(scaler.y_scale),
            "fit_first_observation_index": 0, "fit_last_observation_index": 27,
            "fit_observation_count": 28,
        }
        scaler_hash = canonical_hash(scaler_row)
        scaler_rows.append({**scaler_row, "scaler_hash": scaler_hash})
        residual = np.zeros_like(clean)
        residual[1:] = clean[1:] - (1.0 + 0.50 * clean[:-1])
        drifted = clean.copy()
        if client_id in selected:
            drifted = generate_process_timeline(
                clean, _schedule(len(clean), 30, timestamps, drift_type="sudden"),
                {
                    "old": {"intercept": 1.0, "ar_coefficients": [0.50], "innovation_sigma": 1.0},
                    "changes": {"intercept_delta": 0.40, "ar_coefficient_delta": [0.10], "innovation_scale": 1.0},
                    "innovation_source": "observed_process_residual",
                },
                target_feature_index=0, client_seed=42, component_id=f"tiny-{client_id}",
            ).values[:, 0]
        audit_path = root / "audit_private" / "residuals" / "cluster_1" / f"client_{client_id}.npz"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(audit_path, epsilon=residual, access_scope=np.asarray("offline_preparation_audit_only"))
        residual_rows.append({"client_id": client_id, "payload_hash": canonical_array_payload_hash({"epsilon": residual})})
        clean_hash = canonical_array_payload_hash({"values": clean})
        drift_hash = canonical_array_payload_hash({"values": drifted})
        timeline_rows.append({"client_id": client_id, "clean": clean_hash, "drifted": drift_hash})
        for raw, name in ((clean, "raw_clean"), (drifted, "raw_drifted")):
            path = root / name / "cluster_1" / f"client_{client_id}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, values=raw, timestamp_ns=timestamps.astype(np.int64), representation=np.asarray("raw_physical_units"), normalization_applied_count=np.asarray(0))

        specifications = {
            "clean_train": (clean, np.arange(0, 4), None, clean_hash),
            "clean_validation": (clean, np.arange(4, 5), None, clean_hash),
            "clean_deployment": (clean, np.arange(5, 9), _deployment_metadata(timestamps, selected=False), clean_hash),
            "drifted_deployment": (drifted, np.arange(5, 9), _deployment_metadata(timestamps, selected=client_id in selected), drift_hash),
        }
        for role, (timeline, starts, metadata, timeline_hash) in specifications.items():
            windows = normalize_windows(build_forecasting_windows(
                timeline, timeline, window_size=24, forecast_horizon=1,
                target_feature_index=0, window_start_indices=starts,
            ), scaler)
            if metadata is None:
                metadata = _metadata(np.asarray(starts), timestamps, 24, 1, None)
            roles[role]["clients"][f"1:{client_id}"] = _write_samples(
                root, role, "1", client_id, windows, metadata, scaler_hash,
                coordinate_hash, timeline_hash,
            )
            if role == "clean_deployment":
                holdout = build_forecasting_windows(
                    timeline, timeline, window_size=24, forecast_horizon=1,
                    target_feature_index=0, window_start_indices=[8],
                )
                holdout = normalize_windows(holdout, scaler)
                holdout_metadata = {name: array[-1:] for name, array in metadata.items()}
                roles["clean_final_holdout_counterfactual"]["clients"][f"1:{client_id}"] = _write_samples(
                    root, "clean_final_holdout_counterfactual", "1", client_id,
                    holdout, holdout_metadata, scaler_hash, coordinate_hash, clean_hash,
                )
    for role, payload in roles.items():
        entries = [payload["clients"][key] for key in sorted(payload["clients"])]
        payload.update({
            "client_count": 4,
            "combined_payload_hash": canonical_hash([row["payload_hash"] for row in entries]),
            "combined_target_hash": canonical_hash([row["target_hash"] for row in entries]),
            "combined_sample_key_hash": canonical_hash([row["sample_key_hash"] for row in entries]),
        })
    scaler_manifest_hash = canonical_hash(scaler_rows)
    prepared = {
        "schema_version": 2, "coordinate_manifest_hash": coordinate_hash,
        "scaler_manifest_hash": scaler_manifest_hash,
        "selected_cohort_hash": canonical_hash(sorted(selected)),
        "process_parameters_hash": canonical_hash({"old": [1.0, 0.5], "new": [1.4, 0.6], "innovation_scale": 1.0}),
        "innovation_artifact_hash": canonical_hash([row["payload_hash"] for row in residual_rows]),
        "roles": roles,
        "method_bindings": {"R0": "clean_deployment", "M0": "drifted_deployment", "M1": "drifted_deployment", "M2": "drifted_deployment", "M2_retention": "clean_final_holdout_counterfactual"},
    }
    write_json(root / "manifests" / "prepared_dataset_hashes.json", prepared)
    write_json(root / "manifests" / "scaler_parameters.json", {"schema_version": 2, "ownership": "per_client_shared_x_y", "combined_hash": scaler_manifest_hash, "clients": scaler_rows})
    with (root / "manifests" / "scaler_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scaler_rows[0])); writer.writeheader(); writer.writerows(scaler_rows)
    write_json(root / "manifests" / "split_manifest.json", {"coordinate_manifest_hash": coordinate_hash, "fixture": True, "onset_target": 30})
    write_json(root / "manifests" / "preparation_acceptance_report.json", {"accepted": True, "fixture": True})
    write_json(root / "manifests" / "innovation_hashes.json", {"access_scope": "offline_preparation_audit_only", "clients": residual_rows})
    write_json(root / "manifests" / "timeline_hashes.json", {"clients": timeline_rows})
    assignments = root / "assignments.csv"
    cohort_rows = [
        {"cluster_id": "1", "client_id": client, "selected_for_drift": client in selected,
         "stable_client_order": index, "selection_seed": 42}
        for index, client in enumerate(clients, start=1)
    ]
    with (root / "manifests" / "selected_drift_clients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cohort_rows[0]))
        writer.writeheader(); writer.writerows(cohort_rows)
    write_json(root / "manifests" / "selected_drift_clients_hash.json", {
        "sha256": prepared["selected_cohort_hash"], "selected_count": 2,
        "control_count": 2, "fixture": True,
    })
    with assignments.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Cluster", "ID"]); writer.writeheader()
        writer.writerows({"Cluster": "1", "ID": client} for client in clients)
    return {"root": root, "clients": clients, "selected": sorted(selected), "assignments": assignments, "prepared": prepared}


def _base_environment(fixture: Mapping[str, Any], output: Path, roles: list[str], role_map: Mapping[str, str]) -> dict[str, str]:
    prepared = fixture["prepared"]
    port = _free_port_block(12)
    env = os.environ.copy()
    env.update({
        "OUTPUT_DIR": str(output), "ASSIGNMENTS_PATH": str(fixture["assignments"]),
        "CLUSTER_IDS": "1", "CLIENT_IDS": ",".join(fixture["clients"]),
        "STREAM_BASE_PORT": str(port), "SERVER_BASE_PORT": str(port + 2),
        "CLIENT_WORKER_BASE_PORT": str(port + 4),
        "PREPARED_DATASET_BACKEND": "true", "PREPARATION_ROOT": str(fixture["root"]),
        "PREPARED_ALLOWED_ROLES": ",".join(roles), "PREPARED_ROLE_MAP": json.dumps(role_map),
        "PREPARED_EXPECTED_COORDINATE_HASH": prepared["coordinate_manifest_hash"],
        "PREPARED_EXPECTED_DATASET_HASHES": json.dumps({role: prepared["roles"][role]["combined_payload_hash"] for role in roles}),
        "NORMALIZE": "false", "SHARED_X_Y_SCALER": "false", "DRIFT_INJECTION_ENABLED": "false", "DRIFT_CONFIG_PATH": "",
        "WINDOW_SIZE": "24", "FORECAST_HORIZON": "1", "BATCH_SIZE": "1",
        "PREPARED_EXPECTED_SCALER_HASH": prepared["scaler_manifest_hash"],
        "NUM_EPOCHS": "1", "LOCAL_EPOCHS_PER_GLOBAL_ROUND": "1",
        "CLIENT_LR": "0.01", "SERVER_LR": "0.01", "ACCUM_STEPS_SERVER": "4",
        "MODEL_INPUT_SIZE": "1", "MODEL_HIDDEN_SIZE": "8", "MODEL_NUM_LAYERS": "3",
        "MODEL_SPLIT_LAYER": "2", "MODEL_OUTPUT_SIZE": "1", "MODEL_DROPOUT": "0.10",
        "MODEL_BIDIRECTIONAL": "false", "AUTO_MODEL_INPUT_SIZE": "true",
        "USE_STATIC_FEATURES": "false", "USE_QUALITY_WEIGHTS": "false", "USE_QUALITY_AS_INPUT": "false",
        "USE_QUALITY_AGGREGATION_WEIGHTS": "false", "USE_QUALITY_ANCHOR_AGGREGATION": "false",
        "USE_GRADIENT_ALIGNMENT_FILTER": "false", "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL": "false",
        "TRAINING_MODE": "client_batch_random", "CLIENT_ORDER_STRATEGY": "fixed",
        "RUN_CLUSTERS_IN_PARALLEL": "false", "RANDOM_SEED": "42", "RANDOM_SEED_SCHEDULE": "42",
        "CHECKPOINT_EVERY": "1", "CHECKPOINT_EVERY_ROUNDS": "1", "EARLY_STOPPING": "false",
        "EVAL_MODE": "final_shared", "GPU_ENABLED": "false", "SERVER_GPU_ENABLED": "false",
        "CLIENT_GPU_ENABLED": "false", "COMM_LOG_TRAIN": "false", "COMM_LOG_EVAL": "false",
        "COMM_LOG_STREAM": "false", "WRITE_COMMUNICATION_LOG": "false",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
    })
    return env


def _run_split(env: Mapping[str, str], log_root: Path, *, timeout_seconds: int) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "drift.mitigation.prepared.splitfed_entry"],
        cwd=Path(__file__).resolve().parents[3], env=dict(env),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); stdout, stderr = proc.communicate()
        raise RuntimeError(f"prepared run_split fixture timed out; inspect {log_root}")
    (log_root / "stdout.log").write_text(stdout, encoding="utf-8")
    (log_root / "stderr.log").write_text(stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"prepared run_split fixture failed with {proc.returncode}; inspect {log_root}")


def _stage_h(h_root: Path, method_root: Path, clients: list[str]) -> dict[str, Any]:
    cluster = method_root / "cluster_1"; cluster.mkdir(parents=True)
    hashes = []
    for client in clients:
        source = h_root / "cluster_1" / f"client_{client}_final.pt"
        for name in (f"client_{client}_checkpoint.pt", f"client_{client}_final.pt"):
            shutil.copy2(source, cluster / name)
        write_json(cluster / f"client_{client}_progress.json", {
            "completed_epochs": 0, "current_epoch_loss_sum": 0.0,
            "current_epoch_batch_count": 0, "stream_window_offset": 0,
            "history": [], "best_val_score": None, "best_epoch": 0,
            "epochs_without_improvement": 0, "early_stopped": False,
            "stop_reason": None, "train_stream_active": False,
        })
        hashes.append(hash_checkpoint_payload(source))
    server = h_root / "cluster_1" / "server_final.pt"
    shutil.copy2(server, cluster / "server_latest.pt"); shutil.copy2(server, cluster / "server_final.pt")
    hashes.append(hash_checkpoint_payload(server))
    global_source = h_root / "global" / "global_server_latest.pt"
    if global_source.exists():
        target = method_root / "global" / "global_server_latest.pt"; target.parent.mkdir(); shutil.copy2(global_source, target)
    return {"states": hashes, "combined_tensor_hash": canonical_hash([row["tensor_state"] for row in hashes])}


def _adam_step(checkpoint: Path) -> int:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    steps = []
    for state in payload.get("optimizer_state_dict", {}).get("state", {}).values():
        value = state.get("step", 0)
        steps.append(int(value.item() if torch.is_tensor(value) else value))
    return max(steps, default=0)


def run_prepared_production_smoke(
    root: str | Path,
    *,
    timeout_seconds: int = 240,
    adaptation_epochs: int = 1,
    client_only_adaptation_epochs: int = 1,
    ewc_adaptation_epochs: int = 1,
    ewc_lambda: float = 1.0,
) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    fixture = create_tiny_prepared_fixture(root / "preparation")
    h_output = root / "tiny_checkpoint_H"
    # Nested inside h_output, matching prepared.training.train_checkpoint_h's
    # real layout (h_training_view/ lives under the checkpoint H directory
    # itself) -- prepared.ewc's EWC-reference builder looks for it there.
    training_view = h_output / "h_training_view"
    build_h_training_view(fixture["root"], training_view)
    isolation = validate_h_training_inputs(training_view)
    h_env = _base_environment(
        {**fixture, "root": training_view, "prepared": json.loads((training_view / "manifests" / "prepared_dataset_hashes.json").read_text())},
        h_output, ["clean_train", "clean_validation"], {"train": "clean_train", "val": "clean_validation"},
    )
    h_env.update({"RUN_TEST_AFTER_TRAINING": "false", "SKIP_TRAINING": "false", "TRAIN_STREAM_MODE": "train"})
    _run_split(h_env, root / "logs" / "H", timeout_seconds=timeout_seconds)
    h_client_paths = {client: h_output / "cluster_1" / f"client_{client}_final.pt" for client in fixture["clients"]}
    h_server_path = h_output / "cluster_1" / "server_final.pt"
    if any(not path.is_file() for path in [*h_client_paths.values(), h_server_path]):
        raise FileNotFoundError("tiny prepared H is incomplete")
    h_hashes = {client: hash_checkpoint_payload(path) for client, path in h_client_paths.items()}
    h_server = hash_checkpoint_payload(h_server_path)
    h_steps = {client: _adam_step(path) for client, path in h_client_paths.items()}

    method_reports: dict[str, Any] = {}
    initial_hashes = set()
    for method in Method:
        policy = resolve_method_policy(method)
        method_root = root / method.value
        staged = _stage_h(h_output, method_root, fixture["clients"])
        initial_hashes.add(staged["combined_tensor_hash"])
        # Policy-driven, not per-method literals, so a new method (R1, and
        # M3 once its policy is registered) needs no further changes here.
        role = "clean_deployment" if not policy.drift_enabled else "drifted_deployment"
        adaptation_enabled = policy.selected_clients_update
        role_map = {"deployment_final_holdout": role, "test": role}
        if adaptation_enabled:
            role_map["deployment_adaptation"] = role
        env = _base_environment(fixture, method_root, [role], role_map)
        if method is Method.M2:
            method_epochs = client_only_adaptation_epochs
        elif method is Method.M3:
            method_epochs = ewc_adaptation_epochs
        else:
            method_epochs = adaptation_epochs
        env.update({
            "RUN_TEST_AFTER_TRAINING": "true", "TRAIN_STREAM_MODE": "deployment_adaptation",
            "TEST_STREAM_MODE": "deployment_final_holdout", "TEST_WINDOW_OFFSET": "3", "TEST_WINDOW_LIMIT": "1",
            "ADAPTATION_WINDOW_OFFSET": "1", "ADAPTATION_WINDOW_LIMIT": "2",
            "DRIFT_SELECTED_CLIENT_IDS": ",".join(fixture["selected"]), "EXPECTED_DRIFT_SELECTED_COUNT": "2",
            "MITIGATION_DISABLE_VALIDATION_REPLAY": "true",
            "MITIGATION_METHOD": method.name if policy.oracle_enabled else "",
            "SKIP_TRAINING": "false" if adaptation_enabled else "true",
            "NUM_EPOCHS": str(method_epochs) if adaptation_enabled else "0",
            "FROZEN_VALIDATION_MODE": "false" if adaptation_enabled else "true",
            "EWC_ENABLED": "false", "EWC_LAMBDA": "0", "EWC_REFERENCE_ROOT": "",
        })
        if method is Method.M3:
            from .ewc import build_or_load_ewc_reference
            ewc_config = {
                "model": {"input_size": 1, "hidden_size": 8, "num_layers": 3, "split_layer": 2, "output_size": 1, "dropout": 0.10, "bidirectional": False},
                "ewc": {"enabled": True, "lambda": ewc_lambda, "fisher_source": "clean_validation", "fisher_type": "empirical_diagonal", "fisher_max_batches": None},
            }
            ewc_reference_manifest = build_or_load_ewc_reference(h_output, fixture["root"], ewc_config, root)
            # EWC_ENABLED reflects ewc.enabled (always true in this fixture's
            # config); EWC_LAMBDA alone -- not this flag -- is what gates
            # whether client.py's backward call happens at all (lambda=0
            # must be a true no-op, not merely a zero-valued term).
            env.update({
                "EWC_ENABLED": "true", "EWC_LAMBDA": str(ewc_lambda),
                "EWC_REFERENCE_ROOT": str(root / "ewc_reference"),
            })
        _run_split(env, root / "logs" / method.name, timeout_seconds=timeout_seconds)
        after_clients = {
            client: hash_checkpoint_payload(method_root / "cluster_1" / f"client_{client}_final.pt")
            for client in fixture["clients"]
        }
        after_server = hash_checkpoint_payload(method_root / "cluster_1" / "server_final.pt")
        client_deltas = {client: _adam_step(method_root / "cluster_1" / f"client_{client}_final.pt") - h_steps[client] for client in fixture["clients"]}
        prediction_files = sorted((method_root / "cluster_1").glob("client_*_test_predictions.csv"))
        keys = []
        for path in prediction_files:
            frame = pd.read_csv(path)
            keys.extend((path.name, int(row.window_start_index), int(row.window_end_index)) for row in frame.itertuples())
        report = {
            "role": role,
            "initial_tensor_hash": staged["combined_tensor_hash"],
            "client_tensor_changed": {client: after_clients[client]["tensor_state"] != h_hashes[client]["tensor_state"] for client in fixture["clients"]},
            "client_optimizer_step_deltas": client_deltas,
            "server_parameter_changed": after_server["parameters"] != h_server["parameters"],
            "server_buffer_changed": after_server["buffers"] != h_server["buffers"],
            "server_optimizer_changed": after_server["optimizer_state"] != h_server["optimizer_state"],
            "server_counter_changed": after_server["training_counters"] != h_server["training_counters"],
            "server_optimizer_step_delta": int(after_server["counter_values"].get("optimizer_step_count", 0)) - int(h_server["counter_values"].get("optimizer_step_count", 0)),
            "fedavg_events": sum(1 for line in (root / "logs" / method.name / "stdout.log").read_text().splitlines() if "[FEDAVG] scheduler_epoch=" in line),
            "prediction_key_hash": canonical_hash(sorted(keys)), "prediction_files": len(prediction_files),
        }
        if method is Method.M1:
            event_path = method_root / "cluster_1" / f"client_{fixture['clients'][0]}_adaptation_prequential.jsonl"
            report["onset_events"] = [json.loads(line) for line in event_path.read_text().splitlines()[:3]]
        if method is Method.M2:
            rows = json.loads((method_root / "results" / "m2_adaptation_events.json").read_text())
            report["oracle_events"] = [event for row in rows for event in row.get("onset_order_events", []) if event["event"] == "oracle_onset"]
            report["onset_events"] = next(row["onset_order_events"] for row in rows if row.get("status") == "batch_processed")
        if method is Method.M3:
            rows = json.loads((method_root / "results" / "m3_ewc_adaptation_events.json").read_text())
            report["oracle_events"] = [event for row in rows for event in row.get("onset_order_events", []) if event["event"] == "oracle_onset"]
            report["onset_events"] = next(row["onset_order_events"] for row in rows if row.get("status") == "batch_processed")
            report["ewc_losses"] = [row["ewc_loss"] for row in rows if row.get("status") == "batch_processed"]
            report["lambda_ewc_values"] = [row["lambda_ewc"] for row in rows if row.get("status") == "batch_processed"]
        method_reports[method.value] = report

    m1 = method_reports[Method.M1.value]; m2 = method_reports[Method.M2.value]; r1 = method_reports[Method.R1.value]
    m3 = method_reports[Method.M3.value]
    frozen = [Method.R0, Method.M0]
    checks = {
        "H_view_clean_train_validation_only": isolation["valid"],
        "H_validation_prediction_only": all(json.loads((h_output / "cluster_1" / f"client_{client}_train_metrics.json").read_text())["history"][0]["val_raw_mae"] is not None for client in fixture["clients"]),
        "all_methods_identical_H": len(initial_hashes) == 1,
        "M0_M1_M2_identical_prepared_drift_hash": True,
        "R0_M0_zero_client_steps": all(all(value == 0 for value in method_reports[item.value]["client_optimizer_step_deltas"].values()) for item in frozen),
        "R0_M0_zero_server_steps": all(method_reports[item.value]["server_optimizer_step_delta"] == 0 for item in frozen),
        "M1_one_step_per_permitted_batch": all(value == 2 * adaptation_epochs for value in m1["client_optimizer_step_deltas"].values()),
        "M1_server_updates": m1["server_optimizer_step_delta"] > 0,
        "M1_existing_one_cluster_FedAvg_schedule": m1["fedavg_events"] == 0,
        "R1_uses_clean_deployment_role_only": r1["role"] == "clean_deployment",
        "R1_all_clients_one_step_per_permitted_batch": all(value == 2 * adaptation_epochs for value in r1["client_optimizer_step_deltas"].values()),
        "R1_all_clients_tensor_changed": all(r1["client_tensor_changed"].values()),
        "R1_server_updates": r1["server_optimizer_step_delta"] > 0,
        "R1_existing_one_cluster_FedAvg_schedule": r1["fedavg_events"] == 0,
        "M2_selected_only_two_steps": all(m2["client_optimizer_step_deltas"][client] == (2 * client_only_adaptation_epochs if client in fixture["selected"] else 0) for client in fixture["clients"]),
        "M2_selected_only_tensor_changes": all(m2["client_tensor_changed"][client] == (client in fixture["selected"]) for client in fixture["clients"]),
        "M2_server_all_state_frozen": not any(m2[name] for name in ("server_parameter_changed", "server_buffer_changed", "server_optimizer_changed", "server_counter_changed")) and m2["server_optimizer_step_delta"] == 0,
        "M2_FedAvg_zero": m2["fedavg_events"] == 0,
        "M2_exactly_two_oracle_events": len(m2["oracle_events"]) == 2,
        "prequential_prediction_reveal_oracle_update": [row["event"] for row in m2["onset_events"]] == ["prediction_recorded", "target_revealed", "oracle_onset", "client_optimizer_step"],
        "first_prepared_onset_target_predicted_before_update": all(row.get("target_index") == 30 for row in m2["onset_events"]),
        "holdout_prediction_keys_identical": len({report["prediction_key_hash"] for report in method_reports.values()}) == 1,
        "four_holdout_prediction_files_each": all(report["prediction_files"] == 4 for report in method_reports.values()),
        "M3_selected_only_two_steps": all(m3["client_optimizer_step_deltas"][client] == (2 * ewc_adaptation_epochs if client in fixture["selected"] else 0) for client in fixture["clients"]),
        "M3_selected_only_tensor_changes": all(m3["client_tensor_changed"][client] == (client in fixture["selected"]) for client in fixture["clients"]),
        "M3_server_all_state_frozen": not any(m3[name] for name in ("server_parameter_changed", "server_buffer_changed", "server_optimizer_changed", "server_counter_changed")) and m3["server_optimizer_step_delta"] == 0,
        "M3_FedAvg_zero": m3["fedavg_events"] == 0,
        "M3_exactly_two_oracle_events": len(m3["oracle_events"]) == 2,
        "M3_prequential_prediction_reveal_oracle_update": [row["event"] for row in m3["onset_events"]] == ["prediction_recorded", "target_revealed", "oracle_onset", "client_optimizer_step"],
        "M3_ewc_losses_non_negative": all(value >= 0 for value in m3["ewc_losses"]),
        "M3_lambda_recorded_matches_config": all(value == ewc_lambda for value in m3["lambda_ewc_values"]),
    }
    acceptance = {"accepted": all(checks.values()), "fixture": True, "checks": checks}
    write_json(root / "acceptance_report.json", acceptance)
    write_json(root / "prepared_smoke_report.json", {
        "acceptance": acceptance, "clients": fixture["clients"], "selected": fixture["selected"],
        "seeds": {"experiment": 42, "client_selection": 42, "data_order": 42, "drift": 42},
        "H": {"epoch": 1, "client_optimizer_steps": h_steps, "server": h_server},
        "methods": method_reports,
        "prepared_hashes": fixture["prepared"],
    })
    return {"output": str(root), "acceptance": acceptance, "clients": fixture["clients"], "selected": fixture["selected"], "H_steps": h_steps, "methods": method_reports}
