"""Isolated real run_split.py checkpoint-H integration fixture."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.split_lstm import LSTMConfig
from drift.windowing import fit_clean_training_scalers

from ..core import Method, canonical_hash, file_sha256
from ..testing_support import _free_port_block
from .smoke import _state_snapshot, prepare_smoke_data, run_method


def _write_fixture_data(root: Path, clients: list[str], length: int) -> Path:
    source = Path(__file__).resolve().parents[3] / "dataset" / "output_clean" / "cluster_1.pkl"
    payload = pd.read_pickle(source)
    frame = payload["timeseries"].copy()
    frame["id"] = frame["id"].astype(str)
    timestamp_column = str(payload["timestamp_col"])
    pieces = []
    for client in clients:
        rows = frame.loc[frame["id"] == client].sort_values(timestamp_column).iloc[:length].copy()
        if len(rows) != length:
            raise ValueError(f"source client {client} has insufficient observations")
        rows["id"] = rows["id"].astype(int)
        pieces.append(rows)
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    fixture = {
        "timeseries": pd.concat(pieces, ignore_index=True),
        "client_metadata": None,
        "timestamp_col": payload["timestamp_col"],
        "value_col": payload["value_col"],
        "static_feature_columns": [],
        "input_feature_columns": [payload["value_col"]],
    }
    pd.to_pickle(fixture, data_dir / "cluster_1.pkl")
    return source


def run_tiny_checkpoint_H_fixture(root: str | Path, *, timeout_seconds: int = 240) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"tiny checkpoint fixture output already exists: {root}")
    root.mkdir(parents=True)
    prepared, metadata = prepare_smoke_data()
    clients = list(metadata["clients"])
    source_path = _write_fixture_data(root, clients, 28)
    full_payload = pd.read_pickle(source_path)
    full_frame = full_payload["timeseries"].copy()
    full_frame["id"] = full_frame["id"].astype(str)
    full_client_rows = full_frame.loc[full_frame["id"] == clients[0]].sort_values(str(full_payload["timestamp_col"]))
    full_values = full_client_rows[str(full_payload["value_col"])].to_numpy(dtype=np.float64, copy=True)
    full_total_windows = len(full_values) - 24
    full_train_windows = int(full_total_windows * 0.70)
    full_scaler = fit_clean_training_scalers(
        full_values,
        full_values,
        training_window_starts=range(full_train_windows),
        window_size=24,
        forecast_horizon=1,
        target_feature_index=0,
        method="zscore",
        share_target_scaler=True,
    )
    full_coordinates = list(full_scaler.shared_target_indices)
    full_scaler_proof = {
        "source_observations": len(full_values),
        "total_windows": full_total_windows,
        "train_windows": full_train_windows,
        "first_coordinate": full_coordinates[0],
        "last_coordinate": full_coordinates[-1],
        "coordinate_count": len(full_coordinates),
        "coordinate_6139_excluded": 6139 not in set(full_coordinates),
        "target_scaling_strategy": full_scaler.target_scaling_strategy,
        "x_y_location_equal": float(full_scaler.x_location[0]) == float(full_scaler.y_location),
        "x_y_scale_equal": float(full_scaler.x_scale[0]) == float(full_scaler.y_scale),
        "shared_target_union_hash": full_scaler.shared_target_union_hash,
    }
    if full_scaler_proof["last_coordinate"] != 6138 or not full_scaler_proof["coordinate_6139_excluded"]:
        raise AssertionError(f"full checkpoint-H scaler coordinates are incompatible: {full_scaler_proof}")
    assignments = root / "assignments.csv"
    with assignments.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Cluster", "ID"])
        writer.writeheader()
        writer.writerows({"Cluster": "1", "ID": client} for client in clients)
    drift_config = root / "clean_schema_v2.json"
    drift_config.write_text(json.dumps({
        "schema_version": 2,
        "experiment_id": "tiny-clean-checkpoint-H-seed42",
        "enabled": False,
        "experiment_seed": 42,
        "clients": {"ids": []},
        "modes": ["test"],
        "temporal": {"type": "none"},
        "source": {"type": "process"},
        "windowing": {"window_size": 4, "forecast_horizon": 1, "target_feature_index": 0},
    }, indent=2) + "\n", encoding="utf-8")
    output = root / "run_split_output"
    port_block = _free_port_block(12)
    stream_port = port_block
    server_port = port_block + 2
    client_port = port_block + 4
    repo = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update({
        "OUTPUT_DIR": str(output),
        "ASSIGNMENTS_PATH": str(assignments),
        "CLUSTER_IDS": "1",
        "CLIENT_IDS": ",".join(clients),
        "CLUSTER_DATA_DIR": str(root / "data"),
        "TRAIN_CLUSTER_DATA_DIR": str(root / "data"),
        "VAL_CLUSTER_DATA_DIR": str(root / "data"),
        "TEST_CLUSTER_DATA_DIR": str(root / "data"),
        "TRAIN_TARGET_CLUSTER_DATA_DIR": str(root / "data"),
        "VAL_TARGET_CLUSTER_DATA_DIR": str(root / "data"),
        "TEST_TARGET_CLUSTER_DATA_DIR": str(root / "data"),
        "DRIFT_CONFIG_PATH": str(drift_config),
        "DRIFT_INJECTION_ENABLED": "false",
        "DRIFT_DETECTION_ENABLED": "false",
        "STREAM_BASE_PORT": str(stream_port),
        "SERVER_BASE_PORT": str(server_port),
        "CLIENT_WORKER_BASE_PORT": str(client_port),
        "WINDOW_SIZE": "4",
        "FORECAST_HORIZON": "1",
        "TRAIN_RATIO": "0.4166666666667",
        "VAL_RATIO": "0.1666666666666",
        "TEST_RATIO": "0.4166666666667",
        "NORMALIZE": "true",
        "NORMALIZE_METHOD": "zscore",
        "BATCH_SIZE": "32",
        "NUM_EPOCHS": "1",
        "LOCAL_EPOCHS_PER_GLOBAL_ROUND": "1",
        "CLIENT_LR": "0.01",
        "SERVER_LR": "0.01",
        "ACCUM_STEPS_SERVER": "4",
        "MODEL_INPUT_SIZE": "1",
        "MODEL_HIDDEN_SIZE": "8",
        "MODEL_NUM_LAYERS": "3",
        "MODEL_SPLIT_LAYER": "2",
        "MODEL_OUTPUT_SIZE": "1",
        "MODEL_DROPOUT": "0.10",
        "MODEL_BIDIRECTIONAL": "false",
        "AUTO_MODEL_INPUT_SIZE": "true",
        "USE_STATIC_FEATURES": "false",
        "USE_QUALITY_WEIGHTS": "false",
        "USE_QUALITY_AS_INPUT": "false",
        "USE_QUALITY_AGGREGATION_WEIGHTS": "false",
        "USE_QUALITY_ANCHOR_AGGREGATION": "false",
        "USE_GRADIENT_ALIGNMENT_FILTER": "false",
        "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL": "false",
        "TRAINING_MODE": "client_batch_random",
        "CLIENT_ORDER_STRATEGY": "fixed",
        "RUN_CLUSTERS_IN_PARALLEL": "false",
        "RANDOM_SEED": "42",
        "RANDOM_SEED_SCHEDULE": "42",
        "CHECKPOINT_EVERY": "1",
        "CHECKPOINT_EVERY_ROUNDS": "1",
        "EARLY_STOPPING": "false",
        "EVAL_MODE": "final_shared",
        "GPU_ENABLED": "false",
        "SERVER_GPU_ENABLED": "false",
        "CLIENT_GPU_ENABLED": "false",
        "COMM_LOG_TRAIN": "false",
        "COMM_LOG_EVAL": "false",
        "COMM_LOG_STREAM": "false",
        "WRITE_COMMUNICATION_LOG": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    command = [sys.executable, "run_split.py"]
    proc = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        raise RuntimeError(f"tiny run_split fixture timed out after {timeout_seconds}s")
    (root / "stdout.log").write_text(stdout, encoding="utf-8")
    (root / "stderr.log").write_text(stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"tiny run_split fixture failed with exit {proc.returncode}; see {root / 'stdout.log'}")

    cluster_output = output / "cluster_1"
    client_paths = {client: cluster_output / f"client_{client}_final.pt" for client in clients}
    server_path = cluster_output / "server_final.pt"
    missing = [str(path) for path in [*client_paths.values(), server_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"tiny H did not produce checkpoints: {missing}")
    client_payloads = {client: torch.load(path, map_location="cpu") for client, path in client_paths.items()}
    server_payload = torch.load(server_path, map_location="cpu")
    client_states = {client: deepcopy(payload["model_state_dict"]) for client, payload in client_payloads.items()}
    client_optimizer_states = {client: deepcopy(payload["optimizer_state_dict"]) for client, payload in client_payloads.items()}
    server_state = deepcopy(server_payload["model_state_dict"])
    server_optimizer_state = deepcopy(server_payload["optimizer_state_dict"])
    cfg = LSTMConfig(input_size=1, hidden_size=8, num_layers=3, dropout=0.10, output_size=1, learning_rate=0.01)
    # This generation-1 pipeline predates R1/M3 and has no bespoke handling
    # for them (e.g. no EWC dispatch); it stays closed to the original four
    # methods rather than picking up new Method members automatically.
    method_results = {
        method: run_method(
            method,
            prepared,
            metadata,
            client_states,
            server_state,
            cfg,
            initial_client_optimizer_states=client_optimizer_states,
            initial_server_optimizer_state=server_optimizer_state,
        )
        for method in (Method.R0, Method.M0, Method.M1, Method.M2)
    }
    initial_hashes = {method.value: canonical_hash(result["before"]) for method, result in method_results.items()}
    method_checks: dict[str, dict[str, Any]] = {}
    for method, result in method_results.items():
        client_parameter_changes = {
            client: result["before"]["clients"][client]["parameters"] != result["after"]["clients"][client]["parameters"]
            for client in clients
        }
        method_checks[method.value] = {
            "loaded_H": True,
            "client_parameter_changes": client_parameter_changes,
            "server_parameters_changed": result["before"]["server"]["parameters"] != result["after"]["server"]["parameters"],
            "server_optimizer_changed": result["before"]["server"]["optimizer"] != result["after"]["server"]["optimizer"],
            "oracle_events": len(result["oracle_events"]),
            "client_optimizer_steps": {
                client: int(result["state"].client_counters[client]["optimizer_step_count"])
                for client in clients
            },
            "server_optimizer_steps": int(result["state"].server_counters["optimizer_step_count"]),
        }
    checkpoint_h_adam_steps = {
        client: sorted({
            int(value["step"].item() if torch.is_tensor(value["step"]) else value["step"])
            for value in optimizer_state["state"].values()
            if "step" in value
        })
        for client, optimizer_state in client_optimizer_states.items()
    }
    normalization_path = cluster_output / "drift" / "normalization_manifest.json"
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))["clients"]
    expected_fit = list(range(14))
    normalization_ok = all(
        row["target_scaling_strategy"] == "shared"
        and row["shared_target_indices"] == expected_fit
        and row["x_location"][0] == row["y_location"]
        and row["x_scale"][0] == row["y_scale"]
        for row in normalization
    )
    selected = set(metadata["selected"])
    m2_changes = method_checks[Method.M2.value]["client_parameter_changes"]
    accepted = (
        len(set(initial_hashes.values())) == 1
        and normalization_ok
        and all(not value for value in method_checks[Method.R0.value]["client_parameter_changes"].values())
        and all(not value for value in method_checks[Method.M0.value]["client_parameter_changes"].values())
        and all(m2_changes[client] == (client in selected) for client in clients)
        and not method_checks[Method.M2.value]["server_parameters_changed"]
        and not method_checks[Method.M2.value]["server_optimizer_changed"]
        and all(steps == [1] for steps in checkpoint_h_adam_steps.values())
        and all(method_checks[Method.M1.value]["client_optimizer_steps"][client] == 3 for client in clients)
        and all(
            method_checks[Method.M2.value]["client_optimizer_steps"][client]
            == (3 if client in selected else 0)
            for client in clients
        )
        and method_checks[Method.M2.value]["server_optimizer_steps"] == 0
    )
    report = {
        "accepted": accepted,
        "command": command,
        "returncode": proc.returncode,
        "clients": clients,
        "selected_clients": sorted(selected),
        "seeds": {"experiment": 42, "schedule": 42, "drift": 42, "client_selection": 42},
        "checkpoint_paths": {**{client: str(path) for client, path in client_paths.items()}, "server": str(server_path)},
        "checkpoint_artifact_hashes": {**{client: file_sha256(path) for client, path in client_paths.items()}, "server": file_sha256(server_path)},
        "initial_method_state_hashes": initial_hashes,
        "all_methods_loaded_identical_H": len(set(initial_hashes.values())) == 1,
        "method_checks": method_checks,
        "normalization_manifest": str(normalization_path),
        "checkpoint_H_client_adam_steps": checkpoint_h_adam_steps,
        "checkpoint_H_one_optimizer_step_per_client_batch": all(
            steps == [1]
            for steps in checkpoint_h_adam_steps.values()
        ),
        "normalization_manifest_hash": file_sha256(normalization_path),
        "normalization_fit_indices": expected_fit,
        "normalization_contract_ok": normalization_ok,
        "fixture_geometry": {"observations": 28, "windows": 24, "train_windows": 10, "val_windows": 4, "test_windows": 10},
        "full_H_clean_training_shared_scaler_proof": full_scaler_proof,
        "full_H_data_use_contract": {
            "training_window_numbers_zero_based": [0, 6114],
            "validation_window_numbers_zero_based": [6115, 6987],
            "deployment_test_window_numbers_zero_based": [6988, 8735],
            "checkpoint_selection": "validation_only via ClusteredClientRunner._finalize_epoch",
            "deployment_usage_during_H": "prediction-only test after final checkpoint save",
            "adaptation_or_final_holdout_training": False,
        },
        "source_dataset": str(source_path),
        "source_dataset_hash": file_sha256(source_path),
    }
    (root / "checkpoint_H_acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
