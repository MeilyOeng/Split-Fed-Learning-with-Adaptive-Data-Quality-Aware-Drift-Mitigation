"""Safety-gated production method launcher built around the existing run_split lifecycle."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

from ..checkpoint_registry import hash_checkpoint_payload
from ..core import Method, SUITE_ID, canonical_hash, write_json

FULL_RUN_AUTHORIZATION = "FULL_84_CLIENT_RUN_APPROVED"

def _require_authorization(value: str) -> None:
    if value != FULL_RUN_AUTHORIZATION:
        raise PermissionError("84-client execution requires --authorization FULL_84_CLIENT_RUN_APPROVED")

def load_prepared_suite(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    report_path = root / "preflight_acceptance_report.json"
    if not report_path.is_file(): raise ValueError("production suite has not passed preflight")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("accepted") or not report.get("preflight_only"): raise ValueError("production preflight acceptance is false")
    return {"root": root, "acceptance": report,
            "suite": json.loads((root / "suite_manifest.json").read_text(encoding="utf-8")),
            "checkpoint_H": json.loads((root / "checkpoint_H_manifest.json").read_text(encoding="utf-8"))}

def _activate_staged_h(method: Method, prepared: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(prepared["root"]); method_root = root / method.value
    initial = method_root / "checkpoints" / "initial"
    if not initial.is_dir(): raise ValueError(f"missing staged H for {method.value}")
    if (method_root / "run_manifest.json").exists(): raise FileExistsError(f"method already started: {method_root}")
    rows = []
    for cluster in range(1, 5):
        source_cluster, target_cluster = initial / f"cluster_{cluster}", method_root / f"cluster_{cluster}"
        target_cluster.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_cluster.glob("client_*_final.pt")):
            client_id = source.name.split("_")[1]
            for name in (f"client_{client_id}_checkpoint.pt", f"client_{client_id}_final.pt"):
                destination = target_cluster / name
                shutil.copy2(source, destination)
                rows.append(hash_checkpoint_payload(destination))
            write_json(target_cluster / f"client_{client_id}_progress.json", {
                "completed_epochs": 0, "current_epoch_loss_sum": 0.0, "current_epoch_batch_count": 0,
                "stream_window_offset": 0, "history": [], "best_epoch": 0, "best_val_score": None,
                "epochs_without_improvement": 0, "early_stopped": False, "stop_reason": None,
                "train_stream_active": False, "deployment_source_epoch": 130,
            })
        server = source_cluster / "server_final.pt"
        for name in ("server_latest.pt", "server_final.pt"):
            destination = target_cluster / name
            shutil.copy2(server, destination)
            rows.append(hash_checkpoint_payload(destination))
    global_source = initial / "global" / "global_server_latest.pt"
    global_target = method_root / "global" / "global_server_latest.pt"
    global_target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(global_source, global_target)
    rows.append(hash_checkpoint_payload(global_target))
    return {"activated_checkpoint_count": len(rows), "active_tensor_state_hash": canonical_hash([row["tensor_state"] for row in rows]), "checkpoints": rows}

def _cohort(root: Path) -> tuple[dict[str, list[str]], list[str]]:
    clusters = {str(cluster): [] for cluster in range(1, 5)}; selected = []
    with (root / "selected_drift_clients.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clusters[row["cluster_id"]].append(row["client_id"])
            if str(row["selected_for_drift"]).lower() in {"true", "1"}: selected.append(row["client_id"])
    for clients in clusters.values(): clients.sort()
    if any(len(clients) != 21 for clients in clusters.values()) or len(selected) != 32:
        raise ValueError("prepared cohort no longer has 21 clients/cluster and 32 selected")
    return clusters, selected

def build_method_environment(method: Method, prepared: Mapping[str, Any]) -> dict[str, str]:
    method = Method(method); root = Path(prepared["root"]); method_root = root / method.value
    timeline = root / "common" / ("clean_deployment_timeline" if method is Method.R0 else "drifted_deployment_timeline")
    _, selected = _cohort(root)
    env = os.environ.copy()
    env.update({
        "OUTPUT_DIR": str(method_root), "ASSIGNMENTS_PATH": str(Path("dataset/cluster-assignment.csv").resolve()),
        "CLUSTER_DATA_DIR": str(timeline), "TRAIN_CLUSTER_DATA_DIR": str(timeline), "VAL_CLUSTER_DATA_DIR": str(timeline), "TEST_CLUSTER_DATA_DIR": str(timeline),
        "TRAIN_TARGET_CLUSTER_DATA_DIR": str(timeline), "VAL_TARGET_CLUSTER_DATA_DIR": str(timeline), "TEST_TARGET_CLUSTER_DATA_DIR": str(timeline),
        "MODEL_INPUT_SIZE": "1", "MODEL_HIDDEN_SIZE": "128", "MODEL_NUM_LAYERS": "3", "MODEL_SPLIT_LAYER": "2", "MODEL_OUTPUT_SIZE": "1", "MODEL_DROPOUT": "0.4",
        "WINDOW_SIZE": "24", "FORECAST_HORIZON": "1", "NORMALIZE": "true", "NORMALIZE_METHOD": "zscore", "SHARED_X_Y_SCALER": "true",
        "BATCH_SIZE": "128", "NUM_EPOCHS": "1", "LOCAL_EPOCHS_PER_GLOBAL_ROUND": "1",
        "TRAINING_MODE": "client_batch_random", "CLIENT_ORDER_STRATEGY": "random_per_round",
        "RANDOM_SEED": "42", "RANDOM_SEED_SCHEDULE": "42", "EARLY_STOPPING": "false",
        "TRAIN_STREAM_MODE": "deployment_adaptation", "TEST_STREAM_MODE": "deployment_final_holdout",
        "TEST_WINDOW_OFFSET": "1224", "TEST_WINDOW_LIMIT": "524", "DRIFT_INJECTION_ENABLED": "false", "DRIFT_CONFIG_PATH": "",
        "USE_QUALITY_WEIGHTS": "false", "USE_QUALITY_AS_INPUT": "false", "USE_QUALITY_AGGREGATION_WEIGHTS": "false",
        "USE_QUALITY_ANCHOR_AGGREGATION": "false", "USE_GRADIENT_ALIGNMENT_FILTER": "false", "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL": "false",
        "MITIGATION_METHOD": method.name, "DRIFT_SELECTED_CLIENT_IDS": ",".join(sorted(selected)),
        "SKIP_TRAINING": "true" if method in {Method.R0, Method.M0} else "false",
        "FROZEN_VALIDATION_MODE": "true" if method in {Method.R0, Method.M0} else "false",
    })
    return env

def _copy_prediction_artifacts(method_root: Path, destination: Path) -> None:
    for cluster in range(1, 5):
        source_cluster = method_root / f"cluster_{cluster}"
        target_cluster = destination / f"cluster_{cluster}"
        target_cluster.mkdir(parents=True, exist_ok=True)
        for pattern in ("client_*_test_predictions.csv", "client_*_test_metrics.json"):
            for source in source_cluster.glob(pattern):
                shutil.copy2(source, target_cluster / source.name)


def run_production_method(method: Method | str, prepared_suite: str | Path, *, authorization: str) -> dict[str, Any]:
    _require_authorization(authorization)
    method = Method(method) if str(method) in {item.value for item in Method} else Method[str(method).upper()]
    prepared = load_prepared_suite(prepared_suite)
    activation = _activate_staged_h(method, prepared)
    env = build_method_environment(method, prepared)
    method_root = Path(prepared["root"]) / method.value
    manifest = {"suite_id": SUITE_ID, "method": method.value, "status": "running", "initial_activation": activation,
                "scheduler_contract": "run_split.run_training_schedule_with_fedavg" if method is Method.M1 else None,
                "full_run_authorization_present": True}
    write_json(method_root / "run_manifest.json", manifest)
    stdout_path, stderr_path = method_root / "logs" / "stdout.log", method_root / "logs" / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run([sys.executable, "run_split.py"], cwd=Path.cwd(), env=env, stdout=stdout, stderr=stderr)
    if completed.returncode != 0:
        manifest.update({"status": "failed", "returncode": completed.returncode})
        write_json(method_root / "run_manifest.json", manifest)
        raise RuntimeError(f"{method.value} failed; inspect {stderr_path}")
    if method is Method.M2:
        _copy_prediction_artifacts(method_root, method_root / "persistent_drift_final_holdout")
        clean_timeline = Path(prepared["root"]) / "common" / "clean_deployment_timeline"
        counter_env = dict(env)
        for key in ("CLUSTER_DATA_DIR", "TRAIN_CLUSTER_DATA_DIR", "VAL_CLUSTER_DATA_DIR", "TEST_CLUSTER_DATA_DIR", "TRAIN_TARGET_CLUSTER_DATA_DIR", "VAL_TARGET_CLUSTER_DATA_DIR", "TEST_TARGET_CLUSTER_DATA_DIR"):
            counter_env[key] = str(clean_timeline)
        counter_env.update({"MITIGATION_METHOD": "", "SKIP_TRAINING": "true", "FROZEN_VALIDATION_MODE": "true", "TEST_STREAM_MODE": "deployment_counterfactual_clean_holdout"})
        with (method_root / "logs" / "counterfactual_stdout.log").open("w", encoding="utf-8") as stdout, (method_root / "logs" / "counterfactual_stderr.log").open("w", encoding="utf-8") as stderr:
            counter = subprocess.run([sys.executable, "run_split.py"], cwd=Path.cwd(), env=counter_env, stdout=stdout, stderr=stderr)
        if counter.returncode != 0:
            manifest.update({"status": "failed", "returncode": 0, "counterfactual_returncode": counter.returncode})
            write_json(method_root / "run_manifest.json", manifest)
            raise RuntimeError("M2 counterfactual clean holdout failed")
        _copy_prediction_artifacts(method_root, method_root / "counterfactual_clean_holdout")
        manifest["counterfactual_clean_holdout"] = "complete"
    manifest.update({"status": "complete", "returncode": 0})
    write_json(method_root / "run_manifest.json", manifest)
    return manifest

def run_full_suite(prepared_suite: str | Path, *, authorization: str) -> dict[str, Any]:
    _require_authorization(authorization)
    # This generation-1 pipeline predates R1/M3 and has no bespoke handling
    # for them (e.g. no EWC dispatch); it stays closed to the original four
    # methods rather than picking up new Method members automatically.
    results = {method.value: run_production_method(method, prepared_suite, authorization=authorization) for method in (Method.R0, Method.M0, Method.M1, Method.M2)}
    from .evaluation import evaluate_completed_suite
    evaluation = evaluate_completed_suite(prepared_suite)
    return {"suite_id": SUITE_ID, "methods": results, "evaluation": evaluation, "complete": True}
