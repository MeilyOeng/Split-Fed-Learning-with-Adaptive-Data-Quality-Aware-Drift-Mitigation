#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fed_server.fed_server import run_fedavg_round
from drift.frozen_validation import snapshot_checkpoint_hashes, write_frozen_validation_report
from drift.evaluator import evaluate_run_artifacts
from utils.comm_logging import COMM_LOG_FIELDNAMES
from utils.runtime_utils import (
    client_progress_path,
    client_test_metrics_path,
    client_train_metrics_path,
    cluster_dir,
    compute_model_input_size,
    build_quality_anchor_config,
    derive_dq_mode,
    ensure_dir,
    get_json,
    load_env_file,
    log,
    parse_bool,
    parse_csv_ids,
    post_json,
    read_json,
    wait_for_service,
    write_csv,
    write_json,
)

CLIENT_PATH = "client/client.py"
STREAM_SERVICE_PATH = "data/stream_service.py"
SERVER_PATH = "server/server.py"
ASSIGNMENTS_PATH = Path("dataset") / "cluster-assignment.csv"
# OUTPUT_DIR  = Path("outputs/output-20260613-185635")
OUTPUT_DIR = Path("outputs") / "output-default"
RESULTS_DIR = OUTPUT_DIR / "results"
CLUSTER_DATA_DIR = Path("dataset") / "output"
TRAIN_CLUSTER_DATA_DIR = CLUSTER_DATA_DIR
VAL_CLUSTER_DATA_DIR = CLUSTER_DATA_DIR
TEST_CLUSTER_DATA_DIR = CLUSTER_DATA_DIR
TRAIN_TARGET_CLUSTER_DATA_DIR = TRAIN_CLUSTER_DATA_DIR
VAL_TARGET_CLUSTER_DATA_DIR = VAL_CLUSTER_DATA_DIR
TEST_TARGET_CLUSTER_DATA_DIR = TEST_CLUSTER_DATA_DIR
STREAM_BASE_PORT = 5101
SERVER_BASE_PORT = 5002
CLIENT_WORKER_BASE_PORT = 6101


def configure_output_dirs() -> None:
    global OUTPUT_DIR, RESULTS_DIR
    explicit_output_dir = str(os.environ.get("OUTPUT_DIR", "")).strip()
    if explicit_output_dir:
        OUTPUT_DIR = Path(explicit_output_dir)
    else:
        outputs_root = Path(str(os.environ.get("OUTPUTS_ROOT", "outputs")).strip() or "outputs")
        output_name = str(os.environ.get("OUTPUT_NAME", "")).strip()
        if not output_name:
            output_name = f"output-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        OUTPUT_DIR = outputs_root / output_name
    RESULTS_DIR = OUTPUT_DIR / "results"


def refresh_runtime_settings_from_env() -> None:
    global STREAM_BASE_PORT, SERVER_BASE_PORT, CLIENT_WORKER_BASE_PORT
    global CLUSTER_DATA_DIR, TRAIN_CLUSTER_DATA_DIR, VAL_CLUSTER_DATA_DIR, TEST_CLUSTER_DATA_DIR
    global TRAIN_TARGET_CLUSTER_DATA_DIR, VAL_TARGET_CLUSTER_DATA_DIR, TEST_TARGET_CLUSTER_DATA_DIR
    STREAM_BASE_PORT = int(os.environ.get("STREAM_BASE_PORT", str(STREAM_BASE_PORT)))
    SERVER_BASE_PORT = int(os.environ.get("SERVER_BASE_PORT", str(SERVER_BASE_PORT)))
    CLIENT_WORKER_BASE_PORT = int(os.environ.get("CLIENT_WORKER_BASE_PORT", str(CLIENT_WORKER_BASE_PORT)))
    CLUSTER_DATA_DIR = Path(os.environ.get("CLUSTER_DATA_DIR", str(CLUSTER_DATA_DIR)))
    TRAIN_CLUSTER_DATA_DIR = Path(os.environ.get("TRAIN_CLUSTER_DATA_DIR", str(CLUSTER_DATA_DIR)))
    VAL_CLUSTER_DATA_DIR = Path(os.environ.get("VAL_CLUSTER_DATA_DIR", str(CLUSTER_DATA_DIR)))
    TEST_CLUSTER_DATA_DIR = Path(os.environ.get("TEST_CLUSTER_DATA_DIR", str(CLUSTER_DATA_DIR)))
    TRAIN_TARGET_CLUSTER_DATA_DIR = Path(os.environ.get("TRAIN_TARGET_CLUSTER_DATA_DIR", str(TRAIN_CLUSTER_DATA_DIR)))
    VAL_TARGET_CLUSTER_DATA_DIR = Path(os.environ.get("VAL_TARGET_CLUSTER_DATA_DIR", str(VAL_CLUSTER_DATA_DIR)))
    TEST_TARGET_CLUSTER_DATA_DIR = Path(os.environ.get("TEST_TARGET_CLUSTER_DATA_DIR", str(TEST_CLUSTER_DATA_DIR)))


def _safe_int(raw: Any) -> Optional[int]:
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _resolve_base_input_size(env: dict, *, require_manifest: bool) -> tuple[Optional[int], list[str], Path]:
    manifest_path = TRAIN_CLUSTER_DATA_DIR / "feature_manifest.json"
    use_static_features = parse_bool(env.get("USE_STATIC_FEATURES", "false"), default=False)
    if not use_static_features:
        return 1, [], manifest_path

    if not manifest_path.exists():
        if require_manifest:
            raise FileNotFoundError(
                f"USE_STATIC_FEATURES=true requires {manifest_path}. "
                "Run data/clustering_processing.py first or disable static features."
            )
        return None, [], manifest_path

    manifest = read_json(manifest_path)
    available_static_columns = list(manifest.get("static_feature_columns", []) or [])
    requested_static_columns = _parse_optional_csv(env.get("STATIC_FEATURE_COLUMNS"))
    requested_static_groups = _parse_optional_csv(env.get("STATIC_FEATURE_GROUPS"))
    if requested_static_groups and requested_static_columns:
        raise ValueError("Use either STATIC_FEATURE_COLUMNS or STATIC_FEATURE_GROUPS, not both.")

    if requested_static_groups:
        selected_static_columns: list[str] = []
        normalized_groups = [group.strip().lower() for group in requested_static_groups if group.strip()]
        if "all" in normalized_groups:
            selected_static_columns = list(available_static_columns)
        else:
            for column in available_static_columns:
                if "numeric" in normalized_groups and column.startswith("meta_num_"):
                    selected_static_columns.append(column)
                    continue
                if "category" in normalized_groups and column.startswith("meta_cat_categoria_"):
                    selected_static_columns.append(column)
                    continue
                if "subcategory" in normalized_groups and column.startswith("meta_cat_subcategoria_"):
                    selected_static_columns.append(column)
                    continue
                if "categorical" in normalized_groups and column.startswith("meta_cat_"):
                    selected_static_columns.append(column)
                    continue
        if not selected_static_columns:
            raise ValueError(
                "STATIC_FEATURE_GROUPS did not match any available columns: " + ", ".join(requested_static_groups)
            )
    elif requested_static_columns:
        missing = [col for col in requested_static_columns if col not in available_static_columns]
        if missing:
            raise ValueError(
                "STATIC_FEATURE_COLUMNS contains unknown feature names: " + ", ".join(missing)
            )
        selected_static_columns = requested_static_columns
    else:
        selected_static_columns = available_static_columns
    return 1 + len(selected_static_columns), selected_static_columns, manifest_path


def configure_model_input_size(env: dict) -> dict[str, Any]:
    auto_model_input_size = parse_bool(env.get("AUTO_MODEL_INPUT_SIZE", "true"), default=True)
    use_static_features = parse_bool(env.get("USE_STATIC_FEATURES", "false"), default=False)
    use_quality_weights = parse_bool(env.get("USE_QUALITY_WEIGHTS", "false"), default=False)
    use_quality_as_input = parse_bool(env.get("USE_QUALITY_AS_INPUT", "false"), default=False)
    use_quality_aggregation_weights = parse_bool(env.get("USE_QUALITY_AGGREGATION_WEIGHTS", "false"), default=False)
    quality_anchor_config = build_quality_anchor_config(env)
    quality_input_columns = _parse_optional_csv(env.get("QUALITY_INPUT_COLUMNS", "quality_score")) or ["quality_score"]
    quality_input_dim = len(quality_input_columns)
    dq_mode = derive_dq_mode(
        use_quality_weights,
        use_quality_as_input,
        use_quality_aggregation_weights,
        quality_anchor_config.use_quality_anchor_aggregation,
        quality_anchor_config.use_gradient_alignment_filter,
        quality_anchor_config.use_high_quality_validation_guardrail,
    )
    requested_static_columns = _parse_optional_csv(env.get("STATIC_FEATURE_COLUMNS"))
    requested_static_groups = _parse_optional_csv(env.get("STATIC_FEATURE_GROUPS"))

    base_input_size, selected_static_columns, manifest_path = _resolve_base_input_size(
        env,
        require_manifest=auto_model_input_size,
    )
    if base_input_size is not None:
        env["BASE_INPUT_SIZE"] = str(base_input_size)
    env["DQ_MODE"] = dq_mode
    env["QUALITY_FEATURE_APPENDED"] = "true" if use_quality_as_input else "false"
    env["QUALITY_INPUT_DIM"] = str(quality_input_dim if use_quality_as_input else 0)

    if auto_model_input_size:
        if base_input_size is None:
            raise ValueError("Unable to determine base input size while AUTO_MODEL_INPUT_SIZE=true")
        model_input_size = compute_model_input_size(base_input_size, use_quality_as_input, quality_input_dim)
        previous = str(env.get("MODEL_INPUT_SIZE", "1"))
        env["MODEL_INPUT_SIZE"] = str(model_input_size)
        if previous != str(model_input_size):
            log(
                f"MODEL_INPUT_SIZE overridden from {previous} to {model_input_size} "
                f"(base_input_size={base_input_size}, use_quality_as_input={use_quality_as_input}, "
                f"quality_input_dim={quality_input_dim})"
            )
    else:
        model_input_size = _safe_int(env.get("MODEL_INPUT_SIZE"))

    if not use_static_features:
        return {
            "use_static_features": use_static_features,
            "use_quality_weights": use_quality_weights,
            "use_quality_as_input": use_quality_as_input,
            "use_quality_aggregation_weights": use_quality_aggregation_weights,
            "use_quality_anchor_aggregation": quality_anchor_config.use_quality_anchor_aggregation,
            "use_gradient_alignment_filter": quality_anchor_config.use_gradient_alignment_filter,
            "use_high_quality_validation_guardrail": quality_anchor_config.use_high_quality_validation_guardrail,
            "quality_anchor_top_fraction": quality_anchor_config.top_fraction,
            "quality_anchor_min_clients": quality_anchor_config.min_clients,
            "quality_anchor_update_share": quality_anchor_config.anchor_update_share,
            "quality_alignment_power": quality_anchor_config.alignment_power,
            "quality_negative_alignment_weight": quality_anchor_config.negative_alignment_weight,
            "quality_low_client_norm_cap": quality_anchor_config.low_client_norm_cap,
            "quality_anchor_eps": quality_anchor_config.eps,
            "quality_aggregation_mode": str(env.get("QUALITY_AGGREGATION_MODE", "soft_client_gradient")),
            "quality_aggregation_q_min": float(env.get("QUALITY_AGGREGATION_Q_MIN", "0.5")),
            "quality_aggregation_eps": float(env.get("QUALITY_AGGREGATION_EPS", "1e-8")),
            "dq_mode": dq_mode,
            "quality_feature_appended": use_quality_as_input,
            "quality_input_columns": quality_input_columns if use_quality_as_input else [],
            "quality_input_dim": quality_input_dim if use_quality_as_input else 0,
            "base_input_size": base_input_size,
            "model_input_size": model_input_size,
            "quality_score_path": str(env.get("QUALITY_SCORE_PATH", "")).strip() or None,
        }

    if not manifest_path.exists():
        return {
            "use_static_features": use_static_features,
            "use_quality_weights": use_quality_weights,
            "use_quality_as_input": use_quality_as_input,
            "use_quality_aggregation_weights": use_quality_aggregation_weights,
            "use_quality_anchor_aggregation": quality_anchor_config.use_quality_anchor_aggregation,
            "use_gradient_alignment_filter": quality_anchor_config.use_gradient_alignment_filter,
            "use_high_quality_validation_guardrail": quality_anchor_config.use_high_quality_validation_guardrail,
            "quality_anchor_top_fraction": quality_anchor_config.top_fraction,
            "quality_anchor_min_clients": quality_anchor_config.min_clients,
            "quality_anchor_update_share": quality_anchor_config.anchor_update_share,
            "quality_alignment_power": quality_anchor_config.alignment_power,
            "quality_negative_alignment_weight": quality_anchor_config.negative_alignment_weight,
            "quality_low_client_norm_cap": quality_anchor_config.low_client_norm_cap,
            "quality_anchor_eps": quality_anchor_config.eps,
            "quality_aggregation_mode": str(env.get("QUALITY_AGGREGATION_MODE", "soft_client_gradient")),
            "quality_aggregation_q_min": float(env.get("QUALITY_AGGREGATION_Q_MIN", "0.5")),
            "quality_aggregation_eps": float(env.get("QUALITY_AGGREGATION_EPS", "1e-8")),
            "dq_mode": dq_mode,
            "quality_feature_appended": use_quality_as_input,
            "quality_input_columns": quality_input_columns if use_quality_as_input else [],
            "quality_input_dim": quality_input_dim if use_quality_as_input else 0,
            "base_input_size": base_input_size,
            "model_input_size": model_input_size,
            "quality_score_path": str(env.get("QUALITY_SCORE_PATH", "")).strip() or None,
        }

    if requested_static_columns:
        log(f"Using selected static features: {selected_static_columns}")
    elif requested_static_groups:
        log(f"Using static feature groups {requested_static_groups}: {selected_static_columns}")
    else:
        log(f"Using all static features ({len(selected_static_columns)}) from {manifest_path}")

    return {
        "use_static_features": use_static_features,
        "use_quality_weights": use_quality_weights,
        "use_quality_as_input": use_quality_as_input,
        "use_quality_aggregation_weights": use_quality_aggregation_weights,
        "use_quality_anchor_aggregation": quality_anchor_config.use_quality_anchor_aggregation,
        "use_gradient_alignment_filter": quality_anchor_config.use_gradient_alignment_filter,
        "use_high_quality_validation_guardrail": quality_anchor_config.use_high_quality_validation_guardrail,
        "quality_anchor_top_fraction": quality_anchor_config.top_fraction,
        "quality_anchor_min_clients": quality_anchor_config.min_clients,
        "quality_anchor_update_share": quality_anchor_config.anchor_update_share,
        "quality_alignment_power": quality_anchor_config.alignment_power,
        "quality_negative_alignment_weight": quality_anchor_config.negative_alignment_weight,
        "quality_low_client_norm_cap": quality_anchor_config.low_client_norm_cap,
        "quality_anchor_eps": quality_anchor_config.eps,
        "quality_aggregation_mode": str(env.get("QUALITY_AGGREGATION_MODE", "soft_client_gradient")),
        "quality_aggregation_q_min": float(env.get("QUALITY_AGGREGATION_Q_MIN", "0.5")),
        "quality_aggregation_eps": float(env.get("QUALITY_AGGREGATION_EPS", "1e-8")),
        "dq_mode": dq_mode,
        "quality_feature_appended": use_quality_as_input,
        "quality_input_columns": quality_input_columns if use_quality_as_input else [],
        "quality_input_dim": quality_input_dim if use_quality_as_input else 0,
        "base_input_size": base_input_size,
        "model_input_size": model_input_size,
        "quality_score_path": str(env.get("QUALITY_SCORE_PATH", "")).strip() or None,
    }


def load_cluster_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    cluster_ids = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row.get("Cluster", "")).strip()
            if cid:
                cluster_ids.add(cid)
    return sorted(cluster_ids, key=lambda x: int(x) if x.isdigit() else x)


def load_client_ids_for_cluster(cluster_id: str, path: Path) -> List[str]:
    if not path.exists():
        return []
    client_ids: List[str] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("Cluster", "")).strip() == str(cluster_id):
                client_id = str(row.get("ID", "")).strip()
                if client_id:
                    client_ids.append(client_id)
    return client_ids


def _parse_optional_csv(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def select_gpu_id(role: str, cluster_idx: int, client_idx: int | None = None) -> str | None:
    role_upper = role.upper()
    exact = os.environ.get(f"{role_upper}_GPU_ID")
    if exact not in (None, ""):
        return exact.strip()
    csv_ids = _parse_optional_csv(os.environ.get(f"{role_upper}_GPU_IDS"))
    if csv_ids:
        if role == "server":
            return csv_ids[cluster_idx % len(csv_ids)]
        client_idx = 0 if client_idx is None else client_idx
        return csv_ids[client_idx % len(csv_ids)]
    shared = os.environ.get("GPU_ID")
    if shared not in (None, ""):
        return shared.strip()
    return None


def apply_gpu_env(env: dict, role: str, cluster_idx: int, client_idx: int | None = None) -> tuple[dict, dict]:
    out = env.copy()
    selected_gpu_id = select_gpu_id(role, cluster_idx, client_idx)
    process_role = f"{role}_cluster_{cluster_idx}" if client_idx is None else f"{role}_cluster_{cluster_idx}_client_{client_idx}"
    out["PROCESS_ROLE"] = process_role

    role_upper = role.upper()
    role_gpu_enabled = os.environ.get(f"{role_upper}_GPU_ENABLED")
    if role_gpu_enabled in (None, ""):
        role_gpu_enabled = out.get(f"{role_upper}_GPU_ENABLED")
    if role_gpu_enabled in (None, ""):
        role_gpu_enabled = os.environ.get("GPU_ENABLED", out.get("GPU_ENABLED", "true"))
    out["GPU_ENABLED"] = str(role_gpu_enabled)

    out["GPU_ID"] = "" if selected_gpu_id is None else str(selected_gpu_id)
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility not in (None, ""):
        out["CUDA_VISIBLE_DEVICES"] = visibility
    meta = {
        "role": role,
        "process_role": process_role,
        "gpu_id": selected_gpu_id,
        "cuda_visible_devices": out.get("CUDA_VISIBLE_DEVICES"),
        "gpu_enabled": out.get("GPU_ENABLED"),
    }
    return out, meta


def start_stream_service(cluster_id: str, env: dict, cluster_idx: int):
    port = STREAM_BASE_PORT + cluster_idx
    cdir = cluster_dir(OUTPUT_DIR, cluster_id)
    ensure_dir(cdir)
    log_path = cdir / "stream_service.log"
    log_fh = log_path.open("w", encoding="utf-8")
    proc_env = env.copy()
    proc_env.update(
        {
            "STREAM_CLUSTER_ID": str(cluster_id),
            "CLUSTER_ID": str(cluster_id),
            "STREAM_PORT": str(port),
            "OUTPUT_DIR": str(cdir),
            "CLUSTER_DIR": str(CLUSTER_DATA_DIR),
            "TRAIN_CLUSTER_DATA_DIR": str(TRAIN_CLUSTER_DATA_DIR),
            "VAL_CLUSTER_DATA_DIR": str(VAL_CLUSTER_DATA_DIR),
            "TEST_CLUSTER_DATA_DIR": str(TEST_CLUSTER_DATA_DIR),
            "TRAIN_TARGET_CLUSTER_DATA_DIR": str(TRAIN_TARGET_CLUSTER_DATA_DIR),
            "VAL_TARGET_CLUSTER_DATA_DIR": str(VAL_TARGET_CLUSTER_DATA_DIR),
            "TEST_TARGET_CLUSTER_DATA_DIR": str(TEST_TARGET_CLUSTER_DATA_DIR),
            "PROCESS_ROLE": f"stream_cluster_{cluster_idx}",
            "GPU_ENABLED": "false",
        }
    )
    log(f"[launch] stream cluster={cluster_id} port={port}")
    proc = subprocess.Popen([sys.executable, STREAM_SERVICE_PATH], env=proc_env, stdout=log_fh, stderr=subprocess.STDOUT, cwd=os.getcwd())
    if not wait_for_service(f"http://localhost:{port}/health"):
        try:
            log_fh.flush()
        except Exception:
            pass
        stream_tail = tail_text_file(log_path, n=160)
        raise RuntimeError(
            f"Stream service failed to start for cluster {cluster_id}\n"
            f"===== STREAM SERVICE LOG (tail) =====\n{stream_tail}"
        )
    return proc, log_fh, port


def start_split_server(cluster_id: str, env: dict, cluster_idx: int):
    port = SERVER_BASE_PORT + cluster_idx
    cdir = cluster_dir(OUTPUT_DIR, cluster_id)
    ensure_dir(cdir)
    log_path = cdir / "server.log"
    log_fh = log_path.open("w", encoding="utf-8")
    proc_env, gpu_meta = apply_gpu_env(env, role="server", cluster_idx=cluster_idx)
    proc_env.update(
        {
            "CLUSTER_ID": str(cluster_id),
            "SERVER_PORT": str(port),
            "OUTPUT_DIR": str(cdir),
            "CHECKPOINT_EVERY_ROUNDS": env.get("CHECKPOINT_EVERY_ROUNDS", env.get("CHECKPOINT_EVERY", "1")),
        }
    )
    log(f"[launch] server cluster={cluster_id} port={port} gpu_id={gpu_meta['gpu_id']} CUDA_VISIBLE_DEVICES={gpu_meta['cuda_visible_devices']}")
    proc = subprocess.Popen([sys.executable, SERVER_PATH], env=proc_env, stdout=log_fh, stderr=subprocess.STDOUT, cwd=os.getcwd())
    if not wait_for_service(f"http://localhost:{port}/healthz"):
        raise RuntimeError(f"Server failed to start for cluster {cluster_id}")
    return proc, log_fh, port


def start_client_worker(cluster_id: str, client_id: str, server_port: int, stream_port: int, env: dict, cluster_idx: int, client_idx: int):
    port = CLIENT_WORKER_BASE_PORT + cluster_idx * 100 + client_idx
    cdir = cluster_dir(OUTPUT_DIR, cluster_id)
    ensure_dir(cdir)
    log_path = cdir / f"client_{client_id}_worker.log"
    log_fh = log_path.open("w", encoding="utf-8")
    proc_env, gpu_meta = apply_gpu_env(env, role="client", cluster_idx=cluster_idx, client_idx=client_idx)
    proc_env.update(
        {
            "CLIENT_ID": str(client_id),
            "CLUSTER_ID": str(cluster_id),
            "SERVER_PORT": str(server_port),
            "DATA_STREAM_API_URL": f"http://localhost:{stream_port}",
            "CLIENT_WORKER_PORT": str(port),
            "OUTPUT_DIR": str(cdir),
        }
    )
    log(f"[launch] client cluster={cluster_id} client={client_id} port={port} gpu_id={gpu_meta['gpu_id']} CUDA_VISIBLE_DEVICES={gpu_meta['cuda_visible_devices']}")
    proc = subprocess.Popen([sys.executable, CLIENT_PATH], env=proc_env, stdout=log_fh, stderr=subprocess.STDOUT, cwd=os.getcwd())
    if not wait_for_service(f"http://localhost:{port}/healthz"):
        time.sleep(1.0)
        rc = proc.poll()
        try:
            log_fh.flush()
        except Exception:
            pass
        worker_tail = tail_text_file(log_path, n=160)
        if rc is None:
            proc.terminate()
        msg = (
            f"Client worker failed to start: cluster={cluster_id}, client={client_id}, port={port}, "
            f"returncode={rc}, gpu_enabled={gpu_meta['gpu_enabled']}, gpu_id={gpu_meta['gpu_id']}\n"
            f"===== CLIENT WORKER LOG (tail) =====\n{worker_tail}"
        )
        raise RuntimeError(msg)
    return proc, log_fh, port


def tail_text_file(path: Path, n: int = 120) -> str:
    if not path.exists():
        return f"<missing log: {path}>"
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def dump_failure_context(cluster_id: str, client_id: str | None = None) -> None:
    cdir = cluster_dir(OUTPUT_DIR, cluster_id)
    if client_id is not None:
        log("\n===== CLIENT WORKER LOG (tail) =====")
        log(tail_text_file(cdir / f"client_{client_id}_worker.log"))
    log("===== SERVER LOG (tail) =====")
    log(tail_text_file(cdir / "server.log"))
    log("===== STREAM LOG (tail) =====")
    log(tail_text_file(cdir / "stream_service.log"))


def progress_path(cluster_id: str, client_id: str) -> Path:
    return client_progress_path(OUTPUT_DIR, cluster_id, client_id)


def train_metrics_path(cluster_id: str, client_id: str) -> Path:
    return client_train_metrics_path(OUTPUT_DIR, cluster_id, client_id)


def test_metrics_path(cluster_id: str, client_id: str) -> Path:
    return client_test_metrics_path(OUTPUT_DIR, cluster_id, client_id)


def client_done(cluster_id: str, client_id: str, total_epochs: int) -> bool:
    path = progress_path(cluster_id, client_id)
    if not path.exists():
        return False
    try:
        prog = read_json(path)
    except Exception:
        return False
    return bool(prog.get("early_stopped", False)) or int(prog.get("completed_epochs", 0)) >= total_epochs


def safe_mean(values):
    cleaned = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        cleaned.append(fv)
    return (sum(cleaned) / len(cleaned)) if cleaned else None


def safe_percentile(values, q: float):
    cleaned = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        cleaned.append(fv)
    if not cleaned:
        return None
    cleaned.sort()
    if len(cleaned) == 1:
        return cleaned[0]
    pos = (len(cleaned) - 1) * (q / 100.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return cleaned[lower]
    frac = pos - lower
    return cleaned[lower] * (1.0 - frac) + cleaned[upper] * frac


def safe_sum(values):
    total = 0.0
    found = False
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        total += fv
        found = True
    return total if found else None


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_raw_communication_rows(cluster_ids, cluster_clients):
    rows = []
    for cluster_id in cluster_ids:
        rows.extend(read_csv_rows(cluster_dir(OUTPUT_DIR, cluster_id) / "server_comm.csv"))
        rows.extend(read_csv_rows(cluster_dir(OUTPUT_DIR, cluster_id) / "stream_comm.csv"))
        for client_id in cluster_clients[cluster_id]:
            rows.extend(read_csv_rows(cluster_dir(OUTPUT_DIR, cluster_id) / f"client_{client_id}_comm.csv"))
    rows.extend(read_csv_rows(OUTPUT_DIR / "global" / "fedavg_comm.csv"))
    rows.sort(key=lambda row: str(row.get("timestamp", "")))
    return rows


def _match_rows(rows, **filters):
    out = []
    for row in rows:
        ok = True
        for key, expected in filters.items():
            if str(row.get(key, "")) != str(expected):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def _summary_row(category: str, rows, *, cluster_id=None, scheduler_epoch=None, round_id=None):
    payload_values = [row.get("payload_bytes") for row in rows]
    latency_values = [row.get("latency_ms") for row in rows]
    bandwidth_values = [row.get("bandwidth_MBps") for row in rows]
    return {
        "category": category,
        "cluster_id": cluster_id,
        "scheduler_epoch": scheduler_epoch,
        "round_id": round_id,
        "num_rows": len(rows),
        "total_payload_bytes": safe_sum(payload_values),
        "avg_latency_ms": safe_mean(latency_values),
        "median_latency_ms": safe_percentile(latency_values, 50),
        "p95_latency_ms": safe_percentile(latency_values, 95),
        "avg_bandwidth_MBps": safe_mean(bandwidth_values),
        "median_bandwidth_MBps": safe_percentile(bandwidth_values, 50),
        "p95_bandwidth_MBps": safe_percentile(bandwidth_values, 95),
    }


def build_communication_reports(cluster_ids, cluster_clients):
    raw_rows = collect_raw_communication_rows(cluster_ids, cluster_clients)
    if not raw_rows:
        return raw_rows, [], []

    report_rows = []
    split_upload_rows = _match_rows(
        raw_rows,
        observer="client",
        direction="client_to_server",
        payload_type="activation_payload",
    )
    split_download_rows = _match_rows(
        raw_rows,
        observer="client",
        direction="server_to_client",
        payload_type="gradient_payload",
    )
    fedavg_upload_rows = _match_rows(
        raw_rows,
        observer="fedavg",
        direction="server_to_fed",
        payload_type="server_weights",
    )
    fedavg_broadcast_rows = _match_rows(
        raw_rows,
        observer="fedavg",
        direction="fed_to_server",
        payload_type="avg_server_weights",
    )
    report_rows.append(_summary_row("split_upload_overall", split_upload_rows))
    report_rows.append(_summary_row("split_download_overall", split_download_rows))
    report_rows.append(_summary_row("fedavg_upload_overall", fedavg_upload_rows))
    report_rows.append(_summary_row("fedavg_broadcast_overall", fedavg_broadcast_rows))

    for cluster_id in cluster_ids:
        report_rows.append(_summary_row("split_upload_cluster", [r for r in split_upload_rows if str(r.get("cluster_id")) == str(cluster_id)], cluster_id=cluster_id))
        report_rows.append(_summary_row("split_download_cluster", [r for r in split_download_rows if str(r.get("cluster_id")) == str(cluster_id)], cluster_id=cluster_id))

    epoch_keys = sorted(
        {
            str(row.get("scheduler_epoch"))
            for row in split_upload_rows + split_download_rows
            if str(row.get("scheduler_epoch", "")).strip() not in {"", "None"}
        },
        key=lambda x: int(x),
    )
    for scheduler_epoch in epoch_keys:
        report_rows.append(
            _summary_row(
                "split_upload_epoch",
                [r for r in split_upload_rows if str(r.get("scheduler_epoch")) == scheduler_epoch],
                scheduler_epoch=scheduler_epoch,
            )
        )
        report_rows.append(
            _summary_row(
                "split_download_epoch",
                [r for r in split_download_rows if str(r.get("scheduler_epoch")) == scheduler_epoch],
                scheduler_epoch=scheduler_epoch,
            )
        )

    fedavg_report_rows = []
    for event in sorted({str(row.get("event")) for row in fedavg_upload_rows + fedavg_broadcast_rows if row.get("event")}):
        event_rows = [row for row in fedavg_upload_rows + fedavg_broadcast_rows if str(row.get("event")) == event]
        fedavg_report_rows.append(_summary_row(f"fedavg_{event}", event_rows))

    server_round_rows = _match_rows(raw_rows, observer="server", event="train_step")
    round_groups = defaultdict(list)
    for row in server_round_rows:
        cluster_id = str(row.get("cluster_id", ""))
        round_id = str(row.get("round_id", ""))
        if cluster_id and round_id and round_id not in {"None", ""}:
            round_groups[(cluster_id, round_id)].append(row)
    for (cluster_id, round_id), rows in sorted(round_groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        request_bytes = safe_sum([row.get("payload_bytes") for row in rows]) or 0.0
        response_bytes = safe_sum([row.get("response_payload_bytes") for row in rows]) or 0.0
        report_rows.append(
            {
                **_summary_row("split_round_total", rows, cluster_id=cluster_id, round_id=round_id),
                "total_payload_bytes": request_bytes + response_bytes,
            }
        )

    return raw_rows, report_rows, fedavg_report_rows


def compute_cluster_train_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)
    out = []
    for cluster_id, items in grouped.items():
        out.append(
            {
                "cluster_id": cluster_id,
                "num_clients": len(items),
                "avg_final_train_loss": safe_mean([r.get("final_train_loss") for r in items]),
                "avg_final_val_loss": safe_mean([r.get("final_val_loss") for r in items]),
                "avg_best_val_raw_mae": safe_mean([r.get("best_val_raw_mae") for r in items]),
            }
        )
    return sorted(out, key=lambda x: int(x["cluster_id"]) if str(x["cluster_id"]).isdigit() else str(x["cluster_id"]))


def compute_cluster_test_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)
    out = []
    for cluster_id, items in grouped.items():
        out.append(
            {
                "cluster_id": cluster_id,
                "num_clients": len(items),
                "avg_mae": safe_mean([r.get("mae") for r in items]),
                "avg_rmse": safe_mean([r.get("rmse") for r in items]),
                "avg_mse": safe_mean([r.get("mse") for r in items]),
                "avg_smape": safe_mean([r.get("smape") for r in items]),
                "avg_r2": safe_mean([r.get("r2") for r in items]),
            }
        )
    return sorted(out, key=lambda x: int(x["cluster_id"]) if str(x["cluster_id"]).isdigit() else str(x["cluster_id"]))


def collect_server_config(server_port: int, cluster_id: str):
    config = {"cluster_id": cluster_id, "server_port": server_port}
    try:
        data = get_json(f"http://localhost:{server_port}/config", timeout=5)
        if isinstance(data, dict):
            config.update(data)
    except Exception:
        pass
    return config


def collect_client_config(cluster_id: str, client_id: str):
    cdir = cluster_dir(OUTPUT_DIR, cluster_id)
    path = cdir / f"client_{client_id}_config.json"
    config = {
        "cluster_id": cluster_id,
        "client_id": client_id,
        "checkpoint_path": str(cdir / f"client_{client_id}_checkpoint.pt"),
        "best_checkpoint_path": str(cdir / f"client_{client_id}_best_checkpoint.pt"),
        "final_checkpoint_path": str(cdir / f"client_{client_id}_final.pt"),
    }
    if path.exists():
        try:
            saved = read_json(path)
            if isinstance(saved, dict):
                config.update(saved)
        except Exception:
            pass
    return config


def aggregate_train_rows(cluster_ids, cluster_clients, total_epochs, training_mode, client_order_strategy, random_seed_schedule, train_durations):
    rows = []
    for cid in cluster_ids:
        for client_id in cluster_clients[cid]:
            path = train_metrics_path(cid, client_id)
            if not path.exists():
                continue
            metrics = read_json(path)
            rows.append(
                {
                    "cluster_id": cid,
                    "client_id": client_id,
                    "num_epochs": total_epochs,
                    "training_mode": training_mode,
                    "client_order_strategy": client_order_strategy,
                    "random_seed_schedule": random_seed_schedule,
                    "train_ok": True,
                    "train_duration_sec": round(train_durations.get((cid, client_id), 0.0), 3),
                    "final_train_loss": metrics.get("final_train_loss"),
                    "final_val_loss": metrics.get("final_val_loss"),
                    "best_val_raw_mae": metrics.get("best_val_raw_mae"),
                    "best_epoch": metrics.get("best_epoch"),
                    "num_logged_epochs": len(metrics.get("history", [])),
                    "completed_epochs": metrics.get("completed_epochs"),
                    "early_stopped": metrics.get("early_stopped"),
                    "stop_reason": metrics.get("stop_reason"),
                }
            )
    return rows


def aggregate_test_rows(cluster_ids, cluster_clients, training_mode, client_order_strategy, random_seed_schedule, test_durations):
    rows = []
    for cid in cluster_ids:
        for client_id in cluster_clients[cid]:
            path = test_metrics_path(cid, client_id)
            if not path.exists():
                continue
            metrics = read_json(path)
            raw = metrics.get("raw_metrics", {})
            rows.append(
                {
                    "cluster_id": cid,
                    "client_id": client_id,
                    "training_mode": training_mode,
                    "client_order_strategy": client_order_strategy,
                    "random_seed_schedule": random_seed_schedule,
                    "test_ok": True,
                    "test_duration_sec": round(test_durations.get((cid, client_id), 0.0), 3),
                    "mae": raw.get("mae"),
                    "rmse": raw.get("rmse"),
                    "mse": raw.get("mse"),
                    "smape": raw.get("smape"),
                    "r2": raw.get("r2"),
                    "num_test_batches": metrics.get("num_batches"),
                    "num_test_samples": metrics.get("num_samples"),
                    "eval_mode": metrics.get("eval_mode"),
                }
            )
    return rows


def resolve_cluster_clients(detected_clusters):
    cluster_ids = parse_csv_ids(os.environ["CLUSTER_IDS"]) if os.environ.get("CLUSTER_IDS") else detected_clusters
    client_filter = set(parse_csv_ids(os.environ["CLIENT_IDS"])) if os.environ.get("CLIENT_IDS") else None
    cluster_clients = {}
    for cid in cluster_ids:
        clients = load_client_ids_for_cluster(cid, ASSIGNMENTS_PATH)
        if client_filter is not None:
            clients = [c for c in clients if c in client_filter]
        if not clients:
            raise RuntimeError(f"No clients found for cluster {cid}")
        cluster_clients[cid] = clients
    return cluster_ids, cluster_clients


def cluster_train_sample_weights(cluster_ids, cluster_clients):
    weights = {}
    for cid in cluster_ids:
        total = 0.0
        for client_id in cluster_clients[cid]:
            config = collect_client_config(cid, client_id)
            total += float(config.get("normalization", {}).get("num_train_windows", 0) or 0)
        if total <= 0:
            total = float(len(cluster_clients[cid]))
        weights[str(cid)] = total
    return weights


def ensure_final_client_checkpoints(cluster_ids, cluster_clients):
    for cid in cluster_ids:
        for client_id in cluster_clients[cid]:
            latest = cluster_dir(OUTPUT_DIR, cid) / f"client_{client_id}_checkpoint.pt"
            final = cluster_dir(OUTPUT_DIR, cid) / f"client_{client_id}_final.pt"
            if (not final.exists()) and latest.exists():
                shutil.copy2(latest, final)
                log(f"[finalize] cluster={cid} client={client_id} copied latest -> final")


def fetch_worker_status(client_port: int) -> dict:
    return get_json(f"http://localhost:{client_port}/status", timeout=30.0)


def order_round_clients(clients: List[str], strategy: str, rng: random.Random, scheduler_epoch: int, round_idx: int) -> List[str]:
    ordered = list(clients)
    strategy = str(strategy).strip().lower()
    if strategy in {"fixed", "ordered"}:
        return ordered
    if strategy in {"random", "random_per_round", "shuffle"}:
        rng.shuffle(ordered)
        return ordered
    if strategy == "round_robin":
        if not ordered:
            return ordered
        rotation = (round_idx - 1) % len(ordered)
        return ordered[rotation:] + ordered[:rotation]
    raise ValueError(f"Unsupported CLIENT_ORDER_STRATEGY: {strategy}")


def chunk_round_clients(clients: List[str], accum_steps_server: int) -> List[List[str]]:
    if accum_steps_server <= 0:
        return [list(clients)]
    return [clients[i:i + accum_steps_server] for i in range(0, len(clients), accum_steps_server)]


def finalize_epoch_boundaries(cluster_id: str, client_ports: Dict[str, int], exhausted_clients: List[str], train_durations) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for client_id in exhausted_clients:
        t0 = time.time()
        base = f"http://localhost:{client_ports[client_id]}"
        try:
            result = post_json(f"{base}/command/finalize_epoch", {}, timeout=300.0)
        except requests.HTTPError as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 404:
                # Backward-compatibility path only.
                result = post_json(f"{base}/command/train_one_batch", {}, timeout=300.0)
            else:
                dump_failure_context(cluster_id, client_id)
                raise
        except Exception:
            dump_failure_context(cluster_id, client_id)
            raise
        dt = round(time.time() - t0, 3)
        train_durations[(cluster_id, client_id)] += dt
        results[client_id] = result
        log(
            f"[EPOCH][finalize] cluster={cluster_id} client={client_id} status={result.get('status')} "
            f"duration_sec={dt} completed_epochs={result.get('completed_epochs')} early_stopped={result.get('early_stopped')}"
        )
    return results


def run_round(server_port: int, cluster_id: str, client_ports: Dict[str, int], round_clients: List[str], scheduler_epoch: int, global_round: int, round_idx: int, round_chunk_idx: int, train_durations):
    numeric_round_id = int(round_idx * 1000 + round_chunk_idx)
    round_label = f"epoch{scheduler_epoch}_round{round_idx}_chunk{round_chunk_idx}"
    log(f"[ROUND][start] cluster={cluster_id} scheduler_epoch={scheduler_epoch} round={round_idx} chunk={round_chunk_idx} round_id={round_label} active_clients={round_clients}")
    post_json(
        f"http://localhost:{server_port}/round/start",
        {
            "round_id": numeric_round_id,
            "global_round": global_round,
            "logical_round": round_idx,
            "scheduler_epoch": scheduler_epoch,
            "expected_clients": round_clients,
            "effective_accum_steps_target": len(round_clients),
        },
    )

    results: Dict[str, dict] = {}
    for order_idx, client_id in enumerate(round_clients, start=1):
        t0 = time.time()
        try:
            result = post_json(f"http://localhost:{client_ports[client_id]}/command/train_one_batch", {}, timeout=300.0)
        except Exception:
            dump_failure_context(cluster_id, client_id)
            raise
        dt = round(time.time() - t0, 3)
        train_durations[(cluster_id, client_id)] += dt
        results[client_id] = result
        log(
            f"[ROUND][client] cluster={cluster_id} scheduler_epoch={scheduler_epoch} round={round_idx} chunk={round_chunk_idx} "
            f"order={order_idx}/{len(round_clients)} client={client_id} status={result.get('status')} "
            f"duration_sec={dt} completed_epochs={result.get('completed_epochs')} "
            f"early_stopped={result.get('early_stopped')} train_stream_active={result.get('train_stream_active')}"
        )

    state = get_json(f"http://localhost:{server_port}/round/state")
    log(f"[ROUND][state] cluster={cluster_id} scheduler_epoch={scheduler_epoch} round={round_idx} chunk={round_chunk_idx} state={state}")
    if (
        state.get("pending_accum_steps", 0) != 0
        or state.get("round_total_contrib_count", 0) != 0
        or state.get("window_contrib_count", 0) != 0
        or state.get("expected_clients")
    ):
        raise RuntimeError(f"Round did not flush cleanly: {state}")
    return results


def run_single_scheduler_epoch_for_cluster(cluster_id: str, client_ids: List[str], env: dict, cluster_idx: int, scheduler_epoch: int, stop_event: Event):
    training_mode = env.get("TRAINING_MODE", "client_batch_random").strip().lower()
    if training_mode != "client_batch_random":
        raise ValueError("This persistent-worker scheduler supports TRAINING_MODE=client_batch_random only")

    rng = random.Random(int(env.get("RANDOM_SEED_SCHEDULE", env.get("RANDOM_SEED", "42"))) + cluster_idx)
    order_strategy = env.get("CLIENT_ORDER_STRATEGY", "random_per_round")
    server_port = SERVER_BASE_PORT + cluster_idx
    client_ports = {client_id: CLIENT_WORKER_BASE_PORT + cluster_idx * 100 + idx for idx, client_id in enumerate(client_ids)}
    train_durations = defaultdict(float)

    def _ensure_not_cancelled():
        if stop_event.is_set():
            raise RuntimeError(f"Cluster {cluster_id} cancelled due to failure in another cluster")

    _ensure_not_cancelled()
    total_epochs = int(env.get("NUM_EPOCHS", "100"))
    local_epochs_per_global_round = max(1, int(env.get("LOCAL_EPOCHS_PER_GLOBAL_ROUND", "1")))
    global_round = ((scheduler_epoch - 1) // local_epochs_per_global_round) + 1
    log(f"\n=== Scheduler epoch {scheduler_epoch}/{total_epochs} cluster={cluster_id} ===")
    round_idx = 0
    while True:
        _ensure_not_cancelled()
        worker_status = {}
        active_clients = []
        ready_clients = []
        exhausted_clients = []
        for client_id in client_ids:
            status = fetch_worker_status(client_ports[client_id])
            worker_status[client_id] = status
            if bool(status.get("is_done", False)):
                continue
            active_clients.append(client_id)
            if bool(status.get("has_next_train_batch", False)):
                ready_clients.append(client_id)
            else:
                exhausted_clients.append(client_id)

        if not active_clients:
            log(f"[TRAIN] cluster={cluster_id} scheduler_epoch={scheduler_epoch} no active clients left")
            break

        if ready_clients:
            ordered_ready = order_round_clients(ready_clients, order_strategy, rng, scheduler_epoch, round_idx + 1)
            progress_this_cycle = False
            round_idx += 1
            results = run_round(
                server_port,
                cluster_id,
                client_ports,
                ordered_ready,
                scheduler_epoch,
                global_round,
                round_idx,
                1,
                train_durations,
            )
            for client_id, result in results.items():
                if result.get("status") == "batch_processed":
                    progress_this_cycle = True
            if not progress_this_cycle:
                raise RuntimeError(f"No batch progress detected in cluster {cluster_id} epoch {scheduler_epoch}")

        if exhausted_clients:
            finalize_results = finalize_epoch_boundaries(cluster_id, client_ports, exhausted_clients, train_durations)
            for client_id, result in finalize_results.items():
                status = result.get("status")
                if status not in {"epoch_completed", "training_complete", "done"}:
                    raise RuntimeError(
                        f"Expected epoch boundary finalization for cluster={cluster_id}, client={client_id}, got status={status}"
                    )

        all_reached = True
        client_progress_snapshot = []
        for client_id in client_ids:
            path = progress_path(cluster_id, client_id)
            if not path.exists():
                all_reached = False
                client_progress_snapshot.append({"client_id": client_id, "progress": "missing"})
                continue
            prog = read_json(path)
            completed_epochs = int(prog.get("completed_epochs", 0))
            early_stopped = bool(prog.get("early_stopped", False))
            client_progress_snapshot.append({
                "client_id": client_id,
                "completed_epochs": completed_epochs,
                "early_stopped": early_stopped,
                "epochs_without_improvement": prog.get("epochs_without_improvement"),
            })
            if not early_stopped and completed_epochs < scheduler_epoch:
                all_reached = False
        log(f"[TRAIN][epoch_progress] cluster={cluster_id} scheduler_epoch={scheduler_epoch} rounds={round_idx} progress={client_progress_snapshot}")
        if all_reached:
            log(f"[TRAIN][epoch_complete] cluster={cluster_id} scheduler_epoch={scheduler_epoch} rounds={round_idx}")
            break
    return train_durations


def run_training_schedule_for_cluster(cluster_id: str, client_ids: List[str], env: dict, cluster_idx: int, stop_event: Event):
    merged_durations = defaultdict(float)
    total_epochs = int(env.get("NUM_EPOCHS", "100"))
    for scheduler_epoch in range(1, total_epochs + 1):
        cluster_durations = run_single_scheduler_epoch_for_cluster(
            cluster_id,
            client_ids,
            env,
            cluster_idx,
            scheduler_epoch,
            stop_event,
        )
        for key, value in cluster_durations.items():
            merged_durations[key] += value
    return merged_durations


def run_training_schedule(cluster_ids, cluster_clients, env):
    run_parallel = parse_bool(env.get("RUN_CLUSTERS_IN_PARALLEL"), default=True) and len(cluster_ids) > 1
    log(f"[TRAIN] cluster_parallelism={'enabled' if run_parallel else 'disabled'}")
    merged_durations = defaultdict(float)
    stop_event = Event()

    if not run_parallel:
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            cluster_durations = run_training_schedule_for_cluster(cluster_id, cluster_clients[cluster_id], env, cluster_idx, stop_event)
            for key, value in cluster_durations.items():
                merged_durations[key] += value
        return merged_durations

    first_exception = None
    with ThreadPoolExecutor(max_workers=len(cluster_ids)) as executor:
        futures = {
            executor.submit(run_training_schedule_for_cluster, cluster_id, cluster_clients[cluster_id], env, cluster_idx, stop_event): cluster_id
            for cluster_idx, cluster_id in enumerate(cluster_ids)
        }
        for future in as_completed(futures):
            cluster_id = futures[future]
            try:
                cluster_durations = future.result()
            except Exception as exc:
                stop_event.set()
                log(f"[TRAIN] cluster={cluster_id} failed: {exc}")
                if first_exception is None:
                    first_exception = exc
            else:
                log(f"[TRAIN] cluster={cluster_id} finished")
                for key, value in cluster_durations.items():
                    merged_durations[key] += value

    if first_exception is not None:
        raise first_exception
    return merged_durations


def run_scheduler_epoch_across_clusters(cluster_ids, cluster_clients, env, scheduler_epoch: int, stop_event: Event):
    run_parallel = parse_bool(env.get("RUN_CLUSTERS_IN_PARALLEL"), default=True) and len(cluster_ids) > 1
    merged_durations = defaultdict(float)

    if not run_parallel:
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            cluster_durations = run_single_scheduler_epoch_for_cluster(
                cluster_id,
                cluster_clients[cluster_id],
                env,
                cluster_idx,
                scheduler_epoch,
                stop_event,
            )
            for key, value in cluster_durations.items():
                merged_durations[key] += value
        return merged_durations

    first_exception = None
    with ThreadPoolExecutor(max_workers=len(cluster_ids)) as executor:
        futures = {
            executor.submit(
                run_single_scheduler_epoch_for_cluster,
                cluster_id,
                cluster_clients[cluster_id],
                env,
                cluster_idx,
                scheduler_epoch,
                stop_event,
            ): cluster_id
            for cluster_idx, cluster_id in enumerate(cluster_ids)
        }
        for future in as_completed(futures):
            cluster_id = futures[future]
            try:
                cluster_durations = future.result()
            except Exception as exc:
                stop_event.set()
                log(f"[TRAIN][epoch_barrier] cluster={cluster_id} failed at scheduler_epoch={scheduler_epoch}: {exc}")
                if first_exception is None:
                    first_exception = exc
            else:
                log(f"[TRAIN][epoch_barrier] cluster={cluster_id} reached scheduler_epoch={scheduler_epoch}")
                for key, value in cluster_durations.items():
                    merged_durations[key] += value

    if first_exception is not None:
        raise first_exception
    return merged_durations


def run_server_fedavg_after_epoch(cluster_ids, cluster_clients, scheduler_epoch: int):
    cluster_weights = cluster_train_sample_weights(cluster_ids, cluster_clients)
    server_urls = [f"http://localhost:{SERVER_BASE_PORT + cluster_idx}" for cluster_idx, _ in enumerate(cluster_ids)]
    output_path = OUTPUT_DIR / "global" / "global_server_latest.pt"
    client_output_path = OUTPUT_DIR / "global" / "global_client_latest.pt"
    log(
        f"[FEDAVG] scheduler_epoch={scheduler_epoch} cluster_weights={cluster_weights}"
    )
    record = run_fedavg_round(
        cluster_ids=[str(cid) for cid in cluster_ids],
        server_urls=server_urls,
        checkpoint_dir=str(OUTPUT_DIR),
        output_path=str(output_path),
        client_output_path=str(client_output_path),
        assignments_path=str(ASSIGNMENTS_PATH),
        cluster_weights=cluster_weights,
        aggregate_clients=False,
    )
    if record.error:
        raise RuntimeError(f"FedAvg failed after scheduler_epoch={scheduler_epoch}: {record.error}")
    if record.server_push_fail:
        raise RuntimeError(
            f"FedAvg broadcast failed after scheduler_epoch={scheduler_epoch} for clusters {record.server_push_fail}"
        )
    log(
        f"[FEDAVG] scheduler_epoch={scheduler_epoch} duration_sec={record.duration_sec} "
        f"server_push_ok={record.server_push_ok}"
    )
    return record


def run_training_schedule_with_fedavg(cluster_ids, cluster_clients, env):
    total_epochs = int(env.get("NUM_EPOCHS", "100"))
    local_epochs_per_global_round = max(1, int(env.get("LOCAL_EPOCHS_PER_GLOBAL_ROUND", "1")))
    merged_durations = defaultdict(float)
    stop_event = Event()

    for scheduler_epoch in range(1, total_epochs + 1):
        epoch_durations = run_scheduler_epoch_across_clusters(
            cluster_ids,
            cluster_clients,
            env,
            scheduler_epoch,
            stop_event,
        )
        for key, value in epoch_durations.items():
            merged_durations[key] += value

        should_aggregate = (
            len(cluster_ids) > 1
            and (
                scheduler_epoch % local_epochs_per_global_round == 0
                or scheduler_epoch == total_epochs
            )
        )
        if should_aggregate:
            run_server_fedavg_after_epoch(cluster_ids, cluster_clients, scheduler_epoch)

    return merged_durations


def run_test_phase_for_cluster(cluster_id: str, client_ids: List[str], cluster_idx: int, stop_event: Event):
    test_durations = defaultdict(float)
    for idx, client_id in enumerate(client_ids):
        if stop_event.is_set():
            raise RuntimeError(f"Cluster {cluster_id} cancelled due to failure in another cluster")
        port = CLIENT_WORKER_BASE_PORT + cluster_idx * 100 + idx
        t0 = time.time()
        try:
            result = post_json(f"http://localhost:{port}/command/test", {}, timeout=600.0)
        except Exception:
            dump_failure_context(cluster_id, client_id)
            raise
        dt = round(time.time() - t0, 3)
        test_durations[(cluster_id, client_id)] += dt
        log(
            f"[TEST] cluster={cluster_id} client={client_id} dur={dt}s "
            f"mae={result.get('raw_metrics', {}).get('mae')} rmse={result.get('raw_metrics', {}).get('rmse')}"
        )
    return test_durations


def run_test_phase(cluster_ids, cluster_clients, env):
    run_parallel = parse_bool(env.get("RUN_CLUSTERS_IN_PARALLEL"), default=True) and len(cluster_ids) > 1
    log(f"[TEST] cluster_parallelism={'enabled' if run_parallel else 'disabled'}")
    merged_durations = defaultdict(float)
    stop_event = Event()

    if not run_parallel:
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            cluster_durations = run_test_phase_for_cluster(cluster_id, cluster_clients[cluster_id], cluster_idx, stop_event)
            for key, value in cluster_durations.items():
                merged_durations[key] += value
        return merged_durations

    first_exception = None
    with ThreadPoolExecutor(max_workers=len(cluster_ids)) as executor:
        futures = {
            executor.submit(run_test_phase_for_cluster, cluster_id, cluster_clients[cluster_id], cluster_idx, stop_event): cluster_id
            for cluster_idx, cluster_id in enumerate(cluster_ids)
        }
        for future in as_completed(futures):
            cluster_id = futures[future]
            try:
                cluster_durations = future.result()
            except Exception as exc:
                stop_event.set()
                log(f"[TEST] cluster={cluster_id} failed: {exc}")
                if first_exception is None:
                    first_exception = exc
            else:
                log(f"[TEST] cluster={cluster_id} finished")
                for key, value in cluster_durations.items():
                    merged_durations[key] += value

    if first_exception is not None:
        raise first_exception
    return merged_durations


def main() -> int:
    start_time = time.time()
    dotenv_path = load_env_file()
    configure_output_dirs()
    refresh_runtime_settings_from_env()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(RESULTS_DIR)

    env = os.environ.copy()
    env.update(
        {
            "USE_LOCALHOST": "true",
            "MODEL_INPUT_SIZE": os.environ.get("MODEL_INPUT_SIZE", "1"),
            "MODEL_HIDDEN_SIZE": os.environ.get("MODEL_HIDDEN_SIZE", "128"),
            "MODEL_NUM_LAYERS": os.environ.get("MODEL_NUM_LAYERS", "3"),
            "MODEL_SPLIT_LAYER": os.environ.get("MODEL_SPLIT_LAYER", "2"),
            "MODEL_OUTPUT_SIZE": os.environ.get("MODEL_OUTPUT_SIZE", "1"),
            "MODEL_DROPOUT": os.environ.get("MODEL_DROPOUT", "0.4"),
            "MODEL_BIDIRECTIONAL": os.environ.get("MODEL_BIDIRECTIONAL", "false"),
            "WINDOW_SIZE": os.environ.get("WINDOW_SIZE", "24"),
            "FORECAST_HORIZON": os.environ.get("FORECAST_HORIZON", "1"),
            "TRAIN_RATIO": os.environ.get("TRAIN_RATIO", "0.7"),
            "VAL_RATIO": os.environ.get("VAL_RATIO", "0.1"),
            "TEST_RATIO": os.environ.get("TEST_RATIO", "0.2"),
            "NORMALIZE": os.environ.get("NORMALIZE", "true"),
            "NORMALIZE_METHOD": os.environ.get("NORMALIZE_METHOD", "zscore"),
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "32"),
            "NUM_EPOCHS": os.environ.get("NUM_EPOCHS", "100"),
            "LOCAL_EPOCHS_PER_GLOBAL_ROUND": os.environ.get("LOCAL_EPOCHS_PER_GLOBAL_ROUND", "1"),
            "CLIENT_LR": os.environ.get("CLIENT_LR", os.environ.get("LEARNING_RATE", "0.0005")),
            "SERVER_LR": os.environ.get("SERVER_LR", os.environ.get("LEARNING_RATE", "0.0005")),
            "ACCUM_STEPS_SERVER": os.environ.get("ACCUM_STEPS_SERVER", "1"),
            "RANDOM_SEED": os.environ.get("RANDOM_SEED", "42"),
            "CHECKPOINT_EVERY": os.environ.get("CHECKPOINT_EVERY", "10"),
            "CHECKPOINT_EVERY_ROUNDS": os.environ.get("CHECKPOINT_EVERY_ROUNDS", os.environ.get("CHECKPOINT_EVERY", "10")),
            "TRAINING_MODE": os.environ.get("TRAINING_MODE", "client_batch_random"),
            "CLIENT_ORDER_STRATEGY": os.environ.get("CLIENT_ORDER_STRATEGY", "random_per_round"),
            "RANDOM_SEED_SCHEDULE": os.environ.get("RANDOM_SEED_SCHEDULE", os.environ.get("RANDOM_SEED", "42")),
            "EVAL_MODE": os.environ.get("EVAL_MODE", "final_shared"),
            "EARLY_STOPPING": os.environ.get("EARLY_STOPPING", "true"),
            "EARLY_STOPPING_PATIENCE": os.environ.get("EARLY_STOPPING_PATIENCE", "15"),
            "EARLY_STOPPING_MIN_DELTA": os.environ.get("EARLY_STOPPING_MIN_DELTA", "0.00001"),
            "EARLY_STOPPING_METRIC": os.environ.get("EARLY_STOPPING_METRIC", "val_raw_mae"),
            "COMM_LOG_TRAIN": os.environ.get("COMM_LOG_TRAIN", "true"),
            "COMM_LOG_EVAL": os.environ.get("COMM_LOG_EVAL", "false"),
            "COMM_LOG_STREAM": os.environ.get("COMM_LOG_STREAM", "false"),
            "COMM_LOG_EVERY_N_BATCHES": os.environ.get("COMM_LOG_EVERY_N_BATCHES", "1"),
            "SAVE_PROGRESS_EVERY_N_BATCHES": os.environ.get("SAVE_PROGRESS_EVERY_N_BATCHES", "1"),
            "WRITE_COMMUNICATION_LOG": os.environ.get("WRITE_COMMUNICATION_LOG", "true"),
            "SKIP_TRAINING": os.environ.get("SKIP_TRAINING", "false"),
            "FROZEN_VALIDATION_MODE": os.environ.get("FROZEN_VALIDATION_MODE", "false"),
            "USE_STATIC_FEATURES": os.environ.get("USE_STATIC_FEATURES", "false"),
            "STATIC_FEATURE_COLUMNS": os.environ.get("STATIC_FEATURE_COLUMNS", ""),
            "STATIC_FEATURE_GROUPS": os.environ.get("STATIC_FEATURE_GROUPS", ""),
            "AUTO_MODEL_INPUT_SIZE": os.environ.get("AUTO_MODEL_INPUT_SIZE", "true"),
            "CLUSTER_DATA_DIR": os.environ.get("CLUSTER_DATA_DIR", str(CLUSTER_DATA_DIR)),
            "TRAIN_CLUSTER_DATA_DIR": os.environ.get("TRAIN_CLUSTER_DATA_DIR", str(TRAIN_CLUSTER_DATA_DIR)),
            "VAL_CLUSTER_DATA_DIR": os.environ.get("VAL_CLUSTER_DATA_DIR", str(VAL_CLUSTER_DATA_DIR)),
            "TEST_CLUSTER_DATA_DIR": os.environ.get("TEST_CLUSTER_DATA_DIR", str(TEST_CLUSTER_DATA_DIR)),
            "TRAIN_TARGET_CLUSTER_DATA_DIR": os.environ.get("TRAIN_TARGET_CLUSTER_DATA_DIR", str(TRAIN_TARGET_CLUSTER_DATA_DIR)),
            "VAL_TARGET_CLUSTER_DATA_DIR": os.environ.get("VAL_TARGET_CLUSTER_DATA_DIR", str(VAL_TARGET_CLUSTER_DATA_DIR)),
            "TEST_TARGET_CLUSTER_DATA_DIR": os.environ.get("TEST_TARGET_CLUSTER_DATA_DIR", str(TEST_TARGET_CLUSTER_DATA_DIR)),
            "USE_QUALITY_WEIGHTS": os.environ.get("USE_QUALITY_WEIGHTS", "false"),
            "USE_QUALITY_AS_INPUT": os.environ.get("USE_QUALITY_AS_INPUT", "false"),
            "USE_QUALITY_AGGREGATION_WEIGHTS": os.environ.get("USE_QUALITY_AGGREGATION_WEIGHTS", "false"),
            "USE_QUALITY_ANCHOR_AGGREGATION": os.environ.get("USE_QUALITY_ANCHOR_AGGREGATION", "false"),
            "USE_GRADIENT_ALIGNMENT_FILTER": os.environ.get("USE_GRADIENT_ALIGNMENT_FILTER", "false"),
            "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL": os.environ.get("USE_HIGH_QUALITY_VALIDATION_GUARDRAIL", "false"),
            "QUALITY_ANCHOR_TOP_FRACTION": os.environ.get("QUALITY_ANCHOR_TOP_FRACTION", "0.40"),
            "QUALITY_ANCHOR_MIN_CLIENTS": os.environ.get("QUALITY_ANCHOR_MIN_CLIENTS", "3"),
            "QUALITY_ANCHOR_UPDATE_SHARE": os.environ.get("QUALITY_ANCHOR_UPDATE_SHARE", "0.80"),
            "QUALITY_ALIGNMENT_POWER": os.environ.get("QUALITY_ALIGNMENT_POWER", "1.0"),
            "QUALITY_NEGATIVE_ALIGNMENT_WEIGHT": os.environ.get("QUALITY_NEGATIVE_ALIGNMENT_WEIGHT", "0.0"),
            "QUALITY_LOW_CLIENT_NORM_CAP": os.environ.get("QUALITY_LOW_CLIENT_NORM_CAP", "2.0"),
            "QUALITY_ANCHOR_EPS": os.environ.get("QUALITY_ANCHOR_EPS", "1e-8"),
            "QUALITY_AGGREGATION_DEBUG_LOG": os.environ.get("QUALITY_AGGREGATION_DEBUG_LOG", "false"),
            "QUALITY_AGGREGATION_MODE": os.environ.get("QUALITY_AGGREGATION_MODE", "soft_client_gradient"),
            "QUALITY_AGGREGATION_Q_MIN": os.environ.get("QUALITY_AGGREGATION_Q_MIN", "0.5"),
            "QUALITY_AGGREGATION_EPS": os.environ.get("QUALITY_AGGREGATION_EPS", "1e-8"),
            "QUALITY_SCORE_PATH": os.environ.get("QUALITY_SCORE_PATH", ""),
            "QUALITY_INPUT_COLUMNS": os.environ.get("QUALITY_INPUT_COLUMNS", "quality_score"),
            "QUALITY_WEIGHT_COLUMN": os.environ.get("QUALITY_WEIGHT_COLUMN", "quality_score"),
            "QUALITY_SCORE_GRANULARITY": os.environ.get("QUALITY_SCORE_GRANULARITY", "window"),
            "STRICT_QUALITY_SCORE_MATCHING": os.environ.get("STRICT_QUALITY_SCORE_MATCHING", "true"),
            "QUALITY_SCORE_FALLBACK": os.environ.get("QUALITY_SCORE_FALLBACK", "1.0"),
            "QUALITY_SCORE_MIN": os.environ.get("QUALITY_SCORE_MIN", "0.05"),
            "QUALITY_SCORE_MAX": os.environ.get("QUALITY_SCORE_MAX", "1.0"),
            "QUALITY_SCORE_EPS": os.environ.get("QUALITY_SCORE_EPS", "1e-8"),
        }
    )
    quality_runtime = configure_model_input_size(env)

    total_epochs = int(env["NUM_EPOCHS"])
    detected_clusters = load_cluster_ids(ASSIGNMENTS_PATH)
    if not detected_clusters:
        log("ERROR: No clusters found in dataset/cluster-assignment.csv")
        return 1

    cluster_ids, cluster_clients = resolve_cluster_clients(detected_clusters)
    frozen_validation_mode = parse_bool(env.get("FROZEN_VALIDATION_MODE"), default=False)
    if frozen_validation_mode and not parse_bool(env.get("SKIP_TRAINING"), default=False):
        raise ValueError("FROZEN_VALIDATION_MODE=true requires SKIP_TRAINING=true")
    frozen_hashes_before = snapshot_checkpoint_hashes(OUTPUT_DIR) if frozen_validation_mode else {}

    log("Starting split-learning run")
    log(f".env loaded: {dotenv_path if dotenv_path is not None else '<not found>'} (system env overrides .env)")
    log(f"Epochs: {total_epochs}")
    log(f"Training mode: {env['TRAINING_MODE']}")
    log(f"Client order strategy: {env['CLIENT_ORDER_STRATEGY']}")
    log(f"Schedule seed: {env['RANDOM_SEED_SCHEDULE']}")
    log(f"Eval mode: {env['EVAL_MODE']}")
    log(f"Run output directory: {OUTPUT_DIR}")
    log(f"Cluster data directory: {CLUSTER_DATA_DIR}")
    log(f"Train cluster data directory: {TRAIN_CLUSTER_DATA_DIR}")
    log(f"Val cluster data directory: {VAL_CLUSTER_DATA_DIR}")
    log(f"Test cluster data directory: {TEST_CLUSTER_DATA_DIR}")
    log(f"Train target cluster data directory: {TRAIN_TARGET_CLUSTER_DATA_DIR}")
    log(f"Val target cluster data directory: {VAL_TARGET_CLUSTER_DATA_DIR}")
    log(f"Test target cluster data directory: {TEST_TARGET_CLUSTER_DATA_DIR}")
    log(f"Model layers: {env['MODEL_NUM_LAYERS']}, split at: {env['MODEL_SPLIT_LAYER']}")
    log(f"Window size: {env['WINDOW_SIZE']} horizon: {env['FORECAST_HORIZON']} batch_size: {env['BATCH_SIZE']}")
    log(f"Train/Val/Test ratios: {env['TRAIN_RATIO']}/{env['VAL_RATIO']}/{env['TEST_RATIO']}")
    log(f"Normalization: enabled={env['NORMALIZE']} method={env['NORMALIZE_METHOD']}")
    log(f"Client LR: {env['CLIENT_LR']}")
    log(f"Server LR: {env['SERVER_LR']}")
    log(f"Configured ACCUM_STEPS_SERVER: {env['ACCUM_STEPS_SERVER']} (runtime enforces len(round_clients) per logical round)")
    log(f"Checkpoint every rounds: {env['CHECKPOINT_EVERY_ROUNDS']}")
    log(f"DQ mode: {quality_runtime['dq_mode']}")
    log(f"Quality weights enabled: {quality_runtime['use_quality_weights']}")
    log(f"Quality-as-input enabled: {quality_runtime['use_quality_as_input']}")
    log(f"Quality aggregation weights enabled: {quality_runtime['use_quality_aggregation_weights']}")
    log(f"Quality anchor aggregation enabled: {quality_runtime['use_quality_anchor_aggregation']}")
    log(f"Gradient alignment filter enabled: {quality_runtime['use_gradient_alignment_filter']}")
    log(f"Validation guardrail enabled: {quality_runtime['use_high_quality_validation_guardrail']}")
    log(f"Quality anchor top fraction: {quality_runtime['quality_anchor_top_fraction']}")
    log(f"Quality anchor minimum clients: {quality_runtime['quality_anchor_min_clients']}")
    log(f"Quality anchor update share: {quality_runtime['quality_anchor_update_share']}")
    log(f"Quality alignment power: {quality_runtime['quality_alignment_power']}")
    log(f"Quality negative alignment weight: {quality_runtime['quality_negative_alignment_weight']}")
    log(f"Quality non-anchor norm cap: {quality_runtime['quality_low_client_norm_cap']}")
    log(f"Quality aggregation mode: {quality_runtime['quality_aggregation_mode']}")
    log(f"Quality aggregation q_min: {quality_runtime['quality_aggregation_q_min']}")
    log(f"Quality aggregation eps: {quality_runtime['quality_aggregation_eps']}")
    log(f"Quality score path: {quality_runtime['quality_score_path'] or '<disabled>'}")
    log(f"Quality input columns: {quality_runtime.get('quality_input_columns', [])}")
    log(f"Quality input dim: {quality_runtime.get('quality_input_dim', 0)}")
    log(f"Quality weight column: {env['QUALITY_WEIGHT_COLUMN']}")
    log(f"Quality score granularity: {env['QUALITY_SCORE_GRANULARITY']}")
    log(f"Strict quality score matching: {env['STRICT_QUALITY_SCORE_MATCHING']}")
    if quality_runtime.get("base_input_size") is not None:
        log(f"Base input size: {quality_runtime['base_input_size']}")
    log(f"Quality feature appended: {quality_runtime['quality_feature_appended']}")
    log(f"Final MODEL_INPUT_SIZE: {quality_runtime['model_input_size']}")
    log(
        f"Communication logging: train={env['COMM_LOG_TRAIN']} eval={env['COMM_LOG_EVAL']} "
        f"stream={env['COMM_LOG_STREAM']} every_n_batches={env['COMM_LOG_EVERY_N_BATCHES']}"
    )
    log(f"Progress persistence every_n_batches={env['SAVE_PROGRESS_EVERY_N_BATCHES']}")
    log(f"Write aggregated communication_log.csv={env['WRITE_COMMUNICATION_LOG']}")
    log(f"Skip training: {env['SKIP_TRAINING']}")
    log(f"Cluster parallel local epochs: {parse_bool(env.get('RUN_CLUSTERS_IN_PARALLEL'), default=True) and len(cluster_ids) > 1}")
    log(
        f"Local epochs per global FedAvg round: {max(1, int(env['LOCAL_EPOCHS_PER_GLOBAL_ROUND']))} "
        f"(enabled={len(cluster_ids) > 1})"
    )
    log(f"STREAM_BASE_PORT={STREAM_BASE_PORT} SERVER_BASE_PORT={SERVER_BASE_PORT} CLIENT_WORKER_BASE_PORT={CLIENT_WORKER_BASE_PORT}")
    log(f"GPU_ENABLED={env.get('GPU_ENABLED', os.environ.get('GPU_ENABLED', 'true'))}")
    log(f"SERVER_GPU_ID={os.environ.get('SERVER_GPU_ID')} SERVER_GPU_IDS={os.environ.get('SERVER_GPU_IDS')}")
    log(f"CLIENT_GPU_ID={os.environ.get('CLIENT_GPU_ID')} CLIENT_GPU_IDS={os.environ.get('CLIENT_GPU_IDS')}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for cid in cluster_ids:
        log(f"Cluster {cid}: {len(cluster_clients[cid])} client(s) -> {cluster_clients[cid]}")

    started = []
    model_configs = {"runtime": quality_runtime, "servers": [], "clients": []}
    stop_event = Event()

    try:
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            ensure_dir(cluster_dir(OUTPUT_DIR, cluster_id))
            sp, sfh, stream_port = start_stream_service(cluster_id, env, cluster_idx)
            started.append((sp, sfh))
            srv, srvfh, server_port = start_split_server(cluster_id, env, cluster_idx)
            started.append((srv, srvfh))
            for client_idx, client_id in enumerate(cluster_clients[cluster_id]):
                cp, cfh, worker_port = start_client_worker(cluster_id, client_id, server_port, stream_port, env, cluster_idx, client_idx)
                started.append((cp, cfh))
                status = get_json(f"http://localhost:{worker_port}/status", timeout=20)
                log(f"[worker_ready] cluster={cluster_id} client={client_id} port={worker_port} status={status}")

        train_durations = defaultdict(float)
        if parse_bool(env.get("SKIP_TRAINING"), default=False):
            log("[TRAIN] skipped; using existing checkpoints for test evaluation")
        else:
            train_durations = run_training_schedule_with_fedavg(cluster_ids, cluster_clients, env)

            for cluster_idx, cluster_id in enumerate(cluster_ids):
                post_json(f"http://localhost:{SERVER_BASE_PORT + cluster_idx}/save_final_checkpoint", {}, timeout=30)
            ensure_final_client_checkpoints(cluster_ids, cluster_clients)

        test_durations = defaultdict(float)
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            cluster_test = run_test_phase_for_cluster(cluster_id, cluster_clients[cluster_id], cluster_idx, stop_event)
            for key, value in cluster_test.items():
                test_durations[key] += value
        if parse_bool(env.get("DRIFT_DETECTION_ENABLED"), default=False):
            drift_summary = evaluate_run_artifacts(OUTPUT_DIR, RESULTS_DIR / "drift")
            log(f"[DRIFT] artifact evaluation summary: {drift_summary}")

        # train_durations = run_training_schedule(cluster_ids, cluster_clients, env)

        # for cluster_idx, cluster_id in enumerate(cluster_ids):
        #     post_json(f"http://localhost:{SERVER_BASE_PORT + cluster_idx}/save_final_checkpoint", {}, timeout=30)
        # ensure_final_client_checkpoints(cluster_ids, cluster_clients)

        # test_durations = run_test_phase(cluster_ids, cluster_clients, env)

        for idx, cid in enumerate(cluster_ids):
            model_configs["servers"].append(collect_server_config(SERVER_BASE_PORT + idx, cid))
        for cid in cluster_ids:
            for client_id in cluster_clients[cid]:
                model_configs["clients"].append(collect_client_config(cid, client_id))
        if frozen_validation_mode:
            frozen_hashes_after = snapshot_checkpoint_hashes(OUTPUT_DIR)
            runtime_hashes = {str(item["cluster_id"]): str(item.get("parameter_hash")) for item in model_configs["servers"]}
            write_frozen_validation_report(
                RESULTS_DIR / "frozen_validation_report.json", before=frozen_hashes_before, after=frozen_hashes_after,
                server_runtime_hashes=runtime_hashes, round_start_events=0, fedavg_events=0,
            )

        train_client_rows = aggregate_train_rows(
            cluster_ids,
            cluster_clients,
            total_epochs,
            env["TRAINING_MODE"],
            env["CLIENT_ORDER_STRATEGY"],
            int(env["RANDOM_SEED_SCHEDULE"]),
            train_durations,
        )
        test_client_rows = aggregate_test_rows(
            cluster_ids,
            cluster_clients,
            env["TRAINING_MODE"],
            env["CLIENT_ORDER_STRATEGY"],
            int(env["RANDOM_SEED_SCHEDULE"]),
            test_durations,
        )

        train_cluster_summary = compute_cluster_train_summary(train_client_rows)
        test_cluster_summary = compute_cluster_test_summary(test_client_rows)
        train_global_summary = {
            "num_clients": len(train_client_rows),
            "avg_final_train_loss": safe_mean([r.get("final_train_loss") for r in train_client_rows]),
            "avg_final_val_loss": safe_mean([r.get("final_val_loss") for r in train_client_rows]),
            "avg_best_val_raw_mae": safe_mean([r.get("best_val_raw_mae") for r in train_client_rows]),
        }
        test_global_summary = {
            "num_clients": len(test_client_rows),
            "avg_mae": safe_mean([r.get("mae") for r in test_client_rows]),
            "avg_rmse": safe_mean([r.get("rmse") for r in test_client_rows]),
            "avg_mse": safe_mean([r.get("mse") for r in test_client_rows]),
            "avg_smape": safe_mean([r.get("smape") for r in test_client_rows]),
            "avg_r2": safe_mean([r.get("r2") for r in test_client_rows]),
        }

        train_report = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_epochs": total_epochs,
            "training_mode": env["TRAINING_MODE"],
            "client_order_strategy": env["CLIENT_ORDER_STRATEGY"],
            "random_seed_schedule": int(env["RANDOM_SEED_SCHEDULE"]),
            "eval_mode": env["EVAL_MODE"],
            "model_num_layers": int(env["MODEL_NUM_LAYERS"]),
            "model_split_layer": int(env["MODEL_SPLIT_LAYER"]),
            **quality_runtime,
            "clients": train_client_rows,
            "clusters": train_cluster_summary,
            "global_summary": train_global_summary,
        }
        test_report = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_epochs": total_epochs,
            "training_mode": env["TRAINING_MODE"],
            "client_order_strategy": env["CLIENT_ORDER_STRATEGY"],
            "random_seed_schedule": int(env["RANDOM_SEED_SCHEDULE"]),
            "eval_mode": env["EVAL_MODE"],
            "model_num_layers": int(env["MODEL_NUM_LAYERS"]),
            "model_split_layer": int(env["MODEL_SPLIT_LAYER"]),
            **quality_runtime,
            "clients": test_client_rows,
            "clusters": test_cluster_summary,
            "global_summary": test_global_summary,
        }
        write_json(RESULTS_DIR / "train_report.json", train_report)
        write_json(RESULTS_DIR / "test_report.json", test_report)
        write_json(RESULTS_DIR / "model_configs.json", model_configs)
        if train_client_rows:
            write_csv(RESULTS_DIR / "train_report.csv", train_client_rows, list(train_client_rows[0].keys()))
        if test_client_rows:
            write_csv(RESULTS_DIR / "test_report.csv", test_client_rows, list(test_client_rows[0].keys()))
        comm_raw_rows, comm_report_rows, fedavg_report_rows = build_communication_reports(cluster_ids, cluster_clients)
        if comm_raw_rows and parse_bool(env.get("WRITE_COMMUNICATION_LOG", "true"), default=True):
            write_csv(RESULTS_DIR / "communication_log.csv", comm_raw_rows, COMM_LOG_FIELDNAMES)
        if comm_report_rows:
            write_csv(RESULTS_DIR / "communication_report.csv", comm_report_rows, list(comm_report_rows[0].keys()))
            write_json(RESULTS_DIR / "communication_report.json", {"rows": comm_report_rows})
        if fedavg_report_rows:
            write_csv(RESULTS_DIR / "fedavg_report.csv", fedavg_report_rows, list(fedavg_report_rows[0].keys()))
            write_json(RESULTS_DIR / "fedavg_report.json", {"rows": fedavg_report_rows})

        log("\nTRAIN SUMMARY")
        for row in train_cluster_summary:
            log(str(row))
        log(str(train_global_summary))
        log("\nTEST SUMMARY")
        for row in test_cluster_summary:
            log(str(row))
        log(str(test_global_summary))

    finally:
        for proc, fh in reversed(started):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass

    elapsed = round(time.time() - start_time, 2)
    log(f"\nFinished in {elapsed} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
