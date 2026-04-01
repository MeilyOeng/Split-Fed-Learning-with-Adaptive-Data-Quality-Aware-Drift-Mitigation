#!/usr/bin/env python3
"""
Run split learning end-to-end with scheduler support.

Supported training modes:
- client_full
- client_epoch_ordered
- client_epoch_random
- client_batch_random

Required client commands:
- train
- train_one_epoch
- train_one_batch
- test
"""

import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
import shutil


CLIENT_PATH = "client/client.py"
STREAM_SERVICE_PATH = "data/stream_service.py"
SERVER_PATH = "server/server.py"

ASSIGNMENTS_PATH = Path("dataset") / "cluster-assignment.csv"
OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "results"

STREAM_BASE_PORT = 5101
SERVER_BASE_PORT = 5002
CLIENT_METRICS_BASE = 9100


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cluster_dir(cluster_id: str) -> Path:
    return OUTPUT_DIR / f"cluster_{cluster_id}"


def parse_csv_ids(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def wait_for_service(url: str, max_attempts: int = 60, delay_sec: float = 1.0) -> bool:
    for _ in range(max_attempts):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay_sec)
    return False


def load_cluster_ids(path: Path):
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


def load_client_ids_for_cluster(cluster_id: str, path: Path):
    if not path.exists():
        return []
    client_ids = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = str(row.get("Cluster", "")).strip()
            if cid == str(cluster_id):
                client_id = str(row.get("ID", "")).strip()
                if client_id:
                    client_ids.append(client_id)
    return client_ids


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_optional_csv(raw: str | None):
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
        if client_idx is None:
            client_idx = 0
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
    out["GPU_ENABLED"] = os.environ.get("GPU_ENABLED", out.get("GPU_ENABLED", "true"))
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


def start_stream_service(cluster_id: str, port: int, base_env: dict, cluster_idx: int):
    cdir = cluster_dir(cluster_id)
    ensure_dir(cdir)

    env = base_env.copy()
    env.update(
        {
            "STREAM_CLUSTER_ID": str(cluster_id),
            "CLUSTER_ID": str(cluster_id),
            "STREAM_PORT": str(port),
            "PROCESS_ROLE": f"stream_cluster_{cluster_idx}",
            "GPU_ENABLED": "false",
        }
    )

    log_path = cdir / "stream_service.log"
    fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, STREAM_SERVICE_PATH],
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
    )
    ok = wait_for_service(f"http://localhost:{port}/health")
    if not ok:
        fh.close()
        raise RuntimeError(f"Stream service for cluster {cluster_id} did not start on port {port}")
    return proc, fh


def start_split_server(cluster_id: str, port: int, base_env: dict, cluster_idx: int):
    cdir = cluster_dir(cluster_id)
    ensure_dir(cdir)

    env, gpu_meta = apply_gpu_env(base_env, role="server", cluster_idx=cluster_idx)
    env.update(
        {
            "CLUSTER_ID": str(cluster_id),
            "SERVER_PORT": str(port),
            "OUTPUT_DIR": str(cdir),
        }
    )

    log(f"[launch] server cluster={cluster_id} port={port} gpu_id={gpu_meta['gpu_id']} CUDA_VISIBLE_DEVICES={gpu_meta['cuda_visible_devices']}")

    log_path = cdir / "server.log"
    fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, SERVER_PATH],
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
    )
    ok = wait_for_service(f"http://localhost:{port}/healthz")
    if not ok:
        fh.close()
        raise RuntimeError(f"Split server for cluster {cluster_id} did not start on port {port}")
    return proc, fh


def run_client_command(
    cluster_id: str,
    client_id: str,
    server_port: int,
    stream_port: int,
    metrics_port: int,
    base_env: dict,
    command: str,
    cluster_idx: int,
    client_global_idx: int,
):
    cdir = cluster_dir(cluster_id)
    ensure_dir(cdir)

    env, gpu_meta = apply_gpu_env(base_env, role="client", cluster_idx=cluster_idx, client_idx=client_global_idx)
    env.update(
        {
            "CLIENT_ID": str(client_id),
            "CLUSTER_ID": str(cluster_id),
            "SERVER_PORT": str(server_port),
            "DATA_STREAM_API_URL": f"http://localhost:{stream_port}",
            "CLIENT_METRICS_PORT": str(metrics_port),
        }
    )

    log(f"[launch] client cluster={cluster_id} client={client_id} command={command} gpu_id={gpu_meta['gpu_id']} CUDA_VISIBLE_DEVICES={gpu_meta['cuda_visible_devices']}")

    log_path = cdir / f"client_{client_id}_{command}.log"
    start = time.time()
    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, CLIENT_PATH, command],
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )
    return proc.returncode, round(time.time() - start, 3), log_path


def tail_text_file(path: Path, n: int = 120) -> str:
    if not path.exists():
        return f"<missing log: {path}>"
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def dump_failure_context(cluster_id: str, client_id: str, command: str) -> None:
    cdir = cluster_dir(cluster_id)
    log("\n===== CLIENT LOG (tail) =====")
    log(tail_text_file(cdir / f"client_{client_id}_{command}.log"))
    log("===== SERVER LOG (tail) =====")
    log(tail_text_file(cdir / "server.log"))
    log("===== STREAM LOG (tail) =====")
    log(tail_text_file(cdir / "stream_service.log"))


def collect_server_config(server_port: int, cluster_id: str):
    config = {"cluster_id": cluster_id, "server_port": server_port}
    try:
        resp = requests.get(f"http://localhost:{server_port}/config", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                config.update(data)
    except Exception:
        pass
    return config


def collect_client_config(cluster_id: str, client_id: str):
    cdir = cluster_dir(cluster_id)
    config_path = cdir / f"client_{client_id}_config.json"
    config = {
        "cluster_id": cluster_id,
        "client_id": client_id,
        "checkpoint_path": str(cdir / f"client_{client_id}_checkpoint.pt"),
        "best_checkpoint_path": str(cdir / f"client_{client_id}_best_checkpoint.pt"),
        "final_checkpoint_path": str(cdir / f"client_{client_id}_final.pt"),
    }
    if config_path.exists():
        saved = read_json(config_path)
        if isinstance(saved, dict):
            config.update(saved)
    return config


def progress_path(cluster_id: str, client_id: str) -> Path:
    return cluster_dir(cluster_id) / f"client_{client_id}_progress.json"


def train_metrics_path(cluster_id: str, client_id: str) -> Path:
    return cluster_dir(cluster_id) / f"client_{client_id}_train_metrics.json"


def test_metrics_path(cluster_id: str, client_id: str) -> Path:
    return cluster_dir(cluster_id) / f"client_{client_id}_test_metrics.json"


def step_metrics_path(cluster_id: str, client_id: str) -> Path:
    return cluster_dir(cluster_id) / f"client_{client_id}_step_metrics.json"


def client_done(cluster_id: str, client_id: str, total_epochs: int) -> bool:
    p = progress_path(cluster_id, client_id)
    if not p.exists():
        return False
    try:
        data = read_json(p)
    except Exception:
        return False
    completed_epochs = int(data.get("completed_epochs", 0))
    early_stopped = bool(data.get("early_stopped", False))
    return early_stopped or completed_epochs >= total_epochs


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
                "avg_mape": safe_mean([r.get("mape") for r in items]),
            }
        )
    return sorted(out, key=lambda x: int(x["cluster_id"]) if str(x["cluster_id"]).isdigit() else str(x["cluster_id"]))


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
                    "client_best_checkpoint_path": metrics.get("client_best_checkpoint_path"),
                    "client_final_checkpoint_path": metrics.get("client_final_checkpoint_path"),
                    "server_best_checkpoint_path": metrics.get("server_best_checkpoint_path"),
                    "server_final_checkpoint_path": metrics.get("server_final_checkpoint_path"),
                    "selection_metric": metrics.get("selection_metric"),
                    "train_loss_name": metrics.get("train_loss_name"),
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
                    "mape": raw.get("mape"),
                    "num_test_batches": metrics.get("num_batches"),
                    "num_test_samples": metrics.get("num_samples"),
                    "eval_mode": metrics.get("eval_mode"),
                }
            )
    return rows


def run_training_schedule(cluster_ids, cluster_clients, env):
    training_mode = env["TRAINING_MODE"]
    total_epochs = int(env["NUM_EPOCHS"])
    schedule_seed = int(env["RANDOM_SEED_SCHEDULE"])
    rng = random.Random(schedule_seed)
    global_client_idx = 0
    train_durations = defaultdict(float)

    if training_mode == "client_full":
        for c_idx, cid in enumerate(cluster_ids):
            server_port = SERVER_BASE_PORT + c_idx
            stream_port = STREAM_BASE_PORT + c_idx
            for client_id in cluster_clients[cid]:
                metrics_port = CLIENT_METRICS_BASE + global_client_idx
                if client_done(cid, client_id, total_epochs):
                    log(f"[TRAIN][client_full] cluster={cid} client={client_id} already done; skipping")
                    continue
                rc, dur, _ = run_client_command(
                    cid, client_id, server_port, stream_port, metrics_port,
                    env, "train", c_idx, global_client_idx
                )
                global_client_idx += 1
                train_durations[(cid, client_id)] += dur
                if rc != 0:
                    dump_failure_context(cid, client_id, "train")
                    raise RuntimeError(f"Training failed for cluster={cid}, client={client_id}, exit_code={rc}")
                log(f"[TRAIN][client_full] cluster={cid} client={client_id} done in {dur:.3f}s")
        return train_durations

    if training_mode in {"client_epoch_ordered", "client_epoch_random"}:
        command = "train_one_epoch"
        for scheduler_epoch in range(1, total_epochs + 1):
            log(f"\n=== Scheduler epoch {scheduler_epoch}/{total_epochs} ===")
            any_progress = False
            for c_idx, cid in enumerate(cluster_ids):
                server_port = SERVER_BASE_PORT + c_idx
                stream_port = STREAM_BASE_PORT + c_idx
                order = list(cluster_clients[cid])
                if training_mode == "client_epoch_random":
                    rng.shuffle(order)
                for client_id in order:
                    if client_done(cid, client_id, total_epochs):
                        continue
                    metrics_port = CLIENT_METRICS_BASE + global_client_idx
                    rc, dur, _ = run_client_command(
                        cid, client_id, server_port, stream_port, metrics_port,
                        env, command, c_idx, global_client_idx
                    )
                    global_client_idx += 1
                    train_durations[(cid, client_id)] += dur
                    if rc != 0:
                        dump_failure_context(cid, client_id, command)
                        raise RuntimeError(
                            f"Epoch-wise training failed for cluster={cid}, client={client_id}, scheduler_epoch={scheduler_epoch}, exit_code={rc}"
                        )
                    log(f"[TRAIN][{training_mode}] scheduler_epoch={scheduler_epoch} cluster={cid} client={client_id} done in {dur:.3f}s")
                    any_progress = True
            if not any_progress:
                log(f"[TRAIN][{training_mode}] no remaining active clients at scheduler_epoch={scheduler_epoch}; stopping early")
                break
        return train_durations

    if training_mode == "client_batch_random":
        for scheduler_epoch in range(1, total_epochs + 1):
            log(f"\n=== Scheduler batch-epoch {scheduler_epoch}/{total_epochs} ===")
            cluster_done = {cid: False for cid in cluster_ids}

            while not all(cluster_done.values()):
                for c_idx, cid in enumerate(cluster_ids):
                    if cluster_done[cid]:
                        continue

                    active_clients = [client_id for client_id in cluster_clients[cid] if not client_done(cid, client_id, total_epochs)]
                    if not active_clients:
                        cluster_done[cid] = True
                        continue

                    server_port = SERVER_BASE_PORT + c_idx
                    stream_port = STREAM_BASE_PORT + c_idx
                    rng.shuffle(active_clients)
                    progress_this_round = False

                    for client_id in active_clients:
                        metrics_port = CLIENT_METRICS_BASE + global_client_idx
                        rc, dur, _ = run_client_command(
                            cid, client_id, server_port, stream_port, metrics_port,
                            env, "train_one_batch", c_idx, global_client_idx
                        )
                        global_client_idx += 1
                        train_durations[(cid, client_id)] += dur

                        if rc != 0:
                            dump_failure_context(cid, client_id, "train_one_batch")
                            raise RuntimeError(
                                f"Batch-wise training failed for cluster={cid}, client={client_id}, scheduler_epoch={scheduler_epoch}, exit_code={rc}"
                            )

                        step_path = step_metrics_path(cid, client_id)
                        if step_path.exists():
                            step = read_json(step_path)
                            batch_processed = bool(step.get("batch_processed", False))
                            epoch_completed = bool(step.get("epoch_completed", False))
                            done_all_epochs = bool(step.get("done_all_epochs", False))
                            if batch_processed or epoch_completed:
                                progress_this_round = True
                            log(
                                f"[TRAIN][client_batch_random] scheduler_epoch={scheduler_epoch} cluster={cid} client={client_id} "
                                f"batch_processed={batch_processed} epoch_completed={epoch_completed} "
                                f"done_all_epochs={done_all_epochs} dur={dur:.3f}s"
                            )
                        else:
                            progress_this_round = True
                            log(
                                f"[TRAIN][client_batch_random] scheduler_epoch={scheduler_epoch} cluster={cid} client={client_id} "
                                f"dur={dur:.3f}s (no step metrics file)"
                            )

                    all_reached = True
                    for client_id in cluster_clients[cid]:
                        path = progress_path(cid, client_id)
                        if not path.exists():
                            all_reached = False
                            break
                        prog = read_json(path)
                        completed_epochs = int(prog.get("completed_epochs", 0))
                        early_stopped = bool(prog.get("early_stopped", False))
                        if not early_stopped and completed_epochs < scheduler_epoch:
                            all_reached = False
                            break

                    if all_reached:
                        cluster_done[cid] = True
                    elif not progress_this_round:
                        raise RuntimeError(f"No batch progress detected for cluster={cid} during scheduler_epoch={scheduler_epoch}")
        return train_durations

    raise ValueError(f"Unsupported TRAINING_MODE: {training_mode}")


def save_final_server_checkpoints(cluster_ids):
    for idx, cid in enumerate(cluster_ids):
        resp = requests.post(f"http://localhost:{SERVER_BASE_PORT + idx}/save_final_checkpoint", timeout=20)
        resp.raise_for_status()


def run_test_phase(cluster_ids, cluster_clients, env):
    global_client_idx = 100000
    test_durations = defaultdict(float)

    for c_idx, cid in enumerate(cluster_ids):
        server_port = SERVER_BASE_PORT + c_idx
        stream_port = STREAM_BASE_PORT + c_idx
        for client_id in cluster_clients[cid]:
            metrics_port = CLIENT_METRICS_BASE + global_client_idx
            rc, dur, _ = run_client_command(
                cid, client_id, server_port, stream_port, metrics_port,
                env, "test", c_idx, global_client_idx
            )
            global_client_idx += 1
            test_durations[(cid, client_id)] += dur
            if rc != 0:
                dump_failure_context(cid, client_id, "test")
                raise RuntimeError(f"Testing failed for cluster={cid}, client={client_id}, exit_code={rc}")
            log(f"[TEST] cluster={cid} client={client_id} done in {dur:.3f}s")

    return test_durations

def ensure_final_client_checkpoints(cluster_ids, cluster_clients):
    for cid in cluster_ids:
        for client_id in cluster_clients[cid]:
            latest = cluster_dir(cid) / f"client_{client_id}_checkpoint.pt"
            final = cluster_dir(cid) / f"client_{client_id}_final.pt"
            if (not final.exists()) and latest.exists():
                shutil.copy2(latest, final)
                log(f"[finalize] cluster={cid} client={client_id} copied latest -> final")

def main():
    start_time = time.time()
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
            "LEARNING_RATE": os.environ.get("LEARNING_RATE", "0.0005"),
            "RANDOM_SEED": os.environ.get("RANDOM_SEED", "42"),
            "CHECKPOINT_EVERY": os.environ.get("CHECKPOINT_EVERY", "10"),
            "TRAINING_MODE": os.environ.get("TRAINING_MODE", "client_full"),
            "CLIENT_ORDER_STRATEGY": os.environ.get("CLIENT_ORDER_STRATEGY", "fixed"),
            "RANDOM_SEED_SCHEDULE": os.environ.get("RANDOM_SEED_SCHEDULE", os.environ.get("RANDOM_SEED", "42")),
            "EVAL_MODE": os.environ.get("EVAL_MODE", "final_shared"),
            "EARLY_STOPPING": os.environ.get("EARLY_STOPPING", "true"),
            "EARLY_STOPPING_PATIENCE": os.environ.get("EARLY_STOPPING_PATIENCE", "15"),
            "EARLY_STOPPING_MIN_DELTA": os.environ.get("EARLY_STOPPING_MIN_DELTA", "0.00001"),
            "EARLY_STOPPING_METRIC": os.environ.get("EARLY_STOPPING_METRIC", "val_raw_mae"),
        }
    )

    total_epochs = int(env["NUM_EPOCHS"])
    detected_clusters = load_cluster_ids(ASSIGNMENTS_PATH)
    if not detected_clusters:
        log("ERROR: No clusters found in dataset/cluster-assignment.csv")
        return 1

    cluster_ids, cluster_clients = resolve_cluster_clients(detected_clusters)

    log("Starting split-learning run")
    log(f"Epochs: {total_epochs}")
    log(f"Training mode: {env['TRAINING_MODE']}")
    log(f"Client order strategy: {env['CLIENT_ORDER_STRATEGY']}")
    log(f"Schedule seed: {env['RANDOM_SEED_SCHEDULE']}")
    log(f"Eval mode: {env['EVAL_MODE']}")
    log(f"Model layers: {env['MODEL_NUM_LAYERS']}, split at: {env['MODEL_SPLIT_LAYER']}")
    log(f"GPU_ENABLED={env.get('GPU_ENABLED', os.environ.get('GPU_ENABLED', 'true'))}")
    log(f"SERVER_GPU_ID={os.environ.get('SERVER_GPU_ID')} SERVER_GPU_IDS={os.environ.get('SERVER_GPU_IDS')}")
    log(f"CLIENT_GPU_ID={os.environ.get('CLIENT_GPU_ID')} CLIENT_GPU_IDS={os.environ.get('CLIENT_GPU_IDS')}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    processes = []
    model_configs = {"servers": [], "clients": []}

    try:
        for idx, cid in enumerate(cluster_ids):
            stream_port = STREAM_BASE_PORT + idx
            server_port = SERVER_BASE_PORT + idx
            stream_proc, stream_fh = start_stream_service(cid, stream_port, env, idx)
            processes.append(("stream", cid, stream_proc, stream_fh))
            server_proc, server_fh = start_split_server(cid, server_port, env, idx)
            processes.append(("server", cid, server_proc, server_fh))

        time.sleep(1.0)

        train_durations = run_training_schedule(cluster_ids, cluster_clients, env)
        save_final_server_checkpoints(cluster_ids)
        ensure_final_client_checkpoints(cluster_ids, cluster_clients)
        test_durations = run_test_phase(cluster_ids, cluster_clients, env)

        for idx, cid in enumerate(cluster_ids):
            model_configs["servers"].append(collect_server_config(SERVER_BASE_PORT + idx, cid))
        for cid in cluster_ids:
            for client_id in cluster_clients[cid]:
                model_configs["clients"].append(collect_client_config(cid, client_id))

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
            "avg_mape": safe_mean([r.get("mape") for r in test_client_rows]),
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

        log("\nTRAIN SUMMARY")
        for row in train_cluster_summary:
            log(str(row))
        log(str(train_global_summary))

        log("\nTEST SUMMARY")
        for row in test_cluster_summary:
            log(str(row))
        log(str(test_global_summary))

    finally:
        for _, _, proc, fh in reversed(processes):
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