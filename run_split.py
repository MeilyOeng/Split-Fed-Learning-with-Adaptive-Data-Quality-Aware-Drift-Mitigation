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
from typing import Dict, List

import requests
from dotenv import find_dotenv, load_dotenv
from fed_server.fed_server import run_fedavg_round

CLIENT_PATH = "client/client.py"
STREAM_SERVICE_PATH = "data/stream_service.py"
SERVER_PATH = "server/server.py"
ASSIGNMENTS_PATH = Path("dataset") / "cluster-assignment.csv"
OUTPUT_DIR = Path("output")
RESULTS_DIR = OUTPUT_DIR / "results"
STREAM_BASE_PORT = 5101
SERVER_BASE_PORT = 5002
CLIENT_WORKER_BASE_PORT = 6101


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cluster_dir(cluster_id: str) -> Path:
    return OUTPUT_DIR / f"cluster_{cluster_id}"


def parse_csv_ids(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_env_file() -> Path | None:
    dotenv_path = find_dotenv(".env", usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return Path(dotenv_path)
    fallback = Path(".env")
    if fallback.exists():
        load_dotenv(fallback, override=False)
        return fallback.resolve()
    return None


def refresh_runtime_settings_from_env() -> None:
    global STREAM_BASE_PORT, SERVER_BASE_PORT, CLIENT_WORKER_BASE_PORT
    STREAM_BASE_PORT = int(os.environ.get("STREAM_BASE_PORT", str(STREAM_BASE_PORT)))
    SERVER_BASE_PORT = int(os.environ.get("SERVER_BASE_PORT", str(SERVER_BASE_PORT)))
    CLIENT_WORKER_BASE_PORT = int(os.environ.get("CLIENT_WORKER_BASE_PORT", str(CLIENT_WORKER_BASE_PORT)))


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


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows, fieldnames) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    cdir = cluster_dir(cluster_id)
    ensure_dir(cdir)
    log_path = cdir / "stream_service.log"
    log_fh = log_path.open("w", encoding="utf-8")
    proc_env = env.copy()
    proc_env.update(
        {
            "STREAM_CLUSTER_ID": str(cluster_id),
            "CLUSTER_ID": str(cluster_id),
            "STREAM_PORT": str(port),
            "PROCESS_ROLE": f"stream_cluster_{cluster_idx}",
            "GPU_ENABLED": "false",
        }
    )
    log(f"[launch] stream cluster={cluster_id} port={port}")
    proc = subprocess.Popen([sys.executable, STREAM_SERVICE_PATH], env=proc_env, stdout=log_fh, stderr=subprocess.STDOUT, cwd=os.getcwd())
    if not wait_for_service(f"http://localhost:{port}/health"):
        raise RuntimeError(f"Stream service failed to start for cluster {cluster_id}")
    return proc, log_fh, port


def start_split_server(cluster_id: str, env: dict, cluster_idx: int):
    port = SERVER_BASE_PORT + cluster_idx
    cdir = cluster_dir(cluster_id)
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
    cdir = cluster_dir(cluster_id)
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
    cdir = cluster_dir(cluster_id)
    if client_id is not None:
        log("\n===== CLIENT WORKER LOG (tail) =====")
        log(tail_text_file(cdir / f"client_{client_id}_worker.log"))
    log("===== SERVER LOG (tail) =====")
    log(tail_text_file(cdir / "server.log"))
    log("===== STREAM LOG (tail) =====")
    log(tail_text_file(cdir / "stream_service.log"))


def post_json(url: str, payload: dict | None = None, timeout: float = 120.0):
    resp = requests.post(url, json=payload or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_json(url: str, timeout: float = 30.0):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def progress_path(cluster_id: str, client_id: str) -> Path:
    return cluster_dir(cluster_id) / f"client_{client_id}_progress.json"


def train_metrics_path(cluster_id: str, client_id: str) -> Path:
    return cluster_dir(cluster_id) / f"client_{client_id}_train_metrics.json"


def test_metrics_path(cluster_id: str, client_id: str) -> Path:
    return cluster_dir(cluster_id) / f"client_{client_id}_test_metrics.json"


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
    cdir = cluster_dir(cluster_id)
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
            latest = cluster_dir(cid) / f"client_{client_id}_checkpoint.pt"
            final = cluster_dir(cid) / f"client_{client_id}_final.pt"
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
    """
    Finalize clients that have already been identified as epoch-exhausted.

    Important: do NOT call /command/train_one_batch here. Those clients are at an epoch
    boundary already, so asking them to train again can re-enter the train path, trigger
    validation side-effects at the wrong time, or surface a 500 from a worker that is no
    longer expected to contribute a batch.

    Instead, call the dedicated finalize endpoint. For compatibility with an older worker,
    fall back to /command/train_one_batch only if /command/finalize_epoch returns 404.
    """
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


def run_round(server_port: int, cluster_id: str, client_ports: Dict[str, int], round_clients: List[str], scheduler_epoch: int, round_idx: int, round_chunk_idx: int, train_durations):
    numeric_round_id = int(round_idx * 1000 + round_chunk_idx)
    round_label = f"epoch{scheduler_epoch}_round{round_idx}_chunk{round_chunk_idx}"
    log(f"[ROUND][start] cluster={cluster_id} scheduler_epoch={scheduler_epoch} round={round_idx} chunk={round_chunk_idx} round_id={round_label} active_clients={round_clients}")
    post_json(f"http://localhost:{server_port}/round/start", {"round_id": numeric_round_id, "expected_clients": round_clients})

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
    if state.get("round_contrib_count", 0) != 0 or state.get("pending_accum_steps", 0) != 0:
        raise RuntimeError(f"Round did not flush cleanly: {state}")
    return results


def run_single_scheduler_epoch_for_cluster(cluster_id: str, client_ids: List[str], env: dict, cluster_idx: int, scheduler_epoch: int, stop_event: Event):
    training_mode = env.get("TRAINING_MODE", "client_batch_random").strip().lower()
    if training_mode != "client_batch_random":
        raise ValueError("This persistent-worker scheduler supports TRAINING_MODE=client_batch_random only")

    rng = random.Random(int(env.get("RANDOM_SEED_SCHEDULE", env.get("RANDOM_SEED", "42"))) + cluster_idx)
    order_strategy = env.get("CLIENT_ORDER_STRATEGY", "random_per_round")
    accum_steps_server = max(1, int(env.get("ACCUM_STEPS_SERVER", "1")))
    server_port = SERVER_BASE_PORT + cluster_idx
    client_ports = {client_id: CLIENT_WORKER_BASE_PORT + cluster_idx * 100 + idx for idx, client_id in enumerate(client_ids)}
    train_durations = defaultdict(float)

    def _ensure_not_cancelled():
        if stop_event.is_set():
            raise RuntimeError(f"Cluster {cluster_id} cancelled due to failure in another cluster")

    _ensure_not_cancelled()
    total_epochs = int(env.get("NUM_EPOCHS", "100"))
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
            round_chunks = chunk_round_clients(ordered_ready, accum_steps_server)
            progress_this_cycle = False
            for round_chunk_idx, round_clients in enumerate(round_chunks, start=1):
                round_idx += 1
                results = run_round(server_port, cluster_id, client_ports, round_clients, scheduler_epoch, round_idx, round_chunk_idx, train_durations)
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

        if len(cluster_ids) > 1:
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
        }
    )

    total_epochs = int(env["NUM_EPOCHS"])
    detected_clusters = load_cluster_ids(ASSIGNMENTS_PATH)
    if not detected_clusters:
        log("ERROR: No clusters found in dataset/cluster-assignment.csv")
        return 1

    cluster_ids, cluster_clients = resolve_cluster_clients(detected_clusters)

    log("Starting split-learning run")
    log(f".env loaded: {dotenv_path if dotenv_path is not None else '<not found>'} (system env overrides .env)")
    log(f"Epochs: {total_epochs}")
    log(f"Training mode: {env['TRAINING_MODE']}")
    log(f"Client order strategy: {env['CLIENT_ORDER_STRATEGY']}")
    log(f"Schedule seed: {env['RANDOM_SEED_SCHEDULE']}")
    log(f"Eval mode: {env['EVAL_MODE']}")
    log(f"Model layers: {env['MODEL_NUM_LAYERS']}, split at: {env['MODEL_SPLIT_LAYER']}")
    log(f"Window size: {env['WINDOW_SIZE']} horizon: {env['FORECAST_HORIZON']} batch_size: {env['BATCH_SIZE']}")
    log(f"Train/Val/Test ratios: {env['TRAIN_RATIO']}/{env['VAL_RATIO']}/{env['TEST_RATIO']}")
    log(f"Normalization: enabled={env['NORMALIZE']} method={env['NORMALIZE_METHOD']}")
    log(f"Client LR: {env['CLIENT_LR']}")
    log(f"Server LR: {env['SERVER_LR']}")
    log(f"Server round chunk size (ACCUM_STEPS_SERVER): {env['ACCUM_STEPS_SERVER']}")
    log(f"Checkpoint every rounds: {env['CHECKPOINT_EVERY_ROUNDS']}")
    log(f"Cluster parallel local epochs: {parse_bool(env.get('RUN_CLUSTERS_IN_PARALLEL'), default=True) and len(cluster_ids) > 1}")
    log(f"FedAvg after each local epoch: {len(cluster_ids) > 1}")
    log(f"STREAM_BASE_PORT={STREAM_BASE_PORT} SERVER_BASE_PORT={SERVER_BASE_PORT} CLIENT_WORKER_BASE_PORT={CLIENT_WORKER_BASE_PORT}")
    log(f"GPU_ENABLED={env.get('GPU_ENABLED', os.environ.get('GPU_ENABLED', 'true'))}")
    log(f"SERVER_GPU_ID={os.environ.get('SERVER_GPU_ID')} SERVER_GPU_IDS={os.environ.get('SERVER_GPU_IDS')}")
    log(f"CLIENT_GPU_ID={os.environ.get('CLIENT_GPU_ID')} CLIENT_GPU_IDS={os.environ.get('CLIENT_GPU_IDS')}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for cid in cluster_ids:
        log(f"Cluster {cid}: {len(cluster_clients[cid])} client(s) -> {cluster_clients[cid]}")

    started = []
    model_configs = {"servers": [], "clients": []}
    stop_event = Event()

    try:
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            ensure_dir(cluster_dir(cluster_id))
            sp, sfh, stream_port = start_stream_service(cluster_id, env, cluster_idx)
            started.append((sp, sfh))
            srv, srvfh, server_port = start_split_server(cluster_id, env, cluster_idx)
            started.append((srv, srvfh))
            for client_idx, client_id in enumerate(cluster_clients[cluster_id]):
                cp, cfh, worker_port = start_client_worker(cluster_id, client_id, server_port, stream_port, env, cluster_idx, client_idx)
                started.append((cp, cfh))
                status = get_json(f"http://localhost:{worker_port}/status", timeout=20)
                log(f"[worker_ready] cluster={cluster_id} client={client_id} port={worker_port} status={status}")

        train_durations = run_training_schedule_with_fedavg(cluster_ids, cluster_clients, env)

        for cluster_idx, cluster_id in enumerate(cluster_ids):
            post_json(f"http://localhost:{SERVER_BASE_PORT + cluster_idx}/save_final_checkpoint", {}, timeout=30)
        ensure_final_client_checkpoints(cluster_ids, cluster_clients)

        test_durations = defaultdict(float)
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            cluster_test = run_test_phase_for_cluster(cluster_id, cluster_clients[cluster_id], cluster_idx, stop_event)
            for key, value in cluster_test.items():
                test_durations[key] += value
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
