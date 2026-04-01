#!/usr/bin/env python3
"""
run_training.py — orchestrator (fed server HTTP mode).

CHANGE FROM PREVIOUS VERSION:
  FedAvg is no longer triggered by subprocess.run(federated.py).
  It is triggered by POST /fedavg on the always-running fed_server.py.

  BEFORE:  subprocess.run(["python","federated.py","--clusters",...])
  AFTER:   requests.post("http://localhost:5010/fedavg", json={...})

  The fed server is started in [3/4] alongside stream services and cluster
  servers, then polled at GET /healthz before training begins.

  CLIENT WEIGHT DISTRIBUTION:
    After each FedAvg round the fed server writes:
      output/global/global_client_latest.pt
    Each client subprocess reads this at startup (_maybe_load_checkpoint).
    No HTTP push to clients is needed — they are subprocesses without
    their own HTTP server.
"""

import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

CLIENT_PATH = "client/client.py"
CLIENT_METRICS_BASE_PORT = 9100
LOG_FILE                 = None


class Colors:
    GREEN  = '\033[0;32m'
    BLUE   = '\033[0;34m'
    RED    = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC     = '\033[0m'


def print_colored(msg, color=Colors.NC):
    print(f"{color}{msg}{Colors.NC}")
    if LOG_FILE:
        LOG_FILE.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")
        LOG_FILE.flush()


def check_port(port, timeout=5):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    result = s.connect_ex(("localhost", port))
    s.close()
    return result == 0


def wait_for_service(url, max_attempts=30, delay=1):
    for _ in range(max_attempts):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def trigger_fedavg(fed_server_url, cluster_ids, server_base_port, epoch):
    """
    POST /fedavg to the fed server.

    Body fields tell the fed server:
      - which clusters to aggregate
      - where the running cluster servers are (for /save_checkpoint and /load_weights)
      - where checkpoint files live on disk
      - where to write the global model files

    The call is SYNCHRONOUS — it blocks until the round is fully done,
    meaning the next epoch does not start until FedAvg is complete and all
    servers have received the averaged weights.
    """
    server_urls = [f"http://localhost:{server_base_port+i}" for i in range(len(cluster_ids))]
    payload = {
        "cluster_ids":        cluster_ids,
        "server_urls":        server_urls,
        "checkpoint_dir":     "output",
        "output_path":        "output/global/global_server_latest.pt",
        "client_output_path": "output/global/global_client_latest.pt",
        "assignments_path":   "dataset/cluster-assignment.csv",
        "aggregate_clients":  True,
    }
    url = fed_server_url.rstrip("/") + "/fedavg"
    try:
        print_colored(f"  [FedAvg epoch {epoch}] POST {url}", Colors.BLUE)
        resp = requests.post(url, json=payload, timeout=120.0)
        if resp.status_code == 429:
            print_colored("  [FedAvg] round already in progress", Colors.YELLOW)
            return False
        resp.raise_for_status()
        body       = resp.json()
        r          = body.get("round", {})
        push_ok    = r.get("server_push_ok", [])
        push_fail  = r.get("server_push_fail", [])
        duration   = r.get("duration_sec", "?")
        client_out = r.get("client_output")
        print_colored(
            f"  [FedAvg] ✓ round {r.get('round_num','?')} done in {duration}s "
            f"servers updated: {push_ok}", Colors.GREEN)
        if push_fail:
            print_colored(f"  [FedAvg] ⚠ push failed: {push_fail}", Colors.YELLOW)
        if client_out:
            print_colored(f"  [FedAvg] global client → {client_out}", Colors.GREEN)
        return True
    except requests.HTTPError as e:
        detail = {}
        try: detail = e.response.json()
        except Exception: pass
        print_colored(f"  [FedAvg] ✗ HTTP {e.response.status_code}: {detail.get('error','')}", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"  [FedAvg] ✗ {e}", Colors.RED)
        return False


def load_cluster_ids(path):
    if not path.exists(): return []
    ids = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = str(row.get("Cluster","")).strip()
            if c: ids.add(c)
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def load_client_ids_for_cluster(cluster_id, path):
    if not path.exists(): return []
    ids = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("Cluster","")).strip() == str(cluster_id):
                cid = str(row.get("ID","")).strip()
                if cid: ids.append(cid)
    return ids


def _parse_ids(raw):
    return [c.strip() for c in raw.split(",") if c.strip()]


def _cluster_dir(cid):
    return Path("output") / f"cluster_{cid}"


def main():
    print_colored("="*60, Colors.BLUE)
    print_colored("  Split-Fed Learning — fed server HTTP mode", Colors.BLUE)
    print_colored("="*60, Colors.BLUE)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    global LOG_FILE
    LOG_FILE   = (output_dir / "training.log").open("a", encoding="utf-8")
    start_time = time.time()

    env = os.environ.copy()
    env.update({
        "DATA_STREAM_API_URL": "http://localhost:5001",
        "SERVER_PORT":         "5002",
        "USE_LOCALHOST":       "true",
        "MODEL_INPUT_SIZE":    "1",
        "MODEL_HIDDEN_SIZE":   "64",
        "MODEL_NUM_LAYERS":    "4",
        "MODEL_SPLIT_LAYER":   "2",
        "MODEL_OUTPUT_SIZE":   "1",
        "MODEL_DROPOUT":       "0.0",
        "MODEL_BIDIRECTIONAL": "false",
        "WINDOW_SIZE":         "24",
        "FORECAST_HORIZON":    "1",
        "TRAIN_RATIO":         "0.7",
        "VAL_RATIO":           "0.15",
        "TEST_RATIO":          "0.15",
        "NORMALIZE":           "true",
        "NORMALIZE_METHOD":    "zscore",
        "BATCH_SIZE":          "32",
        "NUM_EPOCHS":          "5",
        "LEARNING_RATE":       "0.001",
        "RANDOM_SEED":         "42",
        "CHECKPOINT_EVERY":    "100",
        "STREAM_BASE_PORT":    "5101",
        "SERVER_BASE_PORT":    "5002",
        "FED_SERVER_PORT":     "5010",
        "RUN_FEDAVG":          "true",
    })

    assignments_path  = Path("dataset") / "cluster-assignment.csv"
    detected_clusters = load_cluster_ids(assignments_path)
    if not detected_clusters:
        print_colored("ERROR: No clusters found.", Colors.RED); return 1

    cluster_ids = _parse_ids(env["CLUSTER_IDS"]) if env.get("CLUSTER_IDS") else detected_clusters
    cluster_clients = {}
    for cid in cluster_ids:
        clients = _parse_ids(env["CLIENT_IDS"]) if env.get("CLIENT_IDS") else load_client_ids_for_cluster(cid, assignments_path)
        if not clients:
            print_colored(f"ERROR: No clients for cluster {cid}", Colors.RED); return 1
        cluster_clients[cid] = clients

    stream_base_port = int(env["STREAM_BASE_PORT"])
    server_base_port = int(env["SERVER_BASE_PORT"])
    fed_server_port  = int(env["FED_SERVER_PORT"])
    fed_server_url   = f"http://localhost:{fed_server_port}"
    total_epochs     = int(env["NUM_EPOCHS"])
    run_fedavg       = env.get("RUN_FEDAVG","true").lower() in ("1","true","yes","y")

    print_colored("\n📋 Plan", Colors.YELLOW)
    print_colored(f"   epochs={total_epochs}  fedavg={'every epoch' if run_fedavg else 'off'}  fed_server={fed_server_url}", Colors.NC)
    for idx, cid in enumerate(cluster_ids):
        print_colored(f"   cluster {cid}: stream=:{stream_base_port+idx}  server=:{server_base_port+idx}  clients={cluster_clients[cid]}", Colors.NC)

    processes         = []
    global_client_idx = 0
    exit_code         = 0

    try:
        # ── Start stream services ─────────────────────────────────────
        print_colored("\n[1/4] Stream services...", Colors.BLUE)
        for idx, cid in enumerate(cluster_ids):
            port = stream_base_port + idx
            _cluster_dir(cid).mkdir(parents=True, exist_ok=True)
            senv = env.copy()
            senv.update({"STREAM_CLUSTER_ID": cid, "CLUSTER_ID": cid, "STREAM_PORT": str(port)})
            log  = (_cluster_dir(cid) / "stream_service.log").open("w")
            proc = subprocess.Popen([sys.executable, "data/stream_service.py"],
                                     env=senv, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
            processes.append((f"stream_svc_{cid}", proc, log))
            if not wait_for_service(f"http://localhost:{port}/health"):
                print_colored(f"  ERROR: stream-svc {cid} not ready", Colors.RED); return 1
            print_colored(f"  ✓ stream-svc {cid} :{port} PID={proc.pid}", Colors.GREEN)

        # ── Start cluster servers ─────────────────────────────────────
        print_colored("\n[2/4] Cluster servers...", Colors.BLUE)
        for idx, cid in enumerate(cluster_ids):
            port = server_base_port + idx
            _cluster_dir(cid).mkdir(parents=True, exist_ok=True)
            senv = env.copy()
            senv.update({"CLUSTER_ID": cid, "SERVER_PORT": str(port), "OUTPUT_DIR": str(_cluster_dir(cid))})
            log  = (_cluster_dir(cid) / "server.log").open("w")
            proc = subprocess.Popen([sys.executable, "server/server.py"],
                                     env=senv, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd())
            processes.append((f"server_{cid}", proc, log))
            if not wait_for_service(f"http://localhost:{port}/healthz"):
                print_colored(f"  ERROR: server {cid} not ready", Colors.RED); return 1
            print_colored(f"  ✓ server {cid} :{port} PID={proc.pid}", Colors.GREEN)

        # ── Start fed server ──────────────────────────────────────────
        # NEW: started here as a persistent background process.
        # Previous version spawned federated.py subprocess after each epoch.
        print_colored("\n[3/4] Fed server...", Colors.BLUE)
        fed_env = env.copy()
        fed_env["FED_SERVER_PORT"] = str(fed_server_port)
        fed_log  = (output_dir / "fed_server.log").open("w")
        # fed_proc = subprocess.Popen([sys.executable, "fed_server.py"],
        #                              env=fed_env, stdout=fed_log, stderr=subprocess.STDOUT, cwd=os.getcwd())
        fed_proc = subprocess.Popen([sys.executable, "fed_server/fed_server.py"],
                             env=fed_env, stdout=fed_log, stderr=subprocess.STDOUT, cwd=os.getcwd())
        processes.append(("fed_server", fed_proc, fed_log))
        if not wait_for_service(f"{fed_server_url}/healthz"):
            print_colored(f"  ERROR: fed server not ready. Check output/fed_server.log", Colors.RED)
            return 1
        print_colored(f"  ✓ fed server :{fed_server_port} PID={fed_proc.pid}", Colors.GREEN)

        time.sleep(2)

        # ── Per-epoch training loop ───────────────────────────────────
        print_colored("\n[4/4] Training...", Colors.BLUE)
        for epoch in range(1, total_epochs + 1):
            print_colored(f"\n{'='*60}\n  EPOCH {epoch}/{total_epochs}\n{'='*60}", Colors.BLUE)

            epoch_env = env.copy()
            epoch_env["NUM_EPOCHS"] = "1"

            for c_idx, cid in enumerate(cluster_ids):
                sv_port = server_base_port + c_idx
                st_port = stream_base_port + c_idx
                print_colored(f"\n  [Cluster {cid}]", Colors.BLUE)

                for client_id in cluster_clients[cid]:
                    metrics_port = CLIENT_METRICS_BASE_PORT + global_client_idx
                    global_client_idx += 1
                    client_env = epoch_env.copy()
                    client_env.update({
                        "CLIENT_ID": client_id, "CLUSTER_ID": cid,
                        "SERVER_PORT": str(sv_port),
                        "DATA_STREAM_API_URL": f"http://localhost:{st_port}",
                        "CLIENT_METRICS_PORT": str(metrics_port),
                    })
                    _cluster_dir(cid).mkdir(parents=True, exist_ok=True)
                    log = (_cluster_dir(cid) / f"client_{client_id}.log").open("a")
                    print_colored(f"    → client {client_id}", Colors.BLUE)
                    t0   = time.time()
                    proc = subprocess.run([sys.executable, CLIENT_PATH, "train"],
                                           env=client_env, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=os.getcwd())
                    log.close()
                    elapsed = time.time() - t0
                    ok = proc.returncode == 0
                    print_colored(f"    {'✓' if ok else '✗'} {client_id} {elapsed:.1f}s (exit={proc.returncode})",
                                   Colors.GREEN if ok else Colors.RED)
                    if not ok:
                        exit_code = proc.returncode; break
                if exit_code != 0: break
            if exit_code != 0: break

            # POST /fedavg to fed server (replaces subprocess call)
            if run_fedavg:
                ok = trigger_fedavg(fed_server_url, cluster_ids, server_base_port, epoch)
                if not ok:
                    print_colored("  WARNING: FedAvg failed — continuing without it.", Colors.YELLOW)

    except KeyboardInterrupt:
        print_colored("\n⚠  Interrupted.", Colors.YELLOW); exit_code = 130
    except Exception as e:
        import traceback
        print_colored(f"\nFatal: {e}", Colors.RED); traceback.print_exc(); exit_code = 1
    finally:
        print_colored("\nShutting down...", Colors.BLUE)
        for name, proc, log_fh in processes:
            try:
                proc.terminate(); proc.wait(timeout=5)
                print_colored(f"  ✓ {name} stopped", Colors.GREEN)
            except subprocess.TimeoutExpired:
                proc.kill(); print_colored(f"  ✗ {name} force-killed", Colors.YELLOW)
            except Exception: pass
            finally:
                try: log_fh.close()
                except Exception: pass

    m, s = divmod(int(time.time()-start_time), 60)
    print_colored(f"\nTotal: {m}m {s}s", Colors.BLUE)
    if exit_code == 0:
        print_colored("Done! Global models → output/global/", Colors.GREEN)
    else:
        print_colored(f"Exit code {exit_code}", Colors.RED)
    if LOG_FILE: LOG_FILE.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())