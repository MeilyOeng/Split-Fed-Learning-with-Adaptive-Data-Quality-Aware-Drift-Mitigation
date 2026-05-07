# fed_server.py
import csv
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import torch
from flask import Flask, jsonify, request
from utils.comm_logging import append_comm_row, calc_bandwidth

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

FED_SERVER_PORT = int(os.environ.get("FED_SERVER_PORT", "5010"))


# ---------------------------------------------------------------------------
# Round history dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    round_num:        int
    timestamp:        str
    cluster_ids:      List[str]
    weights_used:     Dict[str, float]
    server_push_ok:   List[str]
    server_push_fail: List[str]
    output_path:      str
    client_output:    Optional[str]
    duration_sec:     float
    global_model_bytes: Optional[int] = None
    fedavg_compute_time_ms: Optional[float] = None
    error:            Optional[str] = None


# ---------------------------------------------------------------------------
# FedAvg maths
# ---------------------------------------------------------------------------

def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load model_state_dict from a checkpoint file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"  The server for this cluster has not saved yet.\n"
            f"  Check that POST /save_latest_checkpoint was called before /fedavg."
        )
    ckpt  = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return state


def _fedavg(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of model state dicts.

    From the SplitFed paper:  W_global = Σ_k (n_k / n_total) × W_k

    weights: one float per state_dict (e.g. sample counts).
             None → uniform (1/N each).
             Weights are normalised internally.
    """
    if not state_dicts:
        raise ValueError("No state dicts to average")
    keys = state_dicts[0].keys()
    for i, sd in enumerate(state_dicts[1:], 1):
        if sd.keys() != keys:
            raise ValueError(
                f"State dict key mismatch between model 0 and model {i}. "
                f"All clusters must use identical MODEL_* environment variables."
            )
    if weights is None:
        weights = [1.0] * len(state_dicts)
    total = sum(weights)
    norm  = [w / total for w in weights]

    avg: Dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        wt = torch.tensor(norm, dtype=torch.float32)
        for _ in range(stacked.dim() - 1):
            wt = wt.unsqueeze(-1)
        avg[k] = (stacked * wt).sum(dim=0).type_as(state_dicts[0][k])
    return avg


def _load_client_ids(
    assignments_path: str,
    cluster_ids: List[str],
) -> List[Tuple[str, str]]:
    if not os.path.exists(assignments_path):
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")
    cluster_set = set(cluster_ids)
    pairs: List[Tuple[str, str]] = []
    with open(assignments_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c   = str(row.get("Cluster", "")).strip()
            cid = str(row.get("ID", "")).strip()
            if c in cluster_set and cid:
                pairs.append((c, cid))
    return pairs


# ---------------------------------------------------------------------------
# Server communication helpers
# ---------------------------------------------------------------------------

def _fedavg_comm_log_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "global", "fedavg_comm.csv")


def _append_fed_row(log_path: str, row: Dict[str, object]) -> None:
    append_comm_row(log_path, row)


def _state_dict_stats(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    return len(state_dict), sum(int(v.numel()) for v in state_dict.values())


def _serialized_model_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    return len(pickle.dumps({"model_state_dict": state_dict}))

def _force_server_checkpoint(server_url: str, log_path: str, cluster_id: str) -> bool:
    """
    Ask a cluster server to flush its current weights to disk right now.

    Why: servers auto-save every CHECKPOINT_EVERY steps. If an epoch has
    21 batches and CHECKPOINT_EVERY=100, the last auto-save might be 79
    steps stale. This call guarantees the checkpoint is epoch-fresh before
    fedavg reads it from disk.
    """
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            server_url.rstrip("/") + "/save_latest_checkpoint", timeout=10.0
        )
        resp.raise_for_status()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        body = resp.json()
        response_bytes = len(resp.content)
        download_MBps, download_Mbps = calc_bandwidth(response_bytes, latency_ms)
        _append_fed_row(
            log_path,
            {
                "level": "fed",
                "observer": "fedavg",
                "cluster_id": cluster_id,
                "event": "force_server_checkpoint",
                "route": "/save_latest_checkpoint",
                "direction": "fed_to_server_control",
                "payload_type": "checkpoint_request",
                "payload_bytes": 0,
                "response_payload_bytes": response_bytes,
                "latency_ms": latency_ms,
                "bandwidth_MBps": download_MBps,
                "bandwidth_Mbps": download_Mbps,
                "status": "ok",
            },
        )
        print(f"[FedServer]   checkpoint saved on {server_url} "
              f"(path={body.get('path','?')})")
        return True
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        _append_fed_row(
            log_path,
            {
                "level": "fed",
                "observer": "fedavg",
                "cluster_id": cluster_id,
                "event": "force_server_checkpoint",
                "route": "/save_latest_checkpoint",
                "direction": "fed_to_server_control",
                "payload_type": "checkpoint_request",
                "payload_bytes": 0,
                "latency_ms": latency_ms,
                "status": f"error:{type(e).__name__}",
            },
        )
        print(f"[FedServer]   WARNING: could not force checkpoint on {server_url}: {e}")
        return False


def _push_weights(server_url: str, state_dict: Dict[str, torch.Tensor], log_path: str, cluster_id: str) -> bool:
    """
    POST averaged weights to a cluster server's /load_weights endpoint.

    The server replaces its in-memory model weights immediately, so the very
    next training batch uses the globally averaged starting point.
    Without this push, servers only benefit from FedAvg after a restart.
    """
    payload = pickle.dumps({"model_state_dict": state_dict})
    num_tensors, num_parameters = _state_dict_stats(state_dict)
    t0 = time.perf_counter()
    try:
        resp    = requests.post(
            server_url.rstrip("/") + "/load_weights",
            data    = payload,
            headers = {"Content-Type": "application/octet-stream"},
            timeout = 30.0,
        )
        resp.raise_for_status()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        body = resp.json()
        upload_MBps, upload_Mbps = calc_bandwidth(len(payload), latency_ms)
        _append_fed_row(
            log_path,
            {
                "level": "fed",
                "observer": "fedavg",
                "cluster_id": cluster_id,
                "event": "broadcast_weights",
                "route": "/load_weights",
                "direction": "fed_to_server",
                "payload_type": "avg_server_weights",
                "payload_bytes": len(payload),
                "latency_ms": latency_ms,
                "bandwidth_MBps": upload_MBps,
                "bandwidth_Mbps": upload_Mbps,
                "status": "ok",
                "num_tensors": num_tensors,
                "num_parameters": num_parameters,
            },
        )
        print(f"[FedServer]   weights loaded on {server_url} "
              f"(step={body.get('global_step','?')})")
        return True
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        _append_fed_row(
            log_path,
            {
                "level": "fed",
                "observer": "fedavg",
                "cluster_id": cluster_id,
                "event": "broadcast_weights",
                "route": "/load_weights",
                "direction": "fed_to_server",
                "payload_type": "avg_server_weights",
                "payload_bytes": len(payload),
                "latency_ms": latency_ms,
                "status": f"error:{type(e).__name__}",
                "num_tensors": num_tensors,
                "num_parameters": num_parameters,
            },
        )
        print(f"[FedServer]   WARNING: could not push weights to {server_url}: {e}")
        return False


# ---------------------------------------------------------------------------
# Core round function
# ---------------------------------------------------------------------------

def run_fedavg_round(
    cluster_ids:        List[str],
    server_urls:        List[str],
    checkpoint_dir:     str,
    output_path:        str,
    client_output_path: str,
    assignments_path:   str,
    cluster_weights:    Optional[Dict[str, float]] = None,
    aggregate_clients:  bool = True,
) -> RoundRecord:
    """
    One complete FedAvg round — four steps:

    1. Force fresh checkpoints on all cluster servers.
    2. Load, average, and save the global server model.
    3. Push averaged server weights to every live cluster server.
    4. Optionally average client weights. The active split-learning path
       keeps this disabled and only aggregates server weights.
    """
    t_start   = datetime.utcnow()
    push_ok:   List[str] = []
    push_fail: List[str] = []
    error_msg: Optional[str] = None
    client_out: Optional[str] = None
    log_path = _fedavg_comm_log_path(checkpoint_dir)
    global_model_bytes: Optional[int] = None
    fedavg_compute_time_ms: Optional[float] = None

    weights_list = (
        [cluster_weights.get(cid, 1.0) for cid in cluster_ids]
        if cluster_weights else None
    )
    weights_used = (
        dict(zip(cluster_ids, weights_list)) if weights_list
        else {cid: round(1.0 / len(cluster_ids), 4) for cid in cluster_ids}
    )

    try:
        # ── 1. Force checkpoints ─────────────────────────────────────
        print(f"[FedServer] Step 1/4  forcing checkpoints...")
        for cid, url in zip(cluster_ids, server_urls):
            _force_server_checkpoint(url, log_path, cid)

        # ── 2. Average server weights ────────────────────────────────
        print(f"[FedServer] Step 2/4  averaging server weights for clusters {cluster_ids}...")
        server_states = []
        for cid in cluster_ids:
            path = os.path.join(checkpoint_dir, f"cluster_{cid}", "server_latest.pt")
            read_t0 = time.perf_counter()
            state_dict = _load_state_dict(path)
            read_ms = round((time.perf_counter() - read_t0) * 1000.0, 3)
            logical_bytes = _serialized_model_bytes(state_dict)
            num_tensors, num_parameters = _state_dict_stats(state_dict)
            upload_MBps, upload_Mbps = calc_bandwidth(logical_bytes, read_ms)
            _append_fed_row(
                log_path,
                {
                    "level": "fed",
                    "observer": "fedavg",
                    "cluster_id": cid,
                    "event": "load_server_checkpoint",
                    "direction": "server_to_fed",
                    "payload_type": "server_weights",
                    "payload_bytes": logical_bytes,
                    "latency_ms": read_ms,
                    "bandwidth_MBps": upload_MBps,
                    "bandwidth_Mbps": upload_Mbps,
                    "status": "shared_checkpoint_read",
                    "num_tensors": num_tensors,
                    "num_parameters": num_parameters,
                },
            )
            server_states.append(state_dict)

        compute_t0 = time.perf_counter()
        avg_server = _fedavg(server_states, weights=weights_list)
        fedavg_compute_time_ms = round((time.perf_counter() - compute_t0) * 1000.0, 3)
        global_model_bytes = _serialized_model_bytes(avg_server)
        num_tensors, num_parameters = _state_dict_stats(avg_server)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({"model_state_dict": avg_server}, output_path)
        _append_fed_row(
            log_path,
            {
                "level": "fed",
                "observer": "fedavg",
                "cluster_id": "global",
                "event": "fedavg_compute",
                "direction": "local_compute",
                "payload_type": "avg_server_weights",
                "payload_bytes": global_model_bytes,
                "latency_ms": fedavg_compute_time_ms,
                "status": "ok",
                "num_tensors": num_tensors,
                "num_parameters": num_parameters,
            },
        )
        print(f"[FedServer]   global server model → {output_path}")

        # ── 3. Push averaged weights back to live servers ────────────
        print(f"[FedServer] Step 3/4  pushing to {len(server_urls)} live server(s)...")
        for cid, url in zip(cluster_ids, server_urls):
            ok = _push_weights(url, avg_server, log_path, cid)
            (push_ok if ok else push_fail).append(cid)

        if push_fail:
            print(f"[FedServer]   WARNING: push failed for clusters {push_fail}. "
                  f"Those servers will use averaged weights after next restart.")

        # ── 4. Average client weights (saved to disk) ────────────────
        if aggregate_clients:
            print(f"[FedServer] Step 4/4  averaging client weights...")
            pairs  = _load_client_ids(assignments_path, cluster_ids)
            client_states = []
            missing       = []
            for cluster_id, client_id in pairs:
                path = os.path.join(
                    checkpoint_dir,
                    f"cluster_{cluster_id}",
                    f"client_{client_id}_checkpoint.pt",
                )
                try:
                    client_states.append(_load_state_dict(path))
                except FileNotFoundError as exc:
                    print(f"[FedServer]   WARNING: {exc}")
                    missing.append(path)

            if client_states:
                client_avg = _fedavg(client_states)
                os.makedirs(os.path.dirname(client_output_path), exist_ok=True)
                torch.save(
                    {"model_state_dict": client_avg},
                    client_output_path,
                )
                client_out = client_output_path
                print(f"[FedServer]   global client model → {client_output_path}")
                if missing:
                    print(f"[FedServer]   ({len(missing)} missing, "
                          f"averaged over {len(client_states)} available)")
            else:
                print("[FedServer]   no client checkpoints found — skipping")

    except Exception as exc:
        import traceback
        error_msg = str(exc)
        print(f"[FedServer] ERROR: {exc}")
        traceback.print_exc()

    duration = (datetime.utcnow() - t_start).total_seconds()
    return RoundRecord(
        round_num        = 0,
        timestamp        = t_start.isoformat(),
        cluster_ids      = cluster_ids,
        weights_used     = weights_used,
        server_push_ok   = push_ok,
        server_push_fail = push_fail,
        output_path      = output_path,
        client_output    = client_out,
        duration_sec     = round(duration, 2),
        global_model_bytes = global_model_bytes,
        fedavg_compute_time_ms = fedavg_compute_time_ms,
        error            = error_msg,
    )


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app            = Flask(__name__)
    round_lock     = threading.Lock()
    round_history: List[RoundRecord] = []
    global_round   = [0]

    @app.route("/healthz", methods=["GET"])
    def healthz():
        """Health check — run_training.py polls this before starting training."""
        return jsonify({
            "status":      "ok",
            "rounds_done": global_round[0],
            "port":        FED_SERVER_PORT,
        })

    @app.route("/status", methods=["GET"])
    def status():
        """Return the last N round records."""
        n    = int(request.args.get("n", 5))
        last = round_history[-n:] if round_history else []
        return jsonify({
            "rounds_done": global_round[0],
            "last_rounds": [asdict(r) for r in last],
        })

    @app.route("/fedavg", methods=["POST"])
    def fedavg_route():
        """
        Trigger one FedAvg round.  Runs synchronously — caller blocks until done.

        JSON body fields:
          cluster_ids        list[str]         required
          server_urls        list[str]         required, same order as cluster_ids
          checkpoint_dir     str               default "output"
          output_path        str               default "output/global/global_server_latest.pt"
          client_output_path str               default "output/global/global_client_latest.pt"
          assignments_path   str               default "dataset/cluster-assignment.csv"
          aggregate_clients  bool              default false
          weights            dict[str, float]  optional  e.g. {"0": 1000, "1": 500, "2": 500}

        Returns 200 + round summary on success.
        Returns 429 if a round is already running.
        Returns 500 + error if the round fails.
        """
        body = request.get_json(force=True)

        cluster_ids        = body.get("cluster_ids", [])
        server_urls        = body.get("server_urls", [])
        checkpoint_dir     = body.get("checkpoint_dir", "output")
        output_path        = body.get("output_path",
                                      "output/global/global_server_latest.pt")
        client_output_path = body.get("client_output_path",
                                      "output/global/global_client_latest.pt")
        assignments_path   = body.get("assignments_path",
                                      "dataset/cluster-assignment.csv")
        aggregate_clients  = bool(body.get("aggregate_clients", False))
        raw_weights        = body.get("weights")

        if not cluster_ids:
            return jsonify({"error": "cluster_ids is required"}), 400
        if server_urls and len(server_urls) != len(cluster_ids):
            return jsonify({
                "error": f"server_urls ({len(server_urls)}) must have the same "
                         f"length as cluster_ids ({len(cluster_ids)})"
            }), 400

        cluster_weights: Optional[Dict[str, float]] = None
        if raw_weights:
            cluster_weights = {str(k): float(v) for k, v in raw_weights.items()}

        if not round_lock.acquire(blocking=False):
            return jsonify({
                "error": "A FedAvg round is already in progress. Retry shortly."
            }), 429

        try:
            global_round[0] += 1
            print(f"\n[FedServer] ══ Round {global_round[0]} "
                  f"(clusters={cluster_ids}) ══")

            record           = run_fedavg_round(
                cluster_ids        = cluster_ids,
                server_urls        = server_urls,
                checkpoint_dir     = checkpoint_dir,
                output_path        = output_path,
                client_output_path = client_output_path,
                assignments_path   = assignments_path,
                cluster_weights    = cluster_weights,
                aggregate_clients  = aggregate_clients,
            )
            record.round_num = global_round[0]
            round_history.append(record)

            if record.error:
                return jsonify({
                    "error": record.error,
                    "round": asdict(record),
                }), 500

            return jsonify({"status": "ok", "round": asdict(record)})
        finally:
            round_lock.release()

    @app.route("/reset", methods=["POST"])
    def reset_route():
        """Clear round history and reset counter (for testing)."""
        global_round[0] = 0
        round_history.clear()
        return jsonify({"status": "reset"})

    return app


if __name__ == "__main__":
    flask_app = create_app()
    print(f"[FedServer] Starting on port {FED_SERVER_PORT}")
    flask_app.run(host="0.0.0.0", port=FED_SERVER_PORT)
