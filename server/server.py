#!/usr/bin/env python3
import os
import pickle
import sys
import time
import json
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from flask import Flask, Response, jsonify, request
from torch import nn, optim

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.split_lstm import build_split_lstm
from utils.comm_logging import append_comm_row, calc_bandwidth, tensor_nbytes, tensor_shape
from utils.runtime_utils import (
    _get_env,
    build_lstm_config_from_env,
    get_device_info,
    parse_bool,
    resolve_device,
    save_torch_checkpoint,
    try_load_torch_checkpoint,
)


app = Flask(__name__)


def server_output_dir(cluster_id: str) -> str:
    output_dir = _get_env("OUTPUT_DIR", os.path.join("output", f"cluster_{cluster_id}"))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def server_checkpoint_path(cluster_id: str, filename: str) -> str:
    return os.path.join(server_output_dir(cluster_id), f"server_{filename}")


def server_comm_log_path(cluster_id: str) -> str:
    return os.path.join(server_output_dir(cluster_id), "server_comm.csv")


@dataclass
class ServerRuntimeConfig:
    cluster_id: str
    learning_rate: float
    accum_steps_target: int
    save_every_rounds: int
    output_dir: str


class SplitLearningServerState:
    def __init__(self) -> None:
        self.runtime_cfg = ServerRuntimeConfig(
            cluster_id=_get_env("CLUSTER_ID", "0"),
            learning_rate=float(_get_env("SERVER_LR", _get_env("LEARNING_RATE", "1e-3"))),
            accum_steps_target=max(1, int(_get_env("ACCUM_STEPS_SERVER", "1"))), # to make sure it's at least 1
            save_every_rounds=max(0, int(_get_env("CHECKPOINT_EVERY_ROUNDS", "1"))),
            output_dir=_get_env("OUTPUT_DIR", server_output_dir(_get_env("CLUSTER_ID", "0"))),
        )
        self.device = resolve_device()
        self.device_info = get_device_info(self.device, process_role_default="server")
        cfg, split_layer = build_lstm_config_from_env()
        _, server_model = build_split_lstm(cfg, split_layer)
        self.model = server_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.runtime_cfg.learning_rate)
        self.criterion = nn.MSELoss()
        self.lock = Lock()
        self.comm_log_train = parse_bool(_get_env("COMM_LOG_TRAIN", "true"))
        self.comm_log_eval = parse_bool(_get_env("COMM_LOG_EVAL", "false"))
        self.comm_log_path = server_comm_log_path(self.runtime_cfg.cluster_id)

        self.global_step = 0
        self.optimizer_step_count = 0
        self.pending_accum_steps = 0
        self.round_id = 0
        self.expected_clients: List[str] = []
        self.expected_clients_set: Set[str] = set()
        self.seen_clients_total: Set[str] = set()
        self.seen_clients_window: Set[str] = set()
        self.round_total_contrib_count = 0
        self.window_contrib_count = 0
        # try loading latest checkpoint on startup
        latest_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt") 
        extra, ok = try_load_torch_checkpoint(
            self.model,
            self.optimizer,
            latest_path,
            map_location=str(self.device),
            strict=True,
            log_prefix="Server",
        )
        if ok and extra:
            self.global_step = int(extra.get("global_step", 0))
            self.optimizer_step_count = int(extra.get("optimizer_step_count", 0))
            self.round_id = int(extra.get("round_id", 0))
        self.optimizer.zero_grad(set_to_none=True)

        print(
            f"[Server cluster={self.runtime_cfg.cluster_id}] device_info={self.device_info} "
            f"server_lr={self.runtime_cfg.learning_rate} accum_steps_target={self.runtime_cfg.accum_steps_target} output_dir={self.runtime_cfg.output_dir}",
            flush=True,
        )

    def _append_comm_row(self, row: Dict[str, Any]) -> None:
        append_comm_row(self.comm_log_path, row)

    @staticmethod
    def _ensure_lstm_input_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # Backward-compatibility path for older clients that still send only
            # the final timestep. The active client should send [B, T, H].
            return x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected activations with 2 or 3 dims, got shape={tuple(x.shape)}")
        return x
    # saving training progress counters into a dict with checkpoint
    def _checkpoint_extra(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "optimizer_step_count": self.optimizer_step_count,
            "round_id": self.round_id,
        }
    # averaging gradients if effective_accum_steps > 1 to ensure consistent scale regardless of how many steps were accumulated
    def _average_parameter_gradients(self, effective_accum_steps: int) -> None:
        if effective_accum_steps <= 1:
            return
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(effective_accum_steps)

    def _reset_window_tracking(self) -> None:
        self.pending_accum_steps = 0
        self.seen_clients_window.clear()
        self.window_contrib_count = 0

    # just to reset the tracking of each round's progress after a completed logical round
    def _reset_round_tracking(self, increment_round: bool) -> None:
        self.expected_clients = []
        self.expected_clients_set = set()
        self.seen_clients_total.clear()
        self.round_total_contrib_count = 0
        self._reset_window_tracking()
        if increment_round:
            self.round_id += 1
    # check for expected clients, track seen clients and contributions
    def set_round(self, expected_clients: List[str], round_id: Optional[int] = None) -> Dict[str, Any]:
        unique_clients = [str(c) for c in expected_clients]
        if len(unique_clients) != len(set(unique_clients)):
            raise ValueError("expected_clients contains duplicates")
        if self.pending_accum_steps > 0:
            raise RuntimeError("cannot start a new round while gradients are pending")
        if self.expected_clients_set or self.seen_clients_total or self.round_total_contrib_count > 0:
            raise RuntimeError("cannot start a new round while previous logical round is still active")
        self.expected_clients = unique_clients
        self.expected_clients_set = set(unique_clients)
        self.seen_clients_total.clear()
        self.seen_clients_window.clear()
        self.round_total_contrib_count = 0
        self.window_contrib_count = 0
        self.pending_accum_steps = 0
        if round_id is not None:
            self.round_id = int(round_id)
        print(
            f"[Server] round_start cluster={self.runtime_cfg.cluster_id} round_id={self.round_id} expected_clients={self.expected_clients}",
            flush=True,
        )
        return self.round_state()

    def round_state(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.runtime_cfg.cluster_id,
            "round_id": self.round_id,
            "expected_clients": list(self.expected_clients),
            "seen_clients_total": sorted(self.seen_clients_total),
            "seen_clients_window": sorted(self.seen_clients_window),
            "seen_clients": sorted(self.seen_clients_total),
            "round_total_contrib_count": self.round_total_contrib_count,
            "window_contrib_count": self.window_contrib_count,
            "round_contrib_count": self.round_total_contrib_count,
            "pending_accum_steps": self.pending_accum_steps,
            "optimizer_step_count": self.optimizer_step_count,
            "global_step": self.global_step,
        }

    def flush_accumulated_gradients(self, reason: str, *, increment_round: bool) -> Dict[str, Any]:
        if self.pending_accum_steps <= 0:
            return {"flushed": False, **self.round_state(), "reason": reason}
        effective_accum_steps = self.pending_accum_steps
        self._average_parameter_gradients(effective_accum_steps)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_step_count += 1
        finished_round_id = self.round_id
        window_clients = sorted(self.seen_clients_window)
        if increment_round:
            self._reset_round_tracking(increment_round=increment_round)
        else:
            self._reset_window_tracking()
        print(
            f"[Server] flush reason={reason} cluster={self.runtime_cfg.cluster_id} round_id={finished_round_id} "
            f"effective_accum_steps={effective_accum_steps} window_clients={window_clients} optimizer_steps={self.optimizer_step_count}",
            flush=True,
        )
        if increment_round and self.runtime_cfg.save_every_rounds > 0 and finished_round_id % self.runtime_cfg.save_every_rounds == 0:
            self._save_latest(flush=False)
        return {
            "flushed": True,
            "reason": reason,
            "finished_round_id": finished_round_id,
            "effective_accum_steps": effective_accum_steps,
            "window_clients": window_clients,
            **self.round_state(),
        }

    def _flush_before_eval_if_needed(self, reason: str) -> Optional[Dict[str, Any]]:
        if self.pending_accum_steps > 0:
            return self.flush_accumulated_gradients(reason=reason, increment_round=False)
        return None

    def train_step(self, client_id: str, activations_cpu: torch.Tensor, targets_cpu: torch.Tensor) -> Dict[str, Any]:
        client_id = str(client_id)
        if not self.expected_clients_set:               # Round must have been started via /round/start
            raise RuntimeError("No active round configured. Call /round/start first.")
        if client_id not in self.expected_clients_set:  #Client must be in the expected list for this round
            raise RuntimeError(f"Client {client_id} is not part of active round {self.round_id}")
        if client_id in self.seen_clients_total:        # Avoid accepting multiple contributions from the same client in the same round
            raise RuntimeError(f"Duplicate contribution from client {client_id} in round {self.round_id}")

        self.global_step += 1
        activations = self._ensure_lstm_input_3d(activations_cpu.to(self.device))
        targets = targets_cpu.to(self.device)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        activations.requires_grad_(True)
        preds = self.model(activations)
        loss = self.criterion(preds, targets)
        raw_loss = float(loss.item())
        loss.backward()
        # move gradients back to CPU for sending to client
        grad_activations = activations.grad.detach().cpu()

        self.pending_accum_steps += 1
        self.seen_clients_total.add(client_id)
        self.seen_clients_window.add(client_id)
        self.round_total_contrib_count = len(self.seen_clients_total)
        self.window_contrib_count = len(self.seen_clients_window)

        seen_clients_total_snapshot = sorted(self.seen_clients_total)
        seen_clients_window_snapshot = sorted(self.seen_clients_window)
        window_complete = self.pending_accum_steps >= self.runtime_cfg.accum_steps_target
        round_complete = self.seen_clients_total == self.expected_clients_set
        flush_info: Optional[Dict[str, Any]] = None
        finished_round_id: Optional[int] = None

        if window_complete:
            flush_info = self.flush_accumulated_gradients(reason="window_complete", increment_round=False)

        if round_complete:
            finished_round_id = self.round_id
            if self.pending_accum_steps > 0:
                flush_info = self.flush_accumulated_gradients(reason="round_complete_tail", increment_round=False)
            if self.runtime_cfg.save_every_rounds > 0 and finished_round_id % self.runtime_cfg.save_every_rounds == 0:
                self._save_latest(flush=False)
            self._reset_round_tracking(increment_round=True)

        print(
            f"[Server] train_step cluster={self.runtime_cfg.cluster_id} round_id={finished_round_id if finished_round_id is not None else self.round_id} "
            f"client={client_id} loss={raw_loss:.6f} seen_total={seen_clients_total_snapshot} "
            f"seen_window={seen_clients_window_snapshot} pending_accum={self.pending_accum_steps} "
            f"window_complete={window_complete} round_complete={round_complete}",
            flush=True,
        )
        return {
            "grad_activations": grad_activations,
            "loss": raw_loss,
            "window_complete": window_complete,
            "round_complete": round_complete,
            "flush_info": flush_info,
            "finished_round_id": finished_round_id,
            **self.round_state(),
        }

    def validate_step(self, client_id: str, activations_cpu: torch.Tensor, targets_cpu: torch.Tensor) -> Dict[str, Any]:
        self._flush_before_eval_if_needed(reason=f"before_validate:{client_id}") # apply any pending gradients first
        with torch.no_grad():
            activations = self._ensure_lstm_input_3d(activations_cpu.to(self.device))
            targets = targets_cpu.to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            preds = self.model(activations)
            loss = self.criterion(preds, targets)
            return {"predictions": preds.detach().cpu().numpy(), "loss": float(loss.item())}

    def inference_step(self, client_id: str, activations_cpu: torch.Tensor) -> Dict[str, Any]:
        self._flush_before_eval_if_needed(reason=f"before_inference:{client_id}") # apply any pending gradients first
        with torch.no_grad():
            activations = self._ensure_lstm_input_3d(activations_cpu.to(self.device))
            preds = self.model(activations).detach().cpu()
            return {"predictions": preds.numpy()}

    def _save_latest(self, flush: bool = True) -> str:
        if flush:
            self._flush_before_eval_if_needed(reason="save_latest")
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
        save_torch_checkpoint(self.model, self.optimizer, path, extra=self._checkpoint_extra(), log_prefix="Server")
        return path

    def _save_best(self, flush: bool = True) -> str:
        if flush:
            self._flush_before_eval_if_needed(reason="save_best")
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "best.pt")
        save_torch_checkpoint(self.model, self.optimizer, path, extra=self._checkpoint_extra(), log_prefix="Server")
        return path

    def _save_final(self, flush: bool = True) -> str:
        if flush:
            self._flush_before_eval_if_needed(reason="save_final")
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "final.pt")
        save_torch_checkpoint(self.model, self.optimizer, path, extra=self._checkpoint_extra(), log_prefix="Server")
        return path

    def _load_best(self) -> str:
        self._flush_before_eval_if_needed(reason="load_best")
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "best.pt")
        extra, ok = try_load_torch_checkpoint(
            self.model,
            self.optimizer,
            path,
            map_location=str(self.device),
            strict=True,
            log_prefix="Server",
        )
        if not ok:
            raise FileNotFoundError(path)
        if extra:
            self.global_step = int(extra.get("global_step", self.global_step))
            self.optimizer_step_count = int(extra.get("optimizer_step_count", self.optimizer_step_count))
            self.round_id = int(extra.get("round_id", self.round_id))
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_round_tracking(increment_round=False)
        return path

    def _load_latest(self) -> str:
        self._flush_before_eval_if_needed(reason="load_latest")
        latest_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
        extra, ok = try_load_torch_checkpoint(
            self.model,
            self.optimizer,
            latest_path,
            map_location=str(self.device),
            strict=True,
            log_prefix="Server",
        )
        if not ok:
            raise FileNotFoundError(latest_path)
        if extra:
            self.global_step = int(extra.get("global_step", self.global_step))
            self.optimizer_step_count = int(extra.get("optimizer_step_count", self.optimizer_step_count))
            self.round_id = int(extra.get("round_id", self.round_id))
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_round_tracking(increment_round=False)
        return latest_path

    def _load_final(self) -> str:
        self._flush_before_eval_if_needed(reason="load_final")
        final_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "final.pt")
        latest_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
        target = final_path if os.path.exists(final_path) else latest_path                  # fallback to latest if final doesn't exist
        extra, ok = try_load_torch_checkpoint(
            self.model,
            self.optimizer,
            target,
            map_location=str(self.device),
            strict=True,
            log_prefix="Server",
        )
        if not ok:
            raise FileNotFoundError(target)
        if extra:
            self.global_step = int(extra.get("global_step", self.global_step))
            self.optimizer_step_count = int(extra.get("optimizer_step_count", self.optimizer_step_count))
            self.round_id = int(extra.get("round_id", self.round_id))
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_round_tracking(increment_round=False)
        return target

    def load_weights(self, model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self._flush_before_eval_if_needed(reason="load_weights")
        self.model.load_state_dict(model_state_dict)
        # Reset optimizer state after broadcast so moments are consistent with
        # the newly averaged global weights.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.runtime_cfg.learning_rate)
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_round_tracking(increment_round=False)
        return {
            "cluster_id": self.runtime_cfg.cluster_id,
            "global_step": self.global_step,
            "optimizer_step_count": self.optimizer_step_count,
            "round_id": self.round_id,
        }


state = SplitLearningServerState()


def _unpickle_request_data() -> Dict[str, Any]:
    if not request.data:
        raise ValueError("Empty request body")
    payload = pickle.loads(request.data)
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a pickled dict")
    return payload


def _pickle_response(obj: Dict[str, Any]) -> Response:
    return Response(pickle.dumps(obj), content_type="application/octet-stream")


@app.route("/healthz", methods=["GET"])
def healthz():
    with state.lock:
        return jsonify({"status": "ok", **state.round_state(), "device": state.device_info})


@app.route("/config", methods=["GET"])
def config():
    with state.lock:
        return jsonify({
            "cluster_id": state.runtime_cfg.cluster_id,
            "server_lr": state.runtime_cfg.learning_rate,
            "save_every_rounds": state.runtime_cfg.save_every_rounds,
            "device": state.device_info,
            **state.round_state(),
        })


@app.route("/round/start", methods=["POST"])
def round_start():
    data = request.get_json() or {}
    expected_clients = [str(x) for x in data.get("expected_clients", [])]
    round_id = data.get("round_id")
    with state.lock:
        try:
            return jsonify(state.set_round(expected_clients, round_id=round_id))
        except Exception as e:
            return jsonify({"error": str(e), **state.round_state()}), 409


@app.route("/round/state", methods=["GET"])
def round_state():
    with state.lock:
        return jsonify(state.round_state())


@app.route("/train_step", methods=["POST"])
def train_step():
    request_bytes = len(request.get_data(cache=True) or b"")
    t0 = time.perf_counter()
    with state.lock:
        try:
            payload = _unpickle_request_data()
            result = state.train_step(
                client_id=str(payload["client_id"]),
                activations_cpu=payload["activations"],
                targets_cpu=payload["targets"],
            )
            response_blob = pickle.dumps(result)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            upload_MBps, upload_Mbps = calc_bandwidth(request_bytes, latency_ms)
            download_MBps, download_Mbps = calc_bandwidth(len(response_blob), latency_ms)
            if state.comm_log_train:
                state._append_comm_row(
                    {
                        "level": "split",
                        "observer": "server",
                        "cluster_id": state.runtime_cfg.cluster_id,
                        "client_id": str(payload["client_id"]),
                        "scheduler_epoch": payload.get("scheduler_epoch"),
                        "local_epoch": payload.get("local_epoch"),
                        "round_id": result.get("finished_round_id") or result.get("round_id"),
                        "batch_id": payload.get("batch_id"),
                        "event": "train_step",
                        "route": "/train_step",
                        "direction": "server_process",
                        "payload_type": "activation_payload",
                        "payload_shape": tensor_shape(payload.get("activations")),
                        "payload_bytes": request_bytes,
                        "latency_ms": latency_ms,
                        "bandwidth_MBps": upload_MBps,
                        "bandwidth_Mbps": upload_Mbps,
                        "status": "ok",
                        "activation_shape": tensor_shape(payload.get("activations")),
                        "activation_bytes": tensor_nbytes(payload.get("activations")),
                        "target_bytes": tensor_nbytes(payload.get("targets")),
                        "response_payload_bytes": len(response_blob),
                        "grad_shape": tensor_shape(result.get("grad_activations")),
                        "grad_bytes": tensor_nbytes(result.get("grad_activations")),
                        "loss": result.get("loss"),
                        "round_complete": bool(result.get("round_complete")),
                        "flush_triggered": bool(result.get("flush_info")),
                        "effective_accum_steps": (
                            result.get("flush_info", {}) or {}
                        ).get("effective_accum_steps"),
                        "server_compute_time_ms": latency_ms,
                    }
                )
            return Response(response_blob, content_type="application/octet-stream")
        except Exception as e:
            if state.comm_log_train:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                state._append_comm_row(
                    {
                        "level": "split",
                        "observer": "server",
                        "cluster_id": state.runtime_cfg.cluster_id,
                        "client_id": None,
                        "event": "train_step",
                        "route": "/train_step",
                        "direction": "server_process",
                        "payload_type": "activation_payload",
                        "payload_bytes": request_bytes,
                        "latency_ms": latency_ms,
                        "status": f"error:{type(e).__name__}",
                        "server_compute_time_ms": latency_ms,
                    }
                )
            return jsonify({"error": str(e), **state.round_state()}), 409


@app.route("/validate_step", methods=["POST"])
def validate_step():
    request_bytes = len(request.get_data(cache=True) or b"")
    t0 = time.perf_counter()
    with state.lock:
        payload = _unpickle_request_data()
        result = state.validate_step(
            str(payload["client_id"]), 
            payload["activations"], 
            payload["targets"]
        )
        response_blob = pickle.dumps(result)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if state.comm_log_eval:
            upload_MBps, upload_Mbps = calc_bandwidth(request_bytes, latency_ms)
            state._append_comm_row(
                {
                    "level": "split",
                    "observer": "server",
                    "cluster_id": state.runtime_cfg.cluster_id,
                    "client_id": str(payload["client_id"]),
                    "scheduler_epoch": payload.get("scheduler_epoch"),
                    "local_epoch": payload.get("local_epoch"),
                    "batch_id": payload.get("batch_id"),
                    "event": "validate_step",
                    "route": "/validate_step",
                    "direction": "server_process",
                    "payload_type": "validation_payload",
                    "payload_shape": tensor_shape(payload.get("activations")),
                    "payload_bytes": request_bytes,
                    "latency_ms": latency_ms,
                    "bandwidth_MBps": upload_MBps,
                    "bandwidth_Mbps": upload_Mbps,
                    "status": "ok",
                    "activation_shape": tensor_shape(payload.get("activations")),
                    "activation_bytes": tensor_nbytes(payload.get("activations")),
                    "target_bytes": tensor_nbytes(payload.get("targets")),
                    "response_payload_bytes": len(response_blob),
                    "loss": result.get("loss"),
                    "server_compute_time_ms": latency_ms,
                }
            )
        return Response(response_blob, content_type="application/octet-stream")


@app.route("/inference_step", methods=["POST"])
def inference_step():
    request_bytes = len(request.get_data(cache=True) or b"")
    t0 = time.perf_counter()
    with state.lock:
        payload = _unpickle_request_data()
        result = state.inference_step(
            str(payload["client_id"]), 
            payload["activations"])
        response_blob = pickle.dumps(result)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if state.comm_log_eval:
            upload_MBps, upload_Mbps = calc_bandwidth(request_bytes, latency_ms)
            state._append_comm_row(
                {
                    "level": "split",
                    "observer": "server",
                    "cluster_id": state.runtime_cfg.cluster_id,
                    "client_id": str(payload["client_id"]),
                    "scheduler_epoch": payload.get("scheduler_epoch"),
                    "local_epoch": payload.get("local_epoch"),
                    "batch_id": payload.get("batch_id"),
                    "event": "inference_step",
                    "route": "/inference_step",
                    "direction": "server_process",
                    "payload_type": "inference_payload",
                    "payload_shape": tensor_shape(payload.get("activations")),
                    "payload_bytes": request_bytes,
                    "latency_ms": latency_ms,
                    "bandwidth_MBps": upload_MBps,
                    "bandwidth_Mbps": upload_Mbps,
                    "status": "ok",
                    "activation_shape": tensor_shape(payload.get("activations")),
                    "activation_bytes": tensor_nbytes(payload.get("activations")),
                    "response_payload_bytes": len(response_blob),
                    "server_compute_time_ms": latency_ms,
                }
            )
        return Response(response_blob, content_type="application/octet-stream")


@app.route("/flush_gradients", methods=["POST"])
def flush_gradients():
    with state.lock:
        result = state.flush_accumulated_gradients(reason="manual_flush", increment_round=False)
        return jsonify(result)


@app.route("/save_best_checkpoint", methods=["POST"])
def save_best_checkpoint():
    with state.lock:
        path = state._save_best(flush=True)
        return jsonify({"path": path})


@app.route("/save_final_checkpoint", methods=["POST"])
def save_final_checkpoint():
    with state.lock:
        path = state._save_final(flush=True)
        return jsonify({"path": path})


@app.route("/save_latest_checkpoint", methods=["POST"])
def save_latest_checkpoint():
    with state.lock:
        path = state._save_latest(flush=True)
        return jsonify({"path": path})


@app.route("/load_latest_checkpoint", methods=["POST"])
def load_latest_checkpoint():
    with state.lock:
        path = state._load_latest()
        return jsonify({"path": path})


@app.route("/load_best_checkpoint", methods=["POST"])
def load_best_checkpoint():
    with state.lock:
        path = state._load_best()
        return jsonify({"path": path})


@app.route("/load_final_checkpoint", methods=["POST"])
def load_final_checkpoint():
    with state.lock:
        path = state._load_final()
        return jsonify({"path": path})


@app.route("/load_weights", methods=["POST"])
def load_weights():
    request_bytes = len(request.get_data(cache=True) or b"")
    t0 = time.perf_counter()
    with state.lock:
        payload = _unpickle_request_data()
        meta = state.load_weights(payload["model_state_dict"])
        response_body = {"status": "ok", **meta}
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        state._append_comm_row(
            {
                "level": "fed",
                "observer": "server",
                "cluster_id": state.runtime_cfg.cluster_id,
                "event": "load_weights",
                "route": "/load_weights",
                "direction": "fed_to_server",
                "payload_type": "avg_server_weights",
                "payload_bytes": request_bytes,
                "response_payload_bytes": len(json.dumps(response_body).encode("utf-8")),
                "latency_ms": latency_ms,
                "bandwidth_MBps": calc_bandwidth(request_bytes, latency_ms)[0],
                "bandwidth_Mbps": calc_bandwidth(request_bytes, latency_ms)[1],
                "status": "ok",
                "num_tensors": len(payload["model_state_dict"]),
                "num_parameters": sum(int(v.numel()) for v in payload["model_state_dict"].values()),
                "server_compute_time_ms": latency_ms,
            }
        )
        return jsonify(response_body)


if __name__ == "__main__":
    port = int(_get_env("SERVER_PORT", "5002"))
    print(f"[Server] Starting cluster={state.runtime_cfg.cluster_id} on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port)
