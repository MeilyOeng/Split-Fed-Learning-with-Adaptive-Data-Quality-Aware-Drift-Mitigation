#!/usr/bin/env python3
import os
import pickle
import sys
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from flask import Flask, Response, jsonify, request
from torch import nn, optim

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.split_lstm import LSTMConfig, build_split_lstm


app = Flask(__name__)


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val


def parse_bool(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "y")


def resolve_device() -> torch.device:
    gpu_enabled = parse_bool(_get_env("GPU_ENABLED", "true"))
    gpu_id = _get_env("GPU_ID", "")
    if gpu_enabled and torch.cuda.is_available():
        if gpu_id not in (None, ""):
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "selected_device": str(device),
        "gpu_enabled": parse_bool(_get_env("GPU_ENABLED", "true")),
        "gpu_id_env": _get_env("GPU_ID", ""),
        "cuda_visible_devices": _get_env("CUDA_VISIBLE_DEVICES", ""),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "process_role": _get_env("PROCESS_ROLE", "server"),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "cuda_index": idx,
                "cuda_name": torch.cuda.get_device_name(idx),
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
            }
        )
    return info


def server_output_dir(cluster_id: str) -> str:
    output_dir = _get_env("OUTPUT_DIR", os.path.join("output", f"cluster_{cluster_id}"))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def server_checkpoint_path(cluster_id: str, filename: str) -> str:
    return os.path.join(server_output_dir(cluster_id), f"server_{filename}")


def save_torch_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[Server] Saved -> {path}", flush=True)


def try_load_server_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], path: str, map_location: str = "cpu") -> Tuple[Optional[Dict[str, Any]], bool]:
    if not os.path.exists(path):
        return None, False
    try:
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[Server] Loaded <- {path}", flush=True)
        extra = {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict")}
        return extra, True
    except Exception as e:
        print(f"[Server] WARNING: skipping incompatible checkpoint {path}: {e}", flush=True)
        return None, False


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
        self.device_info = get_device_info(self.device)
        cfg = LSTMConfig(
            input_size=int(_get_env("MODEL_INPUT_SIZE", "1")),
            hidden_size=int(_get_env("MODEL_HIDDEN_SIZE", "128")),
            num_layers=int(_get_env("MODEL_NUM_LAYERS", "3")),
            output_size=int(_get_env("MODEL_OUTPUT_SIZE", "1")),
            dropout=float(_get_env("MODEL_DROPOUT", "0.4")),
            bidirectional=parse_bool(_get_env("MODEL_BIDIRECTIONAL", "false")),
        )
        split_layer = int(_get_env("MODEL_SPLIT_LAYER", "2"))
        _, server_model = build_split_lstm(cfg, split_layer)
        self.model = server_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.runtime_cfg.learning_rate)
        self.criterion = nn.MSELoss()
        self.lock = Lock()

        self.global_step = 0
        self.optimizer_step_count = 0
        self.pending_accum_steps = 0
        self.round_id = 0
        self.expected_clients: List[str] = []
        self.expected_clients_set: Set[str] = set()
        self.seen_clients: Set[str] = set()
        self.round_contrib_count = 0
        # try loading latest checkpoint on startup
        latest_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt") 
        extra, ok = try_load_server_checkpoint(self.model, self.optimizer, latest_path, map_location=str(self.device))
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

    @staticmethod
    def _ensure_lstm_input_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
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
    # just to reset the tracking of each round's progress after a flush
    def _reset_round_tracking(self, increment_round: bool) -> None:
        self.pending_accum_steps = 0
        self.seen_clients.clear()
        self.round_contrib_count = 0
        if increment_round:
            self.round_id += 1
    # check for expected clients, track seen clients and contributions
    def set_round(self, expected_clients: List[str], round_id: Optional[int] = None) -> Dict[str, Any]:
        unique_clients = [str(c) for c in expected_clients]
        if len(unique_clients) != len(set(unique_clients)):
            raise ValueError("expected_clients contains duplicates")
        if self.pending_accum_steps > 0:
            raise RuntimeError("cannot start a new round while gradients are pending")
        self.expected_clients = unique_clients
        self.expected_clients_set = set(unique_clients)
        self.seen_clients.clear()
        self.round_contrib_count = 0
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
            "seen_clients": sorted(self.seen_clients),
            "round_contrib_count": self.round_contrib_count,
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
        seen_clients = sorted(self.seen_clients)
        self._reset_round_tracking(increment_round=increment_round)
        print(
            f"[Server] flush reason={reason} cluster={self.runtime_cfg.cluster_id} round_id={finished_round_id} "
            f"effective_accum_steps={effective_accum_steps} seen_clients={seen_clients} optimizer_steps={self.optimizer_step_count}",
            flush=True,
        )
        if increment_round and self.runtime_cfg.save_every_rounds > 0 and finished_round_id % self.runtime_cfg.save_every_rounds == 0:
            self._save_latest(flush=False)
        return {
            "flushed": True,
            "reason": reason,
            "finished_round_id": finished_round_id,
            "effective_accum_steps": effective_accum_steps,
            "seen_clients": seen_clients,
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
        if client_id in self.seen_clients:              # Avoid accepting multiple contributions from the same client in the same round
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
        if grad_activations.dim() == 3 and grad_activations.size(1) == 1:
            grad_activations = grad_activations.squeeze(1)

        self.pending_accum_steps += 1
        self.seen_clients.add(client_id)
        self.round_contrib_count = len(self.seen_clients)

        round_complete = self.seen_clients == self.expected_clients_set
        flush_info: Optional[Dict[str, Any]] = None
        if round_complete:
            flush_info = self.flush_accumulated_gradients(reason="round_complete", increment_round=True)

        print(
            f"[Server] train_step cluster={self.runtime_cfg.cluster_id} round_id={self.round_id if not round_complete else flush_info['finished_round_id']} "
            f"client={client_id} loss={raw_loss:.6f} seen={sorted(self.seen_clients) if not round_complete else flush_info['seen_clients']} "
            f"pending_accum={self.pending_accum_steps}",
            flush=True,
        )
        return {
            "grad_activations": grad_activations,
            "loss": raw_loss,
            "round_complete": round_complete,
            "flush_info": flush_info,
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
        save_torch_checkpoint(self.model, self.optimizer, path, extra=self._checkpoint_extra())
        return path

    def _save_best(self, flush: bool = True) -> str:
        if flush:
            self._flush_before_eval_if_needed(reason="save_best")
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "best.pt")
        save_torch_checkpoint(self.model, self.optimizer, path, extra=self._checkpoint_extra())
        return path

    def _save_final(self, flush: bool = True) -> str:
        if flush:
            self._flush_before_eval_if_needed(reason="save_final")
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "final.pt")
        save_torch_checkpoint(self.model, self.optimizer, path, extra=self._checkpoint_extra())
        return path

    def _load_best(self) -> str:
        self._flush_before_eval_if_needed(reason="load_best")
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "best.pt")
        extra, ok = try_load_server_checkpoint(self.model, self.optimizer, path, map_location=str(self.device))
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
        extra, ok = try_load_server_checkpoint(self.model, self.optimizer, latest_path, map_location=str(self.device))
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
        extra, ok = try_load_server_checkpoint(self.model, self.optimizer, target, map_location=str(self.device))
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
    with state.lock:
        try:
            payload = _unpickle_request_data()
            result = state.train_step(
                client_id=str(payload["client_id"]),
                activations_cpu=payload["activations"],
                targets_cpu=payload["targets"],
            )
            return _pickle_response(result)
        except Exception as e:
            return jsonify({"error": str(e), **state.round_state()}), 409


@app.route("/validate_step", methods=["POST"])
def validate_step():
    with state.lock:
        payload = _unpickle_request_data()
        result = state.validate_step(
            str(payload["client_id"]), 
            payload["activations"], 
            payload["targets"]
        )
        return _pickle_response(result)


@app.route("/inference_step", methods=["POST"])
def inference_step():
    with state.lock:
        payload = _unpickle_request_data()
        result = state.inference_step(
            str(payload["client_id"]), 
            payload["activations"])
        return _pickle_response(result)


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
    with state.lock:
        payload = _unpickle_request_data()
        meta = state.load_weights(payload["model_state_dict"])
        return jsonify({"status": "ok", **meta})


if __name__ == "__main__":
    port = int(_get_env("SERVER_PORT", "5002"))
    print(f"[Server] Starting cluster={state.runtime_cfg.cluster_id} on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port)
