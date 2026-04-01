# # server/server.py
# """
# Modified split-learning server.

# Key changes:
# - Defaults aligned to a 3-layer LSTM with split at layer 2
# - Adds best-checkpoint save/load endpoints so test uses the same paired client/server snapshot
# - Keeps train/validate/inference routes unchanged in spirit
# """

# import os
# import pickle
# import sys
# from dataclasses import dataclass
# from threading import Lock
# from typing import Optional, Tuple

# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# import requests
# import torch
# from flask import Flask, Response, jsonify, request
# from torch import nn, optim

# from models.split_lstm import LSTMConfig, build_split_lstm


# def server_checkpoint_path(cluster_id: str, filename: str) -> str:
#     output_dir = os.path.join("output", f"cluster_{cluster_id}")
#     os.makedirs(output_dir, exist_ok=True)
#     return os.path.join(output_dir, f"server_{filename}")


# def save_torch_checkpoint(model, optimizer, path, extra=None):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     ckpt = {
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#     }
#     if extra:
#         ckpt.update(extra)
#     torch.save(ckpt, path)
#     print(f"[Server] Saved -> {path}")


# def load_torch_checkpoint(model, optimizer, path, map_location="cpu"):
#     if not os.path.exists(path):
#         return None
#     ckpt = torch.load(path, map_location=map_location)
#     model.load_state_dict(ckpt["model_state_dict"])
#     if optimizer is not None and "optimizer_state_dict" in ckpt:
#         optimizer.load_state_dict(ckpt["optimizer_state_dict"])
#     print(f"[Server] Loaded <- {path}")
#     return {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "optimizer_state_dict")}




# def try_load_server_checkpoint(model, optimizer, path, map_location="cpu"):
#     if not os.path.exists(path):
#         return None, False
#     try:
#         extra = load_torch_checkpoint(model, optimizer, path, map_location)
#         return extra, True
#     except Exception as e:
#         print(f"[Server] WARNING: skipping incompatible checkpoint {path}: {e}")
#         return None, False

# @dataclass
# class ServerRuntimeConfig:
#     cluster_id: str
#     learning_rate: float
#     save_every_steps: int
#     fed_server_url: str
#     num_samples: int


# def _get_env(name: str, default: str) -> str:
#     return os.environ.get(name, default)


# def parse_bool(val: str) -> bool:
#     return str(val).lower() in ("1", "true", "yes", "y")


# def build_runtime_config_from_env() -> ServerRuntimeConfig:
#     return ServerRuntimeConfig(
#         cluster_id=_get_env("CLUSTER_ID", "0"),
#         learning_rate=float(_get_env("LEARNING_RATE", "1e-3")),
#         save_every_steps=int(_get_env("CHECKPOINT_EVERY", _get_env("SAVE_EVERY_STEPS", "500"))),
#         fed_server_url=_get_env("FED_SERVER_URL", "http://localhost:5200"),
#         num_samples=int(_get_env("NUM_SAMPLES", "0")),
#     )


# def build_lstm_config_from_env() -> Tuple[LSTMConfig, int]:
#     cfg = LSTMConfig(
#         input_size=int(_get_env("MODEL_INPUT_SIZE", "1")),
#         hidden_size=int(_get_env("MODEL_HIDDEN_SIZE", "128")),
#         num_layers=int(_get_env("MODEL_NUM_LAYERS", "3")),
#         output_size=int(_get_env("MODEL_OUTPUT_SIZE", "1")),
#         dropout=float(_get_env("MODEL_DROPOUT", "0.4")),
#         bidirectional=parse_bool(_get_env("MODEL_BIDIRECTIONAL", "false")),
#     )
#     split_layer = int(_get_env("MODEL_SPLIT_LAYER", "2"))
#     if split_layer <= 0 or split_layer >= cfg.num_layers:
#         raise ValueError(
#             f"MODEL_SPLIT_LAYER must be in [1, num_layers-1]. Got split={split_layer}, num_layers={cfg.num_layers}"
#         )
#     return cfg, split_layer



# def resolve_device() -> torch.device:
#     gpu_enabled = parse_bool(_get_env("GPU_ENABLED", "true"))
#     gpu_id = _get_env("GPU_ID", "")
#     if gpu_enabled and torch.cuda.is_available():
#         if gpu_id not in (None, ""):
#             return torch.device(f"cuda:{gpu_id}")
#         return torch.device("cuda")
#     return torch.device("cpu")


# def get_device_info(device: torch.device) -> dict:
#     info = {
#         "selected_device": str(device),
#         "gpu_enabled": parse_bool(_get_env("GPU_ENABLED", "true")),
#         "gpu_id_env": _get_env("GPU_ID", ""),
#         "cuda_visible_devices": _get_env("CUDA_VISIBLE_DEVICES", ""),
#         "cuda_available": torch.cuda.is_available(),
#         "cuda_device_count": torch.cuda.device_count(),
#         "process_role": _get_env("PROCESS_ROLE", "server"),
#     }
#     if device.type == "cuda" and torch.cuda.is_available():
#         idx = device.index if device.index is not None else torch.cuda.current_device()
#         props = torch.cuda.get_device_properties(idx)
#         info.update({
#             "cuda_index": idx,
#             "cuda_name": torch.cuda.get_device_name(idx),
#             "cuda_capability": f"{props.major}.{props.minor}",
#             "cuda_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
#         })
#     return info


# class SplitLearningServerState:
#     def __init__(self):
#         self.runtime_cfg = build_runtime_config_from_env()
#         self.lstm_cfg, self.split_layer = build_lstm_config_from_env()
#         self.device = resolve_device()
#         self.device_info = get_device_info(self.device)

#         _, server_model = build_split_lstm(self.lstm_cfg, self.split_layer)
#         self.model = server_model.to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.runtime_cfg.learning_rate)
#         self.criterion = nn.MSELoss()
#         self.global_step = 0
#         self.lock = Lock()

#         latest_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
#         extra, ok = try_load_server_checkpoint(self.model, self.optimizer, latest_path, str(self.device))
#         if ok and extra and "global_step" in extra:
#             self.global_step = int(extra["global_step"])

#         print(f"[Server cluster={self.runtime_cfg.cluster_id}] device_info={self.device_info}", flush=True)

#     @staticmethod
#     def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
#         return x.unsqueeze(1) if x.dim() == 2 else x

#     def train_step(self, client_id, acts_cpu, tgts_cpu):
#         self.global_step += 1
#         acts = self._ensure_3d(acts_cpu.to(self.device))
#         tgts = tgts_cpu.to(self.device)
#         if tgts.dim() == 1:
#             tgts = tgts.unsqueeze(1)
#         acts.requires_grad_(True)
#         preds = self.model(acts)
#         loss = self.criterion(preds, tgts)
#         self.optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         self.optimizer.step()
#         grad = acts.grad.detach().cpu()
#         if grad.dim() == 3 and grad.size(1) == 1:
#             grad = grad.squeeze(1)
#         if self.runtime_cfg.save_every_steps > 0 and self.global_step % self.runtime_cfg.save_every_steps == 0:
#             self._save_latest()
#         return {"grad_activations": grad, "loss": float(loss.item())}

#     def validate_step(self, client_id, acts_cpu, tgts_cpu):
#         with torch.no_grad():
#             acts = self._ensure_3d(acts_cpu.to(self.device))
#             tgts = tgts_cpu.to(self.device)
#             if tgts.dim() == 1:
#                 tgts = tgts.unsqueeze(1)
#             preds = self.model(acts)
#             loss = self.criterion(preds, tgts)
#             return {"predictions": preds.detach().cpu().numpy(), "loss": float(loss.item())}

#     def inference_step(self, client_id, acts_cpu):
#         with torch.no_grad():
#             acts = self._ensure_3d(acts_cpu.to(self.device))
#             preds = self.model(acts).detach().cpu()
#             return {"predictions": preds.numpy(), "metrics": {"mean_prediction": float(preds.mean())}}

#     def _save_latest(self) -> str:
#         path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
#         save_torch_checkpoint(self.model, self.optimizer, path, extra={"global_step": self.global_step})
#         return path

#     def _save_best(self) -> str:
#         path = server_checkpoint_path(self.runtime_cfg.cluster_id, "best.pt")
#         save_torch_checkpoint(self.model, self.optimizer, path, extra={"global_step": self.global_step})
#         return path

#     def _load_best(self) -> str:
#         path = server_checkpoint_path(self.runtime_cfg.cluster_id, "best.pt")
#         if not os.path.exists(path):
#             raise FileNotFoundError(path)
#         extra, ok = try_load_server_checkpoint(self.model, self.optimizer, path, str(self.device))
#         if not ok:
#             raise FileNotFoundError(path)
#         return path

#     def load_weights_from_state_dict(self, state_dict):
#         self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
#         print(f"[Server cluster={self.runtime_cfg.cluster_id}] Global weights loaded")

#     def submit_weights_to_fed(self, num_samples: Optional[int] = None) -> bool:
#         n = num_samples or self.runtime_cfg.num_samples or 1
#         url = self.runtime_cfg.fed_server_url.rstrip("/") + "/collect_server_weights"
#         try:
#             payload = pickle.dumps(
#                 {
#                     "cluster_id": self.runtime_cfg.cluster_id,
#                     "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
#                     "num_samples": n,
#                 }
#             )
#             resp = requests.post(url, data=payload, headers={"Content-Type": "application/octet-stream"}, timeout=30.0)
#             resp.raise_for_status()
#             print(f"[Server cluster={self.runtime_cfg.cluster_id}] Weights submitted to fed server (n={n})")
#             return True
#         except Exception as e:
#             print(f"[Server cluster={self.runtime_cfg.cluster_id}] WARNING: could not submit to fed server: {e}")
#             return False


# def _unpickle():
#     if not request.data:
#         raise ValueError("Empty request")
#     p = pickle.loads(request.data)
#     if not isinstance(p, dict):
#         raise ValueError("Expected pickled dict")
#     return p


# def _pkl(obj):
#     return Response(pickle.dumps(obj), content_type="application/octet-stream")


# def create_app() -> Flask:
#     app = Flask(__name__)
#     state = SplitLearningServerState()

#     @app.route("/healthz", methods=["GET"])
#     def healthz():
#         return jsonify({"status": "ok", "cluster_id": state.runtime_cfg.cluster_id, "global_step": state.global_step, "device": state.device_info})

#     @app.route("/config", methods=["GET"])
#     def config():
#         return jsonify({
#             "component": "server",
#             "cluster_id": state.runtime_cfg.cluster_id,
#             "global_step": state.global_step,
#             "device": state.device_info,
#             "runtime": {
#                 "learning_rate": state.runtime_cfg.learning_rate,
#                 "save_every_steps": state.runtime_cfg.save_every_steps,
#                 "fed_server_url": state.runtime_cfg.fed_server_url,
#                 "num_samples": state.runtime_cfg.num_samples,
#             },
#             "model": {
#                 "input_size": state.lstm_cfg.input_size,
#                 "hidden_size": state.lstm_cfg.hidden_size,
#                 "num_layers": state.lstm_cfg.num_layers,
#                 "output_size": state.lstm_cfg.output_size,
#                 "dropout": state.lstm_cfg.dropout,
#                 "bidirectional": state.lstm_cfg.bidirectional,
#                 "split_layer": state.split_layer,
#             },
#             "checkpoint_path": server_checkpoint_path(state.runtime_cfg.cluster_id, "latest.pt"),
#         })

#     @app.route("/train_step", methods=["POST"])
#     def train_step_route():
#         with state.lock:
#             p = _unpickle()
#             r = state.train_step(str(p.get("client_id", "?")), p["activations"], p["targets"])
#             return _pkl({"grad_activations": r["grad_activations"], "loss": r["loss"]})

#     @app.route("/validate_step", methods=["POST"])
#     def validate_step_route():
#         with state.lock:
#             p = _unpickle()
#             r = state.validate_step(str(p.get("client_id", "?")), p["activations"], p["targets"])
#             return _pkl({"predictions": r["predictions"], "loss": r["loss"]})

#     @app.route("/inference_step", methods=["POST"])
#     def inference_step_route():
#         with state.lock:
#             p = _unpickle()
#             r = state.inference_step(str(p.get("client_id", "?")), p["activations"])
#             return _pkl({"predictions": r["predictions"], "metrics": r["metrics"]})

#     @app.route("/save_checkpoint", methods=["POST"])
#     def save_checkpoint_route():
#         with state.lock:
#             path = state._save_latest()
#         return jsonify({"status": "saved", "path": path, "global_step": state.global_step})

#     @app.route("/save_best_checkpoint", methods=["POST"])
#     def save_best_checkpoint_route():
#         with state.lock:
#             path = state._save_best()
#         return jsonify({"status": "saved", "path": path, "global_step": state.global_step})

#     @app.route("/load_best_checkpoint", methods=["POST"])
#     def load_best_checkpoint_route():
#         with state.lock:
#             path = state._load_best()
#         return jsonify({"status": "loaded", "path": path, "global_step": state.global_step})

#     @app.route("/load_weights", methods=["POST"])
#     def load_weights_route():
#         with state.lock:
#             p = _unpickle()
#             sd = p.get("model_state_dict")
#             if sd is None:
#                 return jsonify({"error": "missing model_state_dict"}), 400
#             state.load_weights_from_state_dict(sd)
#         return jsonify({"status": "loaded", "global_step": state.global_step})

#     @app.route("/submit_to_fed", methods=["POST"])
#     def submit_to_fed_route():
#         body = request.get_json(silent=True) or {}
#         n = body.get("num_samples") or state.runtime_cfg.num_samples
#         with state.lock:
#             ok = state.submit_weights_to_fed(num_samples=n)
#         return jsonify({"status": "submitted" if ok else "failed", "cluster_id": state.runtime_cfg.cluster_id, "global_step": state.global_step})

#     return app


# if __name__ == "__main__":
#     port = int(_get_env("SERVER_PORT", "5002"))
#     create_app().run(host="0.0.0.0", port=port)
# server/server.py
"""
Split-learning server with:
- shared live server model per cluster
- paired best checkpoint endpoints
- final shared checkpoint endpoints
- GPU/device reporting
"""

import os
import pickle
import sys
from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import requests
import torch
from flask import Flask, Response, jsonify, request
from torch import nn, optim

from models.split_lstm import LSTMConfig, build_split_lstm


def server_checkpoint_path(cluster_id: str, filename: str) -> str:
    output_dir = os.path.join("output", f"cluster_{cluster_id}")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"server_{filename}")


def save_torch_checkpoint(model, optimizer, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    print(f"[Server] Saved -> {path}")


def load_torch_checkpoint(model, optimizer, path, map_location="cpu"):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Server] Loaded <- {path}")
    return {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "optimizer_state_dict")}


def try_load_server_checkpoint(model, optimizer, path, map_location="cpu"):
    if not os.path.exists(path):
        return None, False
    try:
        extra = load_torch_checkpoint(model, optimizer, path, map_location)
        return extra, True
    except Exception as e:
        print(f"[Server] WARNING: skipping incompatible checkpoint {path}: {e}")
        return None, False


@dataclass
class ServerRuntimeConfig:
    cluster_id: str
    learning_rate: float
    save_every_steps: int
    fed_server_url: str
    num_samples: int


def _get_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_bool(val: str) -> bool:
    return str(val).lower() in ("1", "true", "yes", "y")


def build_runtime_config_from_env() -> ServerRuntimeConfig:
    return ServerRuntimeConfig(
        cluster_id=_get_env("CLUSTER_ID", "0"),
        learning_rate=float(_get_env("LEARNING_RATE", "0.0005")),
        save_every_steps=int(_get_env("CHECKPOINT_EVERY", _get_env("SAVE_EVERY_STEPS", "500"))),
        fed_server_url=_get_env("FED_SERVER_URL", "http://localhost:5200"),
        num_samples=int(_get_env("NUM_SAMPLES", "0")),
    )


def build_lstm_config_from_env() -> Tuple[LSTMConfig, int]:
    cfg = LSTMConfig(
        input_size=int(_get_env("MODEL_INPUT_SIZE", "1")),
        hidden_size=int(_get_env("MODEL_HIDDEN_SIZE", "128")),
        num_layers=int(_get_env("MODEL_NUM_LAYERS", "3")),
        output_size=int(_get_env("MODEL_OUTPUT_SIZE", "1")),
        dropout=float(_get_env("MODEL_DROPOUT", "0.4")),
        bidirectional=parse_bool(_get_env("MODEL_BIDIRECTIONAL", "false")),
    )
    split_layer = int(_get_env("MODEL_SPLIT_LAYER", "2"))
    if split_layer <= 0 or split_layer >= cfg.num_layers:
        raise ValueError(f"MODEL_SPLIT_LAYER must be in [1, num_layers-1]. Got split={split_layer}, num_layers={cfg.num_layers}")
    return cfg, split_layer


def resolve_device() -> torch.device:
    gpu_enabled = parse_bool(_get_env("GPU_ENABLED", "true"))
    gpu_id = _get_env("GPU_ID", "")
    if gpu_enabled and torch.cuda.is_available():
        if gpu_id not in (None, ""):
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    info = {
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


class SplitLearningServerState:
    def __init__(self):
        self.runtime_cfg = build_runtime_config_from_env()
        self.lstm_cfg, self.split_layer = build_lstm_config_from_env()
        self.device = resolve_device()
        self.device_info = get_device_info(self.device)

        _, server_model = build_split_lstm(self.lstm_cfg, self.split_layer)
        self.model = server_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.runtime_cfg.learning_rate)
        self.criterion = nn.MSELoss()
        self.global_step = 0
        self.client_step_counts = {}
        self.lock = Lock()

        latest_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
        extra, ok = try_load_server_checkpoint(self.model, self.optimizer, latest_path, str(self.device))
        if ok and extra and "global_step" in extra:
            self.global_step = int(extra["global_step"])

        print(f"[Server cluster={self.runtime_cfg.cluster_id}] device_info={self.device_info}", flush=True)

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x

    def train_step(self, client_id, acts_cpu, tgts_cpu):
        self.global_step += 1
        self.client_step_counts[client_id] = int(self.client_step_counts.get(client_id, 0)) + 1

        acts = self._ensure_3d(acts_cpu.to(self.device))
        tgts = tgts_cpu.to(self.device)
        if tgts.dim() == 1:
            tgts = tgts.unsqueeze(1)

        acts.requires_grad_(True)
        preds = self.model(acts)
        loss = self.criterion(preds, tgts)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        grad = acts.grad.detach().cpu()
        if grad.dim() == 3 and grad.size(1) == 1:
            grad = grad.squeeze(1)

        if self.runtime_cfg.save_every_steps > 0 and self.global_step % self.runtime_cfg.save_every_steps == 0:
            self._save_latest()

        return {"grad_activations": grad, "loss": float(loss.item())}

    def validate_step(self, client_id, acts_cpu, tgts_cpu):
        with torch.no_grad():
            acts = self._ensure_3d(acts_cpu.to(self.device))
            tgts = tgts_cpu.to(self.device)
            if tgts.dim() == 1:
                tgts = tgts.unsqueeze(1)
            preds = self.model(acts)
            loss = self.criterion(preds, tgts)
            return {"predictions": preds.detach().cpu().numpy(), "loss": float(loss.item())}

    def inference_step(self, client_id, acts_cpu):
        with torch.no_grad():
            acts = self._ensure_3d(acts_cpu.to(self.device))
            preds = self.model(acts).detach().cpu()
            return {"predictions": preds.numpy(), "metrics": {"mean_prediction": float(preds.mean())}}

    def _save_latest(self) -> str:
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
        save_torch_checkpoint(self.model, self.optimizer, path, extra={"global_step": self.global_step})
        return path

    def _save_best(self) -> str:
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "best.pt")
        save_torch_checkpoint(self.model, self.optimizer, path, extra={"global_step": self.global_step})
        return path

    def _load_best(self) -> str:
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "best.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        extra, ok = try_load_server_checkpoint(self.model, self.optimizer, path, str(self.device))
        if not ok:
            raise FileNotFoundError(path)
        if extra and "global_step" in extra:
            self.global_step = int(extra["global_step"])
        return path

    def _save_final(self) -> str:
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "final.pt")
        save_torch_checkpoint(self.model, self.optimizer, path, extra={"global_step": self.global_step})
        return path

    def _load_final(self) -> str:
        path = server_checkpoint_path(self.runtime_cfg.cluster_id, "final.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        extra, ok = try_load_server_checkpoint(self.model, self.optimizer, path, str(self.device))
        if not ok:
            raise FileNotFoundError(path)
        if extra and "global_step" in extra:
            self.global_step = int(extra["global_step"])
        return path

    def load_weights_from_state_dict(self, state_dict):
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})
        print(f"[Server cluster={self.runtime_cfg.cluster_id}] Global weights loaded")

    def submit_weights_to_fed(self, num_samples: Optional[int] = None) -> bool:
        n = num_samples or self.runtime_cfg.num_samples or 1
        url = self.runtime_cfg.fed_server_url.rstrip("/") + "/collect_server_weights"
        try:
            payload = pickle.dumps(
                {
                    "cluster_id": self.runtime_cfg.cluster_id,
                    "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                    "num_samples": n,
                }
            )
            resp = requests.post(url, data=payload, headers={"Content-Type": "application/octet-stream"}, timeout=30.0)
            resp.raise_for_status()
            print(f"[Server cluster={self.runtime_cfg.cluster_id}] Weights submitted to fed server (n={n})")
            return True
        except Exception as e:
            print(f"[Server cluster={self.runtime_cfg.cluster_id}] WARNING: could not submit to fed server: {e}")
            return False


def _unpickle():
    if not request.data:
        raise ValueError("Empty request")
    p = pickle.loads(request.data)
    if not isinstance(p, dict):
        raise ValueError("Expected pickled dict")
    return p


def _pkl(obj):
    return Response(pickle.dumps(obj), content_type="application/octet-stream")


def create_app() -> Flask:
    app = Flask(__name__)
    state = SplitLearningServerState()

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify(
            {
                "status": "ok",
                "cluster_id": state.runtime_cfg.cluster_id,
                "global_step": state.global_step,
                "device": state.device_info,
            }
        )

    @app.route("/config", methods=["GET"])
    def config():
        return jsonify(
            {
                "component": "server",
                "cluster_id": state.runtime_cfg.cluster_id,
                "global_step": state.global_step,
                "device": state.device_info,
                "client_step_counts": state.client_step_counts,
                "runtime": {
                    "learning_rate": state.runtime_cfg.learning_rate,
                    "save_every_steps": state.runtime_cfg.save_every_steps,
                    "fed_server_url": state.runtime_cfg.fed_server_url,
                    "num_samples": state.runtime_cfg.num_samples,
                    "train_loss_name": "MSELoss",
                    "selection_metric": "raw MAE on client validation",
                },
                "model": {
                    "input_size": state.lstm_cfg.input_size,
                    "hidden_size": state.lstm_cfg.hidden_size,
                    "num_layers": state.lstm_cfg.num_layers,
                    "output_size": state.lstm_cfg.output_size,
                    "dropout": state.lstm_cfg.dropout,
                    "bidirectional": state.lstm_cfg.bidirectional,
                    "split_layer": state.split_layer,
                },
                "checkpoint_paths": {
                    "latest": server_checkpoint_path(state.runtime_cfg.cluster_id, "latest.pt"),
                    "best": server_checkpoint_path(state.runtime_cfg.cluster_id, "best.pt"),
                    "final": server_checkpoint_path(state.runtime_cfg.cluster_id, "final.pt"),
                },
            }
        )

    @app.route("/train_step", methods=["POST"])
    def train_step_route():
        with state.lock:
            p = _unpickle()
            r = state.train_step(str(p.get("client_id", "?")), p["activations"], p["targets"])
            return _pkl({"grad_activations": r["grad_activations"], "loss": r["loss"]})

    @app.route("/validate_step", methods=["POST"])
    def validate_step_route():
        with state.lock:
            p = _unpickle()
            r = state.validate_step(str(p.get("client_id", "?")), p["activations"], p["targets"])
            return _pkl({"predictions": r["predictions"], "loss": r["loss"]})

    @app.route("/inference_step", methods=["POST"])
    def inference_step_route():
        with state.lock:
            p = _unpickle()
            r = state.inference_step(str(p.get("client_id", "?")), p["activations"])
            return _pkl({"predictions": r["predictions"], "metrics": r["metrics"]})

    @app.route("/save_checkpoint", methods=["POST"])
    def save_checkpoint_route():
        with state.lock:
            path = state._save_latest()
        return jsonify({"status": "saved", "path": path, "global_step": state.global_step})

    @app.route("/save_best_checkpoint", methods=["POST"])
    def save_best_checkpoint_route():
        with state.lock:
            path = state._save_best()
        return jsonify({"status": "saved", "path": path, "global_step": state.global_step})

    @app.route("/load_best_checkpoint", methods=["POST"])
    def load_best_checkpoint_route():
        with state.lock:
            path = state._load_best()
        return jsonify({"status": "loaded", "path": path, "global_step": state.global_step})

    @app.route("/save_final_checkpoint", methods=["POST"])
    def save_final_checkpoint_route():
        with state.lock:
            path = state._save_final()
        return jsonify({"status": "saved", "path": path, "global_step": state.global_step})

    @app.route("/load_final_checkpoint", methods=["POST"])
    def load_final_checkpoint_route():
        with state.lock:
            path = state._load_final()
        return jsonify({"status": "loaded", "path": path, "global_step": state.global_step})

    @app.route("/load_weights", methods=["POST"])
    def load_weights_route():
        with state.lock:
            p = _unpickle()
            sd = p.get("model_state_dict")
            if sd is None:
                return jsonify({"error": "missing model_state_dict"}), 400
            state.load_weights_from_state_dict(sd)
        return jsonify({"status": "loaded", "global_step": state.global_step})

    @app.route("/submit_to_fed", methods=["POST"])
    def submit_to_fed_route():
        body = request.get_json(silent=True) or {}
        n = body.get("num_samples") or state.runtime_cfg.num_samples
        with state.lock:
            ok = state.submit_weights_to_fed(num_samples=n)
        return jsonify(
            {
                "status": "submitted" if ok else "failed",
                "cluster_id": state.runtime_cfg.cluster_id,
                "global_step": state.global_step,
            }
        )

    return app


if __name__ == "__main__":
    port = int(_get_env("SERVER_PORT", "5002"))
    create_app().run(host="0.0.0.0", port=port)