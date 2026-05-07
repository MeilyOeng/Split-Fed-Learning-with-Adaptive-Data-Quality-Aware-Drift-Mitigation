from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, optim

from models.split_lstm import LSTMConfig


def parse_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y")


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return get_env(name, default)


def resolve_device() -> torch.device:
    gpu_enabled = parse_bool(get_env("GPU_ENABLED", "true"), default=True)
    gpu_id = get_env("GPU_ID", "")
    if gpu_enabled and torch.cuda.is_available():
        if gpu_id not in (None, ""):
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: torch.device, *, process_role_default: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "selected_device": str(device),
        "gpu_enabled": parse_bool(get_env("GPU_ENABLED", "true"), default=True),
        "gpu_id_env": get_env("GPU_ID", ""),
        "cuda_visible_devices": get_env("CUDA_VISIBLE_DEVICES", ""),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "process_role": get_env("PROCESS_ROLE", process_role_default),
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


def build_lstm_config_from_env() -> Tuple[LSTMConfig, int]:
    cfg = LSTMConfig(
        input_size=int(get_env("MODEL_INPUT_SIZE", "1")),
        hidden_size=int(get_env("MODEL_HIDDEN_SIZE", "128")),
        num_layers=int(get_env("MODEL_NUM_LAYERS", "3")),
        output_size=int(get_env("MODEL_OUTPUT_SIZE", "1")),
        dropout=float(get_env("MODEL_DROPOUT", "0.4")),
        bidirectional=parse_bool(get_env("MODEL_BIDIRECTIONAL", "false")),
    )
    split_layer = int(get_env("MODEL_SPLIT_LAYER", "2"))
    if split_layer <= 0 or split_layer >= cfg.num_layers:
        raise ValueError(
            f"MODEL_SPLIT_LAYER must be in [1, num_layers-1], got split={split_layer}, num_layers={cfg.num_layers}"
        )
    return cfg, split_layer


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str | Path, rows, fieldnames) -> None:
    ensure_parent(path)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_torch_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    path: str | Path,
    *,
    extra: Optional[Dict[str, Any]] = None,
    log_prefix: str = "Checkpoint",
) -> None:
    payload = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))
    print(f"[{log_prefix}] Saved -> {path}", flush=True)


def try_load_torch_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    path: str | Path,
    *,
    map_location: str = "cpu",
    strict: bool = True,
    log_prefix: str = "Checkpoint",
) -> Tuple[Optional[Dict[str, Any]], bool]:
    path = str(path)
    if not os.path.exists(path):
        return None, False
    try:
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"[{log_prefix}] WARNING: optimizer state incompatible for {path}: {e}", flush=True)
        print(f"[{log_prefix}] Loaded <- {path}", flush=True)
        extra = {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict")}
        return extra, True
    except Exception as e:
        print(f"[{log_prefix}] WARNING: skipping incompatible checkpoint {path}: {e}", flush=True)
        return None, False
