from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import requests
import torch
from dotenv import find_dotenv, load_dotenv
from torch import nn, optim

from models.split_lstm import LSTMConfig


def parse_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y")


def derive_dq_mode(
    use_quality_weights: bool,
    use_quality_as_input: bool,
    use_quality_aggregation_weights: bool = False,
    use_quality_anchor_aggregation: bool = False,
    use_gradient_alignment_filter: bool = False,
    use_high_quality_validation_guardrail: bool = False,
) -> str:
    active = []
    if use_quality_as_input:
        active.append("input_feature")
    if use_quality_weights:
        active.append("window_loss_weight")
    if use_quality_aggregation_weights:
        active.append("soft_client_gradient_weight")
    if use_quality_anchor_aggregation:
        active.append("quality_anchor_aggregation")
    if use_gradient_alignment_filter:
        active.append("gradient_alignment_filter")
    if use_high_quality_validation_guardrail:
        active.append("high_quality_validation_guardrail")
    return "none" if not active else "_and_".join(active)


def quality_scores_needed(
    use_quality_weights: bool,
    use_quality_as_input: bool,
    use_quality_aggregation_weights: bool = False,
    use_quality_anchor_aggregation: bool = False,
) -> bool:
    return bool(
        use_quality_weights
        or use_quality_as_input
        or use_quality_aggregation_weights
        or use_quality_anchor_aggregation
    )


@dataclass(frozen=True)
class QualityAnchorConfig:
    use_quality_anchor_aggregation: bool
    use_gradient_alignment_filter: bool
    use_high_quality_validation_guardrail: bool
    top_fraction: float
    min_clients: int
    anchor_update_share: float
    alignment_power: float
    negative_alignment_weight: float
    low_client_norm_cap: float
    eps: float
    debug_logging: bool


def build_quality_anchor_config(env: Mapping[str, Any]) -> QualityAnchorConfig:
    config = QualityAnchorConfig(
        use_quality_anchor_aggregation=parse_bool(env.get("USE_QUALITY_ANCHOR_AGGREGATION", "false")),
        use_gradient_alignment_filter=parse_bool(env.get("USE_GRADIENT_ALIGNMENT_FILTER", "false")),
        use_high_quality_validation_guardrail=parse_bool(
            env.get("USE_HIGH_QUALITY_VALIDATION_GUARDRAIL", "false")
        ),
        top_fraction=float(env.get("QUALITY_ANCHOR_TOP_FRACTION", "0.40")),
        min_clients=int(env.get("QUALITY_ANCHOR_MIN_CLIENTS", "3")),
        anchor_update_share=float(env.get("QUALITY_ANCHOR_UPDATE_SHARE", "0.80")),
        alignment_power=float(env.get("QUALITY_ALIGNMENT_POWER", "1.0")),
        negative_alignment_weight=float(env.get("QUALITY_NEGATIVE_ALIGNMENT_WEIGHT", "0.0")),
        low_client_norm_cap=float(env.get("QUALITY_LOW_CLIENT_NORM_CAP", "2.0")),
        eps=float(env.get("QUALITY_ANCHOR_EPS", "1e-8")),
        debug_logging=parse_bool(env.get("QUALITY_AGGREGATION_DEBUG_LOG", "false")),
    )

    if config.use_high_quality_validation_guardrail:
        raise ValueError(
            "USE_HIGH_QUALITY_VALIDATION_GUARDRAIL is reserved for the next implementation stage."
        )
    if config.use_gradient_alignment_filter and not config.use_quality_anchor_aggregation:
        raise ValueError(
            "USE_GRADIENT_ALIGNMENT_FILTER=true requires USE_QUALITY_ANCHOR_AGGREGATION=true."
        )

    legacy_flags = {
        "USE_QUALITY_AS_INPUT": parse_bool(env.get("USE_QUALITY_AS_INPUT", "false")),
        "USE_QUALITY_WEIGHTS": parse_bool(env.get("USE_QUALITY_WEIGHTS", "false")),
        "USE_QUALITY_AGGREGATION_WEIGHTS": parse_bool(
            env.get("USE_QUALITY_AGGREGATION_WEIGHTS", "false")
        ),
    }
    active_legacy = [name for name, enabled in legacy_flags.items() if enabled]
    if config.use_quality_anchor_aggregation and active_legacy:
        raise ValueError(
            "USE_QUALITY_ANCHOR_AGGREGATION=true is mutually exclusive with legacy quality modes: "
            + ", ".join(active_legacy)
        )

    if not (0.0 < config.top_fraction <= 1.0):
        raise ValueError("QUALITY_ANCHOR_TOP_FRACTION must be within (0, 1].")
    if config.min_clients < 1:
        raise ValueError("QUALITY_ANCHOR_MIN_CLIENTS must be at least 1.")
    if not (0.0 < config.anchor_update_share <= 1.0):
        raise ValueError("QUALITY_ANCHOR_UPDATE_SHARE must be within (0, 1].")
    if config.alignment_power <= 0.0:
        raise ValueError("QUALITY_ALIGNMENT_POWER must be positive.")
    if not (0.0 <= config.negative_alignment_weight <= 1.0):
        raise ValueError("QUALITY_NEGATIVE_ALIGNMENT_WEIGHT must be within [0, 1].")
    if config.low_client_norm_cap <= 0.0:
        raise ValueError("QUALITY_LOW_CLIENT_NORM_CAP must be positive.")
    if config.eps <= 0.0:
        raise ValueError("QUALITY_ANCHOR_EPS must be positive.")
    return config


def compute_model_input_size(
    base_input_size: int,
    use_quality_as_input: bool,
    quality_input_dim: int = 1,
) -> int:
    return int(base_input_size) + (int(quality_input_dim) if use_quality_as_input else 0)


def augment_sequence_inputs_with_quality(
    X: torch.Tensor,
    quality_scores: Any,
    *,
    enabled: bool,
    mode: str,
) -> torch.Tensor:
    if not enabled:
        return X
    if X.dim() != 3:
        raise ValueError(
            f"Quality-as-input requires X with shape [B, T, F], got shape={tuple(X.shape)} during mode={mode}"
        )
    if quality_scores is None:
        raise ValueError(f"quality_scores missing while USE_QUALITY_AS_INPUT=true during mode={mode}")

    if not torch.is_tensor(quality_scores):
        quality_tensor = torch.as_tensor(quality_scores, dtype=X.dtype, device=X.device)
    else:
        quality_tensor = quality_scores.to(device=X.device, dtype=X.dtype)

    batch_size = int(X.size(0))
    if quality_tensor.dim() == 0:
        quality_tensor = quality_tensor.reshape(1, 1)
    elif quality_tensor.dim() == 1:
        quality_tensor = quality_tensor.view(-1, 1)
    elif quality_tensor.dim() == 2:
        pass
    else:
        quality_tensor = quality_tensor.reshape(batch_size, -1)

    if quality_tensor.size(0) != batch_size:
        raise ValueError(
            f"quality_scores size mismatch during mode={mode}: expected batch_size={batch_size}, "
            f"got first_dim={quality_tensor.size(0)}"
        )
    if not torch.isfinite(quality_tensor).all():
        raise ValueError(f"quality_scores contain non-finite values during mode={mode}")
    if torch.any((quality_tensor < 0.0) | (quality_tensor > 1.0)):
        raise ValueError(
            f"quality_scores must be within [0, 1] during mode={mode}; "
            f"min={float(quality_tensor.min().item()):.6f} max={float(quality_tensor.max().item()):.6f}"
        )

    q = quality_tensor.view(batch_size, 1, -1).expand(-1, X.size(1), -1)
    return torch.cat([X, q], dim=-1)


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return val


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return get_env(name, default)


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def cluster_dir(output_dir: str | Path, cluster_id: str) -> Path:
    return Path(output_dir) / f"cluster_{cluster_id}"


def parse_csv_ids(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_env_file(filename: str = ".env") -> Path | None:
    dotenv_path = find_dotenv(filename, usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return Path(dotenv_path)
    fallback = Path(filename)
    if fallback.exists():
        load_dotenv(fallback, override=False)
        return fallback.resolve()
    return None


def wait_for_service(url: str, max_attempts: int = 60, delay_sec: float = 1.0) -> bool:
    for _ in range(max_attempts):
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay_sec)
    return False


def post_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 120.0):
    response = requests.post(url, json=payload or {}, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = (response.text or "").strip()
        if body:
            raise RuntimeError(f"POST {url} failed with HTTP {response.status_code}: {body}") from exc
        raise
    return response.json()


def get_json(url: str, timeout: float = 30.0):
    response = requests.get(url, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = (response.text or "").strip()
        if body:
            raise RuntimeError(f"GET {url} failed with HTTP {response.status_code}: {body}") from exc
        raise
    return response.json()


def client_progress_path(output_dir: str | Path, cluster_id: str, client_id: str) -> Path:
    return cluster_dir(output_dir, cluster_id) / f"client_{client_id}_progress.json"


def client_train_metrics_path(output_dir: str | Path, cluster_id: str, client_id: str) -> Path:
    return cluster_dir(output_dir, cluster_id) / f"client_{client_id}_train_metrics.json"


def client_test_metrics_path(output_dir: str | Path, cluster_id: str, client_id: str) -> Path:
    return cluster_dir(output_dir, cluster_id) / f"client_{client_id}_test_metrics.json"


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
