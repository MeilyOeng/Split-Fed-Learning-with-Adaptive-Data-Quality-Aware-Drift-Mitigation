from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


COMM_LOG_FIELDNAMES = [
    "timestamp",
    "level",
    "observer",
    "cluster_id",
    "client_id",
    "scheduler_epoch",
    "local_epoch",
    "round_id",
    "batch_id",
    "event",
    "route",
    "direction",
    "payload_type",
    "payload_shape",
    "payload_bytes",
    "latency_ms",
    "bandwidth_MBps",
    "bandwidth_Mbps",
    "status",
    "activation_shape",
    "activation_bytes",
    "target_bytes",
    "response_payload_bytes",
    "grad_shape",
    "grad_bytes",
    "loss",
    "round_complete",
    "flush_triggered",
    "effective_accum_steps",
    "server_compute_time_ms",
    "batch_size_requested",
    "batch_size_returned",
    "start_idx",
    "end_idx",
    "x_shape",
    "y_shape",
    "num_parameters",
    "num_tensors",
]


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def tensor_shape(value: Any) -> Optional[str]:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return "x".join(str(int(dim)) for dim in tuple(shape))
    except Exception:
        return str(shape)


def tensor_nbytes(value: Any) -> Optional[int]:
    if value is None:
        return None
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel() * element_size())
        except Exception:
            return None
    return None


def calc_bandwidth(payload_bytes: Optional[int], latency_ms: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if payload_bytes in (None, 0) or latency_ms in (None, 0):
        return None, None
    if latency_ms <= 0:
        return None, None
    seconds = float(latency_ms) / 1000.0
    mbps = (float(payload_bytes) / (1024.0 * 1024.0)) / seconds
    return mbps, mbps * 8.0


def should_log_index(index_1based: int, every_n: int) -> bool:
    return max(1, int(every_n)) > 0 and index_1based % max(1, int(every_n)) == 0


def append_comm_row(path: str | Path, row: Dict[str, Any], fieldnames: Iterable[str] = COMM_LOG_FIELDNAMES) -> None:
    ensure_parent(path)
    normalized = {key: row.get(key) for key in fieldnames}
    if not normalized.get("timestamp"):
        normalized["timestamp"] = iso_utc_now()
    path = Path(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)
        f.flush()
