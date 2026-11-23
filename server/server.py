#!/usr/bin/env python3
import os
import pickle
import sys
import time
import json
import hashlib
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
try:
    from server.frozen_gradient import frozen_server_activation_step
except ModuleNotFoundError:  # direct execution as server/server.py
    from frozen_gradient import frozen_server_activation_step
from utils.quality_aggregation import (
    ClientGradientContribution,
    aggregate_anchor_gradients,
    gradient_norm,
)
from utils.comm_logging import append_comm_row, calc_bandwidth, iso_utc_now, should_log_index, tensor_nbytes, tensor_shape
from utils.runtime_utils import (
    _get_env,
    build_lstm_config_from_env,
    build_quality_anchor_config,
    derive_dq_mode,
    get_device_info,
    load_checkpoint_base_reference,
    parse_bool,
    QualityAnchorConfig,
    resolve_device,
    resolve_server_checkpoint_fallback,
    save_torch_checkpoint,
    try_load_torch_checkpoint,
    try_load_torch_checkpoint_with_fallback,
)


app = Flask(__name__)
FROZEN_VALIDATION_MODE = parse_bool(_get_env("FROZEN_VALIDATION_MODE", "false"))


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
    use_quality_weights: bool
    use_quality_as_input: bool
    use_quality_aggregation_weights: bool
    quality_aggregation_mode: str
    quality_aggregation_q_min: float
    quality_aggregation_eps: float
    dq_mode: str
    quality_score_path: str
    quality_score_min: float
    quality_score_max: float
    quality_score_eps: float
    quality_anchor: QualityAnchorConfig


class SplitLearningServerState:
    def __init__(self) -> None:
        use_quality_weights = parse_bool(_get_env("USE_QUALITY_WEIGHTS", "false"))
        use_quality_as_input = parse_bool(_get_env("USE_QUALITY_AS_INPUT", "false"))
        use_quality_aggregation_weights = parse_bool(_get_env("USE_QUALITY_AGGREGATION_WEIGHTS", "false"))
        quality_anchor_config = build_quality_anchor_config(os.environ)
        quality_aggregation_mode = str(_get_env("QUALITY_AGGREGATION_MODE", "soft_client_gradient") or "soft_client_gradient").strip()
        quality_aggregation_q_min = float(_get_env("QUALITY_AGGREGATION_Q_MIN", "0.5"))
        quality_aggregation_eps = float(_get_env("QUALITY_AGGREGATION_EPS", "1e-8"))
        if use_quality_aggregation_weights and quality_aggregation_mode != "soft_client_gradient":
            raise ValueError("QUALITY_AGGREGATION_MODE currently supports only 'soft_client_gradient'.")
        if not (0.0 <= quality_aggregation_q_min <= 1.0):
            raise ValueError("QUALITY_AGGREGATION_Q_MIN must be within [0, 1].")
        if quality_aggregation_eps <= 0.0:
            raise ValueError("QUALITY_AGGREGATION_EPS must be positive.")
        self.runtime_cfg = ServerRuntimeConfig(
            cluster_id=_get_env("CLUSTER_ID", "0"),
            learning_rate=float(_get_env("SERVER_LR", _get_env("LEARNING_RATE", "1e-3"))),
            accum_steps_target=max(1, int(_get_env("ACCUM_STEPS_SERVER", "1"))),
            save_every_rounds=max(0, int(_get_env("CHECKPOINT_EVERY_ROUNDS", "1"))),
            output_dir=_get_env("OUTPUT_DIR", server_output_dir(_get_env("CLUSTER_ID", "0"))),
            use_quality_weights=use_quality_weights,
            use_quality_as_input=use_quality_as_input,
            use_quality_aggregation_weights=use_quality_aggregation_weights,
            quality_aggregation_mode=quality_aggregation_mode,
            quality_aggregation_q_min=quality_aggregation_q_min,
            quality_aggregation_eps=quality_aggregation_eps,
            dq_mode=derive_dq_mode(
                use_quality_weights,
                use_quality_as_input,
                use_quality_aggregation_weights,
                quality_anchor_config.use_quality_anchor_aggregation,
                quality_anchor_config.use_gradient_alignment_filter,
                quality_anchor_config.use_high_quality_validation_guardrail,
            ),
            quality_score_path=str(_get_env("QUALITY_SCORE_PATH", "") or "").strip(),
            quality_score_min=float(_get_env("QUALITY_SCORE_MIN", "0.05")),
            quality_score_max=float(_get_env("QUALITY_SCORE_MAX", "1.0")),
            quality_score_eps=float(_get_env("QUALITY_SCORE_EPS", "1e-8")),
            quality_anchor=quality_anchor_config,
        )
        self.device = resolve_device()
        self.device_info = get_device_info(self.device, process_role_default="server")
        cfg, split_layer = build_lstm_config_from_env()
        _, server_model = build_split_lstm(cfg, split_layer)
        self.model = server_model.to(self.device)
        self.trainable_server_parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        self.optimizer = self._build_optimizer()
        self.criterion = nn.MSELoss()
        self.criterion_per_sample = nn.MSELoss(reduction="none")
        self.lock = Lock()
        self.comm_log_train = parse_bool(_get_env("COMM_LOG_TRAIN", "true"))
        self.comm_log_eval = parse_bool(_get_env("COMM_LOG_EVAL", "false"))
        self.comm_log_every_n_batches = max(1, int(_get_env("COMM_LOG_EVERY_N_BATCHES", "1")))
        self.comm_log_path = server_comm_log_path(self.runtime_cfg.cluster_id)

        self.global_step = 0
        self.optimizer_step_count = 0
        self.pending_accum_steps = 0
        self.round_id = 0
        self.global_round = 0
        self.logical_round = 0
        self.scheduler_epoch = 0
        self.active_accum_steps_target = self.runtime_cfg.accum_steps_target
        self.expected_clients: List[str] = []
        self.expected_clients_set: Set[str] = set()
        self.seen_clients_total: Set[str] = set()
        self.seen_clients_window: Set[str] = set()
        self.round_total_contrib_count = 0
        self.window_contrib_count = 0
        self.accum_quality_weight_sum = 0.0
        self.quality_anchor_contributions: List[ClientGradientContribution] = []
        # try loading latest checkpoint on startup
        latest_path = server_checkpoint_path(self.runtime_cfg.cluster_id, "latest.pt")
        # CHECKPOINT_BASE_REFERENCE_PATH, when set, is the exact declared
        # path to this method's checkpoints/base_reference.json -- never
        # searched for. Unset for every legacy run and for legacy_full-mode
        # runs, so the fallback branch below is provably unreached for them.
        base_reference_path = _get_env("CHECKPOINT_BASE_REFERENCE_PATH", "")
        base_reference = load_checkpoint_base_reference(base_reference_path or None)
        fallback = None
        if base_reference is not None:
            fallback = resolve_server_checkpoint_fallback(base_reference, base_reference_path, self.runtime_cfg.cluster_id)
        extra, ok = try_load_torch_checkpoint_with_fallback(
            self.model,
            self.optimizer,
            latest_path,
            base_reference=base_reference,
            fallback=fallback,
            map_location=str(self.device),
            strict=True,
            log_prefix="Server",
        )
        if ok and extra:
            self.global_step = int(extra.get("global_step", 0))
            self.optimizer_step_count = int(extra.get("optimizer_step_count", 0))
            self.round_id = int(extra.get("round_id", 0))
        self.optimizer.zero_grad(set_to_none=True)
        self._log_optimizer_event(
            "created_at_startup",
            reason="server_startup",
            restored_from_checkpoint=bool(ok),
            checkpoint_path=latest_path if ok else None,
        )

        print(
            f"[Server cluster={self.runtime_cfg.cluster_id}] device_info={self.device_info} "
            f"server_lr={self.runtime_cfg.learning_rate} accum_steps_target={self.runtime_cfg.accum_steps_target} "
            f"use_quality_weights={self.runtime_cfg.use_quality_weights} "
            f"use_quality_as_input={self.runtime_cfg.use_quality_as_input} "
            f"use_quality_aggregation_weights={self.runtime_cfg.use_quality_aggregation_weights} "
            f"use_quality_anchor_aggregation={self.runtime_cfg.quality_anchor.use_quality_anchor_aggregation} "
            f"use_gradient_alignment_filter={self.runtime_cfg.quality_anchor.use_gradient_alignment_filter} "
            f"quality_aggregation_mode={self.runtime_cfg.quality_aggregation_mode} "
            f"quality_aggregation_q_min={self.runtime_cfg.quality_aggregation_q_min} "
            f"dq_mode={self.runtime_cfg.dq_mode} "
            f"quality_score_path={self.runtime_cfg.quality_score_path or '<disabled>'} "
            f"output_dir={self.runtime_cfg.output_dir}",
            flush=True,
        )

    def _append_comm_row(self, row: Dict[str, Any]) -> None:
        append_comm_row(self.comm_log_path, row)

    def _append_quality_anchor_log(self, payload: Dict[str, Any]) -> None:
        record = {"timestamp": iso_utc_now(), **payload}
        with open(
            os.path.join(self.runtime_cfg.output_dir, "quality_anchor_aggregation.jsonl"),
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")
            log_file.flush()

    def _should_log_batch(self, batch_id: Any) -> bool:
        try:
            return should_log_index(int(batch_id), self.comm_log_every_n_batches)
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _optimizer_reset_policy() -> str:
        return "create_once_at_startup_reuse_across_split_rounds_reset_only_after_global_model_load"

    def _build_optimizer(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=self.runtime_cfg.learning_rate)

    def _log_optimizer_event(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "cluster": self.runtime_cfg.cluster_id,
            "optimizer_id": id(self.optimizer),
            "round_id": self.round_id,
            "global_step": self.global_step,
            "optimizer_step_count": self.optimizer_step_count,
            "policy": self._optimizer_reset_policy(),
            **fields,
        }
        details = " ".join(f"{key}={value}" for key, value in payload.items())
        print(f"[Server][optimizer] {details}", flush=True)

    @staticmethod
    def _ensure_lstm_input_3d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # Active split contract: client sends only the last hidden state [B, H].
            # Keep the server-side LSTM path by treating it as a length-1 sequence.
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
    # Normalize gradients by the current accumulation window denominator before optimizer.step().
    def _normalize_parameter_gradients(self, normalizer: float) -> None:
        if normalizer <= 0.0:
            raise ValueError(f"Gradient normalizer must be positive, got {normalizer}")
        if abs(float(normalizer) - 1.0) <= 1e-12:
            return
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(float(normalizer))

    def _reset_window_tracking(self) -> None:
        self.pending_accum_steps = 0
        self.accum_quality_weight_sum = 0.0
        self.seen_clients_window.clear()
        self.window_contrib_count = 0
        self.quality_anchor_contributions.clear()

    # just to reset the tracking of each round's progress after a completed logical round
    def _reset_round_tracking(self, increment_round: bool) -> None:
        self.expected_clients = []
        self.expected_clients_set = set()
        self.active_accum_steps_target = self.runtime_cfg.accum_steps_target
        self.seen_clients_total.clear()
        self.round_total_contrib_count = 0
        self._reset_window_tracking()
        if increment_round:
            self.round_id += 1

    def _prepare_quality_scores(
        self,
        quality_scores_cpu: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        quality_required = (
            self.runtime_cfg.use_quality_weights
            or self.runtime_cfg.use_quality_aggregation_weights
            or self.runtime_cfg.quality_anchor.use_quality_anchor_aggregation
        )
        if quality_scores_cpu is None:
            if quality_required:
                raise ValueError(
                    "quality_scores missing while server-side quality mode is enabled "
                    "(USE_QUALITY_WEIGHTS=true, USE_QUALITY_AGGREGATION_WEIGHTS=true, "
                    "or USE_QUALITY_ANCHOR_AGGREGATION=true)"
                )
            return torch.ones(batch_size, dtype=torch.float32, device=self.device)

        quality_scores = quality_scores_cpu.to(self.device, dtype=torch.float32)
        if quality_scores.dim() == 0:
            quality_scores = quality_scores.repeat(batch_size)
        elif quality_scores.dim() == 2 and quality_scores.size(-1) == 1:
            quality_scores = quality_scores.squeeze(-1)
        else:
            quality_scores = quality_scores.reshape(-1)
        if quality_scores.numel() != batch_size:
            raise ValueError(
                f"quality_scores size mismatch: expected batch_size={batch_size}, got {quality_scores.numel()}"
            )
        strict_quality = (
            self.runtime_cfg.use_quality_aggregation_weights
            or self.runtime_cfg.quality_anchor.use_quality_anchor_aggregation
        )
        if strict_quality:
            if not torch.isfinite(quality_scores).all():
                raise ValueError("quality_scores contain non-finite values while server quality aggregation is enabled")
            if torch.any((quality_scores < 0.0) | (quality_scores > 1.0)):
                raise ValueError(
                    "quality_scores must be within [0, 1] while server quality aggregation is enabled; "
                    f"min={float(quality_scores.min().item()):.6f} max={float(quality_scores.max().item()):.6f}"
                )
        if strict_quality:
            return quality_scores
        return quality_scores.clamp(
            min=self.runtime_cfg.quality_score_min,
            max=self.runtime_cfg.quality_score_max,
        )

    # check for expected clients, track seen clients and contributions
    def set_round(
        self,
        expected_clients: List[str],
        round_id: Optional[int] = None,
        effective_accum_steps_target: Optional[int] = None,
        global_round: Optional[int] = None,
        logical_round: Optional[int] = None,
        scheduler_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        unique_clients = [str(c) for c in expected_clients]
        if len(unique_clients) != len(set(unique_clients)):
            raise ValueError("expected_clients contains duplicates")
        if self.pending_accum_steps > 0:
            raise RuntimeError("cannot start a new round while gradients are pending")
        if self.expected_clients_set or self.seen_clients_total or self.round_total_contrib_count > 0:
            raise RuntimeError("cannot start a new round while previous logical round is still active")
        requested_accum_steps_target = max(
            1,
            int(effective_accum_steps_target or self.runtime_cfg.accum_steps_target),
        )
        if (
            self.runtime_cfg.quality_anchor.use_quality_anchor_aggregation
            and unique_clients
            and requested_accum_steps_target != len(unique_clients)
        ):
            raise ValueError(
                "Quality-anchor aggregation requires one complete logical-round buffer: "
                f"effective_accum_steps_target={requested_accum_steps_target}, "
                f"expected_client_count={len(unique_clients)}."
            )
        self.expected_clients = unique_clients
        self.expected_clients_set = set(unique_clients)
        self.seen_clients_total.clear()
        self.seen_clients_window.clear()
        self.round_total_contrib_count = 0
        self.window_contrib_count = 0
        self.pending_accum_steps = 0
        self.active_accum_steps_target = requested_accum_steps_target
        if round_id is not None:
            self.round_id = int(round_id)
        if global_round is not None:
            self.global_round = int(global_round)
        if logical_round is not None:
            self.logical_round = int(logical_round)
        if scheduler_epoch is not None:
            self.scheduler_epoch = int(scheduler_epoch)
        print(
            f"[Server] round_start cluster={self.runtime_cfg.cluster_id} round_id={self.round_id} "
            f"expected_clients={self.expected_clients} effective_accum_steps_target={self.active_accum_steps_target}",
            flush=True,
        )
        return self.round_state()

    def round_state(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.runtime_cfg.cluster_id,
            "round_id": self.round_id,
            "global_round": self.global_round,
            "logical_round": self.logical_round,
            "scheduler_epoch": self.scheduler_epoch,
            "expected_clients": list(self.expected_clients),
            "seen_clients_total": sorted(self.seen_clients_total),
            "seen_clients_window": sorted(self.seen_clients_window),
            "seen_clients": sorted(self.seen_clients_total),
            "round_total_contrib_count": self.round_total_contrib_count,
            "window_contrib_count": self.window_contrib_count,
            "round_contrib_count": self.round_total_contrib_count,
            "pending_accum_steps": self.pending_accum_steps,
            "active_accum_steps_target": self.active_accum_steps_target,
            "use_quality_weights": self.runtime_cfg.use_quality_weights,
            "use_quality_as_input": self.runtime_cfg.use_quality_as_input,
            "use_quality_aggregation_weights": self.runtime_cfg.use_quality_aggregation_weights,
            "use_quality_anchor_aggregation": self.runtime_cfg.quality_anchor.use_quality_anchor_aggregation,
            "use_gradient_alignment_filter": self.runtime_cfg.quality_anchor.use_gradient_alignment_filter,
            "use_high_quality_validation_guardrail": self.runtime_cfg.quality_anchor.use_high_quality_validation_guardrail,
            "quality_anchor_contribution_count": len(self.quality_anchor_contributions),
            "quality_aggregation_mode": self.runtime_cfg.quality_aggregation_mode,
            "quality_aggregation_q_min": self.runtime_cfg.quality_aggregation_q_min,
            "quality_aggregation_eps": self.runtime_cfg.quality_aggregation_eps,
            "accum_quality_weight_sum": self.accum_quality_weight_sum,
            "dq_mode": self.runtime_cfg.dq_mode,
            "optimizer_step_count": self.optimizer_step_count,
            "global_step": self.global_step,
        }

    @staticmethod
    def _quality_stats(values: Tuple[float, ...]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "min": None, "max": None}
        return {
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    def abort_quality_anchor_round(self, reason: str) -> None:
        if not self.runtime_cfg.quality_anchor.use_quality_anchor_aggregation:
            return
        contribution_count = len(self.quality_anchor_contributions)
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_round_tracking(increment_round=False)
        print(
            f"[Server] quality_anchor_round_aborted cluster={self.runtime_cfg.cluster_id} "
            f"reason={reason} cleared_contributions={contribution_count}",
            flush=True,
        )

    def _flush_quality_anchor_gradients(
        self,
        reason: str,
        *,
        increment_round: bool,
    ) -> Dict[str, Any]:
        if self.pending_accum_steps <= 0:
            return {"flushed": False, **self.round_state(), "reason": reason}
        if self.seen_clients_total != self.expected_clients_set:
            seen_clients = sorted(self.seen_clients_total)
            expected_clients = sorted(self.expected_clients_set)
            self.abort_quality_anchor_round(reason="incomplete_logical_round_flush")
            raise RuntimeError(
                "Cannot aggregate an incomplete quality-anchor logical round: "
                f"seen_clients={seen_clients}, expected_clients={expected_clients}."
            )
        if len(self.quality_anchor_contributions) != self.pending_accum_steps:
            self.abort_quality_anchor_round(reason="inconsistent_contribution_buffer")
            raise RuntimeError(
                "Quality-anchor contribution buffer is inconsistent with pending accumulation steps."
            )

        effective_accum_steps = self.pending_accum_steps
        round_id_snapshot = self.round_id
        global_round_snapshot = self.global_round
        logical_round_snapshot = self.logical_round
        scheduler_epoch_snapshot = self.scheduler_epoch
        window_clients = sorted(self.seen_clients_window)

        try:
            aggregation = aggregate_anchor_gradients(
                tuple(self.quality_anchor_contributions),
                self.runtime_cfg.quality_anchor,
            )
            if len(aggregation.final_gradients) != len(self.trainable_server_parameters):
                raise RuntimeError("Final server-gradient tuple length does not match trainable parameters.")

            self.optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(
                self.trainable_server_parameters,
                aggregation.final_gradients,
            ):
                if parameter.shape != gradient.shape:
                    raise RuntimeError("Final server-gradient tensor shape mismatch.")
                parameter.grad = gradient.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ).clone()

            self.optimizer.step()
            self.optimizer_step_count += 1
            self._log_optimizer_event(
                "step_during_quality_anchor_training",
                reason=reason,
                effective_accum_steps=effective_accum_steps,
                increment_round=increment_round,
                round_id_snapshot=round_id_snapshot,
            )
            self.optimizer.zero_grad(set_to_none=True)
            self._log_optimizer_event(
                "zero_grad_after_quality_anchor_update",
                reason=reason,
                effective_accum_steps=effective_accum_steps,
                increment_round=increment_round,
                round_id_snapshot=round_id_snapshot,
            )

            anchor_stats = self._quality_stats(aggregation.anchor_qualities)
            non_anchor_stats = self._quality_stats(aggregation.non_anchor_qualities)
            summary = {
                "event": "quality_anchor_round",
                "aggregation_mode": (
                    "anchor_plus_alignment"
                    if self.runtime_cfg.quality_anchor.use_gradient_alignment_filter
                    else "anchor_only"
                ),
                "global_round": global_round_snapshot,
                "logical_round": logical_round_snapshot,
                "scheduler_epoch": scheduler_epoch_snapshot,
                "round_id": round_id_snapshot,
                "cluster_id": self.runtime_cfg.cluster_id,
                "split_server_id": self.runtime_cfg.cluster_id,
                "number_of_client_contributions": effective_accum_steps,
                "anchor_client_count": len(aggregation.anchor_client_ids),
                "anchor_client_ids": list(aggregation.anchor_client_ids),
                "anchor_quality_mean": anchor_stats["mean"],
                "anchor_quality_min": anchor_stats["min"],
                "anchor_quality_max": anchor_stats["max"],
                "non_anchor_quality_mean": non_anchor_stats["mean"],
                "non_anchor_quality_min": non_anchor_stats["min"],
                "non_anchor_quality_max": non_anchor_stats["max"],
                "anchor_gradient_norm": aggregation.anchor_gradient_norm,
                "non_anchor_aggregate_norm": aggregation.non_anchor_gradient_norm,
                "candidate_gradient_norm": aggregation.final_gradient_norm,
                "final_gradient_norm": aggregation.final_gradient_norm,
                "anchor_update_share": self.runtime_cfg.quality_anchor.anchor_update_share,
                "accepted_non_anchor_count": aggregation.accepted_non_anchor_count,
                "rejected_non_anchor_count": aggregation.rejected_non_anchor_count,
            }
            self._append_quality_anchor_log(summary)
            if self.runtime_cfg.quality_anchor.debug_logging:
                for record in aggregation.non_anchor_records:
                    self._append_quality_anchor_log(
                        {
                            "event": "quality_anchor_client_alignment",
                            "global_round": global_round_snapshot,
                            "logical_round": logical_round_snapshot,
                            "scheduler_epoch": scheduler_epoch_snapshot,
                            "round_id": round_id_snapshot,
                            "cluster_id": self.runtime_cfg.cluster_id,
                            "client_id": record.client_id,
                            "mean_quality": record.mean_quality,
                            "gradient_norm": record.gradient_norm,
                            "cosine_with_anchor": record.cosine_with_anchor,
                            "alignment_weight": record.alignment_weight,
                            "norm_clip_scale": record.norm_clip_scale,
                            "accepted_or_rejected": "accepted" if record.accepted else "rejected",
                            "reason": record.reason,
                        }
                    )
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            self._reset_round_tracking(increment_round=False)
            raise

        finished_round_id = self.round_id
        if increment_round:
            self._reset_round_tracking(increment_round=True)
        else:
            self._reset_window_tracking()
        print(
            f"[Server] quality_anchor_flush reason={reason} cluster={self.runtime_cfg.cluster_id} "
            f"round_id={finished_round_id} effective_accum_steps={effective_accum_steps} "
            f"anchor_clients={list(aggregation.anchor_client_ids)} "
            f"accepted_non_anchor={aggregation.accepted_non_anchor_count} "
            f"optimizer_steps={self.optimizer_step_count}",
            flush=True,
        )
        if (
            increment_round
            and self.runtime_cfg.save_every_rounds > 0
            and finished_round_id % self.runtime_cfg.save_every_rounds == 0
        ):
            self._save_latest(flush=False)
        state_after_flush = self.round_state()
        return {
            **state_after_flush,
            "flushed": True,
            "reason": reason,
            "finished_round_id": finished_round_id,
            "effective_accum_steps": effective_accum_steps,
            "effective_accumulation_normalizer": None,
            "flushed_accum_quality_weight_sum": 0.0,
            "window_clients": window_clients,
            "anchor_client_ids": list(aggregation.anchor_client_ids),
            "accepted_non_anchor_count": aggregation.accepted_non_anchor_count,
            "rejected_non_anchor_count": aggregation.rejected_non_anchor_count,
        }

    def flush_accumulated_gradients(self, reason: str, *, increment_round: bool) -> Dict[str, Any]:
        if self.runtime_cfg.quality_anchor.use_quality_anchor_aggregation:
            return self._flush_quality_anchor_gradients(reason, increment_round=increment_round)
        if self.pending_accum_steps <= 0:
            return {"flushed": False, **self.round_state(), "reason": reason}
        effective_accum_steps = self.pending_accum_steps
        round_id_snapshot = self.round_id
        accum_quality_weight_sum_snapshot = float(self.accum_quality_weight_sum)
        if self.runtime_cfg.use_quality_aggregation_weights:
            effective_normalizer = accum_quality_weight_sum_snapshot + self.runtime_cfg.quality_aggregation_eps
        else:
            effective_normalizer = float(effective_accum_steps)
        self._normalize_parameter_gradients(effective_normalizer)
        self.optimizer.step()
        self.optimizer_step_count += 1
        self._log_optimizer_event(
            "step_during_normal_split_training",
            reason=reason,
            effective_accum_steps=effective_accum_steps,
            effective_accumulation_normalizer=effective_normalizer,
            accum_quality_weight_sum=accum_quality_weight_sum_snapshot,
            increment_round=increment_round,
            pending_accum_steps_before_reset=effective_accum_steps,
            round_id_snapshot=round_id_snapshot,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._log_optimizer_event(
            "zero_grad_after_normal_update",
            reason=reason,
            effective_accum_steps=effective_accum_steps,
            effective_accumulation_normalizer=effective_normalizer,
            accum_quality_weight_sum=accum_quality_weight_sum_snapshot,
            increment_round=increment_round,
            round_id_snapshot=round_id_snapshot,
        )
        finished_round_id = self.round_id
        window_clients = sorted(self.seen_clients_window)
        if increment_round:
            self._reset_round_tracking(increment_round=increment_round)
        else:
            self._reset_window_tracking()
        print(
            f"[Server] flush reason={reason} cluster={self.runtime_cfg.cluster_id} round_id={finished_round_id} "
            f"effective_accum_steps={effective_accum_steps} active_target={self.active_accum_steps_target} "
            f"effective_normalizer={effective_normalizer:.6f} "
            f"accum_quality_weight_sum={accum_quality_weight_sum_snapshot:.6f} "
            f"window_clients={window_clients} optimizer_steps={self.optimizer_step_count}",
            flush=True,
        )
        if increment_round and self.runtime_cfg.save_every_rounds > 0 and finished_round_id % self.runtime_cfg.save_every_rounds == 0:
            self._save_latest(flush=False)
        state_after_flush = self.round_state()
        return {
            **state_after_flush,
            "flushed": True,
            "reason": reason,
            "finished_round_id": finished_round_id,
            "effective_accum_steps": effective_accum_steps,
            "effective_accumulation_normalizer": effective_normalizer,
            "flushed_accum_quality_weight_sum": accum_quality_weight_sum_snapshot,
            "window_clients": window_clients,
        }

    def _flush_before_eval_if_needed(self, reason: str) -> Optional[Dict[str, Any]]:
        if self.pending_accum_steps > 0:
            return self.flush_accumulated_gradients(reason=reason, increment_round=False)
        return None

    def _train_step_quality_anchor(
        self,
        client_id: str,
        activations_cpu: torch.Tensor,
        targets_cpu: torch.Tensor,
        quality_scores_cpu: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        client_id = str(client_id)
        try:
            if not self.expected_clients_set:
                raise RuntimeError("No active round configured. Call /round/start first.")
            if client_id not in self.expected_clients_set:
                raise RuntimeError(f"Client {client_id} is not part of active round {self.round_id}")
            if client_id in self.seen_clients_total:
                raise RuntimeError(f"Duplicate contribution from client {client_id} in round {self.round_id}")

            self.global_step += 1
            input_was_2d = activations_cpu.dim() == 2
            activations = self._ensure_lstm_input_3d(activations_cpu.to(self.device))
            targets = targets_cpu.to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)

            activations.requires_grad_(True)
            predictions = self.model(activations)
            per_sample_loss = self.criterion_per_sample(predictions, targets)
            if per_sample_loss.dim() > 1:
                per_sample_loss = per_sample_loss.view(per_sample_loss.size(0), -1).mean(dim=1)
            else:
                per_sample_loss = per_sample_loss.reshape(-1)
            ordinary_loss = per_sample_loss.mean()
            ordinary_loss_value = float(ordinary_loss.item())
            quality_scores = self._prepare_quality_scores(
                quality_scores_cpu,
                int(per_sample_loss.numel()),
            )
            client_mean_quality = float(quality_scores.mean().item())

            extracted_gradients = torch.autograd.grad(
                ordinary_loss,
                (activations, *self.trainable_server_parameters),
                retain_graph=False,
                create_graph=False,
            )
            grad_activations = extracted_gradients[0].detach().cpu()
            if input_was_2d and grad_activations.dim() == 3 and grad_activations.size(1) == 1:
                grad_activations = grad_activations.squeeze(1)

            stored_server_gradients = tuple(
                gradient.detach().to(device="cpu").clone()
                for gradient in extracted_gradients[1:]
            )
            server_gradient_norm = gradient_norm(stored_server_gradients)
            contribution = ClientGradientContribution(
                client_id=client_id,
                cluster_id=self.runtime_cfg.cluster_id,
                global_round=self.global_round,
                logical_round=self.logical_round,
                mean_quality=client_mean_quality,
                sample_count=int(per_sample_loss.numel()),
                server_gradients=stored_server_gradients,
                server_gradient_norm=server_gradient_norm,
            )
            self.quality_anchor_contributions.append(contribution)

            self.pending_accum_steps += 1
            self.seen_clients_total.add(client_id)
            self.seen_clients_window.add(client_id)
            self.round_total_contrib_count = len(self.seen_clients_total)
            self.window_contrib_count = len(self.seen_clients_window)

            active_target = self.active_accum_steps_target
            seen_total = sorted(self.seen_clients_total)
            seen_window = sorted(self.seen_clients_window)
            window_complete = self.pending_accum_steps >= active_target
            round_complete = self.seen_clients_total == self.expected_clients_set
            flush_info: Optional[Dict[str, Any]] = None
            finished_round_id: Optional[int] = None

            if window_complete:
                flush_info = self.flush_accumulated_gradients(
                    reason="window_complete",
                    increment_round=False,
                )

            if round_complete:
                finished_round_id = self.round_id
                if self.pending_accum_steps > 0:
                    flush_info = self.flush_accumulated_gradients(
                        reason="round_complete_tail",
                        increment_round=False,
                    )
                if (
                    self.runtime_cfg.save_every_rounds > 0
                    and finished_round_id % self.runtime_cfg.save_every_rounds == 0
                ):
                    self._save_latest(flush=False)
                self._reset_round_tracking(increment_round=True)

            print(
                f"[Server] quality_anchor_train_step cluster={self.runtime_cfg.cluster_id} "
                f"round_id={finished_round_id if finished_round_id is not None else self.round_id} "
                f"client={client_id} loss={ordinary_loss_value:.6f} "
                f"quality_mean={client_mean_quality:.6f} server_gradient_norm={server_gradient_norm:.6f} "
                f"seen_total={seen_total} seen_window={seen_window} "
                f"target_accum={active_target} window_complete={window_complete} "
                f"round_complete={round_complete}",
                flush=True,
            )
            return {
                "grad_activations": grad_activations,
                "loss": ordinary_loss_value,
                "weighted_loss": ordinary_loss_value,
                "effective_loss": ordinary_loss_value,
                "client_mean_quality": client_mean_quality,
                "client_soft_quality": 1.0,
                "accum_quality_weight_sum": 0.0,
                "quality_score_min": float(quality_scores.min().item()),
                "quality_score_max": float(quality_scores.max().item()),
                "quality_score_mean": client_mean_quality,
                "server_gradient_norm": server_gradient_norm,
                "window_complete": window_complete,
                "round_complete": round_complete,
                "flush_info": flush_info,
                "finished_round_id": finished_round_id,
                **self.round_state(),
                "active_accum_steps_target": active_target,
            }
        except Exception as error:
            self.abort_quality_anchor_round(reason=f"{type(error).__name__}:{error}")
            raise

    def train_step(
        self,
        client_id: str,
        activations_cpu: torch.Tensor,
        targets_cpu: torch.Tensor,
        quality_scores_cpu: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.runtime_cfg.quality_anchor.use_quality_anchor_aggregation:
            return self._train_step_quality_anchor(
                client_id,
                activations_cpu,
                targets_cpu,
                quality_scores_cpu,
            )
        client_id = str(client_id)
        if not self.expected_clients_set:
            raise RuntimeError("No active round configured. Call /round/start first.")
        if client_id not in self.expected_clients_set:
            raise RuntimeError(f"Client {client_id} is not part of active round {self.round_id}")
        if client_id in self.seen_clients_total:
            raise RuntimeError(f"Duplicate contribution from client {client_id} in round {self.round_id}")

        self.global_step += 1
        input_was_2d = activations_cpu.dim() == 2
        activations = self._ensure_lstm_input_3d(activations_cpu.to(self.device))
        targets = targets_cpu.to(self.device)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        activations.requires_grad_(True)
        preds = self.model(activations)
        raw_loss = float(self.criterion(preds, targets).item())
        per_sample_loss = self.criterion_per_sample(preds, targets)
        if per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.view(per_sample_loss.size(0), -1).mean(dim=1)
        else:
            per_sample_loss = per_sample_loss.reshape(-1)
        quality_scores = self._prepare_quality_scores(quality_scores_cpu, int(per_sample_loss.numel()))
        loss_quality_scores = quality_scores.clamp(
            min=self.runtime_cfg.quality_score_min,
            max=self.runtime_cfg.quality_score_max,
        )
        if self.runtime_cfg.use_quality_weights:
            weighted_loss = (per_sample_loss * loss_quality_scores).sum() / loss_quality_scores.sum().clamp_min(
                self.runtime_cfg.quality_score_eps
            )
        else:
            weighted_loss = per_sample_loss.mean()
        weighted_loss_value = float(weighted_loss.item())
        client_mean_quality = float(quality_scores.mean().item())
        client_soft_quality = 1.0
        effective_loss = weighted_loss
        if self.runtime_cfg.use_quality_aggregation_weights:
            q_min = self.runtime_cfg.quality_aggregation_q_min
            client_soft_quality = float(q_min + (1.0 - q_min) * client_mean_quality)
            # Design A: clients receive gradients scaled by their own soft quality immediately.
            # Server parameter gradients are normalized later by the current accumulation-window
            # soft-quality sum, because the full denominator is only known at flush time.
            effective_loss = weighted_loss * client_soft_quality
            self.accum_quality_weight_sum += client_soft_quality
        effective_loss_value = float(effective_loss.item())
        effective_loss.backward()
        grad_activations = activations.grad.detach().cpu()
        if input_was_2d and grad_activations.dim() == 3 and grad_activations.size(1) == 1:
            grad_activations = grad_activations.squeeze(1)

        accum_quality_weight_sum_snapshot = float(self.accum_quality_weight_sum)

        self.pending_accum_steps += 1
        self.seen_clients_total.add(client_id)
        self.seen_clients_window.add(client_id)
        self.round_total_contrib_count = len(self.seen_clients_total)
        self.window_contrib_count = len(self.seen_clients_window)

        active_accum_steps_target_snapshot = self.active_accum_steps_target
        seen_clients_total_snapshot = sorted(self.seen_clients_total)
        seen_clients_window_snapshot = sorted(self.seen_clients_window)
        window_complete = self.pending_accum_steps >= active_accum_steps_target_snapshot
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
            f"client={client_id} raw_loss={raw_loss:.6f} weighted_loss={weighted_loss_value:.6f} "
            f"effective_loss={effective_loss_value:.6f} quality_mean={client_mean_quality:.6f} "
            f"soft_quality={client_soft_quality:.6f} accum_quality_weight_sum={accum_quality_weight_sum_snapshot:.6f} seen_total={seen_clients_total_snapshot} "
            f"seen_window={seen_clients_window_snapshot} pending_accum={self.pending_accum_steps} "
            f"target_accum={active_accum_steps_target_snapshot} window_complete={window_complete} round_complete={round_complete}",
            flush=True,
        )
        return {
            "grad_activations": grad_activations,
            "loss": raw_loss,
            "predictions": preds.detach().cpu(),
            "weighted_loss": weighted_loss_value,
            "effective_loss": effective_loss_value,
            "client_mean_quality": client_mean_quality,
            "client_soft_quality": client_soft_quality,
            "accum_quality_weight_sum": accum_quality_weight_sum_snapshot,
            "quality_score_min": float(quality_scores.min().item()),
            "quality_score_max": float(quality_scores.max().item()),
            "quality_score_mean": client_mean_quality,
            "window_complete": window_complete,
            "round_complete": round_complete,
            "flush_info": flush_info,
            "finished_round_id": finished_round_id,
            **self.round_state(),
            "active_accum_steps_target": active_accum_steps_target_snapshot,
        }

    def frozen_activation_step(self, client_id: str, activations_cpu: torch.Tensor, targets_cpu: torch.Tensor) -> Dict[str, Any]:
        """M2-only activation derivative with every server-owned state frozen."""
        before = {
            "parameters": self.parameter_hash(),
            "optimizer_steps": self.optimizer_step_count,
            "global_step": self.global_step,
            "round_id": self.round_id,
            "optimizer_state": pickle.dumps(self.optimizer.state_dict()),
        }
        result = frozen_server_activation_step(
            self.model, self.criterion, activations_cpu, targets_cpu, device=self.device
        )
        after = {
            "parameters": self.parameter_hash(),
            "optimizer_steps": self.optimizer_step_count,
            "global_step": self.global_step,
            "round_id": self.round_id,
            "optimizer_state": pickle.dumps(self.optimizer.state_dict()),
        }
        if before != after:
            raise RuntimeError("M2 frozen server call mutated server-owned state")
        return {**result, "client_id": str(client_id), "server_state_unchanged": True}

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
        if FROZEN_VALIDATION_MODE:
            self.model.eval()
        else:
            self._flush_before_eval_if_needed(reason=f"before_inference:{client_id}")
        with torch.no_grad():
            activations = self._ensure_lstm_input_3d(activations_cpu.to(self.device))
            preds_device = self.model(activations)
            if FROZEN_VALIDATION_MODE and (torch.is_grad_enabled() or preds_device.requires_grad):
                raise RuntimeError("frozen inference unexpectedly enabled gradients")
            preds = preds_device.detach().cpu()
            return {"predictions": preds.numpy(), "gradients_enabled": False}

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

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

    def _load_checkpoint_with_base_reference_fallback(self, primary_path: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        # Mirrors the fallback the constructor already uses (see __init__
        # above): when SKIP_TRAINING left this run's own OUTPUT_DIR empty,
        # re-load the same H artifact the server loaded at startup instead
        # of failing closed against a local-only path.
        base_reference_path = _get_env("CHECKPOINT_BASE_REFERENCE_PATH", "")
        base_reference = load_checkpoint_base_reference(base_reference_path or None)
        if base_reference is None:
            return None, False
        fallback = resolve_server_checkpoint_fallback(base_reference, base_reference_path, self.runtime_cfg.cluster_id)
        return try_load_torch_checkpoint_with_fallback(
            self.model,
            self.optimizer,
            primary_path,
            base_reference=base_reference,
            fallback=fallback,
            map_location=str(self.device),
            strict=True,
            log_prefix="Server",
        )

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
            extra, ok = self._load_checkpoint_with_base_reference_fallback(path)
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
            extra, ok = self._load_checkpoint_with_base_reference_fallback(latest_path)
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
            extra, ok = self._load_checkpoint_with_base_reference_fallback(target)
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
        old_optimizer_id = id(self.optimizer)
        self.optimizer = self._build_optimizer()
        self.optimizer.zero_grad(set_to_none=True)
        self._log_optimizer_event(
            "reset_after_global_model_load",
            reason="fedavg_broadcast",
            old_optimizer_id=old_optimizer_id,
        )
        self._reset_round_tracking(increment_round=False)
        return {
            "cluster_id": self.runtime_cfg.cluster_id,
            "global_step": self.global_step,
            "optimizer_step_count": self.optimizer_step_count,
            "round_id": self.round_id,
            "optimizer_id": id(self.optimizer),
            "optimizer_reset_policy": self._optimizer_reset_policy(),
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
        return jsonify({"status": "ok", **state.round_state(), "device": state.device_info, "frozen_validation_mode": FROZEN_VALIDATION_MODE, "parameter_hash": state.parameter_hash()})


@app.route("/config", methods=["GET"])
def config():
    with state.lock:
        return jsonify({
            "cluster_id": state.runtime_cfg.cluster_id,
            "frozen_validation_mode": FROZEN_VALIDATION_MODE,
            "parameter_hash": state.parameter_hash(),
            "server_lr": state.runtime_cfg.learning_rate,
            "save_every_rounds": state.runtime_cfg.save_every_rounds,
            "use_quality_weights": state.runtime_cfg.use_quality_weights,
            "use_quality_as_input": state.runtime_cfg.use_quality_as_input,
            "use_quality_aggregation_weights": state.runtime_cfg.use_quality_aggregation_weights,
            "use_quality_anchor_aggregation": state.runtime_cfg.quality_anchor.use_quality_anchor_aggregation,
            "use_gradient_alignment_filter": state.runtime_cfg.quality_anchor.use_gradient_alignment_filter,
            "use_high_quality_validation_guardrail": state.runtime_cfg.quality_anchor.use_high_quality_validation_guardrail,
            "quality_anchor_top_fraction": state.runtime_cfg.quality_anchor.top_fraction,
            "quality_anchor_min_clients": state.runtime_cfg.quality_anchor.min_clients,
            "quality_anchor_update_share": state.runtime_cfg.quality_anchor.anchor_update_share,
            "quality_alignment_power": state.runtime_cfg.quality_anchor.alignment_power,
            "quality_negative_alignment_weight": state.runtime_cfg.quality_anchor.negative_alignment_weight,
            "quality_low_client_norm_cap": state.runtime_cfg.quality_anchor.low_client_norm_cap,
            "quality_anchor_eps": state.runtime_cfg.quality_anchor.eps,
            "quality_aggregation_mode": state.runtime_cfg.quality_aggregation_mode,
            "quality_aggregation_q_min": state.runtime_cfg.quality_aggregation_q_min,
            "quality_aggregation_eps": state.runtime_cfg.quality_aggregation_eps,
            "dq_mode": state.runtime_cfg.dq_mode,
            "quality_score_path": state.runtime_cfg.quality_score_path or None,
            "optimizer_id": id(state.optimizer),
            "optimizer_reset_policy": state._optimizer_reset_policy(),
            "device": state.device_info,
            **state.round_state(),
        })


@app.route("/round/start", methods=["POST"])
def round_start():
    if FROZEN_VALIDATION_MODE:
        return jsonify({"error": "round start is disabled in frozen validation mode"}), 409
    data = request.get_json() or {}
    expected_clients = [str(x) for x in data.get("expected_clients", [])]
    round_id = data.get("round_id")
    effective_accum_steps_target = data.get("effective_accum_steps_target")
    with state.lock:
        try:
            return jsonify(
                state.set_round(
                    expected_clients,
                    round_id=round_id,
                    effective_accum_steps_target=effective_accum_steps_target,
                    global_round=data.get("global_round"),
                    logical_round=data.get("logical_round"),
                    scheduler_epoch=data.get("scheduler_epoch"),
                )
            )
        except Exception as e:
            return jsonify({"error": str(e), **state.round_state()}), 409


@app.route("/round/state", methods=["GET"])
def round_state():
    with state.lock:
        return jsonify(state.round_state())


@app.route("/train_step", methods=["POST"])
def train_step():
    if FROZEN_VALIDATION_MODE:
        return jsonify({"error": "training is disabled in frozen validation mode"}), 409
    request_bytes = len(request.get_data(cache=True) or b"")
    t0 = time.perf_counter()
    with state.lock:
        try:
            payload = _unpickle_request_data()
            result = state.train_step(
                client_id=str(payload["client_id"]),
                activations_cpu=payload["activations"],
                targets_cpu=payload["targets"],
                quality_scores_cpu=payload.get("quality_scores"),
            )
            response_blob = pickle.dumps(result)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            upload_MBps, upload_Mbps = calc_bandwidth(request_bytes, latency_ms)
            download_MBps, download_Mbps = calc_bandwidth(len(response_blob), latency_ms)
            if state.comm_log_train and state._should_log_batch(payload.get("batch_id")):
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
                        "weighted_loss": result.get("weighted_loss"),
                        "effective_loss": result.get("effective_loss"),
                        "client_mean_quality": result.get("client_mean_quality"),
                        "client_soft_quality": result.get("client_soft_quality"),
                        "accum_quality_weight_sum": result.get("accum_quality_weight_sum"),
                        "quality_score_min": result.get("quality_score_min"),
                        "quality_score_max": result.get("quality_score_max"),
                        "quality_score_mean": result.get("quality_score_mean"),
                        "round_complete": bool(result.get("round_complete")),
                        "flush_triggered": bool(result.get("flush_info")),
                        "effective_accum_steps": (
                            result.get("flush_info", {}) or {}
                        ).get("effective_accum_steps"),
                        "active_accum_steps_target": result.get("active_accum_steps_target"),
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


@app.route("/frozen_activation_step", methods=["POST"])
def frozen_activation_step_route():
    request_bytes = len(request.get_data(cache=True) or b"")
    t0 = time.perf_counter()
    with state.lock:
        payload = _unpickle_request_data()
        result = state.frozen_activation_step(
            str(payload["client_id"]), payload["activations"], payload["targets"]
        )
        response_blob = pickle.dumps(result)
        if state.comm_log_train and state._should_log_batch(payload.get("batch_id")):
            state._append_comm_row({
                "level": "split", "observer": "server", "cluster_id": state.runtime_cfg.cluster_id,
                "client_id": str(payload["client_id"]), "event": "frozen_activation_step",
                "route": "/frozen_activation_step", "direction": "server_process",
                "payload_type": "activation_payload", "payload_bytes": request_bytes,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3), "status": "ok",
                "server_optimizer_steps": 0, "server_state_unchanged": True,
            })
        return Response(response_blob, content_type="application/octet-stream")


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
        if state.comm_log_eval and state._should_log_batch(payload.get("batch_id")):
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
        if state.comm_log_eval and state._should_log_batch(payload.get("batch_id")):
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
    if FROZEN_VALIDATION_MODE:
        return jsonify({"error": "gradient flushing is disabled in frozen validation mode"}), 409
    with state.lock:
        result = state.flush_accumulated_gradients(reason="manual_flush", increment_round=False)
        return jsonify(result)


@app.route("/save_best_checkpoint", methods=["POST"])
def save_best_checkpoint():
    if FROZEN_VALIDATION_MODE:
        return jsonify({"error": "checkpoint saving is disabled in frozen validation mode"}), 409
    with state.lock:
        path = state._save_best(flush=True)
        return jsonify({"path": path})


@app.route("/save_final_checkpoint", methods=["POST"])
def save_final_checkpoint():
    if FROZEN_VALIDATION_MODE:
        return jsonify({"error": "checkpoint saving is disabled in frozen validation mode"}), 409
    with state.lock:
        path = state._save_final(flush=True)
        return jsonify({"path": path})


@app.route("/save_latest_checkpoint", methods=["POST"])
def save_latest_checkpoint():
    if FROZEN_VALIDATION_MODE:
        return jsonify({"error": "checkpoint saving is disabled in frozen validation mode"}), 409
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
    if FROZEN_VALIDATION_MODE:
        return jsonify({"error": "FedAvg weight loading is disabled in frozen validation mode"}), 409
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
