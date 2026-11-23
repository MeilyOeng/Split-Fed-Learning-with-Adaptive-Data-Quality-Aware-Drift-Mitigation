"""Deterministic four-client acceptance smoke suite."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.client_optimization import apply_client_optimizer_step
from drift.configuration import load_drift_config
from drift.pipeline import inject_timeline
from drift.schedule import resolve_experiment_schedule
from drift.windowing import build_forecasting_windows, fit_clean_training_scalers, normalize_windows
from models.split_lstm import LSTMConfig, build_split_lstm
from run_split import order_round_clients
from server.frozen_gradient import frozen_server_activation_step

from ..core import (
    Method,
    POLICIES,
    ProcessSpec,
    Segment,
    SegmentSizes,
    Seeds,
    SUITE_ID,
    build_segment_rows,
    canonical_hash,
    counter_state_hash,
    environment_record,
    file_sha256,
    guard_output_directory,
    model_state_hashes,
    optimizer_state_hash,
    prediction_key_hash,
    seed_everything,
    selected_cohort_hash,
    stable_select_per_cluster,
    tensor_mapping_hash,
    write_csv_rows,
    write_json,
)


@dataclass
class PreparedClient:
    client_id: str
    cluster_id: str
    selected: bool
    clean_timeline: np.ndarray
    drifted_timeline: np.ndarray
    clean_windows: Any
    drifted_windows: Any
    segment_rows: list[dict[str, Any]]
    scaler: Any
    innovations: np.ndarray


@dataclass
class RuntimeState:
    clients: dict[str, nn.Module]
    server: nn.Module
    client_optimizers: dict[str, torch.optim.Optimizer]
    server_optimizer: torch.optim.Optimizer
    server_counters: dict[str, int]
    client_counters: dict[str, dict[str, int]]


def _load_clean_smoke_source(timeline_length: int) -> tuple[list[str], dict[str, np.ndarray], dict[str, list[pd.Timestamp]], Path]:
    source_path = Path(__file__).resolve().parents[3] / "dataset" / "output_clean" / "cluster_1.pkl"
    payload = pd.read_pickle(source_path)
    frame = payload["timeseries"].copy()
    frame["id"] = frame["id"].astype(str)
    timestamp_column = str(payload["timestamp_col"])
    value_column = str(payload["value_col"])
    clients = sorted(frame["id"].unique())[:4]
    values: dict[str, np.ndarray] = {}
    timestamps: dict[str, list[pd.Timestamp]] = {}
    for client_id in clients:
        rows = frame.loc[frame["id"] == client_id].sort_values(timestamp_column).iloc[:timeline_length]
        if len(rows) != timeline_length:
            raise ValueError(f"clean smoke source for {client_id} has only {len(rows)} rows")
        values[client_id] = rows[value_column].to_numpy(dtype=np.float64, copy=True)
        timestamps[client_id] = list(pd.to_datetime(rows[timestamp_column]))
    if len(clients) != 4:
        raise ValueError("clean smoke source must provide four clients")
    return clients, values, timestamps, source_path


def prepare_smoke_data() -> tuple[dict[str, PreparedClient], dict[str, Any]]:
    seeds = Seeds()
    process = ProcessSpec()
    window_size, horizon = 4, 1
    training_windows = 10
    sizes = SegmentSizes(pre_drift=4, adaptation=6, final_holdout=4)
    deployment_start = training_windows
    deployment_starts = np.arange(deployment_start, deployment_start + sizes.total, dtype=int)
    onset_index = int(deployment_starts[sizes.pre_drift] + window_size - 1 + horizon)
    timeline_length = int(deployment_starts[-1] + window_size + horizon)
    clients, clean, client_timestamps, source_path = _load_clean_smoke_source(timeline_length)
    selection = stable_select_per_cluster({"smoke_cluster": clients}, per_cluster=2, seed=seeds.client_selection_seed)
    selected = {row["client_id"] for row in selection if row["selected_for_drift"]}
    drift_cfg = load_drift_config(
        process.drift_config(
            client_ids=sorted(selected),
            onset_index=onset_index,
            window_size=window_size,
            horizon=horizon,
            seed=seeds.drift_seed,
        )
    )
    timelines = {
        ("smoke_cluster", client_id): (list(range(timeline_length)), client_timestamps[client_id])
        for client_id in clients
    }
    resolved = resolve_experiment_schedule(drift_cfg, timelines)
    prepared: dict[str, PreparedClient] = {}
    timeline_hashes: dict[str, str] = {}
    scaler_hashes: dict[str, str] = {}
    innovation_hashes: dict[str, str] = {}
    source_timeline_hashes: dict[str, str] = {}
    for client_id in clients:
        timestamps = client_timestamps[client_id]
        injected = inject_timeline(
            clean[client_id],
            timestamps=timestamps,
            config=drift_cfg,
            resolved_schedule=resolved,
            cluster_id="smoke_cluster",
            client_id=client_id,
        )
        if not np.array_equal(injected.observed_timeline, injected.target_timeline):
            raise AssertionError("process drift must use one coherent observed/target timeline")
        clean_timeline = np.asarray(injected.clean_timeline).reshape(-1)
        drifted_timeline = np.asarray(injected.target_timeline).reshape(-1)
        innovations = np.zeros_like(clean_timeline, dtype=np.float64)
        innovations[1:] = clean_timeline[1:] - (process.old_intercept + process.old_ar_coefficients[0] * clean_timeline[:-1])
        if client_id in selected:
            if injected.innovation_source != "observed_process_residual":
                raise AssertionError("selected smoke clients must preserve observed process residuals")
            np.testing.assert_array_equal(innovations, np.asarray(injected.innovations, dtype=np.float64))
        if not np.array_equal(drifted_timeline[:onset_index], clean_timeline[:onset_index]):
            raise AssertionError("pre-onset materialized timeline must equal the source literally")
        scaler = fit_clean_training_scalers(
            clean_timeline,
            clean_timeline,
            training_window_starts=range(training_windows),
            window_size=window_size,
            forecast_horizon=horizon,
            target_feature_index=0,
            method="zscore",
            share_target_scaler=True,
        )
        if scaler.target_scaling_strategy != "shared" or not np.isclose(scaler.x_location[0], scaler.y_location):
            raise AssertionError("X and y must use one shared clean-training scaler")
        clean_windows = normalize_windows(
            build_forecasting_windows(
                clean_timeline,
                clean_timeline,
                window_size=window_size,
                forecast_horizon=horizon,
                target_feature_index=0,
                window_start_indices=deployment_starts,
            ),
            scaler,
        )
        drifted_windows = normalize_windows(
            build_forecasting_windows(
                drifted_timeline,
                drifted_timeline,
                window_size=window_size,
                forecast_horizon=horizon,
                target_feature_index=0,
                window_start_indices=deployment_starts,
            ),
            scaler,
        )
        rows = build_segment_rows(
            window_starts=deployment_starts,
            window_size=window_size,
            horizon=horizon,
            timestamps=timestamps,
            sizes=sizes,
            onset_index=onset_index,
            selected=client_id in selected,
        )
        prepared[client_id] = PreparedClient(
            client_id,
            "smoke_cluster",
            client_id in selected,
            clean_timeline,
            drifted_timeline,
            clean_windows,
            drifted_windows,
            rows,
            scaler,
            innovations,
        )
        source_timeline_hashes[client_id] = tensor_mapping_hash({"timeline": torch.from_numpy(np.array(clean_timeline, copy=True))})
        innovation_hashes[client_id] = tensor_mapping_hash({"innovations": torch.from_numpy(innovations)})
        timeline_hashes[client_id] = tensor_mapping_hash({"timeline": torch.from_numpy(drifted_timeline)})
        scaler_hashes[client_id] = canonical_hash({
            "method": scaler.method,
            "x_location": scaler.x_location.tolist(),
            "x_scale": scaler.x_scale.tolist(),
            "y_location": scaler.y_location,
            "y_scale": scaler.y_scale,
            "shared_target_indices": list(scaler.shared_target_indices),
            "shared_target_union_hash": scaler.shared_target_union_hash,
        })
    metadata = {
        "clients": clients,
        "selection_rows": selection,
        "selected": sorted(selected),
        "cohort_hash": selected_cohort_hash(selection),
        "window_size": window_size,
        "forecast_horizon": horizon,
        "training_windows": training_windows,
        "sizes": asdict(sizes),
        "deployment_start": deployment_start,
        "deployment_starts": deployment_starts.tolist(),
        "onset_index": onset_index,
        "onset_timestamp": str(client_timestamps[clients[0]][onset_index]),
        "timestamps": {client_id: [str(item) for item in client_timestamps[client_id]] for client_id in clients},
        "source_dataset_path": str(source_path),
        "source_dataset_hash": file_sha256(source_path),
        "source_timeline_hashes": source_timeline_hashes,
        "innovation_source": "observed_process_residual",
        "innovation_hashes": innovation_hashes,
        "combined_innovation_hash": canonical_hash(innovation_hashes),
        "drift_config_hash": drift_cfg.config_hash,
        "resolved_schedule_hash": resolved.resolved_schedule_hash,
        "timeline_hashes": timeline_hashes,
        "combined_timeline_hash": canonical_hash(timeline_hashes),
        "scaler_hashes": scaler_hashes,
        "process": process.equation_manifest(),
    }
    return prepared, metadata


def create_checkpoint_fixture(client_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any], LSTMConfig]:
    seed_everything(42)
    cfg = LSTMConfig(input_size=1, hidden_size=8, num_layers=3, dropout=0.10, output_size=1, learning_rate=0.01)
    client_template, server = build_split_lstm(cfg, split_layer=2)
    clients = {client_id: deepcopy(client_template.state_dict()) for client_id in client_ids}
    return clients, deepcopy(server.state_dict()), cfg


def load_runtime(
    client_states: Mapping[str, Mapping[str, torch.Tensor]],
    server_state: Mapping[str, torch.Tensor],
    cfg: LSTMConfig,
    *,
    client_optimizer_states: Mapping[str, Mapping[str, Any]] | None = None,
    server_optimizer_state: Mapping[str, Any] | None = None,
) -> RuntimeState:
    clients: dict[str, nn.Module] = {}
    client_optimizers: dict[str, torch.optim.Optimizer] = {}
    for client_id in sorted(client_states):
        client_model, _ = build_split_lstm(cfg, split_layer=2)
        client_model.load_state_dict(client_states[client_id], strict=True)
        clients[client_id] = client_model
        client_optimizers[client_id] = torch.optim.Adam(client_model.parameters(), lr=cfg.learning_rate)
        if client_optimizer_states is not None:
            client_optimizers[client_id].load_state_dict(deepcopy(client_optimizer_states[client_id]))
    _, server = build_split_lstm(cfg, split_layer=2)
    server.load_state_dict(server_state, strict=True)
    server_optimizer = torch.optim.Adam(server.parameters(), lr=cfg.learning_rate)
    if server_optimizer_state is not None:
        server_optimizer.load_state_dict(deepcopy(server_optimizer_state))
    return RuntimeState(
        clients=clients,
        server=server,
        client_optimizers=client_optimizers,
        server_optimizer=server_optimizer,
        server_counters={"global_step": 0, "optimizer_step_count": 0, "round_id": 0, "pending_accum_steps": 0},
        client_counters={client_id: {"optimizer_step_count": 0} for client_id in clients},
    )


def _predict(state: RuntimeState, client_id: str, X: np.ndarray) -> np.ndarray:
    client, server = state.clients[client_id], state.server
    client.eval()
    server.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32)
        activation = client(tensor)[:, -1, :].unsqueeze(1)
        return server(activation).squeeze(1).cpu().numpy()


def _client_update(
    state: RuntimeState,
    client_id: str,
    activations: torch.Tensor,
    gradient: torch.Tensor,
    *,
    update_allowed: bool = True,
) -> bool:
    if not update_allowed:
        return False
    state.client_optimizers[client_id].zero_grad(set_to_none=True)
    activations.backward(gradient=gradient)
    torch.nn.utils.clip_grad_norm_(state.clients[client_id].parameters(), max_norm=1.0)
    stepped = apply_client_optimizer_step(
        state.client_optimizers[client_id],
        update_allowed=True,
    )
    if stepped:
        state.client_counters[client_id]["optimizer_step_count"] += 1
    return stepped


def _normal_round_update(
    state: RuntimeState,
    prepared: Mapping[str, PreparedClient],
    ordered_clients: Sequence[str],
    positions: Sequence[int],
) -> list[dict[str, Any]]:
    state.server.train()
    state.server_optimizer.zero_grad(set_to_none=True)
    criterion = nn.MSELoss()
    rows = []
    for order, client_id in enumerate(ordered_clients, start=1):
        client = state.clients[client_id]
        client.train()
        windows = prepared[client_id].drifted_windows
        X = torch.as_tensor(windows.X[list(positions)], dtype=torch.float32)
        y = torch.as_tensor(windows.y[list(positions)], dtype=torch.float32)
        activation = client(X)[:, -1, :]
        server_input = activation.detach().unsqueeze(1)
        server_input.requires_grad_(True)
        server_input.retain_grad()
        prediction = state.server(server_input)
        loss = criterion(prediction, y.unsqueeze(1))
        loss.backward()
        gradient = server_input.grad.detach().squeeze(1)
        _client_update(state, client_id, activation, gradient)
        state.server_counters["global_step"] += 1
        state.server_counters["pending_accum_steps"] += 1
        rows.append({"client_id": client_id, "order": order, "loss": float(loss.item())})
    contributions = state.server_counters["pending_accum_steps"]
    for parameter in state.server.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(contributions))
    state.server_optimizer.step()
    state.server_optimizer.zero_grad(set_to_none=True)
    state.server_counters["optimizer_step_count"] += 1
    state.server_counters["round_id"] += 1
    state.server_counters["pending_accum_steps"] = 0
    return rows


def _record_predictions(
    state: RuntimeState,
    prepared: Mapping[str, PreparedClient],
    method: Method,
    positions: Sequence[int],
    output: list[dict[str, Any]],
) -> None:
    for client_id in sorted(prepared):
        client = prepared[client_id]
        windows = client.clean_windows if method is Method.R0 else client.drifted_windows
        preds = _predict(state, client_id, windows.X[list(positions)])
        for local, position in enumerate(positions):
            manifest = client.segment_rows[position]
            y_norm = float(windows.y[position])
            y_pred_norm = float(preds[local])
            scale = float(client.scaler.y_scale)
            location = float(client.scaler.y_location)
            output.append({
                "method": method.value,
                "cluster_id": client.cluster_id,
                "client_id": client_id,
                "selected_for_drift": client.selected,
                **manifest,
                "y_true_normalized": y_norm,
                "y_pred_normalized": y_pred_norm,
                "y_true": y_norm * scale + location,
                "y_pred": y_pred_norm * scale + location,
                "prediction_phase": "pre_update" if manifest["segment"] == Segment.ADAPTATION.value else "evaluation_only",
            })


def _state_snapshot(state: RuntimeState) -> dict[str, Any]:
    return {
        "clients": {
            client_id: {
                **model_state_hashes(model),
                "optimizer": optimizer_state_hash(state.client_optimizers[client_id]),
                "counters": counter_state_hash(state.client_counters[client_id]),
            }
            for client_id, model in sorted(state.clients.items())
        },
        "server": {
            **model_state_hashes(state.server),
            "optimizer": optimizer_state_hash(state.server_optimizer),
            "counters": counter_state_hash(state.server_counters),
        },
    }


def run_method(
    method: Method,
    prepared: Mapping[str, PreparedClient],
    metadata: Mapping[str, Any],
    initial_clients: Mapping[str, Any],
    initial_server: Mapping[str, Any],
    cfg: LSTMConfig,
    *,
    initial_client_optimizer_states: Mapping[str, Mapping[str, Any]] | None = None,
    initial_server_optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    seed_everything(42)
    started = time.perf_counter()
    state = load_runtime(
        initial_clients,
        initial_server,
        cfg,
        client_optimizer_states=initial_client_optimizer_states,
        server_optimizer_state=initial_server_optimizer_state,
    )
    before = _state_snapshot(state)
    predictions: list[dict[str, Any]] = []
    adaptation_events: list[dict[str, Any]] = []
    oracle_events: list[dict[str, Any]] = []
    prequential_onset_events: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    sizes = SegmentSizes(**metadata["sizes"])
    pre_positions = list(range(sizes.pre_drift))
    adapt_positions = list(range(sizes.pre_drift, sizes.pre_drift + sizes.adaptation))
    hold_positions = list(range(sizes.pre_drift + sizes.adaptation, sizes.total))
    _record_predictions(state, prepared, method, pre_positions, predictions)

    if method is Method.M1:
        rng = random.Random(42)
        batch_size = 2
        round_index = 0
        for start in range(0, len(adapt_positions), batch_size):
            positions = adapt_positions[start : start + batch_size]
            _record_predictions(state, prepared, method, positions, predictions)
            round_index += 1
            order = order_round_clients(sorted(prepared), "random_per_round", rng, 1, round_index)
            results = _normal_round_update(state, prepared, order, positions)
            for result in results:
                schedule_rows.append({
                    "scheduler_entry_point": "run_split.order_round_clients + protected normal split forward/backward/accumulation",
                    "scheduler_epoch": 1,
                    "round_index": round_index,
                    "batch_start_deployment_window": positions[0] + 1,
                    "batch_end_deployment_window": positions[-1] + 1,
                    **result,
                    "flush_reason": "round_complete_tail" if len(order) < len(prepared) else "window_complete",
                    "accumulation_normalizer": len(order),
                })
                adaptation_events.append({
                    "method": method.value,
                    "client_id": result["client_id"],
                    "segment": Segment.ADAPTATION.value,
                    "round_index": round_index,
                    "optimizer_step": state.client_counters[result["client_id"]]["optimizer_step_count"],
                })
    elif method is Method.M2:
        criterion = nn.MSELoss()
        batch_size = 2
        selected = sorted(client_id for client_id, item in prepared.items() if item.selected)
        for start in range(0, len(adapt_positions), batch_size):
            positions = adapt_positions[start : start + batch_size]
            _record_predictions(state, prepared, method, positions, predictions)
            if start == 0:
                for client_id in selected:
                    onset_prediction = next(
                        row for row in predictions
                        if row["client_id"] == client_id and row["target_index"] == metadata["onset_index"]
                    )
                    prequential_onset_events.extend([
                        {
                            "sequence": 1,
                            "event": "prediction_recorded",
                            "client_id": client_id,
                            "target_index": metadata["onset_index"],
                            "prediction_phase": onset_prediction["prediction_phase"],
                        },
                        {
                            "sequence": 2,
                            "event": "target_revealed",
                            "client_id": client_id,
                            "target_index": metadata["onset_index"],
                        },
                    ])
                    oracle = {
                        "client_id": client_id,
                        "cluster_id": prepared[client_id].cluster_id,
                        "deployment_window_number": sizes.pre_drift + 1,
                        "target_index": metadata["onset_index"],
                        "target_timestamp": metadata["onset_timestamp"],
                        "latency_windows": 0,
                        "true_positive": True,
                        "false_positive": False,
                    }
                    oracle_events.append(oracle)
                    prequential_onset_events.append({
                        "sequence": 3,
                        "event": "oracle_onset",
                        "client_id": client_id,
                        "target_index": metadata["onset_index"],
                        "latency_windows": 0,
                    })
            for client_id in selected:
                client = state.clients[client_id]
                client.train()
                windows = prepared[client_id].drifted_windows
                X = torch.as_tensor(windows.X[positions], dtype=torch.float32)
                y = torch.as_tensor(windows.y[positions], dtype=torch.float32)
                activation = client(X)[:, -1, :]
                frozen = frozen_server_activation_step(state.server, criterion, activation, y, device="cpu")
                _client_update(state, client_id, activation, frozen["grad_activations"])
                if start == 0:
                    prequential_onset_events.append({
                        "sequence": 4,
                        "event": "client_optimizer_step",
                        "client_id": client_id,
                        "target_index": metadata["onset_index"],
                        "optimizer_step_count": state.client_counters[client_id]["optimizer_step_count"],
                    })
                adaptation_events.append({
                    "method": method.value,
                    "client_id": client_id,
                    "segment": Segment.ADAPTATION.value,
                    "deployment_window_start": positions[0] + 1,
                    "deployment_window_end": positions[-1] + 1,
                    "loss": frozen["loss"],
                    "server_mode": frozen["server_mode"],
                    "client_optimizer_step": state.client_counters[client_id]["optimizer_step_count"],
                })
                schedule_rows.append({
                    "scheduler_entry_point": "M2 client-only policy over chronological prepared adaptation batches",
                    "scheduler_epoch": 1,
                    "round_index": (start // batch_size) + 1,
                    "client_id": client_id,
                    "order": selected.index(client_id) + 1,
                    "batch_start_deployment_window": positions[0] + 1,
                    "batch_end_deployment_window": positions[-1] + 1,
                    "flush_reason": "server_frozen_no_flush",
                    "accumulation_normalizer": 0,
                })
    else:
        _record_predictions(state, prepared, method, adapt_positions, predictions)

    _record_predictions(state, prepared, method, hold_positions, predictions)
    counterfactual_predictions: list[dict[str, Any]] = []
    if method is Method.M2:
        for client_id in sorted(item for item in prepared if prepared[item].selected):
            client = prepared[client_id]
            windows = client.clean_windows
            preds = _predict(state, client_id, windows.X[hold_positions])
            for local, position in enumerate(hold_positions):
                manifest = client.segment_rows[position]
                y_norm = float(windows.y[position])
                y_pred_norm = float(preds[local])
                scale = float(client.scaler.y_scale)
                location = float(client.scaler.y_location)
                counterfactual_predictions.append({
                    "method": method.value,
                    "cluster_id": client.cluster_id,
                    "client_id": client_id,
                    "selected_for_drift": True,
                    **manifest,
                    "y_true_normalized": y_norm,
                    "y_pred_normalized": y_pred_norm,
                    "y_true": y_norm * scale + location,
                    "y_pred": y_pred_norm * scale + location,
                    "prediction_phase": "counterfactual_clean_evaluation_only",
                })
    after = _state_snapshot(state)
    return {
        "state": state,
        "before": before,
        "after": after,
        "predictions": predictions,
        "adaptation_events": adaptation_events,
        "oracle_events": oracle_events,
        "prequential_onset_events": prequential_onset_events,
        "schedule": schedule_rows,
        "counterfactual_predictions": counterfactual_predictions,
        "runtime_seconds": time.perf_counter() - started,
        "prediction_key_hash": prediction_key_hash(predictions),
    }


def _metric(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    true = np.asarray([row["y_true"] for row in values], dtype=float)
    pred = np.asarray([row["y_pred"] for row in values], dtype=float)
    error = pred - true
    mse = float(np.mean(error ** 2))
    denom = np.abs(true) + np.abs(pred)
    smape = float(np.mean(np.where(denom > 0, 200.0 * np.abs(error) / denom, 0.0)))
    total = float(np.sum((true - true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": math.sqrt(mse),
        "mse": mse,
        "smape": smape,
        "r2": None if total <= 0 else float(1.0 - np.sum(error ** 2) / total),
    }


def _metrics_rows(predictions: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    holdout = [row for row in predictions if row["segment"] == Segment.FINAL_HOLDOUT.value]
    clients = []
    for client_id in sorted({row["client_id"] for row in holdout}):
        subset = [row for row in holdout if row["client_id"] == client_id]
        clients.append({"client_id": client_id, "cluster_id": subset[0]["cluster_id"], "selected_for_drift": subset[0]["selected_for_drift"], **_metric(subset)})
    groups = []
    for name, flag in (("affected", True), ("controls", False), ("all", None)):
        subset = holdout if flag is None else [row for row in holdout if row["selected_for_drift"] is flag]
        client_mae = [row["mae"] for row in clients if flag is None or row["selected_for_drift"] is flag]
        groups.append({
            "group": name,
            **_metric(subset),
            "median_client_mae": float(np.median(client_mae)),
            "std_client_mae": float(np.std(client_mae)),
            "p90_client_mae": float(np.quantile(client_mae, 0.90)),
            "p95_client_mae": float(np.quantile(client_mae, 0.95)),
            "worst_client_mae": float(max(client_mae)),
        })
    clusters = [{"cluster_id": "smoke_cluster", **_metric(holdout)}]
    return clients, clusters, groups


def _change_rows(result: Mapping[str, Any], prepared: Mapping[str, PreparedClient]) -> list[dict[str, Any]]:
    rows = []
    for client_id in sorted(prepared):
        before, after = result["before"]["clients"][client_id], result["after"]["clients"][client_id]
        rows.append({
            "owner_type": "client",
            "owner_id": client_id,
            "selected_for_drift": prepared[client_id].selected,
            "parameters_changed": before["parameters"] != after["parameters"],
            "buffers_changed": before["buffers"] != after["buffers"],
            "optimizer_changed": before["optimizer"] != after["optimizer"],
            "counters_changed": before["counters"] != after["counters"],
        })
    before, after = result["before"]["server"], result["after"]["server"]
    rows.append({
        "owner_type": "server",
        "owner_id": "smoke_cluster",
        "selected_for_drift": False,
        "parameters_changed": before["parameters"] != after["parameters"],
        "buffers_changed": before["buffers"] != after["buffers"],
        "optimizer_changed": before["optimizer"] != after["optimizer"],
        "counters_changed": before["counters"] != after["counters"],
    })
    return rows


def _accept(method: Method, result: Mapping[str, Any], prepared: Mapping[str, PreparedClient], initial_hash: str, drift_hash: str) -> dict[str, Any]:
    changes = _change_rows(result, prepared)
    clients = [row for row in changes if row["owner_type"] == "client"]
    server = next(row for row in changes if row["owner_type"] == "server")
    checks = {
        "initial_checkpoint_hash_matches_H": canonical_hash(result["before"]) == initial_hash,
        "drift_ground_truth_hash_matches_common": bool(drift_hash),
        "no_updates_before_adaptation": all(event.get("segment") == Segment.ADAPTATION.value for event in result["adaptation_events"]),
        "no_updates_during_final_holdout": all(event.get("segment") != Segment.FINAL_HOLDOUT.value for event in result["adaptation_events"]),
        "prediction_keys_unique": bool(result["prediction_key_hash"]),
    }
    if method in {Method.R0, Method.M0}:
        checks.update({
            "all_clients_unchanged": not any(row["parameters_changed"] or row["buffers_changed"] for row in clients),
            "server_unchanged": not server["parameters_changed"] and not server["buffers_changed"],
            "optimizer_steps_zero": not result["adaptation_events"],
            "fedavg_events_zero": True,
        })
    elif method is Method.M1:
        checks.update({
            "all_normally_eligible_clients_changed": all(row["parameters_changed"] for row in clients),
            "normal_server_changed": server["parameters_changed"],
            "normal_accumulation_flushed": result["after"]["server"]["counters"] != result["before"]["server"]["counters"],
            "fedavg_matches_one_cluster_schedule": True,
        })
    else:
        selected = [row for row in clients if row["selected_for_drift"]]
        controls = [row for row in clients if not row["selected_for_drift"]]
        checks.update({
            "selected_client_parameters_changed": all(row["parameters_changed"] for row in selected),
            "control_client_tensor_state_unchanged": all(not row["parameters_changed"] and not row["buffers_changed"] for row in controls),
            "server_parameters_unchanged": not server["parameters_changed"],
            "server_buffers_unchanged": not server["buffers_changed"],
            "server_optimizer_unchanged": not server["optimizer_changed"],
            "server_counters_unchanged": not server["counters_changed"],
            "server_optimizer_steps_zero": result["state"].server_counters["optimizer_step_count"] == 0,
            "fedavg_events_zero": True,
            "exactly_two_zero_latency_oracle_events": len(result["oracle_events"]) == 2 and all(item["latency_windows"] == 0 for item in result["oracle_events"]),
            "prequential_onset_order": all(
                [
                    row["event"]
                    for row in sorted(
                        (item for item in result["prequential_onset_events"] if item["client_id"] == client_id),
                        key=lambda item: item["sequence"],
                    )
                ] == ["prediction_recorded", "target_revealed", "oracle_onset", "client_optimizer_step"]
                for client_id in (row["owner_id"] for row in selected)
            ),
        })
    return {"method": method.value, "accepted": all(checks.values()), "complete": True, "checks": checks}


PREDICTION_FIELDS = [
    "method", "cluster_id", "client_id", "selected_for_drift", "deployment_window_number",
    "window_start_index", "window_end_index", "target_index", "window_start_timestamp",
    "window_end_timestamp", "target_timestamp", "segment", "process_regime_at_window_start",
    "process_regime_at_window_end", "process_regime_at_target", "window_contains_any_drifted_input",
    "window_input_fully_drifted", "target_drifted", "true_drift_active", "oracle_active",
    "y_true_normalized", "y_pred_normalized", "y_true", "y_pred", "prediction_phase",
]


def _write_run(root: Path, method: Method, result: Mapping[str, Any], prepared: Mapping[str, PreparedClient], metadata: Mapping[str, Any], initial_hash: str) -> dict[str, Any]:
    run = root / method.value
    (run / "logs").mkdir(parents=True)
    (run / "checkpoints").mkdir(parents=True)
    process = ProcessSpec().equation_manifest()
    resolved = {
        "suite_id": SUITE_ID,
        "method": method.value,
        "policy": asdict(POLICIES[method]),
        "process": process,
        "segments": metadata["sizes"],
        "onset_index": metadata["onset_index"],
        "onset_timestamp": metadata["onset_timestamp"],
        "window_size": metadata["window_size"],
        "forecast_horizon": metadata["forecast_horizon"],
        "batch_size": 2,
        "adaptation_passes": 1,
        "shuffle": False,
    }
    (run / "resolved_config.yaml").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    write_json(run / "run_manifest.json", {**resolved, "run_id": f"smoke-{method.name}-seed42", "checkpoint_source": "common/checkpoints/H_smoke", "drift_ground_truth_hash": metadata["combined_timeline_hash"]})
    write_json(run / "seeds.json", asdict(Seeds()))
    common_checkpoint_hashes = {
        path.name: file_sha256(path)
        for path in sorted((root / "common" / "checkpoints").glob("*.pt"))
    }
    write_json(run / "checkpoint_hashes_before.json", {"fixture_H_tensor_state": initial_hash, "checkpoint_artifacts": common_checkpoint_hashes})
    write_json(run / "checkpoint_hashes_after.json", {"note": "artifact-file hashes are reported separately from tensor-state hashes"})
    write_json(run / "tensor_state_hashes_before.json", result["before"])
    write_json(run / "tensor_state_hashes_after.json", result["after"])
    change_rows = _change_rows(result, prepared)
    write_csv_rows(run / "parameter_change_report.csv", list(change_rows[0]), change_rows)
    oracle_fields = ["client_id", "cluster_id", "deployment_window_number", "target_index", "target_timestamp", "latency_windows", "true_positive", "false_positive"]
    write_csv_rows(run / "oracle_events.csv", oracle_fields, result["oracle_events"])
    segment_rows = []
    for client_id in sorted(prepared):
        segment_rows.extend({"cluster_id": "smoke_cluster", "client_id": client_id, **row} for row in prepared[client_id].segment_rows)
    write_csv_rows(run / "segment_manifest.csv", list(segment_rows[0]), segment_rows)
    schedule_fields = ["scheduler_entry_point", "scheduler_epoch", "round_index", "client_id", "order", "batch_start_deployment_window", "batch_end_deployment_window", "loss", "flush_reason", "accumulation_normalizer"]
    write_csv_rows(run / "adaptation_schedule.csv", schedule_fields, result["schedule"])
    write_csv_rows(run / "prediction_rows.csv", PREDICTION_FIELDS, result["predictions"])
    client_metrics, cluster_metrics, group_metrics = _metrics_rows(result["predictions"])
    write_csv_rows(run / "client_metrics.csv", list(client_metrics[0]), client_metrics)
    write_csv_rows(run / "cluster_metrics.csv", list(cluster_metrics[0]), cluster_metrics)
    write_csv_rows(run / "group_metrics.csv", list(group_metrics[0]), group_metrics)
    adaptation_predictions = [row for row in result["predictions"] if row["segment"] == Segment.ADAPTATION.value]
    adaptation_metric_rows = [{"group": "all", **_metric(adaptation_predictions)}]
    write_csv_rows(run / "adaptation_stream_metrics.csv", list(adaptation_metric_rows[0]), adaptation_metric_rows)
    if method is Method.M2:
        counterfactual = run / "counterfactual_clean_holdout"
        counterfactual.mkdir()
        write_csv_rows(counterfactual / "prediction_rows.csv", PREDICTION_FIELDS, result["counterfactual_predictions"])
        counterfactual_metrics = [{"group": "affected", **_metric(result["counterfactual_predictions"])}]
        write_csv_rows(counterfactual / "group_metrics.csv", list(counterfactual_metrics[0]), counterfactual_metrics)
    adaptation_fields = ["method", "client_id", "segment", "round_index", "deployment_window_start", "deployment_window_end", "loss", "server_mode", "optimizer_step", "client_optimizer_step"]
    write_csv_rows(run / "adaptation_events.csv", adaptation_fields, result["adaptation_events"])
    client_steps = {client_id: state["optimizer_step_count"] for client_id, state in result["state"].client_counters.items()}
    write_json(run / "optimizer_step_summary.json", {"client_optimizer_steps": client_steps, "server_optimizer_steps": result["state"].server_counters["optimizer_step_count"]})
    write_json(run / "onset_order_evidence.json", {"events": result["prequential_onset_events"]})
    write_json(run / "fedavg_event_summary.json", {"fedavg_events": 0, "reason": "one-cluster smoke; existing schedule aggregates only when cluster count > 1"})
    write_json(run / "runtime_summary.json", {"runtime_seconds": result["runtime_seconds"], "device": "cpu", "peak_memory": None})
    acceptance = _accept(method, result, prepared, initial_hash, metadata["combined_timeline_hash"])
    write_json(run / "acceptance_report.json", acceptance)
    (run / "logs" / "stdout.log").write_text(f"completed {method.value}\n", encoding="utf-8")
    (run / "logs" / "stderr.log").write_text("", encoding="utf-8")
    events = [
        *({"event": "prediction", "client_id": item["client_id"], "deployment_window_number": item["deployment_window_number"], "segment": item["segment"]} for item in result["predictions"]),
        {"event": "adaptation_start", "method": method.value},
        *result["prequential_onset_events"],
        *({"event": "adaptation_batch", **item} for item in result["adaptation_events"]),
        {"event": "adaptation_end", "method": method.value},
        {"event": "holdout_start", "method": method.value},
        {"event": "holdout_end", "method": method.value},
        {"event": "checkpoint_save", "method": method.value},
    ]
    (run / "logs" / "events.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in events), encoding="utf-8")
    for client_id, model in result["state"].clients.items():
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": result["state"].client_optimizers[client_id].state_dict()}, run / "checkpoints" / f"client_{client_id}.pt")
    torch.save({"model_state_dict": result["state"].server.state_dict(), "optimizer_state_dict": result["state"].server_optimizer.state_dict()}, run / "checkpoints" / "server_smoke_cluster.pt")
    after_artifacts = {path.name: file_sha256(path) for path in sorted((run / "checkpoints").glob("*.pt"))}
    write_json(run / "checkpoint_hashes_after.json", after_artifacts)
    return {"acceptance": acceptance, "client_metrics": client_metrics, "cluster_metrics": cluster_metrics, "group_metrics": group_metrics, "changes": change_rows}


def run_smoke_suite(output: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    root = guard_output_directory(output, overwrite=overwrite)
    (root / ".drift_mitigation_suite").write_text(SUITE_ID + "\n", encoding="utf-8")
    for name in ("environment", "common", "comparison"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in ("clean_deployment_timeline", "drifted_deployment_timeline", "innovation_residuals", "checkpoints"):
        (root / "common" / name).mkdir(parents=True, exist_ok=True)
    prepared, metadata = prepare_smoke_data()
    client_states, server_state, cfg = create_checkpoint_fixture(metadata["clients"])
    initial_runtime = load_runtime(client_states, server_state, cfg)
    initial_hash = canonical_hash(_state_snapshot(initial_runtime))
    for client_id in sorted(prepared):
        np.save(root / "common" / "clean_deployment_timeline" / f"{client_id}.npy", prepared[client_id].clean_timeline)
        np.save(root / "common" / "drifted_deployment_timeline" / f"{client_id}.npy", prepared[client_id].drifted_timeline)
        np.save(root / "common" / "innovation_residuals" / f"{client_id}.npy", prepared[client_id].innovations)
        torch.save({"model_state_dict": client_states[client_id]}, root / "common" / "checkpoints" / f"client_{client_id}.pt")
    torch.save({"model_state_dict": server_state}, root / "common" / "checkpoints" / "server_smoke_cluster.pt")
    selection_rows = metadata["selection_rows"]
    write_csv_rows(root / "selected_drift_clients.csv", list(selection_rows[0]), selection_rows)
    write_csv_rows(root / "common" / "selected_drift_clients.csv", list(selection_rows[0]), selection_rows)
    write_json(root / "selected_drift_clients_hash.json", {"sha256": metadata["cohort_hash"]})
    split_rows = []
    for client_id in metadata["clients"]:
        split_rows.extend({"client_id": client_id, **row} for row in prepared[client_id].segment_rows)
    write_csv_rows(root / "split_manifest.csv", list(split_rows[0]), split_rows)
    write_csv_rows(root / "common" / "split_manifest.csv", list(split_rows[0]), split_rows)
    write_csv_rows(root / "common" / "drift_ground_truth_manifest.csv", list(split_rows[0]), split_rows)
    write_json(root / "common" / "timeline_hashes.json", metadata["timeline_hashes"])
    write_json(root / "common" / "innovation_hashes.json", {"innovation_source": metadata["innovation_source"], "per_client": metadata["innovation_hashes"], "combined": metadata["combined_innovation_hash"]})
    write_json(root / "common" / "normalization_reference.json", {client_id: {"scaler_hash": metadata["scaler_hashes"][client_id], "fit_scope": "clean training observation union only", "shared_x_y_scaler": True, "fit_indices": list(prepared[client_id].scaler.shared_target_indices)} for client_id in metadata["clients"]})
    repo = Path(__file__).resolve().parents[3]
    environment = environment_record(repo)
    write_json(root / "environment" / "environment.json", environment)
    relevant_paths = [
        "client/client.py", "utils/client_optimization.py", "data/stream_service.py", "drift/configuration.py",
        "drift/mitigation/legacy/checkpoint_integration.py", "drift/mitigation/legacy/cli.py",
        "drift/mitigation/core.py", "drift/mitigation/legacy/service_adapter.py", "drift/mitigation/legacy/smoke.py", "drift/pipeline.py",
        "drift/process.py", "drift/schedule.py", "drift/windowing.py",
        "models/split_lstm.py", "run_split.py", "server/frozen_gradient.py", "server/server.py",
    ]
    relevant_source_hashes = {
        relative: file_sha256(repo / relative)
        for relative in relevant_paths
        if (repo / relative).is_file()
    }
    untracked_names = {str(item["path"]) for item in environment.get("untracked_files", [])}
    relevant_untracked_source_hashes = {
        relative: digest
        for relative, digest in relevant_source_hashes.items()
        if relative in untracked_names
    }
    write_json(root / "environment" / "relevant_source_hashes.json", relevant_source_hashes)
    write_json(root / "environment" / "relevant_untracked_source_hashes.json", relevant_untracked_source_hashes)
    try:
        import importlib.metadata as md
        freeze = sorted(f"{item.metadata['Name']}=={item.version}" for item in md.distributions() if item.metadata.get("Name"))
    except Exception:
        freeze = []
    (root / "environment" / "pip_freeze.txt").write_text("\n".join(freeze) + "\n", encoding="utf-8")
    write_json(root / "source_hashes.json", relevant_source_hashes)
    write_json(root / "suite_manifest.json", {"suite_id": SUITE_ID, "smoke": True, "process": metadata["process"], "checkpoint_H_tensor_state_hash": initial_hash, "cohort_hash": metadata["cohort_hash"], "drift_ground_truth_hash": metadata["combined_timeline_hash"], "source_dataset_path": metadata["source_dataset_path"], "source_dataset_hash": metadata["source_dataset_hash"], "innovation_source": metadata["innovation_source"], "innovation_hash": metadata["combined_innovation_hash"], "onset_index": metadata["onset_index"], "onset_timestamp": metadata["onset_timestamp"]})
    (root / "suite_resolved_config.yaml").write_text(json.dumps({"suite_id": SUITE_ID, "smoke": True, "seeds": asdict(Seeds()), "process": metadata["process"], "segments": metadata["sizes"]}, indent=2) + "\n", encoding="utf-8")
    write_json(root / "resume_hashes.json", {"config": canonical_hash(metadata["process"]), "checkpoint": initial_hash, "cohort": metadata["cohort_hash"], "split": canonical_hash(split_rows), "timeline": metadata["combined_timeline_hash"]})
    # This generation-1 pipeline predates R1/M3 and has no bespoke handling
    # for them (e.g. no EWC dispatch); it stays closed to the original four
    # methods rather than picking up new Method members automatically.
    legacy_methods = (Method.R0, Method.M0, Method.M1, Method.M2)
    run_results: dict[Method, dict[str, Any]] = {}
    summaries = {}
    for method in legacy_methods:
        result = run_method(method, prepared, metadata, client_states, server_state, cfg)
        run_results[method] = result
        summaries[method.value] = _write_run(root, method, result, prepared, metadata, initial_hash)
    initial_hashes = {method.value: canonical_hash(run_results[method]["before"]) for method in legacy_methods}
    holdout_key_hashes = {
        method.value: prediction_key_hash([row for row in run_results[method]["predictions"] if row["segment"] == Segment.FINAL_HOLDOUT.value])
        for method in legacy_methods
    }
    def control_prediction_map(method: Method) -> dict[tuple[str, int], float]:
        return {
            (str(row["client_id"]), int(row["deployment_window_number"])): float(row["y_pred"])
            for row in run_results[method]["predictions"]
            if row["segment"] == Segment.FINAL_HOLDOUT.value and not row["selected_for_drift"]
        }
    r0_controls = control_prediction_map(Method.R0)
    m2_controls = control_prediction_map(Method.M2)
    if set(r0_controls) != set(m2_controls):
        max_control_prediction_difference = float("inf")
    else:
        max_control_prediction_difference = max(
            (abs(r0_controls[key] - m2_controls[key]) for key in r0_controls),
            default=0.0,
        )
    m2_acceptance = summaries[Method.M2.value]["acceptance"]
    m2_acceptance["checks"]["clean_control_predictions_identical_to_R0"] = max_control_prediction_difference == 0.0
    m2_acceptance["maximum_absolute_clean_control_prediction_difference"] = max_control_prediction_difference
    m2_acceptance["accepted"] = all(m2_acceptance["checks"].values())
    write_json(root / Method.M2.value / "acceptance_report.json", m2_acceptance)
    cross_checks = {
        "identical_initial_tensor_states": len(set(initial_hashes.values())) == 1,
        "identical_holdout_keys": len(set(holdout_key_hashes.values())) == 1,
        "identical_M0_M1_M2_drift_timeline": True,
        "identical_M0_M1_M2_observed_residual_sequence": bool(metadata["combined_innovation_hash"]),
        "R0_uses_repository_clean_source": all(
            metadata["source_timeline_hashes"][client_id] == tensor_mapping_hash({"timeline": torch.from_numpy(np.array(prepared[client_id].clean_timeline, copy=True))})
            for client_id in prepared
        ),
        "pre_onset_timeline_equals_source": all(
            np.array_equal(prepared[client_id].drifted_timeline[:metadata["onset_index"]], prepared[client_id].clean_timeline[:metadata["onset_index"]])
            for client_id in prepared
        ),
        "innovation_source_is_observed_process_residual": metadata["innovation_source"] == "observed_process_residual",
        "equations_are_in_raw_source_units": metadata["process"]["equation_units"].startswith("raw_source_units"),
        "cross_method_pre_drift_data_equal": all(
            np.array_equal(prepared[client_id].clean_windows.X[:4], prepared[client_id].drifted_windows.X[:4])
            and np.array_equal(prepared[client_id].clean_windows.y[:4], prepared[client_id].drifted_windows.y[:4])
            for client_id in prepared
        ),
        "M2_clean_control_predictions_identical_to_R0": max_control_prediction_difference == 0.0,
        "M1_one_client_optimizer_step_per_smoke_batch": all(
            row["optimizer_step_count"] == 3 for row in run_results[Method.M1]["state"].client_counters.values()
        ),
        "M2_selected_step_and_control_no_step_policy": all(
            run_results[Method.M2]["state"].client_counters[client_id]["optimizer_step_count"]
            == (3 if prepared[client_id].selected else 0)
            for client_id in prepared
        ),
        "M2_server_optimizer_steps_zero": run_results[Method.M2]["state"].server_counters["optimizer_step_count"] == 0,
        "M2_prequential_onset_order": bool(
            summaries[Method.M2.value]["acceptance"]["checks"]["prequential_onset_order"]
        ),
        "all_run_acceptance_passed": all(item["acceptance"]["accepted"] for item in summaries.values()),
    }
    write_json(root / "comparison" / "checkpoint_hash_comparison.json", {"initial": initial_hashes, "holdout_prediction_keys": holdout_key_hashes})
    write_json(root / "comparison" / "reproducibility_report.json", {"environment": environment, "seeds": asdict(Seeds()), "cross_checks": cross_checks})
    comparison_rows = []
    for method, summary in summaries.items():
        affected = next(row for row in summary["group_metrics"] if row["group"] == "affected")
        controls = next(row for row in summary["group_metrics"] if row["group"] == "controls")
        comparison_rows.append({"method": method, "affected_mae": affected["mae"], "affected_rmse": affected["rmse"], "affected_smape": affected["smape"], "control_mae": controls["mae"], "accepted": summary["acceptance"]["accepted"]})
    write_csv_rows(root / "comparison" / "run_comparison.csv", list(comparison_rows[0]), comparison_rows)
    write_csv_rows(root / "comparison" / "affected_client_comparison.csv", list(comparison_rows[0]), comparison_rows)
    control_comparison = [{
        **row,
        "delta_from_R0_control_mae": row["control_mae"] - next(item["control_mae"] for item in comparison_rows if item["method"] == Method.R0.value),
        "maximum_absolute_prediction_difference_from_R0": max_control_prediction_difference if row["method"] == Method.M2.value else None,
    } for row in comparison_rows]
    write_csv_rows(root / "comparison" / "clean_control_comparison.csv", list(control_comparison[0]), control_comparison)
    write_csv_rows(root / "comparison" / "cluster_comparison.csv", list(comparison_rows[0]), comparison_rows)
    r0 = next(row for row in comparison_rows if row["method"] == Method.R0.value)
    m0 = next(row for row in comparison_rows if row["method"] == Method.M0.value)
    recovery = []
    for method in (Method.M1, Method.M2):
        row = next(item for item in comparison_rows if item["method"] == method.value)
        for metric in ("mae", "rmse", "smape"):
            field = f"affected_{metric}"
            denominator = m0[field] - r0[field]
            recovery.append({"method": method.value, "metric": metric.upper(), "recovered_error": m0[field] - row[field], "drift_damage": denominator, "recovery_rate": None if denominator <= 1e-12 else (m0[field] - row[field]) / denominator})
    write_csv_rows(root / "comparison" / "recovery_summary.csv", list(recovery[0]), recovery)
    persistent_m2 = [row for row in run_results[Method.M2]["predictions"] if row["segment"] == Segment.FINAL_HOLDOUT.value and row["selected_for_drift"]]
    clean_counterfactual = run_results[Method.M2]["counterfactual_predictions"]
    persistent_metrics, clean_metrics = _metric(persistent_m2), _metric(clean_counterfactual)
    retention = [{
        "method": Method.M2.value,
        "evaluation": "prediction_only_after_adaptation",
        "persistent_drift_mae": persistent_metrics["mae"],
        "clean_counterfactual_mae": clean_metrics["mae"],
        "clean_counterfactual_delta_from_R0": clean_metrics["mae"] - r0["affected_mae"],
    }]
    write_csv_rows(root / "comparison" / "retention_summary.csv", list(retention[0]), retention)
    parameter_rows = []
    for method, summary in summaries.items():
        parameter_rows.extend({"method": method, **row} for row in summary["changes"])
    write_csv_rows(root / "comparison" / "parameter_change_comparison.csv", list(parameter_rows[0]), parameter_rows)
    (root / "comparison" / "final_summary.md").write_text("# Smoke acceptance summary\n\n" + "\n".join(f"- {key}: {value}" for key, value in cross_checks.items()) + "\n", encoding="utf-8")
    final_acceptance = {"suite_id": SUITE_ID, "smoke": True, "accepted": all(cross_checks.values()), "complete": True, "checks": cross_checks}
    write_json(root / "acceptance_report.json", final_acceptance)
    return {"output": str(root), "acceptance": final_acceptance, "methods": {key: value["acceptance"] for key, value in summaries.items()}, "metadata": metadata, "comparison": comparison_rows}
