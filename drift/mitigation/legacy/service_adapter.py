"""Policy adapter for the mitigation suite's production SplitFed services.

M1 delegates to the unchanged run_split scheduler. Frozen and client-only methods
must not enter that scheduler because it owns normal server updates and FedAvg.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import torch

import fed_server.fed_server as fedavg_service
import run_split

from ..core import Method, POLICIES, canonical_hash, tensor_mapping_hash


PRODUCTION_EXECUTION_TRACE = (
    "data.stream_service.stream_reset(window_offset)",
    "data.stream_service.stream_next",
    "run_split.run_training_schedule_with_fedavg",
    "run_split.run_scheduler_epoch_across_clusters",
    "run_split.run_single_scheduler_epoch_for_cluster",
    "run_split.run_round",
    "client.client.ClusteredClientRunner._train_single_batch",
    "server.server.SplitLearningServerState.train_step",
    "server.server.SplitLearningServerState.flush_accumulated_gradients",
    "client.client.ClusteredClientRunner.optimizer (created once and restored from checkpoint)",
    "server.server.SplitLearningServerState.optimizer (created once and restored from checkpoint)",
    "run_split.run_server_fedavg_after_epoch",
    "fed_server.fed_server.run_fedavg_round",
    "utils.runtime_utils.try_load_torch_checkpoint",
    "utils.runtime_utils.save_torch_checkpoint",
    "run_split.run_test_phase(window_offset=final_holdout_start)",
    "client.client.ClusteredClientRunner.test",
    "client.client.ClusteredClientRunner._run_eval",
    "server.server.SplitLearningServerState.inference_step",
)

@dataclass(frozen=True)
class AdapterResult:
    method: str
    scheduler_invoked: bool
    fedavg_enabled: bool
    durations: Mapping[str, float]
    execution_trace: tuple[str, ...]


def validate_full_population(cluster_clients: Mapping[str, Sequence[str]]) -> None:
    if len(cluster_clients) != 4:
        raise ValueError("full mitigation adapter requires exactly four clusters")
    counts = {str(cluster): len(tuple(clients)) for cluster, clients in cluster_clients.items()}
    if any(count != 21 for count in counts.values()) or sum(counts.values()) != 84:
        raise ValueError(f"full mitigation adapter requires 21 clients per cluster (84 total), got {counts}")


def run_production_scheduler_adapter(
    method: Method,
    cluster_ids: Sequence[str],
    cluster_clients: Mapping[str, Sequence[str]],
    env: Mapping[str, str],
    *,
    require_full_population: bool = True,
) -> AdapterResult:
    """Route M1 through the production scheduler without duplicating its maths."""
    method = Method(method)
    if require_full_population:
        validate_full_population(cluster_clients)
    policy = POLICIES[method]
    if method is not Method.M1:
        return AdapterResult(method.value, False, False, {}, PRODUCTION_EXECUTION_TRACE)
    durations = run_split.run_training_schedule_with_fedavg(
        [str(item) for item in cluster_ids],
        {str(cluster): [str(item) for item in clients] for cluster, clients in cluster_clients.items()},
        dict(env),
    )
    serializable_durations = {
        "/".join(str(part) for part in key) if isinstance(key, tuple) else str(key): float(value)
        for key, value in durations.items()
    }
    return AdapterResult(method.value, True, policy.fedavg_enabled, serializable_durations, PRODUCTION_EXECUTION_TRACE)



def run_prediction_only_final_holdout(
    cluster_ids: Sequence[str],
    cluster_clients: Mapping[str, Sequence[str]],
    env: Mapping[str, str],
    *,
    final_holdout_window_offset: int = 1224,
    final_holdout_window_limit: int = 524,
    stream_mode: str = "deployment_final_holdout",
    require_full_population: bool = True,
) -> Mapping[tuple[str, str], float]:
    """Use production inference endpoints starting at the untouched 524-window holdout."""
    if require_full_population:
        validate_full_population(cluster_clients)
    if (int(final_holdout_window_offset), int(final_holdout_window_limit)) != (1224, 524) and require_full_population:
        raise ValueError("full suite final holdout must be exactly offset 1224, limit 524")
    return dict(run_split.run_test_phase(
        [str(item) for item in cluster_ids],
        {str(cluster): [str(item) for item in clients] for cluster, clients in cluster_clients.items()},
        dict(env),
        window_offset=int(final_holdout_window_offset),
        window_limit=int(final_holdout_window_limit),
        stream_mode=stream_mode,
    ))

def _scheduler_fixture(*, clusters: int, real_fedavg: bool) -> dict[str, Any]:
    """Execute production orchestration with deterministic in-memory service endpoints."""
    cluster_ids = [str(index + 1) for index in range(clusters)]
    cluster_clients = {cluster: [f"{cluster}01", f"{cluster}02"] for cluster in cluster_ids}
    env = {
        "NUM_EPOCHS": "1",
        "LOCAL_EPOCHS_PER_GLOBAL_ROUND": "1",
        "RUN_CLUSTERS_IN_PARALLEL": "false",
        "TRAINING_MODE": "client_batch_random",
        "CLIENT_ORDER_STRATEGY": "fixed",
        "RANDOM_SEED": "42",
        "RANDOM_SEED_SCHEDULE": "42",
    }
    events: list[dict[str, Any]] = []
    pushes: dict[str, dict[str, torch.Tensor]] = {}
    fedavg_events: list[dict[str, Any]] = []
    with TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp)
        old_output = run_split.OUTPUT_DIR
        old_results = run_split.RESULTS_DIR
        old_assignments = run_split.ASSIGNMENTS_PATH
        run_split.OUTPUT_DIR = root
        run_split.RESULTS_DIR = root / "results"
        run_split.ASSIGNMENTS_PATH = root / "assignments.csv"
        run_split.ASSIGNMENTS_PATH.write_text(
            "Cluster,ID\n" + "".join(f"{cluster},{client}\n" for cluster in cluster_ids for client in cluster_clients[cluster]),
            encoding="utf-8",
        )
        port_to_client: dict[int, tuple[str, str]] = {}
        client_state: dict[tuple[str, str], dict[str, bool]] = {}
        for cluster_index, cluster in enumerate(cluster_ids):
            for client_index, client in enumerate(cluster_clients[cluster]):
                port = run_split.CLIENT_WORKER_BASE_PORT + cluster_index * 100 + client_index
                port_to_client[port] = (cluster, client)
                client_state[(cluster, client)] = {"ready": True, "done": False}

        def status(port: int) -> dict[str, Any]:
            key = port_to_client[int(port)]
            state = client_state[key]
            return {"is_done": state["done"], "has_next_train_batch": state["ready"]}

        def identify(url: str) -> tuple[int, str]:
            tail = url.split("localhost:", 1)[1]
            port_text, route = tail.split("/", 1)
            return int(port_text), "/" + route

        def post_json(url: str, payload: Mapping[str, Any], timeout: float = 30.0) -> dict[str, Any]:
            port, route = identify(url)
            if route == "/round/start":
                events.append({"event": "round_start", "server_port": port, **dict(payload)})
                return {"status": "ok"}
            cluster, client = port_to_client[port]
            if route == "/command/train_one_batch":
                client_state[(cluster, client)]["ready"] = False
                events.append({"event": "train_batch", "cluster_id": cluster, "client_id": client})
                return {"status": "batch_processed", "completed_epochs": 0, "early_stopped": False, "train_stream_active": True}
            if route == "/command/finalize_epoch":
                client_state[(cluster, client)]["done"] = True
                run_split.write_json(
                    run_split.progress_path(cluster, client),
                    {"completed_epochs": 1, "early_stopped": False},
                )
                events.append({"event": "finalize_epoch", "cluster_id": cluster, "client_id": client})
                return {"status": "training_complete", "completed_epochs": 1, "early_stopped": False}
            raise AssertionError(f"unexpected fixture POST {url}")

        def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
            _, route = identify(url)
            if route == "/round/state":
                return {"pending_accum_steps": 0, "round_total_contrib_count": 0, "window_contrib_count": 0, "expected_clients": []}
            raise AssertionError(f"unexpected fixture GET {url}")

        stack.enter_context(patch.object(run_split, "fetch_worker_status", status))
        stack.enter_context(patch.object(run_split, "post_json", post_json))
        stack.enter_context(patch.object(run_split, "get_json", get_json))
        if real_fedavg:
            server_states = {
                "1": {"weight": torch.tensor([1.0, 3.0])},
                "2": {"weight": torch.tensor([5.0, 7.0])},
            }
            for cluster, state_dict in server_states.items():
                path = root / f"cluster_{cluster}" / "server_latest.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": state_dict}, path)

            def push(url: str, state_dict: Mapping[str, torch.Tensor], log_path: str, cluster_id: str) -> bool:
                pushes[str(cluster_id)] = {name: tensor.detach().clone() for name, tensor in state_dict.items()}
                return True

            stack.enter_context(patch.object(fedavg_service, "_force_server_checkpoint", lambda *args, **kwargs: True))
            stack.enter_context(patch.object(fedavg_service, "_push_weights", push))
            stack.enter_context(patch.object(run_split, "cluster_train_sample_weights", lambda *args: {"1": 1.0, "2": 3.0}))
        else:
            def record_fedavg(cluster_ids_arg, cluster_clients_arg, scheduler_epoch):
                fedavg_events.append({"scheduler_epoch": scheduler_epoch, "clusters": list(cluster_ids_arg)})
                return None
            stack.enter_context(patch.object(run_split, "run_server_fedavg_after_epoch", record_fedavg))

        try:
            result = run_production_scheduler_adapter(
                Method.M1,
                cluster_ids,
                cluster_clients,
                env,
                require_full_population=False,
            )
            global_state = None
            global_path = root / "global" / "global_server_latest.pt"
            if global_path.exists():
                global_state = torch.load(global_path, map_location="cpu")["model_state_dict"]
            report = {
                "adapter": asdict(result),
                "events": events,
                "fedavg_event_count": 1 if global_state is not None else len(fedavg_events),
                "global_tensor_hash": None if global_state is None else tensor_mapping_hash(global_state),
                "pushed_tensor_hashes": {cluster: tensor_mapping_hash(state) for cluster, state in pushes.items()},
                "expected_weight": None if global_state is None else [4.0, 6.0],
                "actual_weight": None if global_state is None else global_state["weight"].tolist(),
                "fedavg_parity": global_state is None or torch.equal(global_state["weight"], torch.tensor([4.0, 6.0])),
            }
        finally:
            run_split.OUTPUT_DIR = old_output
            run_split.RESULTS_DIR = old_results
            run_split.ASSIGNMENTS_PATH = old_assignments
    return report


def run_full_adapter_dry_run() -> dict[str, Any]:
    report = _scheduler_fixture(clusters=1, real_fedavg=False)
    policy_results = {
        method.value: asdict(run_production_scheduler_adapter(
            method,
            ["fixture"],
            {"fixture": ["a", "b"]},
            {"NUM_EPOCHS": "1"},
            require_full_population=False,
        ))
        for method in (Method.R0, Method.M0, Method.M2)
    }
    report["non_M1_policies"] = policy_results
    report["accepted"] = (
        report["adapter"]["scheduler_invoked"]
        and sum(item["event"] == "train_batch" for item in report["events"]) == 2
        and all(not item["scheduler_invoked"] and not item["fedavg_enabled"] for item in policy_results.values())
    )
    return report


def run_two_cluster_fedavg_fixture() -> dict[str, Any]:
    report = _scheduler_fixture(clusters=2, real_fedavg=True)
    report["accepted"] = (
        report["fedavg_event_count"] == 1
        and report["fedavg_parity"]
        and len(set(report["pushed_tensor_hashes"].values())) == 1
        and sum(item["event"] == "train_batch" for item in report["events"]) == 4
    )
    return report
