"""Shared, dependency-light utilities for the diagnostics package.

Everything here is read-only with respect to production artifacts: it loads
checkpoints / prepared windows / EWC references via ``torch.load`` /
``np.load`` / ``pandas.read_csv`` and never writes into a suite root,
checkpoint H root, or preparation root. See the package docstring for why
the EWC penalty formula and the regression metrics are duplicated here
instead of imported from client.py / evaluation.py.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.split_lstm import LSTMConfig, SplitLSTMClient, SplitLSTMServer, build_split_lstm
from server.frozen_gradient import frozen_server_activation_step

from drift.mitigation.core import canonical_hash, file_sha256, git_record

SAMPLE_KEY_COLUMNS: tuple[str, ...] = (
    "cluster_id", "client_id", "window_start_index", "window_end_index", "target_index", "timestamp",
)


# ---------------------------------------------------------------------------
# Architecture / checkpoint loading (read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelArchitectureSpec:
    """Mirrors drift.mitigation.core.ModelArchitecture's fields without
    importing it, so this package has no import-time dependency on the
    mitigation core module's own validation/dataclass machinery -- just the
    plain field values needed to reconstruct a SplitLSTMClient/Server.
    """

    input_size: int = 1
    hidden_size: int = 128
    num_layers: int = 3
    split_layer: int = 2
    output_size: int = 1
    dropout: float = 0.4
    bidirectional: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ModelArchitectureSpec":
        raw = dict(raw or {})
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in raw.items() if key in known})

    def lstm_config(self) -> LSTMConfig:
        return LSTMConfig(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, batch_first=True, bidirectional=self.bidirectional,
            output_size=self.output_size,
        )


def build_client_server(arch: ModelArchitectureSpec) -> tuple[SplitLSTMClient, SplitLSTMServer]:
    """Fresh, untrained client/server pair with the requested architecture.
    Reused only as a shell to load a checkpoint's state_dict into.
    """
    return build_split_lstm(arch.lstm_config(), arch.split_layer)


def load_model_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Read-only checkpoint load: never writes, never restores optimizer
    state, never mutates the file on disk.
    """
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def load_client_model(checkpoint_path: str | Path, arch: ModelArchitectureSpec) -> SplitLSTMClient:
    client, _ = build_client_server(arch)
    client.load_state_dict(load_model_state_dict(checkpoint_path))
    client.eval()
    return client


def load_server_model(checkpoint_path: str | Path, arch: ModelArchitectureSpec) -> SplitLSTMServer:
    _, server = build_client_server(arch)
    server.load_state_dict(load_model_state_dict(checkpoint_path))
    server.eval()
    return server


def cut_layer_representation(client_model: SplitLSTMClient, X: torch.Tensor) -> torch.Tensor:
    """The exact tensor production transmits to the server.

    This is ``client_model(X)[:, -1, :]`` -- copied verbatim (read, not
    imported) from client/client.py's ``oracle_adapt_one_batch`` so the
    diagnostic measures the real smashed representation, not an
    accidentally-different hidden state. See
    tests/test_diagnostics_common.py for a regression test that pins this
    against the client architecture's known output shape.
    """
    client_model.eval()
    with torch.no_grad():
        return client_model(X)[:, -1, :]


def server_predict(
    server_model: SplitLSTMServer, activations: torch.Tensor, y: torch.Tensor, *, criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Pure, read-only reuse of production's own frozen-activation-gradient
    function. Server parameters/buffers/optimizer are never touched --
    this is the same guarantee production's own M2 code path relies on.
    """
    criterion = criterion or nn.MSELoss()
    return frozen_server_activation_step(server_model, criterion, activations, y, device="cpu")


# ---------------------------------------------------------------------------
# EWC penalty -- deliberately duplicated, not imported (see package docstring)
# ---------------------------------------------------------------------------


def ewc_penalty_loss_diagnostic(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    theta_star: Mapping[str, torch.Tensor],
    fisher: Mapping[str, torch.Tensor],
    lambda_: float,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """L_ewc = (lambda / 2) * sum_i F_i * (theta_i - theta_star_i)^2, over
    client parameters only.

    This is a standalone re-implementation of the same closed-form
    expression client.py's ``_ewc_penalty_loss`` computes -- diagnostics
    must not import from or modify client.py, so this is intentionally a
    second, independently-written copy of the formula rather than a shared
    helper. See test_diagnostics_common.py::test_ewc_penalty_matches_hand_computed_value
    for a hand-computed-fixture regression pinning it to the exact formula.
    """
    penalty = torch.zeros((), device=device)
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if name not in theta_star or name not in fisher:
            continue
        theta_star_t = theta_star[name].to(device)
        fisher_t = fisher[name].to(device)
        penalty = penalty + (fisher_t * (param - theta_star_t) ** 2).sum()
    return 0.5 * float(lambda_) * penalty


# ---------------------------------------------------------------------------
# Regression metrics -- deliberately duplicated, not imported from evaluation.py
# ---------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(true) != len(pred):
        raise ValueError(f"y_true/y_pred length mismatch: {len(true)} vs {len(pred)}")
    error = pred - true
    mse = float(np.mean(error ** 2)) if len(true) else float("nan")
    denom = np.abs(true) + np.abs(pred)
    smape = (
        float(np.mean(np.divide(2 * np.abs(error), denom, out=np.zeros_like(error), where=denom > 0)) * 100)
        if len(true) else float("nan")
    )
    return {
        "n": int(len(true)),
        "mae": float(np.mean(np.abs(error))) if len(true) else float("nan"),
        "mse": mse,
        "rmse": float(np.sqrt(mse)) if len(true) else float("nan"),
        "smape": smape,
    }


def pearson_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Descriptive-only association measures. Never interpret as causal or
    as a significance test -- callers must report sample count alongside.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    n = int(len(x_arr))
    if n < 2 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return {"n": n, "pearson_r": None, "spearman_r": None}
    pearson = float(np.corrcoef(x_arr, y_arr)[0, 1])
    rank_x = pd.Series(x_arr).rank().to_numpy()
    rank_y = pd.Series(y_arr).rank().to_numpy()
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return {"n": n, "pearson_r": pearson, "spearman_r": spearman}


# ---------------------------------------------------------------------------
# Prepared-data / prediction artifact loaders (read-only)
# ---------------------------------------------------------------------------


def load_prepared_window(preparation_root: str | Path, role: str, cluster_id: str, client_id: str) -> dict[str, np.ndarray]:
    path = Path(preparation_root) / "prepared_windows" / role / f"cluster_{cluster_id}" / f"client_{client_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_test_predictions_csv(method_root: str | Path, cluster_id: str, client_id: str) -> pd.DataFrame:
    """Reads a client_ID_test_predictions.csv (timestamp, window_start_index,
    window_end_index, y_true, y_pred, abs_error, sq_error) as written by
    client.py's write_prediction_points_csv, and derives target_index the
    same way evaluation.py does (window_end_index + 1, horizon=1).
    """
    path = Path(method_root) / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
    frame = pd.read_csv(path)
    frame.insert(0, "client_id", str(client_id))
    frame.insert(0, "cluster_id", str(cluster_id))
    frame["target_index"] = frame["window_end_index"].astype(int) + 1
    return frame


def load_adaptation_predictions_csv(method_root: str | Path, cluster_id: str, client_id: str) -> pd.DataFrame:
    """Reads a client_ID_adaptation_predictions.csv (already carries
    cluster_id/client_id/target_index/timestamp natively).
    """
    path = Path(method_root) / f"cluster_{cluster_id}" / f"client_{client_id}_adaptation_predictions.csv"
    frame = pd.read_csv(path)
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    frame["client_id"] = frame["client_id"].astype(str)
    return frame


def load_preparation_cohort(preparation_root: str | Path) -> pd.DataFrame:
    """Reads <preparation_root>/manifests/selected_drift_clients.csv -- the
    location prepare_offline_data itself writes it to. Distinct from a
    SUITE root's cohort file, which prepared.suite_identity copies to
    <suite_root>/selected_drift_clients.csv directly (no manifests/ prefix,
    see drift.mitigation.diagnostics.representation.load_cohort).
    """
    frame = pd.read_csv(Path(preparation_root) / "manifests" / "selected_drift_clients.csv", dtype={"cluster_id": str, "client_id": str})
    frame["selected_for_drift"] = frame["selected_for_drift"].astype(str).str.lower().isin(["true", "1"])
    return frame


def exact_key_align(reference: pd.DataFrame, other: pd.DataFrame, *, key_columns: Sequence[str] = SAMPLE_KEY_COLUMNS) -> pd.DataFrame:
    """Fail-closed exact sample-identity join, mirroring the pattern already
    established for M2/M3 counterfactual retention matching: full key-tuple
    equality (never target_index alone), never a partial/fuzzy join.
    """
    key_columns = list(key_columns)
    ref_keys = set(map(tuple, reference[key_columns].itertuples(index=False, name=None)))
    other_keys = set(map(tuple, other[key_columns].itertuples(index=False, name=None)))
    if ref_keys != other_keys:
        missing = sorted(ref_keys - other_keys)[:5]
        extra = sorted(other_keys - ref_keys)[:5]
        raise ValueError(
            f"sample identity mismatch on {key_columns}: missing={missing} extra={extra} "
            "(fail-closed; refusing to compare mismatched samples)"
        )
    return reference.merge(other, on=key_columns, how="inner", validate="one_to_one", suffixes=("_ref", "_other"))


# ---------------------------------------------------------------------------
# Reproducibility / provenance
# ---------------------------------------------------------------------------


def matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def diagnostic_provenance(
    *,
    diagnostic_name: str,
    repo_root: str | Path,
    parameters: Mapping[str, Any],
    source_suite_root: str | Path | None = None,
    checkpoint_h: str | Path | None = None,
    preparation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Reproducibility manifest stamped into every diagnostic output's
    metadata.json: source identity (read from already-existing manifests,
    never recomputed), requested parameters, git state, timestamp, and the
    exact command line. Read-only: opens manifest files, never writes into
    the suite/checkpoint/preparation roots themselves.
    """
    import datetime as _dt

    record: dict[str, Any] = {
        "diagnostic_name": diagnostic_name,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "parameters": dict(parameters),
    }
    try:
        record["git"] = git_record(repo_root)
    except Exception as exc:  # pragma: no cover - environment without git
        record["git"] = {"error": str(exc)}

    if source_suite_root is not None:
        suite_root = Path(source_suite_root)
        record["source_suite_root"] = str(suite_root)
        cohort_hash_path = suite_root / "manifests" / "selected_drift_clients_hash.json"
        if cohort_hash_path.is_file():
            record["source_selected_cohort"] = json.loads(cohort_hash_path.read_text(encoding="utf-8"))
        resolved_config_path = suite_root / "suite_resolved_config.yaml"
        if resolved_config_path.is_file():
            record["source_resolved_config_sha256"] = file_sha256(resolved_config_path)

    if preparation_root is not None:
        prep_root = Path(preparation_root)
        record["preparation_root"] = str(prep_root)
        hashes_path = prep_root / "manifests" / "prepared_dataset_hashes.json"
        if hashes_path.is_file():
            record["preparation_hashes"] = json.loads(hashes_path.read_text(encoding="utf-8"))

    if checkpoint_h is not None:
        record["checkpoint_h"] = str(checkpoint_h)

    record["record_hash"] = canonical_hash({k: v for k, v in record.items() if k != "record_hash"})
    return record


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


def run_canonical_cli(argv: Sequence[str], *, repo_root: str | Path) -> subprocess.CompletedProcess:
    """Invoke the existing canonical CLI as a subprocess (never in-process),
    so a sweep exercises exactly the same code path a direct CLI user would,
    with zero diagnostic-package state able to leak into the run.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "drift.mitigation.prepared.cli", *argv],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    return completed


def snapshot_tree_hashes(root: str | Path) -> dict[str, str]:
    """sha256 of every file under root, keyed by relative path -- used by
    tests to prove a diagnostic run left an existing suite/checkpoint/
    preparation root byte-for-byte unchanged.
    """
    root = Path(root)
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }