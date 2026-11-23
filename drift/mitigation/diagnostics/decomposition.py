"""Part L (correction 10, high priority): M1 client/server hybrid
counterfactual decomposition.

On the identical drifted final-holdout X/y, evaluates all four
(client, server) combinations of checkpoint H and M1's own adapted
checkpoints:

    1. H_client  + H_server   (== R0/M0's own frozen baseline behavior)
    2. M1_client + H_server   (client adaptation only, counterfactual)
    3. H_client  + M1_server  (server adaptation only, counterfactual)
    4. M1_client + M1_server  (== M1's actual production behavior)

This is explicitly a post-hoc parameter-combination diagnostic, not a valid
training method: combinations 2 and 3 never existed as trained states and
must never be reported as mitigation results. Inference only -- every
checkpoint is loaded read-only and no model is ever updated.

    python -m drift.mitigation.diagnostics.decomposition \
        --checkpoint-h <checkpoint_h_root> --suite-root <suite_root> \
        --preparation <preparation_root> --output diagnostics/m1_decomposition
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import common
from .representation import (
    _final_holdout_predictions_dir,
    _h_client_checkpoint,
    _h_server_checkpoint,
    _method_client_checkpoint,
    _method_server_checkpoint,
    _select_npz_rows,
    load_cohort,
)

M1_METHOD = "M1_continued_splitfed"

COMBINATIONS = (
    ("H_client_H_server", "h", "h"),
    ("M1_client_H_server", "m1", "h"),
    ("H_client_M1_server", "h", "m1"),
    ("M1_client_M1_server", "m1", "m1"),
)


def run_decomposition_diagnostic(
    *, checkpoint_h: str | Path, suite_root: str | Path, preparation_root: str | Path, output: str | Path,
    arch: common.ModelArchitectureSpec | None = None,
) -> dict[str, Any]:
    checkpoint_h, suite_root, preparation_root, output = Path(checkpoint_h), Path(suite_root), Path(preparation_root), Path(output)
    arch = arch or common.ModelArchitectureSpec()
    cohort = load_cohort(suite_root)
    selected = cohort.loc[cohort.selected_for_drift]

    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        cluster_id, client_id = row["cluster_id"], row["client_id"]
        m1_client_path = _method_client_checkpoint(suite_root, M1_METHOD, cluster_id, client_id)
        m1_server_path = _method_server_checkpoint(suite_root, M1_METHOD, cluster_id)
        if not (m1_client_path.is_file() and m1_server_path.is_file()):
            continue

        pred_dir = _final_holdout_predictions_dir(suite_root, M1_METHOD, oracle_enabled=False)
        pred_path = pred_dir / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
        if not pred_path.is_file():
            continue
        predictions = common.load_test_predictions_csv(pred_dir, cluster_id, client_id)
        npz = common.load_prepared_window(preparation_root, "drifted_deployment", cluster_id, client_id)
        X = torch.as_tensor(_select_npz_rows(npz, predictions), dtype=torch.float32)
        y = torch.as_tensor(predictions["y_true"].to_numpy(dtype=np.float32))

        clients = {
            "h": common.load_client_model(_h_client_checkpoint(checkpoint_h, cluster_id, client_id), arch),
            "m1": common.load_client_model(m1_client_path, arch),
        }
        servers = {
            "h": common.load_server_model(_h_server_checkpoint(checkpoint_h, cluster_id), arch),
            "m1": common.load_server_model(m1_server_path, arch),
        }

        for combo_name, client_key, server_key in COMBINATIONS:
            representation = common.cut_layer_representation(clients[client_key], X)
            result = common.server_predict(servers[server_key], representation, y)
            preds = result["predictions"].reshape(-1)
            metrics = common.regression_metrics(y.numpy(), preds.numpy())
            rows.append({
                "cluster_id": cluster_id, "client_id": client_id, "combination": combo_name,
                "client_source": client_key, "server_source": server_key,
                "diagnostic_kind": "post_hoc_parameter_combination_not_a_trained_method",
                **metrics,
            })

    result_frame = pd.DataFrame(rows)
    common.write_csv(output / "m1_client_server_decomposition.csv", result_frame)

    metadata = common.diagnostic_provenance(
        diagnostic_name="m1_client_server_decomposition", repo_root=Path(__file__).resolve().parents[3],
        parameters={}, source_suite_root=suite_root, checkpoint_h=checkpoint_h, preparation_root=preparation_root,
    )
    metadata["warning"] = (
        "combinations 'M1_client_H_server' and 'H_client_M1_server' are post-hoc, "
        "never-trained parameter combinations -- inference-only counterfactuals, "
        "not valid mitigation results"
    )
    common.write_json(output / "metadata.json", metadata)
    return {"output": str(output), "rows": len(result_frame)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1 client/server hybrid counterfactual decomposition (inference-only, read-only).")
    parser.add_argument("--checkpoint-h", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/m1_decomposition"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_decomposition_diagnostic(
        checkpoint_h=args.checkpoint_h, suite_root=args.suite_root, preparation_root=args.preparation, output=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())