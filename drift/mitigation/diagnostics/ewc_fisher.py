"""Part D1/F: read-only inspection of the saved EWC reference (Fisher
diagonal + theta_star), as already written by
drift.mitigation.prepared.ewc.build_or_load_ewc_reference.

Never recomputes Fisher (that would require importing/duplicating training
machinery beyond a simple read); a separate diagnostic-only recomputation
for comparison purposes lives in fisher_compare.py instead. This module
only opens the existing ``<suite_root>/ewc_reference/`` artifacts via
``torch.load``/``json.loads`` and reports statistics -- it never writes
into that directory.

    python -m drift.mitigation.diagnostics.ewc_fisher \
        --suite-root <suite_root> --output diagnostics/ewc_fisher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from . import common


def _tensor_stats(values: np.ndarray, *, near_zero_threshold: float) -> dict[str, float]:
    if values.size == 0:
        return {
            "n_params": 0, "fisher_mean": float("nan"), "fisher_median": float("nan"), "fisher_std": float("nan"),
            "fisher_min": float("nan"), "fisher_max": float("nan"), "fisher_p50": float("nan"),
            "fisher_p90": float("nan"), "fisher_p95": float("nan"), "fisher_p99": float("nan"),
            "fraction_exactly_zero": float("nan"), "fraction_near_zero": float("nan"),
        }
    return {
        "n_params": int(values.size),
        "fisher_mean": float(values.mean()),
        "fisher_median": float(np.median(values)),
        "fisher_std": float(values.std()),
        "fisher_min": float(values.min()),
        "fisher_max": float(values.max()),
        "fisher_p50": float(np.percentile(values, 50)),
        "fisher_p90": float(np.percentile(values, 90)),
        "fisher_p95": float(np.percentile(values, 95)),
        "fisher_p99": float(np.percentile(values, 99)),
        "fraction_exactly_zero": float(np.mean(values == 0.0)),
        "fraction_near_zero": float(np.mean(np.abs(values) < near_zero_threshold)),
    }


def _layer_name(tensor_name: str) -> str:
    parts = tensor_name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else tensor_name


def fisher_granularity_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Part F: state exactly how the saved Fisher was computed, without
    changing it. Labeled "batch_mean" (option B in the task spec) whenever
    the manifest's own fisher_type says so -- this repository's current
    implementation always does, per drift/mitigation/prepared/ewc.py's own
    ``"empirical_diagonal_squared_batch_gradient"`` fisher_type.
    """
    fisher_type = str(manifest.get("fisher_type", ""))
    if "batch" in fisher_type:
        granularity = "batch_mean"
        note = (
            "batch gradients are averaged before squaring; sample-gradient "
            "cancellation may reduce the empirical Fisher scale"
        )
    elif "sample" in fisher_type:
        granularity = "per_sample"
        note = None
    else:
        granularity = "unknown"
        note = f"unrecognized fisher_type={fisher_type!r}; cannot classify granularity"
    result: dict[str, Any] = {
        "fisher_gradient_granularity": granularity,
        "fisher_type_raw": fisher_type,
        "fisher_batch_size": manifest.get("fisher_batch_size"),
        "fisher_max_batches": manifest.get("fisher_max_batches"),
        "loss_definition": manifest.get("loss_definition"),
    }
    if note:
        result["diagnostic_observation"] = note
    return result


def run_fisher_diagnostic(*, suite_root: str | Path, output: str | Path, near_zero_threshold: float = 1e-8) -> dict[str, Any]:
    suite_root, output = Path(suite_root), Path(output)
    manifest_path = suite_root / "ewc_reference" / "ewc_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    per_tensor_rows: list[dict[str, Any]] = []
    per_client_rows: list[dict[str, Any]] = []
    layer_values: dict[str, list[np.ndarray]] = {}
    population_values: list[np.ndarray] = []
    per_client_loss_definition: str | None = None

    for entry in manifest["clients"]:
        cluster_id, client_id = str(entry["cluster_id"]), str(entry["client_id"])
        ref_path = suite_root / "ewc_reference" / entry["relative_path"]
        payload = torch.load(ref_path, map_location="cpu", weights_only=False)
        per_client_loss_definition = payload.get("loss_definition", per_client_loss_definition)
        client_values: list[np.ndarray] = []
        for name, tensor in payload["fisher_diagonal"].items():
            values = tensor.detach().flatten().numpy().astype(float)
            stats = _tensor_stats(values, near_zero_threshold=near_zero_threshold)
            per_tensor_rows.append({
                "cluster_id": cluster_id, "client_id": client_id, "tensor_name": name,
                "layer": _layer_name(name), **stats,
            })
            client_values.append(values)
            layer_values.setdefault(_layer_name(name), []).append(values)
        combined = np.concatenate(client_values) if client_values else np.asarray([])
        population_values.append(combined)
        per_client_rows.append({
            "cluster_id": cluster_id, "client_id": client_id,
            "fisher_sample_count": payload.get("fisher_sample_count"),
            "fisher_batch_count": payload.get("fisher_batch_count"),
            **_tensor_stats(combined, near_zero_threshold=near_zero_threshold),
        })

    summary_rows: list[dict[str, Any]] = []
    for layer, arrays in layer_values.items():
        summary_rows.append({"scope": "layer", "name": layer, **_tensor_stats(np.concatenate(arrays), near_zero_threshold=near_zero_threshold)})
    all_population = np.concatenate(population_values) if population_values else np.asarray([])
    summary_rows.append({"scope": "population", "name": "all_clients", **_tensor_stats(all_population, near_zero_threshold=near_zero_threshold)})

    per_tensor = pd.DataFrame(per_tensor_rows)
    per_client = pd.DataFrame(per_client_rows)
    summary = pd.DataFrame(summary_rows)

    common.write_csv(output / "ewc_fisher_per_tensor.csv", per_tensor)
    common.write_csv(output / "ewc_fisher_per_client.csv", per_client)
    common.write_csv(output / "ewc_fisher_summary.csv", summary)

    granularity = fisher_granularity_report(manifest)
    metadata = common.diagnostic_provenance(
        diagnostic_name="ewc_fisher", repo_root=Path(__file__).resolve().parents[3],
        parameters={"near_zero_threshold": near_zero_threshold}, source_suite_root=suite_root,
    )
    metadata.update(granularity)
    common.write_json(output / "metadata.json", metadata)

    return {"output": str(output), "per_tensor_rows": len(per_tensor), "per_client_rows": len(per_client), **granularity}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Fisher/EWC-reference diagnostic (never recomputes Fisher).")
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/ewc_fisher"))
    parser.add_argument("--near-zero-threshold", type=float, default=1e-8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_fisher_diagnostic(suite_root=args.suite_root, output=args.output, near_zero_threshold=args.near_zero_threshold)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())