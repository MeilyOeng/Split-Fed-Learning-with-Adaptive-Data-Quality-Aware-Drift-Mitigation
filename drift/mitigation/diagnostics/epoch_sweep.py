"""Part J (+ correction 8): adaptation-epoch sweep via the existing
canonical CLI, with method-separated output directories.

Sweeps M1's ``adaptation.adaptation_epochs``, M2's
``adaptation.client_only_adaptation_epochs``, and M3's
``adaptation.ewc_adaptation_epochs`` independently -- each method gets its
own derived config per epoch value and its own isolated output directory
(``epoch_sweep/<suite_id>/{M1,M2,M3}/epoch_<N>/``), never mixed into one
shared ``epoch_N/`` directory. Never edits the canonical YAML (deep copy +
single-field override, diff-verified, exactly like lambda_sweep.py). Never
reruns R0/M0: every epoch/method combination's recovery and retention are
computed against the SOURCE suite's own already-completed R0/M0
predictions and comparison/run_comparison.csv, after the same identity
validation lambda_sweep.py performs.

``--matched-client-only-epochs`` (default on) sweeps M2 and M3 over the
SAME epoch list, so each swept epoch value E produces a directly comparable
M2(E) vs M3(E) pair. Pass ``--no-matched-client-only-epochs`` together with
separate ``--m2-epochs``/``--m3-epochs`` to sweep them independently. M1
always sweeps its own list (``--m1-epochs``, defaults to ``--epochs``).

Representation-movement and adaptation-curve diagnostics are NOT computed
inline here -- each epoch/method output directory is itself a fully valid,
completed suite-shaped method root, so representation.py (etc.) can be run
directly against e.g. ``epoch_sweep/<suite_id>/M2/epoch_3`` without this
module needing to import or duplicate that logic.

    python -m drift.mitigation.diagnostics.epoch_sweep \
        --base-config <yaml> --preparation <root> --checkpoint-h <H_root> \
        --source-suite-root <already-completed suite with R0/M0> \
        --output diagnostics/epoch_sweep --epochs 1 2 3
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import common
from .lambda_sweep import (
    R0_METHOD, _diff_paths, _group_mae, _method_group_mae_from_predictions,
    run_method_via_cli, validate_identity_against_source, validate_final_holdout_sample_keys,
)

METHOD_EPOCH_FIELD = {
    "M1": "adaptation_epochs",
    "M2": "client_only_adaptation_epochs",
    "M3": "ewc_adaptation_epochs",
}
METHOD_DIR_NAME = {"M1": "M1_continued_splitfed", "M2": "M2_oracle_client_only", "M3": "M3_oracle_client_ewc"}


def derive_epoch_config(base_config: dict[str, Any], *, method: str, epoch_value: int) -> dict[str, Any]:
    derived = copy.deepcopy(base_config)
    field = METHOD_EPOCH_FIELD[method]
    derived.setdefault("adaptation", {})[field] = epoch_value
    derived["suite_id"] = f"{base_config['suite_id']}_diagepoch{method.lower()}{epoch_value}"
    return derived


def write_derived_epoch_config(base_config_path: Path, *, method: str, epoch_value: int, configs_dir: Path) -> Path:
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    derived = derive_epoch_config(base_config, method=method, epoch_value=epoch_value)
    diffs = _diff_paths(base_config, derived)
    allowed = {"suite_id", f"adaptation.{METHOD_EPOCH_FIELD[method]}"}
    unexpected = set(diffs) - allowed
    if unexpected or "suite_id" not in diffs:
        raise ValueError(f"derived config for method={method} epoch={epoch_value} changed unexpected fields: {sorted(unexpected)}")
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / f"{method.lower()}_epoch_{epoch_value}.yaml"
    path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")
    return path


def _client_improvement_fraction(method_root: Path, cohort: pd.DataFrame, source_frozen_root: Path) -> float:
    """Fraction of affected clients whose per-client MAE improved (is lower)
    relative to the source suite's M0 (frozen baseline) per-client MAE.
    """
    improved = 0
    total = 0
    for _, row in cohort.loc[cohort.selected_for_drift].iterrows():
        cluster_id, client_id = row["cluster_id"], row["client_id"]
        method_pred_dir = method_root / "persistent_drift_final_holdout"
        method_path = method_pred_dir / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
        frozen_path = source_frozen_root / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
        if not (method_path.is_file() and frozen_path.is_file()):
            continue
        method_frame = common.load_test_predictions_csv(method_pred_dir, cluster_id, client_id)
        frozen_frame = common.load_test_predictions_csv(source_frozen_root, cluster_id, client_id)
        method_mae = common.regression_metrics(method_frame["y_true"], method_frame["y_pred"])["mae"]
        frozen_mae = common.regression_metrics(frozen_frame["y_true"], frozen_frame["y_pred"])["mae"]
        total += 1
        if method_mae < frozen_mae - 1e-9:
            improved += 1
    return improved / total if total else float("nan")


def run_epoch_sweep_for_method(
    *, method: str, epochs: list[int], base_config: Path, preparation: Path, checkpoint_h: Path,
    source_suite_root: Path, output: Path, cohort: pd.DataFrame, source_run_comparison: pd.DataFrame,
    repo_root: Path,
) -> pd.DataFrame:
    method_output = output / METHOD_DIR_NAME[method].split("_", 1)[0]
    configs_dir = method_output / "configs"
    e_clean = _group_mae(source_run_comparison, R0_METHOD)
    e_frozen = _group_mae(source_run_comparison, "M0_drift_frozen")
    m0_pred_root = source_suite_root / "M0_drift_frozen"

    rows: list[dict[str, Any]] = []
    for epoch_value in epochs:
        config_path = write_derived_epoch_config(base_config, method=method, epoch_value=epoch_value, configs_dir=configs_dir)
        epoch_root = method_output / f"epoch_{epoch_value}"
        run_method_via_cli(config_path=config_path, preparation=preparation, checkpoint_h=checkpoint_h, suite_root=epoch_root, method=method, repo_root=repo_root)
        validate_identity_against_source(source_suite_root, epoch_root)

        method_dir = epoch_root / METHOD_DIR_NAME[method]
        group_mae = _method_group_mae_from_predictions(method_dir, cohort, nested="persistent_drift_final_holdout")
        e_method = group_mae["affected"]
        drift_damage = e_frozen - e_clean
        recovered_error = e_frozen - e_method
        recovery_rate = None if drift_damage <= 1e-12 else recovered_error / drift_damage
        improvement_fraction = _client_improvement_fraction(method_dir, cohort, m0_pred_root)

        retention_delta = None
        if method in ("M2", "M3"):
            validate_final_holdout_sample_keys(source_suite_root, epoch_root, cohort) if method == "M3" else None
            try:
                retention_group_mae = _method_group_mae_from_predictions(method_dir, cohort.loc[cohort.selected_for_drift], nested="counterfactual_clean_holdout")
                r0_clean_group_mae = _method_group_mae_from_predictions(source_suite_root / R0_METHOD, cohort.loc[cohort.selected_for_drift], nested=None)
                retention_delta = retention_group_mae["affected"] - r0_clean_group_mae["affected"]
            except (FileNotFoundError, ValueError):
                pass

        rows.append({
            "method": method, "epoch": epoch_value, "E_clean_R0": e_clean, "E_frozen_M0": e_frozen,
            "E_method_affected": e_method, "recovery_rate": recovery_rate,
            "client_improvement_fraction": improvement_fraction, "clean_retention_delta": retention_delta,
        })
    return pd.DataFrame(rows)


def run_epoch_sweep(
    *, base_config: str | Path, preparation: str | Path, checkpoint_h: str | Path, source_suite_root: str | Path,
    output: str | Path, epochs: list[int] | None = None, m1_epochs: list[int] | None = None,
    matched_client_only_epochs: bool = True, m2_epochs: list[int] | None = None, m3_epochs: list[int] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    base_config, preparation, checkpoint_h, source_suite_root, output = (
        Path(base_config), Path(preparation), Path(checkpoint_h), Path(source_suite_root), Path(output)
    )
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    epochs = epochs if epochs is not None else [1, 2, 3]
    m1_epochs = m1_epochs if m1_epochs is not None else epochs
    if matched_client_only_epochs:
        m2_epochs, m3_epochs = epochs, epochs
    else:
        m2_epochs = m2_epochs if m2_epochs is not None else epochs
        m3_epochs = m3_epochs if m3_epochs is not None else epochs

    from .representation import load_cohort
    cohort = load_cohort(source_suite_root)
    source_run_comparison = pd.read_csv(source_suite_root / "comparison" / "run_comparison.csv")

    suite_id = base_config.stem
    suite_output = output / suite_id
    frames = {
        "M1": run_epoch_sweep_for_method(method="M1", epochs=m1_epochs, base_config=base_config, preparation=preparation, checkpoint_h=checkpoint_h, source_suite_root=source_suite_root, output=suite_output, cohort=cohort, source_run_comparison=source_run_comparison, repo_root=repo_root),
        "M2": run_epoch_sweep_for_method(method="M2", epochs=m2_epochs, base_config=base_config, preparation=preparation, checkpoint_h=checkpoint_h, source_suite_root=source_suite_root, output=suite_output, cohort=cohort, source_run_comparison=source_run_comparison, repo_root=repo_root),
        "M3": run_epoch_sweep_for_method(method="M3", epochs=m3_epochs, base_config=base_config, preparation=preparation, checkpoint_h=checkpoint_h, source_suite_root=source_suite_root, output=suite_output, cohort=cohort, source_run_comparison=source_run_comparison, repo_root=repo_root),
    }
    combined = pd.concat(frames.values(), ignore_index=True)

    comparison_rows: list[dict[str, Any]] = []
    if matched_client_only_epochs:
        for epoch_value in epochs:
            m2_row = frames["M2"].loc[frames["M2"].epoch == epoch_value]
            m3_row = frames["M3"].loc[frames["M3"].epoch == epoch_value]
            m1_row = frames["M1"].loc[frames["M1"].epoch == epoch_value]
            comparison_rows.append({
                "epoch": epoch_value,
                "M1_minus_M2_gap": (float(m1_row["E_method_affected"].iloc[0]) - float(m2_row["E_method_affected"].iloc[0])) if not m1_row.empty and not m2_row.empty else None,
                "M2_minus_M3_gap": (float(m2_row["E_method_affected"].iloc[0]) - float(m3_row["E_method_affected"].iloc[0])) if not m2_row.empty and not m3_row.empty else None,
            })
    gap_frame = pd.DataFrame(comparison_rows)

    comparison_dir = suite_output / "comparison"
    common.write_csv(comparison_dir / "epoch_sweep_summary.csv", combined)
    common.write_csv(comparison_dir / "epoch_sweep_method_gaps.csv", gap_frame)
    metadata = common.diagnostic_provenance(
        diagnostic_name="epoch_sweep", repo_root=repo_root,
        parameters={"epochs": epochs, "m1_epochs": m1_epochs, "m2_epochs": m2_epochs, "m3_epochs": m3_epochs, "matched_client_only_epochs": matched_client_only_epochs},
        source_suite_root=source_suite_root, checkpoint_h=checkpoint_h, preparation_root=preparation,
    )
    common.write_json(comparison_dir / "metadata.json", metadata)
    return {"output": str(suite_output), "summary_rows": len(combined)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptation-epoch sweep via the canonical CLI, method-separated output, reusing source R0/M0.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--checkpoint-h", type=Path, required=True)
    parser.add_argument("--source-suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/epoch_sweep"))
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--m1-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--m2-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--m3-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--no-matched-client-only-epochs", dest="matched", action="store_false", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_epoch_sweep(
        base_config=args.base_config, preparation=args.preparation, checkpoint_h=args.checkpoint_h,
        source_suite_root=args.source_suite_root, output=args.output, epochs=args.epochs,
        m1_epochs=args.m1_epochs, matched_client_only_epochs=args.matched, m2_epochs=args.m2_epochs, m3_epochs=args.m3_epochs,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())