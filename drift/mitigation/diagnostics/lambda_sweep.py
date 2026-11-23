"""Leakage-safe EWC lambda selection through the canonical M3 runner.

Each candidate adapts on ADAPTATION only. Candidate ranking uses the final
quarter of pre-update adaptation predictions plus positive forgetting
measured by a frozen replay of the earlier PRE_DRIFT segment. Neither the
persistent-drift FINAL_HOLDOUT nor its clean counterfactual is executed,
read, or summarized.

Derived YAMLs change only ``ewc.lambda`` and ``suite_id``; suite identity
checks hold preparation, checkpoint H, and cohort fixed. The emitted audit
fails closed if any final-holdout artifact directory exists.

    python -m drift.mitigation.diagnostics.lambda_sweep \
        --base-config <yaml> --preparation <root> --checkpoint-h <H_root> \
        --output diagnostics/ewc_lambda_selection_preholdout \
        --lambdas 0 200000 1000000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import common

M2_METHOD = "M2_oracle_client_only"
M3_METHOD = "M3_oracle_client_ewc"
R0_METHOD = "R0_clean_frozen"
IDENTITY_FIELDS_TO_MATCH = (
    "prepared_root", "prepared_dataset_manifest_hash", "drifted_deployment_hash",
    "selected_cohort_hash", "checkpoint_H", "checkpoint_H_tensor_hash",
    "checkpoint_H_optimizer_hash", "baseline_h_contract_hash",
)


def _format_lambda(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "neg")


def _diff_paths(a: Any, b: Any, prefix: str = "") -> list[str]:
    diffs: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                diffs.append(path)
            else:
                diffs.extend(_diff_paths(a[key], b[key], path))
    elif a != b:
        diffs.append(prefix)
    return diffs


def derive_lambda_config(base_config: dict[str, Any], *, lambda_value: float) -> dict[str, Any]:
    derived = copy.deepcopy(base_config)
    derived.setdefault("ewc", {})["lambda"] = lambda_value
    derived["suite_id"] = f"{base_config['suite_id']}_diaglambda{_format_lambda(lambda_value)}"
    return derived


def write_derived_configs(base_config_path: Path, lambdas: list[float], configs_dir: Path) -> dict[float, Path]:
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    configs_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[float, Path] = {}
    diff_manifest: dict[str, Any] = {"base_config": str(base_config_path), "lambdas": {}}
    for lambda_value in lambdas:
        derived = derive_lambda_config(base_config, lambda_value=lambda_value)
        diffs = _diff_paths(base_config, derived)
        # suite_id always changes (the _diaglambda suffix); ewc.lambda only
        # shows up as a diff when it actually differs from the base config's
        # own value (e.g. sweeping a lambda equal to the base's existing
        # lambda is legitimate and yields no lambda diff) -- so this is a
        # subset check, not an exact-set check, but "suite_id" must always
        # be present and nothing outside {suite_id, ewc.lambda} is allowed.
        allowed = {"suite_id", "ewc.lambda"}
        unexpected = set(diffs) - allowed
        if unexpected or "suite_id" not in diffs:
            raise ValueError(f"derived config for lambda={lambda_value} changed unexpected fields: {sorted(unexpected)}")
        path = configs_dir / f"lambda_{_format_lambda(lambda_value)}.yaml"
        path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")
        paths[lambda_value] = path
        diff_manifest["lambdas"][str(lambda_value)] = {"config_path": str(path), "changed_fields": sorted(diffs)}
    common.write_json(configs_dir / "diff_manifest.json", diff_manifest)
    return paths


def _suite_identity(suite_root: Path) -> dict[str, Any]:
    manifest_path = suite_root / "suite_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))["identity"]


def validate_identity_against_source(source_suite_root: Path, candidate_suite_root: Path) -> dict[str, Any]:
    """Compares prepared.suite_identity's own identity block between the
    source suite and a lambda-derived suite root: same preparation,
    checkpoint H, and selected cohort. config_hash is expected to differ
    (that's exactly what a lambda sweep varies) and is deliberately excluded.
    """
    source_identity = _suite_identity(source_suite_root)
    candidate_identity = _suite_identity(candidate_suite_root)
    mismatched = [
        field for field in IDENTITY_FIELDS_TO_MATCH
        if source_identity.get(field) != candidate_identity.get(field)
    ]
    if mismatched:
        raise ValueError(
            f"lambda-derived suite root {candidate_suite_root} does not share the source suite's "
            f"preparation/checkpoint-H/cohort identity on: {mismatched} (fail-closed; sweep results "
            "would not be comparable to the source suite's R0/M0)"
        )
    return {"validated": True, "fields_checked": list(IDENTITY_FIELDS_TO_MATCH)}


def _final_holdout_keys(method_root: Path, cluster_id: str, client_id: str, *, nested: str | None) -> pd.DataFrame:
    pred_dir = method_root / nested if nested else method_root
    frame = common.load_test_predictions_csv(pred_dir, cluster_id, client_id)
    return frame[["cluster_id", "client_id", "window_start_index", "window_end_index", "target_index", "timestamp"]]


def validate_final_holdout_sample_keys(source_suite_root: Path, candidate_suite_root: Path, cohort: pd.DataFrame) -> None:
    """Explicit sample-key-set comparison between the source suite's R0
    final holdout and this lambda's M3 final holdout, per correction 7 --
    not just trusting the identity-hash comparison above.
    """
    for _, row in cohort.loc[cohort.selected_for_drift].iterrows():
        cluster_id, client_id = row["cluster_id"], row["client_id"]
        source_keys = _final_holdout_keys(source_suite_root / R0_METHOD, cluster_id, client_id, nested=None)
        candidate_keys = _final_holdout_keys(candidate_suite_root / M3_METHOD, cluster_id, client_id, nested="persistent_drift_final_holdout")
        source_set = set(map(tuple, source_keys.itertuples(index=False, name=None)))
        candidate_set = set(map(tuple, candidate_keys.itertuples(index=False, name=None)))
        if source_set != candidate_set:
            raise ValueError(f"final-holdout sample keys differ for cluster={cluster_id} client={client_id} (fail-closed)")


def run_method_via_cli(*, config_path: Path, preparation: Path, checkpoint_h: Path, suite_root: Path, method: str, repo_root: Path) -> None:
    # --legacy-full-checkpoints is required here, not optional: the default
    # "auto" sparse checkpoint mode only wires its base-reference fallback
    # into the training-start checkpoint load (client.py::_maybe_load_checkpoint),
    # not into client.py::test()'s own direct-path lookup. Every method's
    # pre_drift phase runs with SKIP_TRAINING=true (training only starts in
    # the later adaptation_and_holdout phase), so under "auto" mode that
    # phase's evaluation would find no physical local checkpoint yet and
    # fail closed with "Missing client checkpoint candidates" -- for every
    # method, not just frozen ones. --legacy-full-checkpoints sidesteps this
    # by physically staging H's checkpoints upfront, exactly like every
    # pre-"auto"-mode suite in this repo already does.
    completed = common.run_canonical_cli(
        ["run-method", "--config", str(config_path), "--preparation", str(preparation),
         "--checkpoint-h", str(checkpoint_h), "--suite-root", str(suite_root), "--method", method,
         "--legacy-full-checkpoints"],
        repo_root=repo_root,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"canonical CLI run-method failed for {method} at {suite_root}:\n{completed.stdout}\n{completed.stderr}")


def run_selection_method_via_cli(*, config_path: Path, preparation: Path, checkpoint_h: Path, suite_root: Path, repo_root: Path) -> None:
    """Run canonical M3 adaptation while making final-holdout evaluation
    unreachable. Legacy checkpoint staging is still required by the initial
    PRE_DRIFT prediction-only phase.
    """
    completed = common.run_canonical_cli(
        ["run-method", "--config", str(config_path), "--preparation", str(preparation),
         "--checkpoint-h", str(checkpoint_h), "--suite-root", str(suite_root), "--method", "M3",
         "--legacy-full-checkpoints", "--selection-only"],
        repo_root=repo_root,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"canonical CLI selection-only M3 failed at {suite_root}:\n{completed.stdout}\n{completed.stderr}")


def _group_mae(run_comparison: pd.DataFrame, method: str, group: str = "affected") -> float:
    row = run_comparison.loc[(run_comparison.method == method) & (run_comparison.group == group), "mae"]
    if row.empty:
        raise ValueError(f"{method}/{group} not found in source run_comparison.csv")
    return float(row.iloc[0])


def _method_group_mae_from_predictions(method_root: Path, cohort: pd.DataFrame, *, nested: str | None) -> dict[str, float]:
    frames = []
    for _, row in cohort.iterrows():
        pred_dir = method_root / nested if nested else method_root
        path = pred_dir / f"cluster_{row['cluster_id']}" / f"client_{row['client_id']}_test_predictions.csv"
        if not path.is_file():
            continue
        frame = common.load_test_predictions_csv(pred_dir, row["cluster_id"], row["client_id"])
        frame["selected_for_drift"] = bool(row["selected_for_drift"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return {
        "affected": common.regression_metrics(combined.loc[combined.selected_for_drift, "y_true"], combined.loc[combined.selected_for_drift, "y_pred"])["mae"],
        "controls": common.regression_metrics(combined.loc[~combined.selected_for_drift, "y_true"], combined.loc[~combined.selected_for_drift, "y_pred"])["mae"],
        "all": common.regression_metrics(combined["y_true"], combined["y_pred"])["mae"],
    }


def _selection_prediction_frames(method_root: Path, cohort: pd.DataFrame, *, nested: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    selected = cohort.loc[cohort.selected_for_drift]
    for _, row in selected.iterrows():
        pred_dir = method_root / nested
        path = pred_dir / f"cluster_{row['cluster_id']}" / f"client_{row['client_id']}_test_predictions.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        frames.append(common.load_test_predictions_csv(pred_dir, row["cluster_id"], row["client_id"]))
    if missing:
        raise FileNotFoundError(f"missing selection predictions for {len(missing)} affected clients: {missing[:3]}")
    if not frames:
        raise ValueError("selection cohort has no affected-client prediction rows")
    return pd.concat(frames, ignore_index=True)


def _adaptation_selection_metrics(
    method_root: Path,
    cohort: pd.DataFrame,
    *,
    preparation: Path,
    pre_drift_windows: int,
    adaptation_windows: int,
    tail_fraction: float,
) -> dict[str, Any]:
    expected_windows = set(range(pre_drift_windows + 1, pre_drift_windows + adaptation_windows + 1))
    tail_count = max(1, math.ceil(adaptation_windows * tail_fraction))
    tail_first_window = pre_drift_windows + adaptation_windows - tail_count + 1
    frames: list[pd.DataFrame] = []
    latest_epochs: set[int] = set()
    key_columns = list(common.SAMPLE_KEY_COLUMNS)

    for _, row in cohort.loc[cohort.selected_for_drift].iterrows():
        cluster_id, client_id = row["cluster_id"], row["client_id"]
        pred_dir = method_root / "adaptation_predictions"
        frame = common.load_adaptation_predictions_csv(pred_dir, cluster_id, client_id)
        if frame.empty:
            raise ValueError(
                f"empty adaptation predictions for cluster={cluster_id} client={client_id}"
            )
        frame["adaptation_epoch"] = pd.to_numeric(
            frame["adaptation_epoch"], errors="raise"
        ).astype(int)
        latest_epoch = int(frame["adaptation_epoch"].max())
        latest_epochs.add(latest_epoch)
        latest = frame.loc[frame["adaptation_epoch"] == latest_epoch].copy()

        # The production CSV currently leaves deployment_window_number blank,
        # so recover it only through an exact full-key match against the
        # immutable prepared ADAPTATION slice. This is stronger than deriving
        # a window number from row order or target_index alone.
        prepared = common.load_prepared_window(
            preparation, "drifted_deployment", cluster_id, client_id
        )
        selection_slice = slice(
            pre_drift_windows, pre_drift_windows + adaptation_windows
        )
        expected = pd.DataFrame({
            "cluster_id": str(cluster_id),
            "client_id": str(client_id),
            "window_start_index": prepared["window_start_index"][selection_slice],
            "window_end_index": prepared["window_end_index"][selection_slice],
            "target_index": prepared["target_index"][selection_slice],
            "timestamp": prepared["target_timestamp_ns"][selection_slice].astype(str),
            "deployment_window_number": prepared["deployment_window_number"][selection_slice],
        })
        for column in ("window_start_index", "window_end_index", "target_index"):
            expected[column] = pd.to_numeric(expected[column], errors="raise").astype(int)
            latest[column] = pd.to_numeric(latest[column], errors="raise").astype(int)
        latest["timestamp"] = latest["timestamp"].astype(str)

        expected_tuples = list(
            expected[key_columns].itertuples(index=False, name=None)
        )
        observed_tuples = list(
            latest[key_columns].itertuples(index=False, name=None)
        )
        expected_keys, observed_keys = set(expected_tuples), set(observed_tuples)
        if len(expected_keys) != len(expected_tuples) or len(observed_keys) != len(observed_tuples):
            raise ValueError(
                f"duplicate adaptation sample keys for cluster={cluster_id} client={client_id}"
            )
        if observed_keys != expected_keys:
            missing = sorted(expected_keys - observed_keys)[:3]
            extra = sorted(observed_keys - expected_keys)[:3]
            raise ValueError(
                f"adaptation sample-key mismatch for cluster={cluster_id} "
                f"client={client_id}: missing={missing} extra={extra}"
            )

        window_by_key = dict(zip(
            expected_tuples,
            expected["deployment_window_number"].astype(int),
            strict=True,
        ))
        latest["deployment_window_number"] = [
            window_by_key[key] for key in observed_tuples
        ]
        observed_windows = set(latest["deployment_window_number"])
        if observed_windows != expected_windows:
            missing = sorted(expected_windows - observed_windows)[:5]
            extra = sorted(observed_windows - expected_windows)[:5]
            raise ValueError(
                f"adaptation selection window mismatch for cluster={cluster_id} "
                f"client={client_id}: missing={missing} extra={extra}"
            )
        frames.append(latest)

    if len(latest_epochs) != 1:
        raise ValueError(f"affected clients ended on inconsistent adaptation epochs: {sorted(latest_epochs)}")
    combined = pd.concat(frames, ignore_index=True)
    tail = combined.loc[combined["deployment_window_number"] >= tail_first_window]
    if tail.empty:
        raise ValueError("adaptation selection tail is empty")
    return {
        "adaptation_epoch_scored": next(iter(latest_epochs)),
        "adaptation_prequential_mae": common.regression_metrics(combined["y_true"], combined["y_pred"])["mae"],
        "adaptation_tail_prequential_mae": common.regression_metrics(tail["y_true"], tail["y_pred"])["mae"],
        "adaptation_rows_scored": int(len(combined)),
        "adaptation_tail_rows_scored": int(len(tail)),
        "adaptation_tail_fraction": tail_fraction,
        "adaptation_first_window": pre_drift_windows + 1,
        "adaptation_last_window": pre_drift_windows + adaptation_windows,
        "adaptation_tail_first_window": tail_first_window,
    }


def _pre_drift_retention_metrics(method_root: Path, cohort: pd.DataFrame) -> dict[str, float]:
    before = _selection_prediction_frames(method_root, cohort, nested="pre_drift_predictions")
    after = _selection_prediction_frames(
        method_root, cohort, nested="post_adaptation_pre_drift_predictions"
    )
    aligned = common.exact_key_align(before, after)
    target_delta = (aligned["y_true_ref"] - aligned["y_true_other"]).abs().max()
    if float(target_delta) > 1e-7:
        raise ValueError(f"pre-drift replay targets changed after adaptation: max_delta={target_delta}")
    before_mae = common.regression_metrics(aligned["y_true_ref"], aligned["y_pred_ref"])["mae"]
    after_mae = common.regression_metrics(aligned["y_true_other"], aligned["y_pred_other"])["mae"]
    return {
        "initial_pre_drift_mae": before_mae,
        "post_adaptation_pre_drift_mae": after_mae,
        "pre_drift_forgetting_delta": after_mae - before_mae,
    }


def validate_selection_only_artifacts(lambda_root: Path) -> dict[str, Any]:
    method_root = lambda_root / M3_METHOD
    manifest = json.loads((method_root / "run_manifest.json").read_text(encoding="utf-8"))
    policy = manifest.get("selection_data_policy") or {}
    forbidden = [
        method_root / "persistent_drift_final_holdout",
        method_root / "clean_final_holdout",
        method_root / "counterfactual_clean_holdout",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    checks = {
        "selection_only_manifest": manifest.get("selection_only") is True,
        "final_holdout_evaluated_false": policy.get("final_holdout_evaluated") is False,
        "counterfactual_final_holdout_evaluated_false": (
            policy.get("counterfactual_final_holdout_evaluated") is False
        ),
        "post_adaptation_test_disabled": policy.get("run_test_after_adaptation") == "false",
        "forbidden_final_holdout_directories_absent": not present,
        "adaptation_predictions_present": (method_root / "adaptation_predictions").is_dir(),
        "post_adaptation_pre_drift_predictions_present": (
            method_root / "post_adaptation_pre_drift_predictions"
        ).is_dir(),
    }
    if not all(checks.values()):
        failed = [name for name, accepted in checks.items() if not accepted]
        raise ValueError(
            f"selection-only artifact audit failed for {lambda_root}: {failed}; "
            f"forbidden_present={present}"
        )
    return {"accepted": True, "checks": checks, "forbidden_present": present}


def _reuse_completed_selection_candidate(lambda_root: Path, lambda_value: float) -> bool:
    """Return true only for a complete, matching selection-only candidate.

    An existing but incomplete/mismatched method directory fails closed so a
    resume can never silently mix artifacts from different runs.
    """
    method_root = lambda_root / M3_METHOD
    if not method_root.exists():
        return False
    manifest_path = method_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"cannot resume existing candidate without run manifest: {method_root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_lambda = (manifest.get("ewc_config") or {}).get("lambda")
    if (
        manifest.get("status") != "complete"
        or manifest.get("selection_only") is not True
        or recorded_lambda is None
        or float(recorded_lambda) != float(lambda_value)
    ):
        raise ValueError(
            f"existing candidate is not a complete matching selection-only run: "
            f"{method_root}"
        )
    validate_selection_only_artifacts(lambda_root)
    return True


def run_lambda_sweep(
    *,
    base_config: str | Path,
    preparation: str | Path,
    checkpoint_h: str | Path,
    output: str | Path,
    lambdas: list[float] | None = None,
    source_suite_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    tail_fraction: float = 0.25,
) -> dict[str, Any]:
    """Select EWC lambda without executing or reading FINAL_HOLDOUT.

    The score is fixed before looking at candidate results:
    final-quarter adaptation prequential MAE + positive PRE_DRIFT forgetting.
    Both terms use raw target units and lower is better. PRE_DRIFT is replayed
    prediction-only after adaptation; adaptation predictions are captured
    before each corresponding update.
    """
    base_config = Path(base_config)
    preparation = Path(preparation)
    checkpoint_h = Path(checkpoint_h)
    output = Path(output)
    source_suite_root = Path(source_suite_root) if source_suite_root is not None else None
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    lambdas = lambdas if lambdas is not None else [0.0, 200000.0, 1000000.0]
    if not lambdas or len(set(lambdas)) != len(lambdas):
        raise ValueError("lambdas must be a non-empty unique list")
    if any(value < 0 for value in lambdas):
        raise ValueError("lambdas must be nonnegative")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")

    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    segments = base["segments"]
    pre_drift_windows = int(segments["pre_drift"])
    adaptation_windows = int(segments["adaptation"])
    final_holdout_first_window = pre_drift_windows + adaptation_windows + 1
    cohort = common.load_preparation_cohort(preparation)

    suite_id = base_config.stem
    suite_output = output / suite_id
    config_paths = write_derived_configs(base_config, lambdas, suite_output / "configs")
    identity_reference = (
        source_suite_root
        if source_suite_root is not None and (source_suite_root / "suite_manifest.json").is_file()
        else None
    )

    summary_rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for lambda_value in lambdas:
        lambda_root = suite_output / f"lambda_{_format_lambda(lambda_value)}"
        reused_complete_candidate = _reuse_completed_selection_candidate(
            lambda_root, lambda_value
        )
        if not reused_complete_candidate:
            run_selection_method_via_cli(
                config_path=config_paths[lambda_value],
                preparation=preparation,
                checkpoint_h=checkpoint_h,
                suite_root=lambda_root,
                repo_root=repo_root,
            )
        if identity_reference is None:
            identity_reference = lambda_root
        else:
            validate_identity_against_source(identity_reference, lambda_root)

        audit = validate_selection_only_artifacts(lambda_root)
        method_root = lambda_root / M3_METHOD
        adaptation = _adaptation_selection_metrics(
            method_root,
            cohort,
            preparation=preparation,
            pre_drift_windows=pre_drift_windows,
            adaptation_windows=adaptation_windows,
            tail_fraction=tail_fraction,
        )
        if adaptation["adaptation_last_window"] >= final_holdout_first_window:
            raise ValueError("selection data reaches FINAL_HOLDOUT")
        retention = _pre_drift_retention_metrics(method_root, cohort)
        positive_forgetting = max(0.0, retention["pre_drift_forgetting_delta"])
        selection_score = adaptation["adaptation_tail_prequential_mae"] + positive_forgetting
        summary_rows.append({
            "lambda": lambda_value,
            "suite_id": suite_id,
            **adaptation,
            **retention,
            "positive_pre_drift_forgetting_penalty": positive_forgetting,
            "selection_score": selection_score,
        })
        audits[str(lambda_value)] = {
            **audit,
            "reused_complete_candidate": reused_complete_candidate,
            "adaptation_last_window_scored": adaptation["adaptation_last_window"],
            "final_holdout_first_window": final_holdout_first_window,
        }

    summary = pd.DataFrame(summary_rows)
    ordered = summary.sort_values(
        ["selection_score", "lambda"], ascending=[True, True], kind="stable"
    )
    summary["selection_rank"] = 0
    for rank, index in enumerate(ordered.index, start=1):
        summary.loc[index, "selection_rank"] = rank
    selected_index = ordered.index[0]
    summary["selected_for_main_experiment"] = False
    summary.loc[selected_index, "selected_for_main_experiment"] = True
    selected_lambda = float(summary.loc[selected_index, "lambda"])

    comparison = suite_output / "comparison"
    common.write_csv(comparison / "lambda_selection_summary.csv", summary)
    decision = {
        "selected_lambda": selected_lambda,
        "selection_rule": (
            "minimize adaptation_tail_prequential_mae + "
            "max(0, post_adaptation_pre_drift_mae - initial_pre_drift_mae); "
            "break exact ties toward smaller lambda"
        ),
        "selection_data": ["ADAPTATION pre-update predictions", "PRE_DRIFT frozen replay"],
        "adaptation_tail_fraction": tail_fraction,
        "final_holdout_used": False,
        "warning": (
            "Do not report a final-holdout result from any historical leaked sweep as an "
            "unbiased test result; evaluate the selected lambda once on a genuinely unseen "
            "main-experiment holdout."
        ),
    }
    common.write_json(comparison / "selection_decision.json", decision)
    common.write_json(
        comparison / "selection_data_audit.json",
        {
            "accepted": all(item["accepted"] for item in audits.values()),
            "final_holdout_first_window": final_holdout_first_window,
            "final_holdout_used": False,
            "per_lambda": audits,
        },
    )
    metadata = common.diagnostic_provenance(
        diagnostic_name="lambda_selection_preholdout",
        repo_root=repo_root,
        parameters={
            "lambdas": lambdas,
            "tail_fraction": tail_fraction,
            "selection_rule": decision["selection_rule"],
            "final_holdout_used": False,
        },
        source_suite_root=source_suite_root,
        checkpoint_h=checkpoint_h,
        preparation_root=preparation,
    )
    common.write_json(comparison / "metadata.json", metadata)
    return {
        "output": str(suite_output),
        "summary_rows": len(summary),
        "selected_lambda": selected_lambda,
        "final_holdout_used": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe EWC lambda selection on ADAPTATION/PRE_DRIFT only."
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--checkpoint-h", type=Path, required=True)
    parser.add_argument(
        "--source-suite-root", type=Path,
        help="optional suite used for identity fields only; no predictions or metrics are read",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("diagnostics/ewc_lambda_selection_preholdout")
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.0, 200000.0, 1000000.0]
    )
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_lambda_sweep(
        base_config=args.base_config,
        preparation=args.preparation,
        checkpoint_h=args.checkpoint_h,
        source_suite_root=args.source_suite_root,
        output=args.output,
        lambdas=args.lambdas,
        tail_fraction=args.tail_fraction,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())