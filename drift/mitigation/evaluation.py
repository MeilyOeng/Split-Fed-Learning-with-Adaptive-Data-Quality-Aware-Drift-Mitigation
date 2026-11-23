"""Final-holdout comparison and recovery evaluation for completed production methods."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .core import Method, canonical_hash, registered_method_names, resolve_method_policy, write_json


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    true, pred = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    error = pred - true; mse = float(np.mean(error ** 2)); denom = np.abs(true) + np.abs(pred)
    smape = float(np.mean(np.divide(2 * np.abs(error), denom, out=np.zeros_like(error), where=denom > 0)) * 100)
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {"n": len(frame), "mae": float(np.mean(np.abs(error))), "mse": mse, "rmse": float(np.sqrt(mse)),
            "smape": smape, "r2": None if ss_tot <= 0 else float(1 - np.sum(error ** 2) / ss_tot)}

def _load_cohort(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "selected_drift_clients.csv", dtype={"cluster_id": str, "client_id": str})
    frame["selected_for_drift"] = frame.selected_for_drift.astype(str).str.lower().isin(["true", "1"])
    return frame

def _suite_methods(root: Path) -> list[str]:
    """Methods this suite actually ran, not every registered method.

    Prefers the explicit list prepared.suite_identity stamps into
    suite_manifest.json. Falls back to scanning for registered-method
    directories that exist under root, so legacy suite roots (which predate
    that field and always ran all four built-in methods) still work.
    """
    manifest_path = root / "suite_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = manifest.get("selected_methods")
        if selected:
            return [str(name) for name in selected]
    return [name for name in registered_method_names() if (root / name).is_dir()]

def _load_method_predictions(root: Path, method: str) -> pd.DataFrame:
    method_root = root / method
    policy = resolve_method_policy(method)
    source_root = method_root / "persistent_drift_final_holdout" if policy.oracle_enabled and (method_root / "persistent_drift_final_holdout").exists() else method_root
    rows = []
    for path in sorted(source_root.glob("cluster_*/client_*_test_predictions.csv")):
        cluster_id = path.parent.name.removeprefix("cluster_")
        client_id = path.name.removeprefix("client_").removesuffix("_test_predictions.csv")
        frame = pd.read_csv(path)
        if len(frame) != 524: raise ValueError(f"{method} {cluster_id}/{client_id} has {len(frame)} holdout predictions, expected 524")
        frame.insert(0, "client_id", client_id); frame.insert(0, "cluster_id", cluster_id); rows.append(frame)
    if len(rows) != 84: raise ValueError(f"{method} must have 84 client prediction files, found {len(rows)}")
    result = pd.concat(rows, ignore_index=True)
    result["target_index"] = result.window_end_index.astype(int) + 1
    keys = list(zip(result.cluster_id, result.client_id, result.window_start_index, result.window_end_index, result.target_index))
    if len(keys) != len(set(keys)): raise ValueError(f"duplicate prediction keys in {method}")
    result["method"] = method
    return result

def _pct_change(delta: float | None, reference: float | None) -> float | None:
    if delta is None or reference is None or reference == 0:
        return None
    return delta / reference


def _continued_training_control(clients: pd.DataFrame, groups: pd.DataFrame, comparison: Path, methods: list[str]) -> None:
    """R0-vs-R1 (clean continued-training gain) and M0-vs-M1 (drift
    continued-training gain), paired on the exact same clients.

    R1 is a diagnostic control, not a recovery candidate (see
    method_role="diagnostic_control"): this file is produced only when the
    suite actually ran R1, and never changes recovery_summary.csv's
    existing formula or the R0/M0/M1/M2 comparison outputs.
    """
    if Method.R1.value not in methods:
        return

    def group_value(method: str, group: str, metric: str) -> float | None:
        if method not in methods:
            return None
        row = groups.loc[(groups.method == method) & (groups.group == group), metric]
        return None if row.empty else float(row.iloc[0])

    r0 = clients.loc[clients.method == Method.R0.value, ["cluster_id", "client_id", "selected_for_drift", "mae"]].rename(columns={"mae": "mae_r0"})
    r1 = clients.loc[clients.method == Method.R1.value, ["cluster_id", "client_id", "mae"]].rename(columns={"mae": "mae_r1"})
    paired = r0.merge(r1, on=["cluster_id", "client_id"], validate="one_to_one")
    # Negative delta = R1 error lower than R0 = improved (lower-error convention).
    paired["delta"] = paired.mae_r1 - paired.mae_r0

    rows = []
    for metric in ("mae", "rmse", "smape"):
        for group_name, mask in (
            ("affected", paired.selected_for_drift),
            ("controls", ~paired.selected_for_drift),
            ("all", pd.Series(True, index=paired.index)),
        ):
            e_r0 = group_value(Method.R0.value, group_name, metric)
            e_r1 = group_value(Method.R1.value, group_name, metric)
            e_m0 = group_value(Method.M0.value, group_name, metric)
            e_m1 = group_value(Method.M1.value, group_name, metric)
            clean_gain = None if e_r0 is None or e_r1 is None else e_r0 - e_r1
            drift_gain = None if e_m0 is None or e_m1 is None else e_m0 - e_m1
            subset = paired.loc[mask]
            improved = int((subset.delta < -1e-9).sum())
            worsened = int((subset.delta > 1e-9).sum())
            rows.append({
                "metric": metric, "group": group_name,
                "E_R0_clean_frozen": e_r0, "E_R1_clean_continued": e_r1,
                "clean_continued_gain": clean_gain, "clean_continued_pct_change": _pct_change(clean_gain, e_r0),
                "E_M0_drift_frozen": e_m0, "E_M1_drift_continued": e_m1,
                "drift_continued_gain": drift_gain, "drift_continued_pct_change": _pct_change(drift_gain, e_m0),
                "n_clients": int(len(subset)), "improved_clients": improved, "worsened_clients": worsened,
                "unchanged_clients": int(len(subset)) - improved - worsened,
            })
    pd.DataFrame(rows).to_csv(comparison / "continued_training_control.csv", index=False)


_RETENTION_KEY_COLUMNS = ["cluster_id", "client_id", "window_start_index", "window_end_index", "target_index", "timestamp"]


def _load_counterfactual_predictions(root: Path, method: str) -> pd.DataFrame:
    source_root = root / method / "counterfactual_clean_holdout"
    rows = []
    for path in sorted(source_root.glob("cluster_*/client_*_test_predictions.csv")):
        cluster_id = path.parent.name.removeprefix("cluster_")
        client_id = path.name.removeprefix("client_").removesuffix("_test_predictions.csv")
        frame = pd.read_csv(path)
        frame.insert(0, "client_id", client_id); frame.insert(0, "cluster_id", cluster_id)
        rows.append(frame)
    if not rows:
        raise ValueError(f"{method} has no counterfactual_clean_holdout prediction files at {source_root}")
    result = pd.concat(rows, ignore_index=True)
    result["target_index"] = result.window_end_index.astype(int) + 1
    return result


def _retention_merge(r0_frame: pd.DataFrame, root: Path, method: str) -> pd.DataFrame:
    """R0 clean final-holdout vs. `method`'s counterfactual-clean-holdout,
    for the affected/selected cohort only, matched on the *full* sample
    identity (cluster, client, window bounds, target index, timestamp --
    not target_index alone), with y_true cross-checked for numerical
    agreement. Fails closed (raises) on any identity or value mismatch,
    rather than silently comparing different samples.
    """
    r0_affected = r0_frame.loc[
        r0_frame.selected_for_drift, [*_RETENTION_KEY_COLUMNS, "y_true", "y_pred"]
    ].rename(columns={"y_true": "y_true_r0", "y_pred": "y_pred_r0"})

    cf_frame = _load_counterfactual_predictions(root, method)
    affected_ids = r0_frame.loc[r0_frame.selected_for_drift, ["cluster_id", "client_id"]].drop_duplicates()
    cf_affected = cf_frame.merge(affected_ids, on=["cluster_id", "client_id"], how="inner")
    cf_renamed = cf_affected[[*_RETENTION_KEY_COLUMNS, "y_true", "y_pred"]].rename(
        columns={"y_true": "y_true_cf", "y_pred": "y_pred_cf"}
    )

    ref_keys = set(map(tuple, r0_affected[_RETENTION_KEY_COLUMNS].itertuples(index=False, name=None)))
    other_keys = set(map(tuple, cf_renamed[_RETENTION_KEY_COLUMNS].itertuples(index=False, name=None)))
    if ref_keys != other_keys:
        missing, extra = sorted(ref_keys - other_keys)[:5], sorted(other_keys - ref_keys)[:5]
        raise ValueError(
            f"{method} counterfactual retention sample identity does not exactly match R0's clean "
            f"final-holdout for the affected cohort (missing={missing} extra={extra}); refusing to "
            f"compare mismatched samples (fail-closed)."
        )
    merged = r0_affected.merge(cf_renamed, on=_RETENTION_KEY_COLUMNS, validate="one_to_one")
    if not np.allclose(merged.y_true_r0.to_numpy(float), merged.y_true_cf.to_numpy(float), atol=1e-6):
        raise ValueError(
            f"{method} counterfactual retention: y_true differs from R0's clean final-holdout for "
            f"identically-keyed samples beyond tolerance -- these are not the same underlying "
            f"clean timestamps (fail-closed)."
        )
    return merged


def _retention_per_client(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cluster_id, client_id), group in merged.groupby(["cluster_id", "client_id"]):
        ref_metrics = _metrics(pd.DataFrame({"y_true": group.y_true_r0, "y_pred": group.y_pred_r0}))
        cf_metrics = _metrics(pd.DataFrame({"y_true": group.y_true_cf, "y_pred": group.y_pred_cf}))
        for metric in ("mae", "rmse", "smape"):
            e_ref, e_cf = ref_metrics[metric], cf_metrics[metric]
            delta = e_cf - e_ref
            rows.append({
                "cluster_id": cluster_id, "client_id": client_id, "metric": metric,
                "E_R0_clean": e_ref, "E_counterfactual_clean": e_cf,
                "retention_delta": delta, "retention_pct_change": _pct_change(delta, e_ref),
                "status": "worsened" if delta > 1e-9 else ("improved" if delta < -1e-9 else "unchanged"),
            })
    return pd.DataFrame(rows)


def _retention_aggregate_row(merged: pd.DataFrame, per_client: pd.DataFrame, metric: str) -> dict[str, Any]:
    ref_metrics = _metrics(pd.DataFrame({"y_true": merged.y_true_r0, "y_pred": merged.y_pred_r0}))
    cf_metrics = _metrics(pd.DataFrame({"y_true": merged.y_true_cf, "y_pred": merged.y_pred_cf}))
    e_ref, e_cf = ref_metrics[metric], cf_metrics[metric]
    delta = e_cf - e_ref
    row = {
        "metric": metric, "E_R0_clean": e_ref, "E_counterfactual_clean": e_cf,
        "retention_delta": delta, "retention_pct_change": _pct_change(delta, e_ref),
        "denominator_status": "valid" if e_ref not in (0, None) else "non_positive_or_near_zero",
    }
    if metric == "mae":
        client_deltas = per_client.loc[per_client.metric == "mae", "retention_delta"]
        row.update({
            "client_delta_mean": float(client_deltas.mean()), "client_delta_median": float(client_deltas.median()),
            "client_delta_std": float(client_deltas.std(ddof=0)),
            "client_delta_p90": float(client_deltas.quantile(0.90)), "client_delta_p95": float(client_deltas.quantile(0.95)),
            "worst_degradation": float(client_deltas.max()),
            "n_improved": int((client_deltas < -1e-9).sum()), "n_unchanged": int((client_deltas.abs() <= 1e-9).sum()),
            "n_worsened": int((client_deltas > 1e-9).sum()),
            "pct_worsened": float((client_deltas > 1e-9).mean() * 100.0),
        })
    return row


def _counterfactual_retention(frames: dict[str, pd.DataFrame], root: Path, methods: list[str], comparison: Path) -> None:
    """Quantitative M2/M3 counterfactual-retention analysis: after
    adapting selected clients to the drifted regime, how well do they
    still perform on the CLEAN version of the exact same final-holdout
    timestamps, versus R0's frozen-clean-H baseline on those same clients?

    retention_delta = E_after_adaptation_on_clean - E_clean_reference.
    Positive = forgetting/degradation, zero = exact retention, negative =
    adaptation also improved clean-regime performance -- never forced into
    a 0-100 "retention score".
    """
    oracle_methods = [method for method in methods if resolve_method_policy(method).oracle_enabled]
    r0_frame = frames.get(Method.R0.value)
    if not oracle_methods or r0_frame is None:
        pd.DataFrame([{"status": "missing", "reason": "no oracle-enabled method or no R0 baseline in this suite"}]).to_csv(
            comparison / "retention_summary.csv", index=False
        )
        return

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for method in oracle_methods:
        merged = _retention_merge(r0_frame, root, method)
        per_client = _retention_per_client(merged)
        per_client.to_csv(comparison / f"{_short_method_name(method)}_retention_per_client.csv", index=False)
        for metric in ("mae", "rmse", "smape"):
            agg = _retention_aggregate_row(merged, per_client, metric)
            summary_rows.append({"method": method, **agg})
            comparison_rows.append({
                "metric": metric, "method": method,
                "E_R0_clean": agg["E_R0_clean"], "E_counterfactual_clean": agg["E_counterfactual_clean"],
                "retention_delta": agg["retention_delta"], "retention_pct_change": agg["retention_pct_change"],
            })
    pd.DataFrame(summary_rows).to_csv(comparison / "retention_summary.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison / "client_only_retention_comparison.csv", index=False)


def _short_method_name(method: str) -> str:
    # "M2_oracle_client_only" -> "m2", "M3_oracle_client_ewc" -> "m3" -- for
    # the per-client retention CSV filenames the spec names explicitly (m2_
    # retention_per_client.csv) and their natural M3 counterpart.
    return method.split("_", 1)[0].lower()


def evaluate_completed_suite(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve(); comparison = root / "comparison"; comparison.mkdir(exist_ok=True)
    cohort = _load_cohort(root)
    methods = _suite_methods(root)
    frames = {method: _load_method_predictions(root, method).merge(cohort[["cluster_id", "client_id", "selected_for_drift"]], on=["cluster_id", "client_id"], validate="many_to_one") for method in methods}
    key_hashes = {method: canonical_hash(sorted(zip(frame.cluster_id, frame.client_id, frame.target_index))) for method, frame in frames.items()}
    if len(set(key_hashes.values())) != 1: raise ValueError(f"final holdout keys differ across methods: {key_hashes}")
    client_rows, cluster_rows, group_rows = [], [], []
    for method, frame in frames.items():
        for (cluster, client), group in frame.groupby(["cluster_id", "client_id"]): client_rows.append({"method": method, "cluster_id": cluster, "client_id": client, "selected_for_drift": bool(group.selected_for_drift.iloc[0]), **_metrics(group)})
        for cluster, group in frame.groupby("cluster_id"): cluster_rows.append({"method": method, "cluster_id": cluster, **_metrics(group)})
        for name, mask in (("affected", frame.selected_for_drift), ("controls", ~frame.selected_for_drift), ("all", np.ones(len(frame), dtype=bool))): group_rows.append({"method": method, "group": name, **_metrics(frame.loc[mask])})
    clients = pd.DataFrame(client_rows); clusters = pd.DataFrame(cluster_rows); groups = pd.DataFrame(group_rows)
    clients.loc[clients.selected_for_drift].to_csv(comparison / "affected_client_comparison.csv", index=False)
    clients.loc[~clients.selected_for_drift].to_csv(comparison / "clean_control_comparison.csv", index=False)
    clusters.to_csv(comparison / "cluster_comparison.csv", index=False); groups.to_csv(comparison / "run_comparison.csv", index=False)
    _continued_training_control(clients, groups, comparison, methods)
    roles = {method: resolve_method_policy(method).method_role for method in methods}
    clean_baseline = [method for method, role in roles.items() if role == "clean_baseline"]
    frozen_baseline = [method for method, role in roles.items() if role == "frozen_baseline"]
    candidates = [method for method, role in roles.items() if role == "mitigation_candidate"]
    if len(clean_baseline) != 1 or len(frozen_baseline) != 1:
        raise ValueError(f"recovery-rate comparison requires exactly one clean_baseline and one frozen_baseline method; roles={roles}")
    clean_method, frozen_method = clean_baseline[0], frozen_baseline[0]
    recovery = []
    for metric in ("mae", "rmse", "smape"):
        values = {method: float(groups.loc[(groups.method == method) & (groups.group == "affected"), metric].iloc[0]) for method in methods}
        denominator = values[frozen_method] - values[clean_method]
        for method in candidates:
            recovered = values[frozen_method] - values[method]
            recovery.append({"metric": metric, "method": method, "E_clean": values[clean_method], "E_frozen": values[frozen_method], "E_mitigated": values[method], "drift_damage": denominator, "recovered_error": recovered, "recovery_rate": None if denominator <= 1e-12 else recovered / denominator, "denominator_status": "non_positive_or_near_zero" if denominator <= 1e-12 else "valid"})
    recovery_frame = pd.DataFrame(recovery); recovery_frame.to_csv(comparison / "recovery_summary.csv", index=False)
    _counterfactual_retention(frames, root, methods, comparison)
    (comparison / "parameter_change_comparison.csv").write_text("method,accepted\n", encoding="utf-8")
    write_json(comparison / "checkpoint_hash_comparison.json", {"prediction_key_hashes": key_hashes})
    write_json(comparison / "reproducibility_report.json", {"prediction_keys_identical": True})
    summary = "# Final drift-mitigation summary\n\n" + recovery_frame.to_markdown(index=False) + "\n"
    (comparison / "final_summary.md").write_text(summary, encoding="utf-8")
    acceptance = {"accepted": True, "identical_holdout_keys": True, "prediction_key_hash": next(iter(key_hashes.values())), "primary_metric": "affected-client MAE"}
    write_json(comparison / "acceptance_report.json", acceptance)
    return {"output": str(comparison), "acceptance": acceptance, "recovery": recovery}
