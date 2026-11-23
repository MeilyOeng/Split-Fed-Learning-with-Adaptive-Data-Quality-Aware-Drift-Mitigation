"""Artifact-based drift and forecast evaluation; contains no model or injection code."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence
from collections import defaultdict
from .summary_metrics import aggregate_detector_rows
from .contracts import (
    ClientSelectionRecord,
    DetectorEvent,
    DriftManifestRow,
    DriftEvaluationResult,
    DriftType,
    ForecastSegment,
    ForecastSegmentMetrics,
    InjectionRecord,
    IntervalEvaluation,
    OnsetEvaluation,
)


def _onset(label: str, onset_index: int, alarms: Sequence[int], warmup_end_index: Optional[int]) -> OnsetEvaluation:
    pre = tuple(alarm for alarm in alarms if alarm < onset_index)
    eligible = max(onset_index, warmup_end_index) if warmup_end_index is not None else onset_index
    post = tuple(alarm for alarm in alarms if alarm >= eligible)
    first = post[0] if post else None
    return OnsetEvaluation(
        label=label,
        onset_index=onset_index,
        pre_drift_alarms=pre,
        post_drift_alarms=post,
        first_valid_detection=first,
        successful_detection=first is not None,
        detection_delay=(first - onset_index) if first is not None else None,
        warmup_end_index=warmup_end_index,
        eligible_onset_index=eligible,
        minimum_achievable_delay=max(0, eligible - onset_index),
    )


def _active_intervals(records: Sequence[InjectionRecord]) -> list[tuple[int, int, int]]:
    active = sorted(record.observation_index for record in records if record.regime_active)
    if not active:
        return []
    intervals: list[tuple[int, int, int]] = []
    start = previous = active[0]
    interval_id = 0
    for current in active[1:]:
        if current != previous + 1:
            intervals.append((interval_id, start, previous + 1))
            interval_id += 1
            start = current
        previous = current
    intervals.append((interval_id, start, previous + 1))
    return intervals


def evaluate_detection(
    events: Sequence[DetectorEvent],
    injection_records: Sequence[InjectionRecord] = (),
) -> DriftEvaluationResult:
    """Evaluate one detector/client/stream event sequence against injection ground truth."""
    if not events:
        raise ValueError("events must be non-empty")
    identity = {(e.detector_name, e.stream_name, e.run_id, e.cluster_id, e.client_id, e.mode) for e in events}
    if len(identity) != 1:
        raise ValueError("events must belong to exactly one detector/client/mode/stream")
    ordered = sorted(events, key=lambda event: event.observation_index)
    if len({event.observation_index for event in ordered}) != len(ordered):
        raise ValueError("events contain duplicate observation coordinates")
    alarms = tuple(event.observation_index for event in ordered if event.alarm)
    detector_name, stream_name, *_ = next(iter(identity))
    rate = 1000.0 * len(alarms) / len(ordered)

    warmup_values = {event.detector_min_instances for event in ordered if event.detector_min_instances is not None}
    if len(warmup_values) > 1:
        raise ValueError("events contain inconsistent detector warm-up metadata")
    warmup_observations = next(iter(warmup_values), None)
    warmup_end = ordered[0].observation_index + warmup_observations - 1 if warmup_observations else None

    records = sorted(injection_records, key=lambda record: record.observation_index)
    if records:
        record_identity = {(r.run_id, r.cluster_id, r.client_id, r.mode) for r in records}
        event_identity = {(e.run_id, e.cluster_id, e.client_id, e.mode) for e in ordered}
        if record_identity != event_identity:
            raise ValueError("injection records do not match detector event identity")
    active_records = [record for record in records if record.regime_active]
    if not active_records:
        return DriftEvaluationResult(
            detector_name=detector_name, stream_name=stream_name, observations_seen=len(ordered),
            all_alarms=alarms, first_alarm_anywhere=alarms[0] if alarms else None,
            clean_client_false_positive=bool(alarms), false_alarm_present=bool(alarms),
            false_alarm_count=len(alarms), alarms_per_1000_observations=rate, primary=None,
            warmup_observations=warmup_observations, warmup_end_index=warmup_end,
            eligible_for_detection=None,
        )

    drift_type = DriftType.parse(active_records[0].drift_type)
    onset_results: list[OnsetEvaluation] = []
    interval_results: list[IntervalEvaluation] = []
    inactive_alarms: tuple[int, ...] = ()

    if drift_type is DriftType.RECURRING:
        intervals = _active_intervals(records)
        valid_pairs: list[tuple[int, int]] = []
        active_alarm_set: set[int] = set()
        for interval_id, start, end in intervals:
            inside = tuple(alarm for alarm in alarms if start <= alarm < end)
            active_alarm_set.update(inside)
            eligible_inside = tuple(alarm for alarm in inside if warmup_end is None or alarm >= warmup_end)
            first = eligible_inside[0] if eligible_inside else None
            if first is not None:
                valid_pairs.append((start, first))
            interval_results.append(IntervalEvaluation(
                interval_id=interval_id, start_index=start, end_index_exclusive=end,
                alarms=eligible_inside, first_valid_detection=first,
                successful_detection=first is not None,
                detection_delay=(first - start) if first is not None else None,
            ))
        inactive_alarms = tuple(alarm for alarm in alarms if alarm not in active_alarm_set)
        first_start = intervals[0][1]
        first_pair = valid_pairs[0] if valid_pairs else None
        eligible = max(first_start, warmup_end) if warmup_end is not None else first_start
        primary = OnsetEvaluation(
            label="recurring_active_intervals", onset_index=first_start,
            pre_drift_alarms=tuple(alarm for alarm in alarms if alarm < first_start),
            post_drift_alarms=tuple(pair[1] for pair in valid_pairs),
            first_valid_detection=first_pair[1] if first_pair else None,
            successful_detection=first_pair is not None,
            detection_delay=(first_pair[1] - first_pair[0]) if first_pair else None,
            warmup_end_index=warmup_end, eligible_onset_index=eligible,
            minimum_achievable_delay=max(0, eligible - first_start),
        )
        false_count = len(inactive_alarms)
    else:
        transition_index = int(active_records[0].configured_start_index)
        onset_results.append(_onset("transition", transition_index, alarms, warmup_end))
        if drift_type in (DriftType.GRADUAL, DriftType.INCREMENTAL):
            effective = next((r.observation_index for r in records if r.drift_weight >= 0.5), None)
            if effective is not None:
                onset_results.append(_onset("effective_0.5", effective, alarms, warmup_end))
        primary = onset_results[0]
        false_count = len(primary.pre_drift_alarms)

    return DriftEvaluationResult(
        detector_name=detector_name, stream_name=stream_name, observations_seen=len(ordered),
        all_alarms=alarms, first_alarm_anywhere=alarms[0] if alarms else None,
        clean_client_false_positive=False, false_alarm_present=false_count > 0,
        false_alarm_count=false_count, alarms_per_1000_observations=rate, primary=primary,
        onset_evaluations=tuple(onset_results), interval_evaluations=tuple(interval_results),
        inactive_interval_alarms=inactive_alarms, warmup_observations=warmup_observations,
        warmup_end_index=warmup_end,
        eligible_for_detection=ordered[-1].observation_index >= (primary.eligible_onset_index or primary.onset_index),
    )


def classify_forecast_segment(window_start_index: int, window_end_index: int, drift_start_index: int) -> ForecastSegment:
    if window_start_index < 0 or window_end_index < window_start_index or drift_start_index < 0:
        raise ValueError("forecast coordinates must satisfy 0 <= window_start <= window_end and drift_start >= 0")
    if window_end_index < drift_start_index:
        label = "pre_drift"
    elif window_start_index < drift_start_index <= window_end_index:
        label = "mixed_window"
    else:
        label = "fully_drifted"
    return ForecastSegment(window_start_index, window_end_index, label)


def evaluate_forecasts_by_segment(
    window_start_indices: Sequence[int],
    window_end_indices: Sequence[int],
    targets: Sequence[float],
    predictions: Sequence[float],
    *,
    drift_start_index: int,
) -> tuple[ForecastSegmentMetrics, ...]:
    if not (len(window_start_indices) == len(window_end_indices) == len(targets) == len(predictions)):
        raise ValueError("forecast arrays must have identical lengths")
    groups: dict[str, list[tuple[float, float]]] = {
        name: [] for name in ("pre_drift", "mixed_window", "fully_drifted")
    }
    for start, end, target, prediction in zip(window_start_indices, window_end_indices, targets, predictions):
        if not math.isfinite(float(target)) or not math.isfinite(float(prediction)):
            raise ValueError("forecast values must be finite")
        label = classify_forecast_segment(int(start), int(end), drift_start_index).segment
        groups[label].append((float(target), float(prediction)))
    results = []
    for label, pairs in groups.items():
        if not pairs:
            results.append(ForecastSegmentMetrics(label, 0, None, None))
            continue
        errors = [prediction - target for target, prediction in pairs]
        results.append(ForecastSegmentMetrics(
            label, len(pairs), sum(abs(error) for error in errors) / len(errors),
            math.sqrt(sum(error * error for error in errors) / len(errors)),
        ))
    return tuple(results)


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _optional_int(value: object) -> Optional[int]:
    return None if value is None or str(value).strip() == "" else int(value)


def _optional_float(value: object) -> Optional[float]:
    return None if value is None or str(value).strip() == "" else float(value)


def load_detector_events(path: str | Path) -> list[DetectorEvent]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            DetectorEvent(
                run_id=row["run_id"], cluster_id=row["cluster_id"], client_id=row["client_id"],
                mode=row["mode"], stream_name=row["stream_name"], detector_name=row["detector_name"],
                event_index=int(row["event_index"]), observation_index=int(row["observation_index"]),
                timestamp=row.get("timestamp") or None, detector_value=float(row["detector_value"]),
                alarm=_bool(row["alarm"]), regime_active=_bool(row["regime_active"]) if row.get("regime_active") else None,
                transform_applied=_bool(row["transform_applied"]) if row.get("transform_applied") else None,
                detector_min_instances=_optional_int(row.get("detector_min_instances")),
            ) for row in csv.DictReader(handle)
        ]


def load_injection_records(path: str | Path) -> list[InjectionRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(InjectionRecord(
                run_id=row["run_id"], configuration_id=row["configuration_id"], cluster_id=row["cluster_id"],
                client_id=row["client_id"], mode=row["mode"], observation_index=int(row["observation_index"]),
                timestamp=row.get("timestamp") or None, base_value=float(row["base_value"]),
                runtime_value=float(row["runtime_value"]), drift_type=row["drift_type"],
                drift_source=row["drift_source"],
                configured_start_index=_optional_int(row.get("configured_start_index")),
                configured_end_index=_optional_int(row.get("configured_end_index")),
                drift_weight=float(row["drift_weight"]), regime_active=_bool(row["regime_active"]),
                transform_applied=_bool(row["transform_applied"]), affected_sample_count=int(row["affected_sample_count"]),
                magnitude=float(row["magnitude"]), offset_delta=float(row.get("offset_delta") or 0.0),
                noise_sigma=float(row["noise_sigma"]), label_delta=float(row["label_delta"]),
                period=_optional_int(row.get("period")),
                transition_function=row["transition_function"], random_seed=int(row["random_seed"]),
                active_interval_id=_optional_int(row.get("active_interval_id")),
                scope=row.get("scope") or "test",
                requested_start_ratio=_optional_float(row.get("requested_start_ratio")),
                requested_end_ratio=_optional_float(row.get("requested_end_ratio")),
                segment_start_index=_optional_int(row.get("segment_start_index")),
                segment_end_index_exclusive=_optional_int(row.get("segment_end_index_exclusive")),
            ))
        return rows



def load_component_manifest(path: str | Path) -> list[DriftManifestRow]:
    """Load schema-v2 component-level ground truth without collapsing mixed effects."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            DriftManifestRow(
                experiment_id=row["experiment_id"],
                configuration_id=row["configuration_id"],
                cluster_id=row["cluster_id"],
                client_id=row["client_id"],
                absolute_index=int(row["absolute_index"]),
                timestamp=row.get("timestamp") or None,
                component_id=row["component_id"],
                component_type=row["component_type"],
                drift_type=row["drift_type"],
                drift_source=row["drift_source"],
                configured_regime=row["configured_regime"],
                old_concept_id=row["old_concept_id"],
                new_concept_id=row["new_concept_id"],
                active_concept_id=row["active_concept_id"],
                drift_weight=float(row["drift_weight"]),
                regime_active=_bool(row["regime_active"]),
                change_point_event=_bool(row["change_point_event"]),
                new_regime_selected=_bool(row["new_regime_selected"]),
                transformation_applied=_bool(row["transformation_applied"]),
                affected_features=row["affected_features"],
                transformation_parameters=row["transformation_parameters"],
                experiment_seed=int(row["experiment_seed"]),
                client_seed=int(row["client_seed"]),
                config_hash=row["config_hash"],
                resolved_schedule_hash=row["resolved_schedule_hash"],
                autoregressive_consistency_expected=_bool(row["autoregressive_consistency_expected"]),
                exception_reason=row.get("exception_reason") or None,
                base_values=row["base_values"],
                runtime_values=row["runtime_values"],
            )
            for row in csv.DictReader(handle)
        ]


def load_client_selection_manifest(path: str | Path) -> list[ClientSelectionRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            ClientSelectionRecord(
                experiment_id=row["experiment_id"],
                cluster_id=row["cluster_id"],
                client_id=row["client_id"],
                affected=_bool(row["affected"]),
                selection_method=row["selection_method"],
                experiment_seed=int(row["experiment_seed"]),
                client_seed=int(row["client_seed"]),
                config_hash=row["config_hash"],
                resolved_schedule_hash=row["resolved_schedule_hash"],
            )
            for row in csv.DictReader(handle)
        ]


def evaluate_detection_from_manifest(
    events: Sequence[DetectorEvent],
    manifest_rows: Sequence[DriftManifestRow],
    *,
    affected: bool,
) -> DriftEvaluationResult:
    """Evaluate detector events from schema-v2 component truth and client selection."""
    if not affected:
        return evaluate_detection(events, ())
    if not manifest_rows:
        raise ValueError("an affected client requires schema-v2 component manifest rows")
    identities = {
        (row.experiment_id, row.cluster_id, row.client_id) for row in manifest_rows
    }
    if len(identities) != 1:
        raise ValueError("component manifest rows must belong to one experiment/client")
    event_identities = {(event.run_id, event.cluster_id, event.client_id) for event in events}
    if identities != event_identities:
        raise ValueError("component manifest rows do not match detector event identity")
    drift_types = {row.drift_type for row in manifest_rows}
    if len(drift_types) != 1:
        raise ValueError("mixed components must share one temporal drift type")
    drift_type = DriftType.parse(next(iter(drift_types)))
    by_index: dict[int, list[DriftManifestRow]] = defaultdict(list)
    for row in manifest_rows:
        by_index[row.absolute_index].append(row)
    onset_candidates = sorted({
        row.absolute_index for row in manifest_rows if row.change_point_event
    })
    if not onset_candidates:
        raise ValueError("affected schema-v2 manifest contains no change-point event")
    onset = onset_candidates[0]
    event_mode = events[0].mode if events else "test"
    synthetic: list[InjectionRecord] = []
    for absolute_index, rows in sorted(by_index.items()):
        representative = rows[0]
        if any(row.drift_weight != representative.drift_weight for row in rows):
            raise ValueError("mixed component schedule weights disagree at one timestamp")
        selected_new = any(row.new_regime_selected for row in rows)
        active = selected_new if drift_type is DriftType.RECURRING else any(
            row.regime_active for row in rows
        )
        transformed = any(row.transformation_applied for row in rows)
        synthetic.append(InjectionRecord(
            run_id=representative.experiment_id,
            configuration_id=representative.configuration_id,
            cluster_id=representative.cluster_id,
            client_id=representative.client_id,
            mode=event_mode,
            observation_index=absolute_index,
            timestamp=representative.timestamp,
            base_value=0.0,
            runtime_value=1.0 if transformed else 0.0,
            drift_type=drift_type.value,
            drift_source=representative.drift_source,
            configured_start_index=onset,
            configured_end_index=None,
            drift_weight=representative.drift_weight,
            regime_active=active,
            transform_applied=transformed,
            affected_sample_count=int(transformed),
            magnitude=0.0,
            offset_delta=0.0,
            noise_sigma=0.0,
            label_delta=0.0,
            period=None,
            transition_function="resolved_manifest",
            random_seed=representative.client_seed,
        ))
    return evaluate_detection(events, synthetic)


def evaluate_run_artifacts(run_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Evaluate persisted stream artifacts without loading models, data, or scalers."""
    run_root = Path(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    events_by_identity: dict[tuple[str, str, str, str, str, str], list[DetectorEvent]] = defaultdict(list)
    records_by_client: dict[tuple[str, str, str, str], list[InjectionRecord]] = defaultdict(list)
    components_by_client: dict[tuple[str, str, str], list[DriftManifestRow]] = defaultdict(list)
    selections_by_client: dict[tuple[str, str, str], ClientSelectionRecord] = {}
    for path in sorted(run_root.glob("**/drift/detector_events.csv")):
        for event in load_detector_events(path):
            events_by_identity[(event.run_id, event.cluster_id, event.client_id, event.mode, event.stream_name, event.detector_name)].append(event)
    for path in sorted(run_root.glob("**/drift/injection_manifest.csv")):
        for record in load_injection_records(path):
            records_by_client[(record.run_id, record.cluster_id, record.client_id, record.mode)].append(record)
    for path in sorted(run_root.glob("**/drift/component_manifest.csv")):
        for row in load_component_manifest(path):
            components_by_client[(row.experiment_id, row.cluster_id, row.client_id)].append(row)
    for path in sorted(run_root.glob("**/drift/client_selection_manifest.csv")):
        for row in load_client_selection_manifest(path):
            key = (row.experiment_id, row.cluster_id, row.client_id)
            previous = selections_by_client.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting client-selection rows for {key}")
            selections_by_client[key] = row
    summaries = []
    forecast_rows = []
    for identity, event_rows in sorted(events_by_identity.items()):
        run_id, cluster_id, client_id, mode, stream_name, detector_name = identity
        records = records_by_client.get((run_id, cluster_id, client_id, mode), [])
        schema_key = (run_id, cluster_id, client_id)
        schema_rows = components_by_client.get(schema_key, [])
        selection = selections_by_client.get(schema_key)
        if schema_rows or selection is not None:
            affected = selection.affected if selection is not None else bool(schema_rows)
            result = evaluate_detection_from_manifest(
                event_rows, schema_rows, affected=affected
            )
            drifted_client = affected
        else:
            result = evaluate_detection(event_rows, records)
            drifted_client = bool(records)
        summaries.append({
            "run_id": run_id, "cluster_id": cluster_id, "client_id": client_id, "mode": mode,
            "stream_name": stream_name, "detector_name": detector_name, "drifted_client": drifted_client,
            "observations_seen": result.observations_seen, "alarm_count": len(result.all_alarms),
            "first_alarm_anywhere": result.first_alarm_anywhere,
            "clean_client_false_positive": result.clean_client_false_positive,
            "pre_or_inactive_false_alarm_count": result.false_alarm_count,
            "successful_detection": result.primary.successful_detection if result.primary else None,
            "first_valid_detection": result.primary.first_valid_detection if result.primary else None,
            "detection_delay": result.primary.detection_delay if result.primary else None,
            "alarms_per_1000_observations": result.alarms_per_1000_observations,
            "warmup_observations": result.warmup_observations, "warmup_end_index": result.warmup_end_index,
            "eligible_for_detection": result.eligible_for_detection,
        })
    for client_key, records in sorted(records_by_client.items()):
        _, cluster_id, client_id, _ = client_key
        prediction_path = run_root / f"cluster_{cluster_id}" / f"client_{client_id}_test_predictions.csv"
        if not prediction_path.exists() or not records:
            continue
        with prediction_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        start_index = min(record.configured_start_index for record in records if record.configured_start_index is not None)
        metrics = evaluate_forecasts_by_segment(
            [int(row["window_start_index"]) for row in rows], [int(row["window_end_index"]) for row in rows],
            [float(row["y_true"]) for row in rows], [float(row["y_pred"]) for row in rows], drift_start_index=start_index,
        )
        for item in metrics:
            forecast_rows.append({"cluster_id": cluster_id, "client_id": client_id, **asdict(item)})
    summary_fields = list(summaries[0]) if summaries else ["run_id", "cluster_id", "client_id", "mode", "stream_name", "detector_name"]
    with (destination / "client_detection_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    forecast_fields = list(forecast_rows[0]) if forecast_rows else ["cluster_id", "client_id", "segment", "count", "mae", "rmse"]
    with (destination / "forecast_segment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=forecast_fields)
        writer.writeheader()
        writer.writerows(forecast_rows)
    drifted = [row for row in summaries if row["drifted_client"]]
    clean = [row for row in summaries if not row["drifted_client"]]
    valid_delays = [row["detection_delay"] for row in drifted if row["detection_delay"] is not None]
    rows_by_detector: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        rows_by_detector[str(row["detector_name"])].append(row)
    overall_metrics = aggregate_detector_rows(summaries)
    payload = {
        "detector_client_rows": len(summaries), "drifted_detector_client_rows": len(drifted),
        "clean_detector_client_rows": len(clean),
        "post_drift_detection_rate": (sum(bool(row["successful_detection"]) for row in drifted) / len(drifted)) if drifted else None,
        "clean_client_false_positive_rate": (sum(bool(row["clean_client_false_positive"]) for row in clean) / len(clean)) if clean else None,
        "drifted_client_pre_or_inactive_false_alarm_rate": (sum(row["pre_or_inactive_false_alarm_count"] > 0 for row in drifted) / len(drifted)) if drifted else None,
        "mean_detection_delay": (sum(valid_delays) / len(valid_delays)) if valid_delays else None,
        "event_count": sum(row["observations_seen"] for row in summaries),
        "alarm_count": sum(row["alarm_count"] for row in summaries),
        "overall_metrics": overall_metrics,
        "per_detector": {
            detector_name: aggregate_detector_rows(detector_rows)
            for detector_name, detector_rows in sorted(rows_by_detector.items())
        },
        "metric_definitions": {
            "classification_unit": "one detector-client row",
            "positive_class": "drifted client eligible for post-drift detection",
            "negative_class": "clean client",
            "accuracy": "(true_positive + true_negative) / eligible classified rows",
            "false_positive_rate": "clean clients with at least one alarm / clean clients",
            "ineligible_drifted_clients": "excluded from classification metrics",
        },
    }
    (destination / "experiment_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate detector event artifacts without loading models or data")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--injection", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.run_dir:
        if args.events or not args.results_dir:
            parser.error("--run-dir requires --results-dir and cannot be combined with --events")
        print(json.dumps(evaluate_run_artifacts(args.run_dir, args.results_dir), indent=2))
        return 0
    if not args.events:
        parser.error("either --events or --run-dir is required")
    records = load_injection_records(args.injection) if args.injection else []
    result = evaluate_detection(load_detector_events(args.events), records)
    payload = json.dumps(asdict(result), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
