"""Schema-v2 drift configuration parsing and legacy migration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .contracts import (
    DriftComponentConfig,
    DriftExperimentConfig,
    DriftSource,
    DriftType,
    FeatureTransform,
    TemporalConfig,
    TransformType,
    TransitionFunction,
)


def canonical_json(value: object) -> str:
    """Return deterministic JSON suitable for hashing and manifests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_canonical(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} field(s): {unknown}")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return copy.deepcopy(dict(value))


def _parse_transform(feature_index: int, raw: Mapping[str, Any]) -> FeatureTransform:
    spec = _mapping(raw, f"feature_transforms[{feature_index}]")
    transform_type = TransformType.parse(spec.pop("type", ""))
    allowed_parameters = {
        TransformType.SCALE: {"factor"},
        TransformType.OFFSET: {"delta"},
        TransformType.NOISE: {"sigma"},
        TransformType.NONLINEAR: {"kind", "lower", "upper"},
    }[transform_type]
    _reject_unknown(spec, allowed_parameters, f"{transform_type.value} transform")
    if transform_type is TransformType.NONLINEAR:
        kind = str(spec.get("kind", "")).strip().lower()
        if kind not in {"square", "log1p", "clip"}:
            raise ValueError("nonlinear transform kind must be square, log1p, or clip")
        if kind != "clip" and any(key in spec for key in ("lower", "upper")):
            raise ValueError("nonlinear lower/upper bounds are only valid for kind=clip")
    if transform_type is TransformType.SCALE:
        factor = _finite("scale.factor", spec.get("factor"))
        if factor <= 0 or math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("scale.factor must be > 0 and different from 1")
        spec = {"factor": factor}
    elif transform_type is TransformType.OFFSET:
        delta = _finite("offset.delta", spec.get("delta"))
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("offset.delta must be non-zero")
        spec = {"delta": delta}
    elif transform_type is TransformType.NOISE:
        sigma = _finite("noise.sigma", spec.get("sigma"))
        if sigma <= 0:
            raise ValueError("noise.sigma must be > 0")
        spec = {"sigma": sigma}
    else:
        if kind == "clip":
            if not any(key in spec for key in ("lower", "upper")):
                raise ValueError("nonlinear clip requires at least one finite bound")
            lower = float("-inf") if "lower" not in spec else _finite("nonlinear.lower", spec["lower"])
            upper = float("inf") if "upper" not in spec else _finite("nonlinear.upper", spec["upper"])
            if lower >= upper:
                raise ValueError("nonlinear clip lower must be less than upper")
            spec = {"kind": kind}
            if math.isfinite(lower):
                spec["lower"] = lower
            if math.isfinite(upper):
                spec["upper"] = upper
        else:
            spec = {"kind": kind}
    return FeatureTransform(feature_index, transform_type, spec)


def _parse_feature_transforms(raw: object) -> tuple[FeatureTransform, ...]:
    mapping = _mapping(raw, "feature_transforms")
    result: list[FeatureTransform] = []
    indexed_specs: list[tuple[int, object]] = []
    for raw_index, raw_specs in mapping.items():
        try:
            feature_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature index {raw_index!r} is not an integer") from exc
        if feature_index < 0:
            raise ValueError("feature indices must be >= 0")
        indexed_specs.append((feature_index, raw_specs))
    for feature_index, raw_specs in sorted(indexed_specs):
        specs = raw_specs if isinstance(raw_specs, list) else [raw_specs]
        if not specs:
            raise ValueError(f"feature {feature_index} has no transformations")
        for spec in specs:
            if not isinstance(spec, Mapping):
                raise ValueError(f"feature transform for {feature_index} must be a mapping")
            result.append(_parse_transform(feature_index, spec))
    return tuple(result)


def _parse_temporal(raw: object, *, enabled: bool) -> TemporalConfig:
    data = _mapping(raw, "temporal")
    dtype = DriftType.parse(data.get("type", "none" if not enabled else "sudden"))
    _reject_unknown(data, {
        "type", "start_index", "end_index", "start_timestamp", "end_timestamp",
        "transition", "sigmoid_k", "exp_alpha", "cycle_length", "old_duration",
        "new_duration",
    }, "temporal")
    if "transition" in data and dtype not in {DriftType.INCREMENTAL, DriftType.GRADUAL}:
        raise ValueError("temporal.transition is only compatible with incremental or gradual drift")
    if "sigmoid_k" in data and str(data.get("transition", "linear")).lower() != "sigmoid":
        raise ValueError("temporal.sigmoid_k requires transition=sigmoid")
    if "exp_alpha" in data and str(data.get("transition", "linear")).lower() != "exponential":
        raise ValueError("temporal.exp_alpha requires transition=exponential")
    if dtype in {DriftType.NONE, DriftType.SUDDEN, DriftType.RECURRING} and any(
        key in data for key in ("end_index", "end_timestamp")
    ):
        raise ValueError(f"temporal end coordinates are not compatible with {dtype.value} drift")
    if dtype is DriftType.NONE and any(key in data for key in ("start_index", "start_timestamp")):
        raise ValueError("disabled temporal.type=none cannot configure an onset")
    start_index = data.get("start_index")
    end_index = data.get("end_index")
    start_timestamp = data.get("start_timestamp")
    end_timestamp = data.get("end_timestamp")
    if start_index is not None:
        start_index = int(start_index)
    if end_index is not None:
        end_index = int(end_index)
    if start_index is not None and start_timestamp is not None:
        raise ValueError("use either temporal.start_index or start_timestamp, not both")
    if end_index is not None and end_timestamp is not None:
        raise ValueError("use either temporal.end_index or end_timestamp, not both")
    if dtype is not DriftType.NONE and start_index is None and start_timestamp is None:
        raise ValueError(f"{dtype.value} drift requires an absolute start index or timestamp")
    if start_index is not None and start_index < 0:
        raise ValueError("temporal.start_index must be >= 0")
    if end_index is not None and end_index < 0:
        raise ValueError("temporal.end_index must be >= 0")
    if dtype in (DriftType.INCREMENTAL, DriftType.GRADUAL):
        if end_index is None and end_timestamp is None:
            raise ValueError(f"{dtype.value} drift requires an end index or timestamp")
        if start_index is not None and end_index is not None and end_index <= start_index:
            raise ValueError("temporal.end_index must be greater than start_index")
    transition = TransitionFunction.parse(data.get("transition", "linear"))
    sigmoid_k = _finite("temporal.sigmoid_k", data.get("sigmoid_k", 10.0))
    exp_alpha = _finite("temporal.exp_alpha", data.get("exp_alpha", 3.0))
    if exp_alpha <= 0:
        raise ValueError("temporal.exp_alpha must be > 0")
    if sigmoid_k <= 0:
        raise ValueError("temporal.sigmoid_k must be > 0")

    cycle_length = data.get("cycle_length")
    old_duration = data.get("old_duration")
    new_duration = data.get("new_duration")
    if dtype is DriftType.RECURRING:
        if cycle_length is None:
            raise ValueError("recurring drift requires cycle_length")
        cycle_length = int(cycle_length)
        if cycle_length <= 1:
            raise ValueError("recurring cycle_length must be > 1")
        if old_duration is None or new_duration is None:
            raise ValueError("recurring drift requires old_duration and new_duration")
        old_duration, new_duration = int(old_duration), int(new_duration)
        if old_duration <= 0 or new_duration <= 0:
            raise ValueError("recurring old/new durations must be > 0")
        if old_duration + new_duration != cycle_length:
            raise ValueError("old_duration + new_duration must equal cycle_length")
    elif any(item is not None for item in (cycle_length, old_duration, new_duration)):
        raise ValueError("cycle parameters are only compatible with recurring drift")

    if dtype is DriftType.NONE and enabled:
        raise ValueError("enabled drift configuration cannot use temporal.type=none")
    return TemporalConfig(
        drift_type=dtype,
        start_index=start_index,
        end_index=end_index,
        start_timestamp=None if start_timestamp is None else str(start_timestamp),
        end_timestamp=None if end_timestamp is None else str(end_timestamp),
        transition=transition,
        sigmoid_k=sigmoid_k,
        exp_alpha=exp_alpha,
        cycle_length=cycle_length,
        old_duration=old_duration,
        new_duration=new_duration,
    )


def _validate_process_parameters(data: Mapping[str, Any]) -> None:
    _reject_unknown(
        data,
        {"old", "changes", "seasonal_change", "innovation_source", "innovation_mode", "allow_unstable_ar"},
        "process component",
    )
    old = _mapping(data.get("old"), "process.old")
    _reject_unknown(
        old,
        {
            "intercept",
            "ar_coefficients",
            "seasonal_amplitude",
            "seasonal_phase",
            "seasonal_period",
            "innovation_sigma",
        },
        "process.old",
    )
    coefficients = old.get("ar_coefficients")
    if not isinstance(coefficients, (list, tuple)) or not coefficients:
        raise ValueError("process.old.ar_coefficients must be a non-empty list")
    for index, value in enumerate(coefficients):
        _finite(f"process.old.ar_coefficients[{index}]", value)
    for key in ("intercept", "seasonal_amplitude", "seasonal_phase"):
        if key in old:
            _finite(f"process.old.{key}", old[key])
    period = _finite("process.old.seasonal_period", old.get("seasonal_period", 24.0))
    if period <= 0:
        raise ValueError("process.old.seasonal_period must be > 0")
    sigma = _finite("process.old.innovation_sigma", old.get("innovation_sigma", 0.0))
    if sigma < 0:
        raise ValueError("process.old.innovation_sigma must be >= 0")

    changes = _mapping(data.get("changes"), "process.changes")
    _reject_unknown(
        changes,
        {
            "intercept_delta",
            "ar_coefficient_delta",
            "innovation_scale",
            "level_offset",
            "scale_factor",
        },
        "process.changes",
    )
    deltas = changes.get("ar_coefficient_delta")
    if deltas is not None:
        if not isinstance(deltas, (list, tuple)) or len(deltas) != len(coefficients):
            raise ValueError("process.changes.ar_coefficient_delta must match ar_coefficients")
        for index, value in enumerate(deltas):
            _finite(f"process.changes.ar_coefficient_delta[{index}]", value)
    for key in ("intercept_delta", "level_offset"):
        if key in changes:
            _finite(f"process.changes.{key}", changes[key])
    innovation_scale = _finite("process.changes.innovation_scale", changes.get("innovation_scale", 1.0))
    if innovation_scale < 0:
        raise ValueError("process.changes.innovation_scale must be >= 0")
    scale_factor = _finite("process.changes.scale_factor", changes.get("scale_factor", 1.0))
    if scale_factor <= 0:
        raise ValueError("process.changes.scale_factor must be > 0")

    seasonal = _mapping(data.get("seasonal_change"), "process.seasonal_change")
    _reject_unknown(
        seasonal,
        {"amplitude_delta", "phase_delta", "period_new"},
        "process.seasonal_change",
    )
    for key in ("amplitude_delta", "phase_delta"):
        if key in seasonal:
            _finite(f"process.seasonal_change.{key}", seasonal[key])
    if "period_new" in seasonal:
        period_new = _finite("process.seasonal_change.period_new", seasonal["period_new"])
        if period_new <= 0:
            raise ValueError("process.seasonal_change.period_new must be > 0")

    if "innovation_source" in data and "innovation_mode" in data:
        raise ValueError("configure process.innovation_source, not both innovation_source and innovation_mode")
    if "innovation_source" in data:
        innovation_source = str(data["innovation_source"]).strip().lower()
        if innovation_source not in {"observed_process_residual", "sampled_gaussian"}:
            raise ValueError(
                "process.innovation_source must be observed_process_residual or sampled_gaussian"
            )
    else:
        innovation_mode = str(data.get("innovation_mode", "clean_residual")).strip().lower()
        if innovation_mode not in {"clean_residual", "normal"}:
            raise ValueError("process.innovation_mode must be clean_residual or normal")
    if "allow_unstable_ar" in data and not isinstance(data["allow_unstable_ar"], bool):
        raise ValueError("process.allow_unstable_ar must be a boolean")



def _process_is_effectful(parameters: Mapping[str, Any]) -> bool:
    changes = _mapping(parameters.get("changes"), "process.changes")
    for key, value in changes.items():
        if key == "ar_coefficient_delta":
            if any(not math.isclose(float(item), 0.0, abs_tol=1e-15) for item in value):
                return True
        elif key in {"innovation_scale", "scale_factor"}:
            if not math.isclose(float(value), 1.0, abs_tol=1e-15):
                return True
        elif not math.isclose(float(value), 0.0, abs_tol=1e-15):
            return True
    seasonal = _mapping(parameters.get("seasonal_change"), "process.seasonal_change")
    old = _mapping(parameters.get("old"), "process.old")
    old_amplitude = float(old.get("seasonal_amplitude", 0.0))
    amplitude_delta = float(seasonal.get("amplitude_delta", 0.0))
    if not math.isclose(amplitude_delta, 0.0, abs_tol=1e-15):
        return True
    seasonal_active = not math.isclose(old_amplitude, 0.0, abs_tol=1e-15)
    if seasonal_active and not math.isclose(
        float(seasonal.get("phase_delta", 0.0)), 0.0, abs_tol=1e-15
    ):
        return True
    if seasonal_active and "period_new" in seasonal and not math.isclose(
        float(seasonal["period_new"]), float(old.get("seasonal_period", 24.0)), abs_tol=1e-15
    ):
        return True
    return False


def _parse_component(raw: Mapping[str, Any], default_id: str) -> DriftComponentConfig:
    data = _mapping(raw, f"component {default_id}")
    component_id = str(data.pop("id", default_id)).strip()
    type_value = data.pop("type", None)
    source_value = data.pop("source", None)
    if type_value is not None and source_value is not None:
        if DriftSource.parse(type_value) is not DriftSource.parse(source_value):
            raise ValueError("component type and source fields disagree")
    source = DriftSource.parse(
        type_value if type_value is not None else (
            source_value if source_value is not None else ""
        )
    )
    transforms = _parse_feature_transforms(data.pop("feature_transforms", {}))

    if source is DriftSource.MEASUREMENT:
        if not transforms:
            raise ValueError("measurement drift requires at least one non-identity feature transform")
        _reject_unknown(data, set(), "measurement component")
    elif transforms:
        raise ValueError(f"feature_transforms are not valid for source={source.value}; use MIXED components")

    if source is DriftSource.STRICT_COVARIATE:
        generator_id = str(data.get("generator_id", "")).strip()
        benchmark_kind = str(data.get("benchmark_kind", "")).strip().lower()
        if not generator_id:
            raise ValueError("strict_covariate drift requires generator_id")
        if benchmark_kind not in {"chronological", "supervised"}:
            raise ValueError("strict_covariate benchmark_kind must be chronological or supervised")
        if (
            "guarantees_autoregressive_consistency" in data
            and not isinstance(data["guarantees_autoregressive_consistency"], bool)
        ):
            raise ValueError("guarantees_autoregressive_consistency must be a boolean")
        if benchmark_kind == "chronological" and data.get("guarantees_autoregressive_consistency") is not True:
            raise ValueError(
                "chronological strict_covariate generator must explicitly guarantee autoregressive consistency"
            )
    elif source is DriftSource.PROCESS:
        _validate_process_parameters(data)
        if not _process_is_effectful(data):
            raise ValueError("process drift requires at least one non-zero process change")
    elif source is DriftSource.TARGET_MAPPING:
        _reject_unknown(
            data,
            {"old_mapping_id", "new_mapping_id", "target_fed_back", "supervised_only"},
            "target_mapping component",
        )
        old_id = str(data.get("old_mapping_id", "")).strip()
        new_id = str(data.get("new_mapping_id", "")).strip()
        if not old_id or not new_id or old_id == new_id:
            raise ValueError("target_mapping requires different old_mapping_id and new_mapping_id")
        for key, default in (("target_fed_back", True), ("supervised_only", False)):
            if key in data and not isinstance(data[key], bool):
                raise ValueError(f"target_mapping.{key} must be a boolean")
        feedback = data.get("target_fed_back", True)
        supervised_only = data.get("supervised_only", False)
        if feedback and not supervised_only:
            raise ValueError(
                "isolated target mapping is incoherent when targets feed later inputs; "
                "set supervised_only=true for an explicit benchmark"
            )
    elif source is DriftSource.MIXED:
        raise ValueError("nested mixed components are not supported")

    return DriftComponentConfig(component_id, source, transforms, data)


def load_drift_config(source: str | Path | Mapping[str, Any]) -> DriftExperimentConfig:
    """Parse without mutating the caller's mapping."""
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    else:
        raw = copy.deepcopy(dict(source))
    if not isinstance(raw, Mapping):
        raise ValueError("drift configuration must be a mapping")
    user_config = copy.deepcopy(dict(raw))
    _reject_unknown(
        user_config,
        {
            "schema_version",
            "experiment_id",
            "enabled",
            "experiment_seed",
            "clients",
            "modes",
            "temporal",
            "source",
            "windowing",
        },
        "root configuration",
    )
    if int(user_config.get("schema_version", 0)) != 2:
        raise ValueError("schema_version must be 2 for the timeline drift API")
    config_hash = sha256_canonical(user_config)
    enabled_value = user_config.get("enabled", False)
    if not isinstance(enabled_value, bool):
        raise ValueError("enabled must be a boolean")
    enabled = enabled_value
    experiment_id = str(user_config.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    experiment_seed_value = user_config.get("experiment_seed", 42)
    if not isinstance(experiment_seed_value, int) or isinstance(experiment_seed_value, bool):
        raise ValueError("experiment_seed must be an integer")
    experiment_seed = experiment_seed_value
    temporal = _parse_temporal(user_config.get("temporal"), enabled=enabled)

    clients = _mapping(user_config.get("clients"), "clients")
    _reject_unknown(clients, {"ids", "fraction"}, "clients")
    raw_client_ids = clients.get("ids", [])
    if not isinstance(raw_client_ids, (list, tuple)):
        raise ValueError("clients.ids must be a list")
    normalized_client_ids = [str(item).strip() for item in raw_client_ids]
    if any(not item for item in normalized_client_ids):
        raise ValueError("clients.ids cannot contain empty identifiers")
    if len(set(normalized_client_ids)) != len(normalized_client_ids):
        raise ValueError("clients.ids must be unique")
    client_ids = tuple(sorted(normalized_client_ids))
    fraction = clients.get("fraction")
    if fraction is not None:
        fraction = _finite("clients.fraction", fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("clients.fraction must be in [0, 1]")
    if client_ids and fraction is not None:
        raise ValueError("configure either clients.ids or clients.fraction, not both")
    if enabled and not client_ids and fraction is None:
        raise ValueError("enabled drift requires clients.ids or clients.fraction")
    if enabled and fraction == 0.0:
        raise ValueError("enabled drift cannot select a zero client fraction")

    raw_source = _mapping(user_config.get("source"), "source")
    root_type_value = raw_source.get("type")
    root_source_value = raw_source.get("source")
    if root_type_value is not None and root_source_value is not None:
        if DriftSource.parse(root_type_value) is not DriftSource.parse(root_source_value):
            raise ValueError("source.type and source.source fields disagree")
    source_type = DriftSource.parse(
        root_type_value if root_type_value is not None else (
            root_source_value if root_source_value is not None else "measurement"
        )
    )
    if not enabled:
        components: tuple[DriftComponentConfig, ...] = ()
    elif source_type is DriftSource.MIXED:
        _reject_unknown(raw_source, {"type", "source", "components"}, "mixed source")
        raw_components = raw_source.get("components")
        if not isinstance(raw_components, list) or len(raw_components) < 2:
            raise ValueError("mixed drift requires at least two components")
        components = tuple(_parse_component(item, f"component_{index}") for index, item in enumerate(raw_components))
        if len({item.component_id for item in components}) != len(components):
            raise ValueError("mixed component IDs must be unique")
    else:
        components = (_parse_component(raw_source, source_type.value),)

    windowing = _mapping(user_config.get("windowing"), "windowing")
    _reject_unknown(
        windowing,
        {"window_size", "forecast_horizon", "target_feature_index"},
        "windowing",
    )
    window_size = int(windowing.get("window_size", 24))
    forecast_horizon = int(windowing.get("forecast_horizon", 1))
    target_feature_index = int(windowing.get("target_feature_index", 0))
    if window_size <= 0 or forecast_horizon <= 0 or target_feature_index < 0:
        raise ValueError("window size and horizon must be > 0 and target_feature_index must be >= 0")
    raw_modes = user_config.get("modes", ["test"])
    if not isinstance(raw_modes, (list, tuple)):
        raise ValueError("modes must be a list")
    modes = tuple(str(item).strip().lower() for item in raw_modes)
    if not modes or any(item not in {"train", "val", "test"} for item in modes):
        raise ValueError("modes must contain train, val, and/or test")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must not contain duplicates")

    return DriftExperimentConfig(
        schema_version=2,
        experiment_id=experiment_id,
        enabled=enabled,
        experiment_seed=experiment_seed,
        drift_source=source_type,
        temporal=temporal,
        components=components,
        client_ids=client_ids,
        client_fraction=fraction,
        modes=modes,
        target_feature_index=target_feature_index,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        config_hash=config_hash,
        canonical_config=canonical_json(user_config),
    )


def legacy_environment_config(env: Mapping[str, str], *, series_length: int) -> DriftExperimentConfig:
    """Map the old scalar environment contract to measurement drift without changing its math."""
    dtype_text = str(env.get("DRIFT_TYPE", "sudden")).strip().lower()
    if dtype_text == "seasonal":
        dtype_text = "recurring"
    dtype = DriftType.parse(dtype_text)
    enabled = str(env.get("DRIFT_INJECTION_ENABLED", "false")).lower() in {"1", "true", "yes", "y"}
    start_ratio = float(env.get("DRIFT_START_RATIO", "0.5"))
    end_raw = str(env.get("DRIFT_END_RATIO", "")).strip()
    end_ratio = None if not end_raw else float(end_raw)
    start_index = int(start_ratio * series_length)
    temporal: dict[str, Any] = {"type": dtype.value, "start_index": start_index}
    if dtype in (DriftType.GRADUAL, DriftType.INCREMENTAL):
        if end_ratio is None:
            raise ValueError(f"legacy {dtype.value} drift requires DRIFT_END_RATIO")
        temporal["end_index"] = int(end_ratio * series_length)
        temporal["transition"] = env.get("DRIFT_TRANSITION", "linear")
    if dtype is DriftType.RECURRING:
        period = int(env.get("DRIFT_PERIOD", "168"))
        temporal.update(
            cycle_length=period,
            new_duration=(period + 1) // 2,
            old_duration=period // 2,
        )
    transforms: list[dict[str, Any]] = []
    magnitude = float(env.get("DRIFT_MAGNITUDE", "0.5"))
    offset = float(env.get("DRIFT_OFFSET_DELTA", "0"))
    noise = float(env.get("DRIFT_NOISE_SIGMA", "0"))
    if magnitude != 0:
        transforms.append({"type": "scale", "factor": 1.0 + magnitude})
    if offset != 0:
        transforms.append({"type": "offset", "delta": offset})
    if noise != 0:
        transforms.append({"type": "noise", "sigma": noise})
    mapping = {
        "schema_version": 2,
        "experiment_id": env.get("DRIFT_RUN_ID", "legacy"),
        "enabled": enabled and dtype is not DriftType.NONE,
        "experiment_seed": int(env.get("DRIFT_RANDOM_SEED", "42")),
        "clients": {
            "ids": [item for item in str(env.get("DRIFT_CLIENT_IDS", "")).split(",") if item],
        },
        "temporal": temporal if dtype is not DriftType.NONE else {"type": "none"},
        "source": {
            "type": "measurement",
            "feature_transforms": {"0": transforms},
        },
        "windowing": {
            "window_size": int(env.get("WINDOW_SIZE", "24")),
            "forecast_horizon": int(env.get("FORECAST_HORIZON", "1")),
            "target_feature_index": 0,
        },
        "modes": [env.get("DRIFT_SCOPE", "test")],
    }
    if not mapping["clients"]["ids"]:
        mapping["clients"] = {"fraction": float(env.get("DRIFT_CLIENT_FRACTION", "0.5"))}
    if not mapping["enabled"]:
        mapping["source"]["feature_transforms"] = {}
    return load_drift_config(mapping)
