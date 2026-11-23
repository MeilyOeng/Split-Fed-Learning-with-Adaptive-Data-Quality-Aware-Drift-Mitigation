"""Strict public contracts for standalone drift experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class _StrictEnum(str, Enum):
    @classmethod
    def parse(cls, value: object):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"Unsupported {cls.__name__} {value!r}; expected one of: {allowed}") from exc


class DriftType(_StrictEnum):
    NONE = "none"
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"


class DriftSource(_StrictEnum):
    STRICT_COVARIATE = "strict_covariate"
    MEASUREMENT = "measurement"
    TARGET_MAPPING = "target_mapping"
    PROCESS = "process"
    MIXED = "mixed"
    @classmethod
    def parse(cls, value: object):
        legacy = {
            "input": "measurement",
            "target": "target_mapping",
            "label": "target_mapping",
            "both": "mixed",
        }
        return super().parse(legacy.get(str(value).strip().lower(), value))

    # Deprecated symbolic aliases keep Python callers working while serialized output
    # and enum iteration always use the corrected terminology.
    INPUT = "measurement"
    LABEL = "target_mapping"
    BOTH = "mixed"
    TARGET = "target_mapping"


class TransitionFunction(_StrictEnum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"

class TransformType(_StrictEnum):
    SCALE = "scale"
    OFFSET = "offset"
    NOISE = "noise"
    NONLINEAR = "nonlinear"



class DetectorName(_StrictEnum):
    PAGE_HINKLEY = "page_hinkley"
    ADWIN = "adwin"


def _finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class DriftConfig:
    run_id: str
    configuration_id: str
    cluster_id: str
    client_id: str
    mode: str = "inference"
    scope: str = "test"
    drift_type: DriftType = DriftType.NONE
    drift_source: DriftSource = DriftSource.MEASUREMENT
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    magnitude: float = 0.0
    offset_delta: float = 0.0
    noise_sigma: float = 0.0
    label_delta: float = 0.0
    period: Optional[int] = None
    transition: TransitionFunction = TransitionFunction.LINEAR
    random_seed: int = 42
    sigmoid_k: float = 10.0
    exp_alpha: float = 3.0
    requested_start_ratio: Optional[float] = None
    requested_end_ratio: Optional[float] = None
    segment_start_index: Optional[int] = None
    segment_end_index_exclusive: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "drift_type", DriftType.parse(self.drift_type))
        object.__setattr__(self, "drift_source", DriftSource.parse(self.drift_source))
        object.__setattr__(self, "transition", TransitionFunction.parse(self.transition))
        for name in ("run_id", "configuration_id", "cluster_id", "client_id", "mode", "scope"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("magnitude", "offset_delta", "noise_sigma", "label_delta", "sigmoid_k", "exp_alpha"):
            _finite(name, getattr(self, name))
        if self.magnitude < 0:
            raise ValueError("magnitude must be >= 0")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be >= 0")
        if self.sigmoid_k <= 0:
            raise ValueError("sigmoid_k must be > 0")
        if self.scope != "test":
            raise ValueError("standalone integration currently supports only scope='test'")
        if (self.segment_start_index is None) != (self.segment_end_index_exclusive is None):
            raise ValueError("segment boundaries must be provided together")
        if self.segment_start_index is not None:
            if self.segment_start_index < 0 or self.segment_end_index_exclusive <= self.segment_start_index:
                raise ValueError("segment must satisfy 0 <= start < end_exclusive")
        for name in ("requested_start_ratio", "requested_end_ratio"):
            ratio = getattr(self, name)
            if ratio is not None and not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.requested_start_ratio is not None and self.start_index is None:
            raise ValueError("requested_start_ratio requires a resolved start_index")
        if self.requested_end_ratio is not None and self.end_index is None:
            raise ValueError("requested_end_ratio requires a resolved end_index")
        if self.drift_type is DriftType.NONE:
            if self.start_index is not None and self.start_index < 0:
                raise ValueError("start_index must be >= 0")
            return
        if self.start_index is None or self.start_index < 0:
            raise ValueError(f"{self.drift_type.value} drift requires start_index >= 0")
        if self.drift_type in (DriftType.GRADUAL, DriftType.INCREMENTAL):
            if self.end_index is None or self.end_index <= self.start_index:
                raise ValueError(f"{self.drift_type.value} drift requires end_index > start_index")
        elif self.end_index is not None and self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index when provided")
        if self.drift_type is DriftType.RECURRING:
            if self.period is None or self.period <= 0:
                raise ValueError("recurring drift requires period > 0")
        elif self.period is not None and self.period <= 0:
            raise ValueError("period must be > 0 when provided")

    @property
    def has_input_effect(self) -> bool:
        return any((self.magnitude != 0.0, self.offset_delta != 0.0, self.noise_sigma != 0.0))

    @property
    def has_target_effect(self) -> bool:
        return self.label_delta != 0.0

@dataclass(frozen=True)
class FeatureTransform:
    feature_index: int
    transform_type: TransformType
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform_type", TransformType.parse(self.transform_type))
        object.__setattr__(self, "parameters", dict(self.parameters))
        if self.feature_index < 0:
            raise ValueError("feature_index must be >= 0")


@dataclass(frozen=True)
class TemporalConfig:
    drift_type: DriftType
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    transition: TransitionFunction = TransitionFunction.LINEAR
    sigmoid_k: float = 10.0
    exp_alpha: float = 3.0
    cycle_length: Optional[int] = None
    old_duration: Optional[int] = None
    new_duration: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "drift_type", DriftType.parse(self.drift_type))
        object.__setattr__(self, "transition", TransitionFunction.parse(self.transition))


@dataclass(frozen=True)
class DriftComponentConfig:
    component_id: str
    source: DriftSource
    feature_transforms: tuple[FeatureTransform, ...] = field(default_factory=tuple)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", DriftSource.parse(self.source))
        object.__setattr__(self, "feature_transforms", tuple(self.feature_transforms))
        object.__setattr__(self, "parameters", dict(self.parameters))
        if not self.component_id.strip():
            raise ValueError("component_id must be non-empty")


@dataclass(frozen=True)
class DriftExperimentConfig:
    schema_version: int
    experiment_id: str
    enabled: bool
    experiment_seed: int
    drift_source: DriftSource
    temporal: TemporalConfig
    components: tuple[DriftComponentConfig, ...]
    client_ids: tuple[str, ...] = field(default_factory=tuple)
    client_fraction: Optional[float] = None
    modes: tuple[str, ...] = ("test",)
    target_feature_index: int = 0
    window_size: int = 24
    forecast_horizon: int = 1
    config_hash: str = ""
    canonical_config: str = ""


@dataclass(frozen=True)
class SchedulePoint:
    absolute_index: int
    timestamp: Optional[str]
    configured_regime: str
    old_concept_id: str
    new_concept_id: str
    active_concept_id: str
    drift_weight: float
    regime_active: bool
    change_point_event: bool
    new_regime_selected: bool


@dataclass(frozen=True)
class ClientSelectionRecord:
    experiment_id: str
    cluster_id: str
    client_id: str
    affected: bool
    selection_method: str
    experiment_seed: int
    client_seed: int
    config_hash: str
    resolved_schedule_hash: str


@dataclass(frozen=True)
class DriftManifestRow:
    experiment_id: str
    configuration_id: str
    cluster_id: str
    client_id: str
    absolute_index: int
    timestamp: Optional[str]
    component_id: str
    new_regime_selected: bool
    component_type: str
    drift_type: str
    drift_source: str
    configured_regime: str
    old_concept_id: str
    new_concept_id: str
    active_concept_id: str
    drift_weight: float
    regime_active: bool
    change_point_event: bool
    transformation_applied: bool
    affected_features: str
    transformation_parameters: str
    experiment_seed: int
    client_seed: int
    config_hash: str
    resolved_schedule_hash: str
    autoregressive_consistency_expected: bool
    exception_reason: Optional[str]
    base_values: str
    runtime_values: str


@dataclass(frozen=True)
class TimelineInjectionResult:
    clean_timeline: np.ndarray
    latent_timeline: np.ndarray
    observed_timeline: np.ndarray
    target_timeline: np.ndarray
    manifest: tuple[DriftManifestRow, ...]
    autoregressive_consistency_expected: bool
    exception_reason: Optional[str]
    innovation_source: Optional[str] = None
    innovations: Optional[np.ndarray] = None
    equation_units: str = "raw_source_units"

@dataclass(frozen=True)
class InjectionRecord:
    run_id: str
    configuration_id: str
    cluster_id: str
    client_id: str
    mode: str
    observation_index: int
    timestamp: Optional[str]
    base_value: float
    runtime_value: float
    drift_type: str
    drift_source: str
    configured_start_index: Optional[int]
    configured_end_index: Optional[int]
    drift_weight: float
    regime_active: bool
    transform_applied: bool
    affected_sample_count: int
    magnitude: float
    offset_delta: float
    noise_sigma: float
    label_delta: float
    period: Optional[int]
    transition_function: str
    random_seed: int
    active_interval_id: Optional[int] = None
    scope: str = "test"
    requested_start_ratio: Optional[float] = None
    requested_end_ratio: Optional[float] = None
    segment_start_index: Optional[int] = None
    segment_end_index_exclusive: Optional[int] = None


@dataclass(frozen=True)
class InjectionResult:
    base_values: np.ndarray
    runtime_values: np.ndarray
    records: tuple[InjectionRecord, ...]


@dataclass(frozen=True)
class DetectorKey:
    run_id: str
    cluster_id: str
    client_id: str
    mode: str
    stream_name: str
    detector_name: DetectorName

    def __post_init__(self) -> None:
        object.__setattr__(self, "detector_name", DetectorName.parse(self.detector_name))
        for name in ("run_id", "cluster_id", "client_id", "mode", "stream_name"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True)
class DetectorParameters:
    ph_min_instances: int = 30
    ph_delta: float = 0.005
    ph_threshold: float = 50.0
    ph_alpha: float = 0.9999
    adwin_delta: float = 0.002
    adwin_clock: int = 32
    adwin_max_buckets: int = 5
    adwin_min_window_length: int = 5
    adwin_grace_period: int = 10

    def __post_init__(self) -> None:
        if self.ph_min_instances <= 0 or self.ph_threshold <= 0:
            raise ValueError("Page-Hinkley min_instances and threshold must be > 0")
        if self.ph_delta < 0 or not 0 < self.ph_alpha <= 1:
            raise ValueError("Page-Hinkley delta must be >= 0 and alpha must be in (0, 1]")
        if not 0 < self.adwin_delta < 1:
            raise ValueError("ADWIN delta must be in (0, 1)")
        if min(self.adwin_clock, self.adwin_max_buckets, self.adwin_min_window_length, self.adwin_grace_period) <= 0:
            raise ValueError("ADWIN integer parameters must be > 0")


@dataclass(frozen=True)
class DetectorEvent:
    run_id: str
    cluster_id: str
    client_id: str
    mode: str
    stream_name: str
    detector_name: str
    event_index: int
    observation_index: int
    timestamp: Optional[str]
    detector_value: float
    alarm: bool
    regime_active: Optional[bool] = None
    transform_applied: Optional[bool] = None
    detector_min_instances: Optional[int] = None


@dataclass(frozen=True)
class OnsetEvaluation:
    label: str
    onset_index: int
    pre_drift_alarms: tuple[int, ...]
    post_drift_alarms: tuple[int, ...]
    first_valid_detection: Optional[int]
    successful_detection: bool
    detection_delay: Optional[int]
    warmup_end_index: Optional[int] = None
    eligible_onset_index: Optional[int] = None
    minimum_achievable_delay: Optional[int] = None


@dataclass(frozen=True)
class IntervalEvaluation:
    interval_id: int
    start_index: int
    end_index_exclusive: int
    alarms: tuple[int, ...]
    first_valid_detection: Optional[int]
    successful_detection: bool
    detection_delay: Optional[int]


@dataclass(frozen=True)
class DriftEvaluationResult:
    detector_name: str
    stream_name: str
    observations_seen: int
    all_alarms: tuple[int, ...]
    first_alarm_anywhere: Optional[int]
    clean_client_false_positive: bool
    false_alarm_present: bool
    false_alarm_count: int
    alarms_per_1000_observations: float
    primary: Optional[OnsetEvaluation]
    onset_evaluations: tuple[OnsetEvaluation, ...] = field(default_factory=tuple)
    interval_evaluations: tuple[IntervalEvaluation, ...] = field(default_factory=tuple)
    inactive_interval_alarms: tuple[int, ...] = field(default_factory=tuple)
    warmup_observations: Optional[int] = None
    warmup_end_index: Optional[int] = None
    eligible_for_detection: Optional[bool] = None


@dataclass(frozen=True)
class ForecastSegment:
    window_start_index: int
    window_end_index: int
    segment: str


@dataclass(frozen=True)
class ForecastSegmentMetrics:
    segment: str
    count: int
    mae: Optional[float]
    rmse: Optional[float]


def validate_series(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"series must be one-dimensional, got shape={array.shape}")
    if array.size == 0:
        raise ValueError("series must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError("series contains NaN or infinite values")
    return array
