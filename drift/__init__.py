"""Standalone, deterministic drift injection and evaluation package."""

from .configuration import canonical_json, legacy_environment_config, load_drift_config

from .contracts import (
    ClientSelectionRecord,
    DriftComponentConfig,
    DriftExperimentConfig,
    DriftManifestRow,
    DetectorEvent,
    DetectorKey,
    DetectorName,
    DetectorParameters,
    DriftConfig,
    DriftEvaluationResult,
    ForecastSegment,
    ForecastSegmentMetrics,
    DriftSource,
    FeatureTransform,
    DriftType,
    InjectionRecord,
    InjectionResult,
    IntervalEvaluation,
    OnsetEvaluation,
    TransitionFunction,
    SchedulePoint,
    TemporalConfig,
    TimelineInjectionResult,
    TransformType,
)
from .detectors import DetectorRegistry
from .evaluator import classify_forecast_segment, evaluate_detection, evaluate_forecasts_by_segment, evaluate_run_artifacts
from .evaluator import evaluate_detection_from_manifest, load_client_selection_manifest, load_component_manifest
from .injector import derive_client_seed, inject_series

from .generators import (
    FrozenLinearSupervisedGenerator,
    GeneratorRegistry,
    StrictCovariateGeneration,
    TargetMappingRegistry,
    validate_strict_covariate_generation,
)
from .pipeline import (
    TargetMappingBenchmarkResult,
    generate_supervised_strict_covariate_benchmark,
    generate_target_mapping_benchmark,
    inject_timeline,
)
from .process import generate_process_timeline
from .schedule import (
    ResolvedExperimentSchedule,
    build_drift_schedule,
    derive_stable_seed,
    resolve_experiment_schedule,
)
from .windowing import assert_horizon_one_consistency, build_forecasting_windows, fit_clean_training_scalers, normalize_windows

__all__ = [
    "DetectorEvent",
    "DetectorKey",
    "DetectorName",
    "DetectorParameters",
    "DetectorRegistry",
    "DriftConfig",
    "DriftEvaluationResult",
    "ForecastSegment",
    "ForecastSegmentMetrics",
    "DriftSource",
    "DriftType",
    "InjectionRecord",
    "InjectionResult",
    "IntervalEvaluation",
    "OnsetEvaluation",
    "ClientSelectionRecord",
    "DriftComponentConfig",
    "DriftExperimentConfig",
    "DriftManifestRow",
    "FeatureTransform",
    "SchedulePoint",
    "TemporalConfig",
    "TimelineInjectionResult",
    "TransformType",
    "ResolvedExperimentSchedule",
    "StrictCovariateGeneration",
    "TransitionFunction",
    "classify_forecast_segment",
    "derive_client_seed",
    "evaluate_detection",
    "evaluate_detection_from_manifest",
    "load_client_selection_manifest",
    "load_component_manifest",
    "evaluate_forecasts_by_segment",
    "evaluate_run_artifacts",
    "inject_series",
    "FrozenLinearSupervisedGenerator",
    "GeneratorRegistry",
    "TargetMappingBenchmarkResult",
    "TargetMappingRegistry",
    "assert_horizon_one_consistency",
    "build_drift_schedule",
    "build_forecasting_windows",
    "canonical_json",
    "derive_stable_seed",
    "fit_clean_training_scalers",
    "generate_process_timeline",
    "generate_supervised_strict_covariate_benchmark",
    "generate_target_mapping_benchmark",
    "inject_timeline",
    "legacy_environment_config",
    "load_drift_config",
    "normalize_windows",
    "resolve_experiment_schedule",
    "validate_strict_covariate_generation",
]
