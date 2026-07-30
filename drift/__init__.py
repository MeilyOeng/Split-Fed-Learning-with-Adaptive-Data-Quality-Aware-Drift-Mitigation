"""Standalone, deterministic drift injection and evaluation package."""

from .contracts import (
    DetectorEvent,
    DetectorKey,
    DetectorName,
    DetectorParameters,
    DriftConfig,
    DriftEvaluationResult,
    ForecastSegment,
    ForecastSegmentMetrics,
    DriftSource,
    DriftType,
    InjectionRecord,
    InjectionResult,
    IntervalEvaluation,
    OnsetEvaluation,
    TransitionFunction,
)
from .detectors import DetectorRegistry
from .evaluator import classify_forecast_segment, evaluate_detection, evaluate_forecasts_by_segment, evaluate_run_artifacts
from .injector import derive_client_seed, inject_series

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
    "TransitionFunction",
    "classify_forecast_segment",
    "derive_client_seed",
    "evaluate_detection",
    "evaluate_forecasts_by_segment",
    "evaluate_run_artifacts",
    "inject_series",
]
