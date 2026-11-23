"""Strict-covariate generator contracts and validation guardrails."""

from __future__ import annotations
from collections.abc import Callable

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contracts import SchedulePoint


@dataclass(frozen=True)
class StrictCovariateGeneration:
    benchmark_kind: str
    generator_id_before: str
    generator_id_after: str
    generator_version_before: str
    generator_version_after: str
    residual_distribution_id_before: str
    residual_distribution_id_after: str
    reference_inputs: np.ndarray
    generated_inputs: np.ndarray
    generated_targets: np.ndarray
    residuals: np.ndarray
    chronological_timeline: np.ndarray | None = None


@runtime_checkable
class StrictCovariateGenerator(Protocol):
    generator_id: str
    generator_version: str
    residual_distribution_id: str
    benchmark_kind: str

    def generate(
        self,
        clean_timeline: np.ndarray,
        schedule: Sequence[SchedulePoint],
        *,
        target_feature_index: int,
        window_size: int,
        forecast_horizon: int,
        rng: np.random.Generator,
        parameters: Mapping[str, object],
    ) -> StrictCovariateGeneration:
        ...

    def conditional_mean(self, inputs: np.ndarray) -> np.ndarray:
        ...


class TargetMappingRegistry:
    """Stable-ID registry for explicit supervised old/new target mappings."""

    def __init__(self) -> None:
        self._items: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    def register(self, mapping_id: str, function: Callable[[np.ndarray], np.ndarray]) -> None:
        identifier = str(mapping_id).strip()
        if not identifier or identifier in self._items:
            raise ValueError(f"invalid or duplicate target mapping ID: {mapping_id!r}")
        self._items[identifier] = function

    def require(self, mapping_id: str) -> Callable[[np.ndarray], np.ndarray]:
        try:
            return self._items[str(mapping_id)]
        except KeyError as exc:
            raise ValueError(f"target mapping {mapping_id!r} is not registered") from exc


class GeneratorRegistry:
    def __init__(self) -> None:
        self._items: dict[str, StrictCovariateGenerator] = {}

    def register(self, generator: StrictCovariateGenerator) -> None:
        identifier = str(generator.generator_id).strip()
        if not identifier:
            raise ValueError("generator_id must be non-empty")
        if identifier in self._items:
            raise ValueError(f"generator_id already registered: {identifier}")
        self._items[identifier] = generator

    def require(self, generator_id: str) -> StrictCovariateGenerator:
        try:
            return self._items[str(generator_id)]
        except KeyError as exc:
            raise ValueError(
                f"strict_covariate generator {generator_id!r} is not registered"
            ) from exc


def validate_strict_covariate_generation(
    generation: StrictCovariateGeneration,
    generator: StrictCovariateGenerator,
    *,
    require_chronological: bool,
    target_feature_index: int,
    window_size: int,
    forecast_horizon: int,
    atol: float = 1e-10,
) -> None:
    """Reject generator drift, residual drift, incoherent labels, and false distribution claims."""
    if generation.benchmark_kind not in {"chronological", "supervised"}:
        raise ValueError("strict-covariate generation has an invalid benchmark_kind")
    if require_chronological and generation.benchmark_kind != "chronological":
        raise ValueError(
            "overlapping univariate streaming requires a chronological strict-covariate generator"
        )
    if generation.generator_id_before != generation.generator_id_after:
        raise ValueError("strict-covariate conditional generator changed across regimes")
    if generation.generator_id_before != generator.generator_id:
        raise ValueError("generation metadata does not match the registered frozen generator")
    if generation.generator_version_before != generation.generator_version_after:
        raise ValueError("strict-covariate generator version changed across regimes")
    if generation.generator_version_before != generator.generator_version:
        raise ValueError("generation version does not match the registered generator")
    if generation.residual_distribution_id_before != generation.residual_distribution_id_after:
        raise ValueError("strict-covariate residual distribution changed across regimes")
    if generation.residual_distribution_id_before != generator.residual_distribution_id:
        raise ValueError("generation residual distribution does not match the registered generator")

    reference = np.asarray(generation.reference_inputs, dtype=float)
    generated = np.asarray(generation.generated_inputs, dtype=float)
    targets = np.asarray(generation.generated_targets, dtype=float).reshape(-1)
    residuals = np.asarray(generation.residuals, dtype=float).reshape(-1)
    if reference.shape != generated.shape or generated.shape[0] != targets.size:
        raise ValueError("strict-covariate generated input and target shapes are inconsistent")
    if targets.size != residuals.size:
        raise ValueError("strict-covariate residual count does not match targets")
    if not all(np.isfinite(item).all() for item in (reference, generated, targets, residuals)):
        raise ValueError("strict-covariate output contains non-finite values")
    if np.array_equal(reference, generated):
        raise ValueError("strict-covariate generator did not change the input distribution samples")

    conditional = np.asarray(generator.conditional_mean(generated), dtype=float).reshape(-1)
    if conditional.size != targets.size:
        raise ValueError("conditional generator returned the wrong target shape")
    if not np.allclose(targets, conditional + residuals, rtol=0.0, atol=atol):
        raise ValueError("generated targets do not use the frozen conditional plus unchanged residuals")

    if require_chronological:
        if generation.chronological_timeline is None:
            raise ValueError("chronological strict-covariate generation did not return a timeline")
        from .windowing import build_forecasting_windows, assert_horizon_one_consistency

        windows = build_forecasting_windows(
            generation.chronological_timeline,
            generation.chronological_timeline,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            target_feature_index=target_feature_index,
        )
        if generated.shape != windows.X.shape or not np.allclose(
            generated, windows.X, rtol=0.0, atol=atol
        ):
            raise ValueError(
                "chronological strict-covariate inputs are not the windows of its returned timeline"
            )
        if not np.allclose(targets, windows.y, rtol=0.0, atol=atol):
            raise ValueError(
                "chronological strict-covariate targets are not aligned to its returned timeline"
            )
        if forecast_horizon == 1:
            assert_horizon_one_consistency(
                windows.X,
                windows.y,
                target_feature_index=target_feature_index,
            )


class FrozenLinearSupervisedGenerator:
    """Small deterministic generator used by examples and tests."""

    generator_id = "frozen_linear_supervised_v1"
    generator_version = "1"
    residual_distribution_id = "normal_fixed_sigma_v1"
    benchmark_kind = "supervised"

    def __init__(self, coefficients: Sequence[float], intercept: float = 0.0) -> None:
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.intercept = float(intercept)

    def conditional_mean(self, inputs: np.ndarray) -> np.ndarray:
        array = np.asarray(inputs, dtype=float)
        return self.intercept + array @ self.coefficients

    def generate(
        self,
        clean_timeline: np.ndarray,
        schedule: Sequence[SchedulePoint],
        *,
        target_feature_index: int,
        window_size: int,
        forecast_horizon: int,
        rng: np.random.Generator,
        parameters: Mapping[str, object],
    ) -> StrictCovariateGeneration:
        reference = np.asarray(parameters["reference_inputs"], dtype=float)
        shift = np.asarray(parameters.get("input_shift", np.ones(reference.shape[1])), dtype=float)
        generated = reference + shift
        sigma = float(parameters.get("residual_sigma", 0.1))
        residuals = rng.normal(0.0, sigma, size=reference.shape[0])
        targets = self.conditional_mean(generated) + residuals
        return StrictCovariateGeneration(
            benchmark_kind=self.benchmark_kind,
            generator_id_before=self.generator_id,
            generator_id_after=self.generator_id,
            generator_version_before=self.generator_version,
            generator_version_after=self.generator_version,
            residual_distribution_id_before=self.residual_distribution_id,
            residual_distribution_id_after=self.residual_distribution_id,
            reference_inputs=reference,
            generated_inputs=generated,
            generated_targets=targets,
            residuals=residuals,
        )
