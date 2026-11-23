"""Scientific contracts for the first oracle drift-mitigation suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml


SUITE_ID = "sudden_process_ar_int0p40_phi0p10_innov1p0_seed42_pre30_adapt40_holdout30"


class Segment(str, Enum):
    PRE_DRIFT = "PRE_DRIFT"
    ADAPTATION = "ADAPTATION"
    FINAL_HOLDOUT = "FINAL_HOLDOUT"


class Method(str, Enum):
    R0 = "R0_clean_frozen"
    R1 = "R1_clean_continued_splitfed"
    M0 = "M0_drift_frozen"
    M1 = "M1_continued_splitfed"
    M2 = "M2_oracle_client_only"
    M3 = "M3_oracle_client_ewc"


@dataclass(frozen=True)
class Seeds:
    """Full seed-contract surface. Each field documents its runtime consumer;
    a field marked reserved has no consumer yet and is carried only for
    provenance until the stage that wires it lands.

    experiment_seed        -- provenance/identity tag recorded in
                               suite_resolved_config.yaml; no RNG consumer.
    client_selection_seed  -- drift-cohort selection, core.stable_select_per_cluster.
    data_order_seed        -- per-round client ordering, run_split.order_round_clients
                               via the RANDOM_SEED_SCHEDULE environment variable.
    drift_seed             -- process-timeline materialization,
                               drift.process.generate_process_timeline's client_seed.
                               Under innovation_source="observed_process_residual"
                               (the only source used by the approved suite), the
                               timeline generator never draws from this seed, so
                               drift_seed is currently a no-op for that source; it
                               only affects output under "sampled_gaussian".
    h_training_seed         -- checkpoint-H training run; RANDOM_SEED inside
                               prepared.training.build_h_environment.
    adaptation_seed          -- R0/M0/M1/M2 phase runs; RANDOM_SEED inside
                               prepared.runner.build_method_environment.
    fault_selection_seed    -- reserved for Stage E fault-cohort selection.
    fault_generation_seed   -- reserved for Stage E fault-event generation.
    """

    experiment_seed: int = 42
    client_selection_seed: int = 42
    data_order_seed: int = 42
    drift_seed: int = 42
    h_training_seed: int = 42
    adaptation_seed: int = 42
    fault_selection_seed: int = 42
    fault_generation_seed: int = 42


@dataclass(frozen=True)
class SegmentSizes:
    pre_drift: int = 524
    adaptation: int = 700
    final_holdout: int = 524

    @property
    def total(self) -> int:
        return self.pre_drift + self.adaptation + self.final_holdout

    def validate(self) -> None:
        """Reject a degenerate segmentation rather than pinning to the one
        golden 524/700/524 split, so a differently sized scenario remains
        valid so long as every phase actually has windows in it.
        """
        if self.pre_drift <= 0 or self.adaptation <= 0 or self.final_holdout <= 0:
            raise ValueError(
                f"pre_drift, adaptation, and final_holdout must each be positive: "
                f"{(self.pre_drift, self.adaptation, self.final_holdout)!r}"
            )


@dataclass(frozen=True)
class ProcessSpec:
    drift_type: str = "sudden"
    drift_source: str = "process"
    old_intercept: float = 1.0
    old_ar_coefficients: tuple[float, ...] = (0.50,)
    intercept_delta: float = 0.40
    ar_coefficient_delta: tuple[float, ...] = (0.10,)
    innovation_sigma: float = 1.0
    innovation_scale: float = 1.0
    innovation_source: str = "observed_process_residual"
    # Only meaningful when drift_type == "gradual": length, in windows, of the
    # probabilistic old/new regime-switching transition. None defers to the
    # caller's default (preparation.py uses segments.adaptation, so the new
    # regime is deterministically fully selected exactly when final_holdout
    # begins -- matching sudden's own final_holdout condition for the same
    # magnitude).
    gradual_transition_windows: Optional[int] = None
    # Only meaningful when drift_type == "seasonal": period P (in windows) of
    # the repeating old/new AR-parameter oscillation, and its phase offset.
    seasonal_period: Optional[float] = None
    seasonal_phase: float = 0.0

    @property
    def new_intercept(self) -> float:
        return self.old_intercept + self.intercept_delta

    @property
    def new_ar_coefficients(self) -> tuple[float, ...]:
        return tuple(old + delta for old, delta in zip(self.old_ar_coefficients, self.ar_coefficient_delta))

    def validate(self) -> None:
        """Reject any AR(1) process-drift scenario that is not internally
        well-posed, rather than pinning to one golden set of parameters.
        This intentionally accepts scenarios other than the approved
        production suite (e.g. a different intercept/coefficient delta) so
        long as they describe a genuine, stationary, non-trivial drift.
        """
        if self.drift_type not in ("sudden", "gradual", "seasonal") or self.drift_source != "process":
            raise ValueError("this milestone supports only sudden, gradual, or seasonal process drift")
        if len(self.old_ar_coefficients) != 1 or len(self.ar_coefficient_delta) != 1:
            raise ValueError("this experiment requires one AR(1) coefficient intervention")
        old_ar, new_ar = self.old_ar_coefficients[0], self.new_ar_coefficients[0]
        if not (abs(old_ar) < 1.0 and abs(new_ar) < 1.0):
            raise ValueError(f"AR(1) coefficients must be stationary (|phi| < 1): old={old_ar!r} new={new_ar!r}")
        if self.intercept_delta == 0.0 and self.ar_coefficient_delta[0] == 0.0:
            raise ValueError("process drift requires a non-zero intercept or AR-coefficient change")
        if self.innovation_sigma <= 0.0 or self.innovation_scale <= 0.0:
            raise ValueError("innovation_sigma and innovation_scale must be strictly positive")
        if self.innovation_source not in {"observed_process_residual", "sampled_gaussian"}:
            raise ValueError("innovation_source must be observed_process_residual or sampled_gaussian")
        if self.gradual_transition_windows is not None:
            if self.drift_type != "gradual":
                raise ValueError("gradual_transition_windows is only meaningful for drift_type='gradual'")
            if int(self.gradual_transition_windows) <= 0:
                raise ValueError("gradual_transition_windows must be a positive integer")
        if self.seasonal_period is not None and self.drift_type != "seasonal":
            raise ValueError("seasonal_period is only meaningful for drift_type='seasonal'")
        if self.drift_type == "seasonal":
            if self.seasonal_period is None or float(self.seasonal_period) <= 0.0:
                raise ValueError("seasonal drift requires a positive seasonal_period")
            if not math.isfinite(float(self.seasonal_phase)):
                raise ValueError("seasonal_phase must be finite")

    def equation_manifest(self) -> dict[str, Any]:
        if self.drift_type == "gradual":
            semantics = (
                "conditional process/concept drift; recursive realized-history AR(1); gradual = "
                "probabilistic regime switching, B_t ~ Bernoulli(w_t) with w_t ramping linearly "
                "from 0 to 1 across the transition window -- the AR parameters themselves are "
                "never interpolated, only which discrete regime (old or new) generates z_t"
            )
            transition_equation = "w_t = clip((t - t_start) / (t_end - t_start), 0, 1); B_t ~ Bernoulli(w_t)"
        elif self.drift_type == "seasonal":
            semantics = (
                "conditional process/concept drift; recursive realized-history AR(1); seasonal = "
                "AR parameters interpolate continuously between old and new following a repeating "
                "periodic weight -- unlike gradual/incremental this never settles permanently at "
                "the new regime, it oscillates back toward old every period"
            )
            transition_equation = (
                "w_t = (1 + sin(2*pi*(t - t0)/P + phase)) / 2; "
                "a_t = (1-w_t)*old_intercept + w_t*new_intercept; "
                "phi_t = (1-w_t)*old_coefficient + w_t*new_coefficient"
            )
        else:
            semantics = "conditional process/concept drift; recursive realized-history AR(1); sudden = instant regime switch at onset"
            transition_equation = None
        manifest = {
            "semantics": semantics,
            "pre_drift_equation": "z_t = 1.00 + 0.50*z_(t-1) + epsilon_t",
            "post_drift_equation": "z_t = 1.40 + 0.60*z_(t-1) + epsilon_t",
            "residual_equation": "epsilon_t = x_t - (1.00 + 0.50*x_(t-1))",
            "pre_drift_value_contract": "z_t = x_t exactly",
            "equation_units": "raw_source_units_before_fixed_clean_training_scaler",
            "old_and_new_zero_innovation_fixed_points": {"old": 2.0, "new": 3.5},
            "fixed_point_interpretation": (
                "zero-innovation AR(1) fixed points; not empirical client means unless "
                "the observed residual mean is independently verified as zero"
            ),
            "old_intercept": self.old_intercept,
            "new_intercept": self.new_intercept,
            "old_ar_coefficients": list(self.old_ar_coefficients),
            "new_ar_coefficients": list(self.new_ar_coefficients),
            "intercept_delta": self.intercept_delta,
            "ar_coefficient_delta": list(self.ar_coefficient_delta),
            "innovation_sigma": self.innovation_sigma,
            "innovation_scale": self.innovation_scale,
            "innovation_source": self.innovation_source,
            "previous_value_contract": "previous realized materialized timeline value",
            "legacy_magnitude_used": False,
        }
        if transition_equation is not None:
            manifest["transition_equation"] = transition_equation
            if self.drift_type == "gradual":
                manifest["gradual_transition_windows"] = self.gradual_transition_windows
            elif self.drift_type == "seasonal":
                manifest["seasonal_period"] = self.seasonal_period
                manifest["seasonal_phase"] = self.seasonal_phase
        return manifest

    def drift_config(self, *, client_ids: Sequence[str], onset_index: int, window_size: int, horizon: int, seed: int) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 2,
            "experiment_id": SUITE_ID,
            "enabled": True,
            "experiment_seed": int(seed),
            "clients": {"ids": [str(item) for item in client_ids]},
            "modes": ["test"],
            "temporal": {"type": self.drift_type, "start_index": int(onset_index)},
            "source": {
                "type": self.drift_source,
                "old": {
                    "intercept": self.old_intercept,
                    "ar_coefficients": list(self.old_ar_coefficients),
                    "innovation_sigma": self.innovation_sigma,
                },
                "changes": {
                    "intercept_delta": self.intercept_delta,
                    "ar_coefficient_delta": list(self.ar_coefficient_delta),
                    "innovation_scale": self.innovation_scale,
                },
                "innovation_source": self.innovation_source,
            },
            "windowing": {
                "window_size": int(window_size),
                "forecast_horizon": int(horizon),
                "target_feature_index": 0,
            },
        }


@dataclass(frozen=True)
class UpdatePolicy:
    drift_enabled: bool
    selected_clients_update: bool
    control_clients_update: bool
    server_updates: bool
    fedavg_enabled: bool
    oracle_enabled: bool
    # Role this method plays in the recovery-rate comparison: exactly one
    # method should be "clean_baseline" (E_clean) and one "frozen_baseline"
    # (E_frozen); any number may be "mitigation_candidate" (E_mitigated),
    # including methods not yet imagined when this was written.
    # "diagnostic_control" methods (e.g. R1) are excluded from the
    # candidates list entirely -- they have their own separate comparison
    # (continued_training_control.csv), not a recovery rate.
    method_role: str = "mitigation_candidate"


POLICIES: Mapping[Method, UpdatePolicy] = {
    Method.R0: UpdatePolicy(False, False, False, False, False, False, method_role="clean_baseline"),
    Method.M0: UpdatePolicy(True, False, False, False, False, False, method_role="frozen_baseline"),
    Method.M1: UpdatePolicy(True, True, True, True, True, False, method_role="mitigation_candidate"),
    Method.M2: UpdatePolicy(True, True, False, False, False, True, method_role="mitigation_candidate"),
    # R1: same update mechanics as M1 (clients update, servers update,
    # FedAvg enabled) but bound to clean deployment data instead of
    # drifted_deployment -- a diagnostic control for how much of M1's
    # improvement is just "more training", not drift recovery.
    Method.R1: UpdatePolicy(False, True, True, True, True, False, method_role="diagnostic_control"),
    # M3: identical ownership/update policy to M2 (selected clients only,
    # servers frozen, no FedAvg) -- the only difference is client-side (an
    # added EWC penalty term on top of the same forecasting gradient),
    # which UpdatePolicy has no field for since it doesn't change who is
    # permitted to update.
    Method.M3: UpdatePolicy(True, True, False, False, False, True, method_role="mitigation_candidate"),
}

_METHOD_POLICY_REGISTRY: dict[str, UpdatePolicy] = {method.value: policy for method, policy in POLICIES.items()}


def register_method_policy(name: str, policy: UpdatePolicy) -> None:
    """Register a new mitigation method's policy by name.

    This is the seam a future DQ-aware or otherwise novel adaptation method
    uses to participate in prepared.runner dispatch and evaluation.py's
    recovery-rate comparison without editing the closed R0/M0/M1/M2 enum or
    any of their conditional call sites, which all consult the policy
    instead of the method's identity.
    """
    if str(name) in _METHOD_POLICY_REGISTRY:
        raise ValueError(f"method policy already registered: {name!r}")
    _METHOD_POLICY_REGISTRY[str(name)] = policy


def resolve_method_policy(method: "Method | str") -> UpdatePolicy:
    key = method.value if isinstance(method, Method) else str(method)
    try:
        return _METHOD_POLICY_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"no policy registered for method: {method!r}") from exc


def registered_method_names() -> list[str]:
    """Registration order, not alphabetical: R0/M0/M1/M2 first (their
    declaration order in POLICIES), then any later-registered methods in
    the order they were registered. Consumers that build ordered output
    (e.g. evaluation.py's per-method comparison tables) rely on this to
    keep the built-in methods' row order stable.
    """
    return list(_METHOD_POLICY_REGISTRY)


# Quality-aware SplitFed (M4): registered via the open seam above, not added
# to the closed Method enum / POLICIES dict, but declared here -- not in a
# downstream consumer module -- so it's always available regardless of
# import order. Every consumer (runner, cli, evaluation, mitigation_report,
# standalone scripts) already imports drift.mitigation.core, and this module
# only executes its top-level code once per process, so no double-
# registration guard is needed here. Policy is identical to M1's: normal
# continued SplitFed, no oracle restriction, no EWC -- the only difference
# is optional, configurable DQ-based gradient weighting (see
# QualityWeightingConfig), which is inert unless a suite's
# quality_weighting: block explicitly targets this method.
M4_QUALITY_AWARE_SPLITFED = "M4_quality_aware_splitfed"
register_method_policy(
    M4_QUALITY_AWARE_SPLITFED,
    UpdatePolicy(True, True, True, True, True, False, method_role="mitigation_candidate"),
)


# Scenario-independent abstract names for the dataset roles a method
# environment binds. A future scenario (fault-only, independent drift/fault
# cohorts, etc.) may rebind these without prepared.runner needing any
# "drift"-specific vocabulary. scenario_true and clean_reference alias the
# same concrete roles as scenario_observed/clean_counterfactual today
# because there is no fault operator yet (see drift.mitigation.prepared,
# Stage E): once one exists, "true" and "observed" become distinct roles.
SCENARIO_DATASET_BINDINGS: dict[str, str] = {
    "clean_reference": "clean_deployment",
    "scenario_observed": "drifted_deployment",
    "scenario_true": "drifted_deployment",
    "clean_counterfactual": "clean_final_holdout_counterfactual",
}


@dataclass(frozen=True)
class ModelArchitecture:
    """The split-LSTM architecture shared identically by checkpoint-H
    training and every method's adaptation/evaluation phase.

    Defaults match the architecture every checkpoint H in this repository
    was actually trained with, so a config that omits ``model:`` entirely
    resolves to exactly the previous hardcoded behavior.
    """

    input_size: int = 1
    hidden_size: int = 128
    num_layers: int = 3
    split_layer: int = 2
    output_size: int = 1
    dropout: float = 0.4
    bidirectional: bool = False

    def validate(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0 or self.num_layers <= 0 or self.output_size <= 0:
            raise ValueError(
                f"model input_size, hidden_size, num_layers, and output_size must all be positive: "
                f"{(self.input_size, self.hidden_size, self.num_layers, self.output_size)!r}"
            )
        if not (0 < self.split_layer < self.num_layers):
            raise ValueError(f"model split_layer must satisfy 0 < split_layer < num_layers: split_layer={self.split_layer!r} num_layers={self.num_layers!r}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"model dropout must satisfy 0 <= dropout < 1: {self.dropout!r}")


@dataclass(frozen=True)
class RuntimeConfig:
    """Method-execution runtime parameters for R0/M0/M1/M2, read from the
    scenario config's ``runtime:`` block.

    Like MethodRuntimeConfig, none of these fields affect checkpoint H's
    own trained weights, so changing them never requires retraining H or
    touching BaselineHContract. Adaptation-pass counts live separately in
    :class:`AdaptationConfig` (the ``adaptation:`` block), since M1 and M2
    each need their own independent count.
    """

    batch_size: int = 128
    shuffle: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"runtime batch_size must be positive: {self.batch_size!r}")


@dataclass(frozen=True)
class AdaptationConfig:
    """Independent adaptation-pass counts for M1/R1, M2, and M3, read from
    the scenario config's ``adaptation:`` block.

    ``adaptation_epochs`` is the number of chronological passes
    M1_continued_splitfed (and R1_clean_continued_splitfed, which shares
    it rather than getting its own field) makes over the ADAPTATION
    segment via the standard FedAvg training path. ``client_only_adaptation_epochs``
    is the number of chronological passes M2_oracle_client_only's selected
    clients make over the same segment via the frozen-server, client-only
    adaptation path. ``ewc_adaptation_epochs`` is the same for
    M3_oracle_client_ewc; when a config omits it, ``load_suite_yaml``
    resolves it to ``client_only_adaptation_epochs`` (or 1 if that is also
    absent) before this dataclass is constructed, so the field below only
    ever sees an explicit value once resolution has happened -- its own
    default of 1 only matters for direct construction outside that path
    (e.g. tests). R0/M0 and the PRE_DRIFT/FINAL_HOLDOUT/counterfactual
    phases never consult any of these fields.

    None of these fields affect checkpoint H's own trained weights or
    BaselineHContract compatibility -- changing any of them only requires
    rerunning the corresponding method.
    """

    adaptation_epochs: int = 1
    client_only_adaptation_epochs: int = 1
    ewc_adaptation_epochs: int = 1

    def validate(self) -> None:
        for name in ("adaptation_epochs", "client_only_adaptation_epochs", "ewc_adaptation_epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"adaptation.{name} must be an integer >= 1: {value!r}")


@dataclass(frozen=True)
class EWCConfig:
    """M3's elastic-weight-consolidation runtime parameters, read from the
    scenario config's ``ewc:`` block.

    ``lambda_`` weights the EWC penalty against the forecasting loss (0
    disables the penalty term entirely -- see prepared.runner/client.py for
    why lambda=0 is wired as a true M2 no-op, not merely a zero-valued
    term). ``fisher_source``/``fisher_type``/``fisher_max_batches`` describe
    how the diagonal importance estimate is derived (see
    prepared.ewc.build_or_load_ewc_reference); they affect only the
    *derived EWC reference artifacts* under a suite root, never checkpoint
    H itself and never BaselineHContract.
    """

    enabled: bool = True
    lambda_: float = 1.0
    fisher_source: str = "clean_validation"
    fisher_type: str = "empirical_diagonal"
    fisher_max_batches: int | None = None

    def validate(self) -> None:
        if self.lambda_ < 0:
            raise ValueError(f"ewc.lambda must be >= 0: {self.lambda_!r}")
        if self.fisher_max_batches is not None and (
            not isinstance(self.fisher_max_batches, int) or isinstance(self.fisher_max_batches, bool) or self.fisher_max_batches <= 0
        ):
            raise ValueError(f"ewc.fisher_max_batches must be a positive integer or null: {self.fisher_max_batches!r}")
        if self.fisher_source != "clean_validation":
            raise ValueError(
                f"ewc.fisher_source must be 'clean_validation' -- the only role checkpoint H was "
                f"legitimately allowed to access that is also not gradient-trained: {self.fisher_source!r}"
            )


def ewc_config_from_mapping(raw: Mapping[str, Any] | None) -> EWCConfig:
    """Construct EWCConfig from a raw ``ewc:`` mapping, remapping the YAML
    key ``lambda`` (a Python keyword, so the dataclass field is ``lambda_``).
    The one place this remap needs to happen, reused by load_suite_yaml and
    every other caller that resolves an ``ewc:`` block from config.
    """
    fields = dict(raw or {})
    if "lambda" in fields:
        fields["lambda_"] = fields.pop("lambda")
    return EWCConfig(**fields)


@dataclass(frozen=True)
class FaultConfig:
    """Fault-injection settings, read from the scenario config's ``faults:``
    block.

    ``config_path`` points at the existing, independently-owned
    ``data quality/configs/fault_injection_controlled_window.yaml`` (or an
    equivalent controlled-window fault config) -- that file is read
    read-only and never modified or duplicated here; this block only
    decides *whether* to invoke it and how overlapping cohorts are treated.
    There is deliberately no ``selection_seed`` field: the fault cohort's
    seed is ``seeds.fault_selection_seed`` (already a first-class field on
    :class:`Seeds`), so there is exactly one place a fault-selection seed
    can be configured -- never two values that could silently disagree.
    """

    enabled: bool = False
    config_path: str | None = None
    overlap_policy: str = "independent"

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.config_path:
            raise ValueError("faults.config_path is required when faults.enabled is true")
        if self.overlap_policy != "independent":
            raise ValueError(
                f"faults.overlap_policy must be 'independent' -- drift and fault cohorts are always "
                f"selected independently in this implementation: {self.overlap_policy!r}"
            )


def fault_config_from_mapping(raw: Mapping[str, Any] | None) -> FaultConfig:
    return FaultConfig(**dict(raw or {}))


_QUALITY_WEIGHTING_MODES = ("window", "aggregation", "both")


@dataclass(frozen=True)
class QualityWeightingConfig:
    """Data-quality-based gradient weighting for M4_quality_aware_splitfed,
    read from the scenario config's ``quality_weighting:`` block.

    ``config_path`` points at the existing, independently-owned
    ``data quality/configs/data_quality_scoring.yaml`` (or an equivalent
    WideDemandQualityScorer config) -- read read-only, never modified or
    duplicated here. Requires ``faults.enabled=true`` because the per-window
    score is computed from X_observed, which only exists when the
    fault-injection step ran (enforced in load_suite_yaml, not here).

    ``mode`` controls which of the two independent, pre-existing server
    flags get activated -- "window" (USE_QUALITY_WEIGHTS: genuine per-window
    gradient weighting within a client's own batch), "aggregation"
    (USE_QUALITY_AGGREGATION_WEIGHTS: a single client_soft_quality scalar
    applied to the whole contribution, which also rescales the
    smashed-activation gradient returned to the client -- see
    prepared.runner's build_method_environment), or "both". There is
    deliberately no default: a config that sets enabled=true without an
    explicit mode is a configuration error, not an implicit "window" -- see
    validate().
    """

    enabled: bool = False
    config_path: str | None = None
    mode: str | None = None

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.config_path:
            raise ValueError("quality_weighting.config_path is required when quality_weighting.enabled is true")
        if self.mode is None:
            raise ValueError(
                "quality_weighting.mode is required when quality_weighting.enabled is true -- "
                f"must be one of {_QUALITY_WEIGHTING_MODES!r}, never silently defaulted"
            )
        if self.mode not in _QUALITY_WEIGHTING_MODES:
            raise ValueError(f"quality_weighting.mode must be one of {_QUALITY_WEIGHTING_MODES!r}: {self.mode!r}")


def quality_weighting_config_from_mapping(raw: Mapping[str, Any] | None) -> QualityWeightingConfig:
    return QualityWeightingConfig(**dict(raw or {}))


@dataclass(frozen=True)
class BaselineHContract:
    """Everything that determines whether an existing checkpoint H's trained
    weights remain valid for a given scenario, independent of that
    scenario's drift process, deployment segmentation, fault policy, or
    adaptation method. Two scenarios that resolve to an identical
    BaselineHContract may safely share one H rather than retraining it.

    Model architecture fields are included and gate reuse: changing
    hidden size, layer count, split layer, dropout, or input/output size
    changes what H's own weights mean and shape-checks against, so it must
    require a new H. All architecture fields default to the values every
    checkpoint H in this repository was actually trained with, so an
    existing H's derived contract (see
    ``prepared.suite_identity._resolve_h_baseline_contract``) matches a
    scenario config that leaves ``model:`` unset, exactly as before this
    field set existed.
    """

    window_size: int
    forecast_horizon: int
    h_training_seed: int
    normalization_method: str
    scaler_manifest_hash: str
    coordinate_manifest_hash: str
    clean_train_hash: str
    clean_validation_hash: str
    model_input_size: int = 1
    model_hidden_size: int = 128
    model_num_layers: int = 3
    model_split_layer: int = 2
    model_output_size: int = 1
    model_dropout: float = 0.4
    model_bidirectional: bool = False

    def content_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class MethodRuntimeConfig:
    """Per-run fields that shape only the R0/M0/M1/M2 adaptation and
    evaluation phases. None of these affect checkpoint H's own trained
    weights, so changing them never requires retraining H.
    """

    adaptation_seed: int
    data_order_seed: int
    pre_drift_windows: int
    adaptation_windows: int
    final_holdout_windows: int
    expected_drift_selected_count: int


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_tensor_hash(digest: "hashlib._Hash", name: str, tensor: torch.Tensor) -> None:
    material = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(tuple(material.shape)).encode("ascii") + b"\0")
    digest.update(str(material.dtype).encode("ascii") + b"\0")
    digest.update(material.numpy().tobytes(order="C"))


def tensor_mapping_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        _update_tensor_hash(digest, name, state[name])
    return digest.hexdigest()


def model_state_hashes(model: torch.nn.Module) -> dict[str, str]:
    return {
        "parameters": tensor_mapping_hash(dict(model.named_parameters())),
        "buffers": tensor_mapping_hash(dict(model.named_buffers())),
        "tensor_state": tensor_mapping_hash(model.state_dict()),
    }


def _hash_nested(digest: "hashlib._Hash", prefix: str, value: Any) -> None:
    if torch.is_tensor(value):
        _update_tensor_hash(digest, prefix, value)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _hash_nested(digest, f"{prefix}/{key}", value[key])
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _hash_nested(digest, f"{prefix}/{index}", item)
    else:
        digest.update(prefix.encode("utf-8") + b"\0")
        digest.update(canonical_json(value).encode("utf-8") + b"\0")


def optimizer_state_hash(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _hash_nested(digest, "optimizer", optimizer.state_dict())
    return digest.hexdigest()


def scheduler_state_hash(scheduler: Any | None) -> str:
    return canonical_hash(None if scheduler is None else scheduler.state_dict())


def counter_state_hash(counters: Mapping[str, Any]) -> str:
    return canonical_hash(dict(counters))


def prediction_key_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = [
        (
            str(row["client_id"]),
            int(row["deployment_window_number"]),
            int(row["window_start_index"]),
            int(row["window_end_index"]),
            int(row["target_index"]),
            str(row["segment"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate prediction keys detected")
    return canonical_hash(sorted(keys))


def assert_exact_prediction_keys(rows: Sequence[Mapping[str, Any]], expected: Iterable[tuple[Any, ...]]) -> None:
    actual = {
        (
            str(row["client_id"]),
            int(row["deployment_window_number"]),
            int(row["target_index"]),
            str(row["segment"]),
        )
        for row in rows
    }
    required = {(str(c), int(n), int(t), str(s)) for c, n, t, s in expected}
    if len(actual) != len(rows):
        raise ValueError("duplicate prediction keys detected")
    missing, extra = sorted(required - actual), sorted(actual - required)
    if missing or extra:
        raise ValueError(f"prediction key mismatch: missing={missing[:5]} extra={extra[:5]}")


def stable_select_per_cluster(cluster_clients: Mapping[str, Sequence[str]], *, per_cluster: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for cluster_id in sorted(cluster_clients, key=str):
        clients = sorted(str(item) for item in cluster_clients[cluster_id])
        if per_cluster < 0 or per_cluster > len(clients):
            raise ValueError(f"cannot select {per_cluster} clients from cluster {cluster_id}")
        ranked = sorted(
            clients,
            key=lambda client_id: (
                hashlib.sha256(f"{int(seed)}\0{cluster_id}\0{client_id}".encode("utf-8")).digest(),
                client_id,
            ),
        )
        chosen = set(ranked[:per_cluster])
        for order, client_id in enumerate(clients, start=1):
            selected.append({
                "cluster_id": str(cluster_id),
                "client_id": client_id,
                "selected_for_drift": client_id in chosen,
                "stable_client_order": order,
                "selection_seed": int(seed),
            })
    return selected


def selected_cohort_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    material = [
        {key: row[key] for key in ("cluster_id", "client_id", "selected_for_drift", "stable_client_order")}
        for row in rows
    ]
    return canonical_hash(material)


def segment_for_window(number: int, sizes: SegmentSizes) -> Segment:
    if number < 1 or number > sizes.total:
        raise ValueError("deployment window number is outside the deployment region")
    if number <= sizes.pre_drift:
        return Segment.PRE_DRIFT
    if number <= sizes.pre_drift + sizes.adaptation:
        return Segment.ADAPTATION
    return Segment.FINAL_HOLDOUT


def build_segment_rows(
    *,
    window_starts: Sequence[int],
    window_size: int,
    horizon: int,
    timestamps: Sequence[Any],
    sizes: SegmentSizes,
    onset_index: int,
    selected: bool,
) -> list[dict[str, Any]]:
    if len(window_starts) != sizes.total:
        raise ValueError(f"expected {sizes.total} deployment windows, got {len(window_starts)}")
    rows: list[dict[str, Any]] = []
    for number, raw_start in enumerate(window_starts, start=1):
        start = int(raw_start)
        end = start + int(window_size) - 1
        target = end + int(horizon)
        segment = segment_for_window(number, sizes)
        def regime(index: int) -> str:
            return "NEW" if selected and index >= onset_index else "OLD"
        any_input = bool(selected and end >= onset_index)
        fully_input = bool(selected and start >= onset_index)
        target_drifted = bool(selected and target >= onset_index)
        rows.append({
            "deployment_window_number": number,
            "window_start_index": start,
            "window_end_index": end,
            "target_index": target,
            "window_start_timestamp": str(timestamps[start]),
            "window_end_timestamp": str(timestamps[end]),
            "target_timestamp": str(timestamps[target]),
            "segment": segment.value,
            "process_regime_at_window_start": regime(start),
            "process_regime_at_window_end": regime(end),
            "process_regime_at_target": regime(target),
            "window_contains_any_drifted_input": any_input,
            "window_input_fully_drifted": fully_input,
            "target_drifted": target_drifted,
            "true_drift_active": target_drifted,
            "oracle_active": bool(target_drifted and segment is Segment.ADAPTATION),
        })
    first_adaptation = rows[sizes.pre_drift]
    if first_adaptation["target_index"] != onset_index:
        raise ValueError("onset must equal the first adaptation target index")
    if any(row["true_drift_active"] for row in rows[: sizes.pre_drift]):
        raise ValueError("drift contamination detected before adaptation")
    return rows


def guard_output_directory(path: str | Path, *, resume: bool = False, overwrite: bool = False, expected_hashes: Mapping[str, str] | None = None) -> Path:
    target = Path(path)
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if not target.exists():
        target.mkdir(parents=True)
        return target
    if overwrite:
        marker = target / ".drift_mitigation_suite"
        if not marker.exists():
            raise ValueError(f"refusing to overwrite unmarked directory: {target}")
        import shutil
        shutil.rmtree(target)
        target.mkdir(parents=True)
        return target
    if resume:
        manifest_path = target / "resume_hashes.json"
        if not manifest_path.exists():
            raise ValueError("resume requires resume_hashes.json")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if dict(existing) != dict(expected_hashes or {}):
            raise ValueError("resume hash validation failed")
        if (target / "acceptance_report.json").exists():
            acceptance = json.loads((target / "acceptance_report.json").read_text(encoding="utf-8"))
            if acceptance.get("complete") is True:
                raise ValueError("completed runs cannot be resumed with additional optimizer steps")
        return target
    raise FileExistsError(f"output directory already exists: {target}")


def assert_materialized_timeline_reuse(*, materialized: bool, runtime_injection_requested: bool) -> None:
    """Reject the only state combination that can apply process drift twice."""
    if materialized and runtime_injection_requested:
        raise ValueError("runtime drift injection is forbidden for an already materialized timeline")


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "torch_cuda_seed_all": seed,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def git_record(repo: str | Path) -> dict[str, Any]:
    root = Path(repo)
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE).stdout
    commit = run("rev-parse", "HEAD").strip()
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD", "--", ":(exclude).env")
    untracked = []
    for line in status.splitlines():
        if line.startswith("?? "):
            relative = line[3:]
            if Path(relative).name == ".env" or ".env" in Path(relative).parts:
                continue
            path = root / relative
            if path.is_file():
                untracked.append({"path": relative, "sha256": file_sha256(path)})
    return {
        "commit": commit,
        "status": status,
        "git_diff_hash": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "relevant_untracked_source_hash": canonical_hash(untracked),
        "untracked_files": untracked,
    }


def environment_record(repo: str | Path) -> dict[str, Any]:
    git = git_record(repo)
    gpu = None
    if torch.cuda.is_available():
        gpu = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        **git,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": gpu,
        "operating_system": platform.platform(),
        "reproducibility_class": "bitwise reproducible" if not torch.cuda.is_available() else "seed-controlled but hardware-dependent",
    }


def load_suite_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("suite configuration must be a mapping")
    def reject_magnitude(value: Any, location: str = "root") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() == "magnitude":
                    raise ValueError(f"obsolete magnitude field is forbidden at {location}")
                reject_magnitude(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                reject_magnitude(item, f"{location}[{index}]")
    reject_magnitude(data)
    suite_id = data.get("suite_id")
    if not isinstance(suite_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_]*", suite_id):
        raise ValueError(
            f"suite_id must be a non-empty lowercase identifier (letters, digits, underscores): {suite_id!r}"
        )
    ProcessSpec(
        drift_type=str(data["process"]["drift_type"]),
        drift_source=str(data["process"]["drift_source"]),
        old_intercept=float(data["process"]["old_intercept"]),
        old_ar_coefficients=tuple(float(item) for item in data["process"]["old_ar_coefficients"]),
        intercept_delta=float(data["process"]["intercept_delta"]),
        ar_coefficient_delta=tuple(float(item) for item in data["process"]["ar_coefficient_delta"]),
        innovation_sigma=float(data["process"]["innovation_sigma"]),
        innovation_scale=float(data["process"]["innovation_scale"]),
        innovation_source=str(data["process"].get("innovation_source", "observed_process_residual")),
        gradual_transition_windows=(
            int(data["process"]["gradual_transition_windows"])
            if data["process"].get("gradual_transition_windows") is not None else None
        ),
        seasonal_period=(
            float(data["process"]["seasonal_period"])
            if data["process"].get("seasonal_period") is not None else None
        ),
        seasonal_phase=float(data["process"].get("seasonal_phase", 0.0)),
    ).validate()
    SegmentSizes(**data["segments"]).validate()
    model = ModelArchitecture(**dict(data.get("model") or {}))
    model.validate()
    runtime = RuntimeConfig(**dict(data.get("runtime") or {}))
    runtime.validate()
    adaptation_fields = dict(data.get("adaptation") or {})
    if "ewc_adaptation_epochs" not in adaptation_fields:
        # Default: same cadence as M2's client-only adaptation, or 1 if the
        # config declares neither -- keeps an old M1/M2-only config's
        # meaning unchanged even though AdaptationConfig now has a third field.
        adaptation_fields["ewc_adaptation_epochs"] = adaptation_fields.get("client_only_adaptation_epochs", 1)
    adaptation = AdaptationConfig(**adaptation_fields)
    adaptation.validate()
    ewc = ewc_config_from_mapping(data.get("ewc"))
    ewc.validate()
    faults = fault_config_from_mapping(data.get("faults"))
    faults.validate()
    quality_weighting = quality_weighting_config_from_mapping(data.get("quality_weighting"))
    quality_weighting.validate()
    if quality_weighting.enabled and not faults.enabled:
        raise ValueError(
            "quality_weighting.enabled requires faults.enabled=true -- the per-window quality "
            "score is computed from X_observed, which only exists when fault injection ran"
        )
    resolved = dict(data)
    resolved["seeds"] = asdict(Seeds(**dict(data.get("seeds") or {})))
    resolved["model"] = asdict(model)
    resolved["runtime"] = asdict(runtime)
    resolved["adaptation"] = asdict(adaptation)
    resolved["ewc"] = asdict(ewc)
    resolved["faults"] = asdict(faults)
    resolved["quality_weighting"] = asdict(quality_weighting)
    return resolved


def write_csv_rows(path: str | Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
