"""Validated deployment slices used by mitigation without changing ordinary splits."""
from __future__ import annotations
from dataclasses import dataclass

DEPLOYMENT_TOTAL_WINDOWS = 1748

@dataclass(frozen=True)
class DeploymentSlice:
    requested_mode: str
    source_mode: str
    window_offset: int
    window_limit: int | None
    materialized: bool
    runtime_drift_injection: bool
    training_allowed: bool

_PREPARED = {
    "deployment_pre_drift": (0, 524, False),
    "deployment_adaptation": (524, 700, True),
    "deployment_final_holdout": (1224, 524, False),
    "deployment_counterfactual_clean_holdout": (1224, 524, False),
}

def resolve_deployment_slice(mode: str, *, window_offset: int | None = None, window_limit: int | None = None,
                             materialized: bool = False, runtime_drift_injection: bool = False,
                             for_training: bool = False) -> DeploymentSlice:
    requested = str(mode).strip().lower()
    if requested not in _PREPARED:
        if requested not in {"train", "val", "test"}: raise ValueError(f"unsupported stream mode: {requested}")
        offset = 0 if window_offset is None else int(window_offset)
        if offset < 0: raise ValueError("window_offset must be non-negative")
        limit = None if window_limit is None else int(window_limit)
        if limit is not None and limit <= 0: raise ValueError("window_limit must be positive")
        return DeploymentSlice(requested, requested, offset, limit, materialized, runtime_drift_injection, requested == "train")
    fixed_offset, fixed_limit, training_allowed = _PREPARED[requested]
    if not materialized: raise ValueError("prepared deployment modes require materialized=true")
    if runtime_drift_injection: raise ValueError("runtime drift injection is forbidden for materialized timelines")
    if window_offset is not None and int(window_offset) != fixed_offset: raise ValueError(f"{requested} requires window_offset={fixed_offset}")
    if window_limit is not None and int(window_limit) != fixed_limit: raise ValueError(f"{requested} requires window_limit={fixed_limit}")
    if fixed_offset < 0 or fixed_offset + fixed_limit > DEPLOYMENT_TOTAL_WINDOWS: raise ValueError("deployment slice is outside the 1,748-window region")
    if for_training and not training_allowed: raise ValueError(f"training is forbidden for stream mode {requested}")
    return DeploymentSlice(requested, "test", fixed_offset, fixed_limit, True, False, training_allowed)