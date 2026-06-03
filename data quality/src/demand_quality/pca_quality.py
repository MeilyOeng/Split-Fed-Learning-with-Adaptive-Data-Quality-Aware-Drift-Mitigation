"""PCA-based composite quality score for per-client dimension scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .io import save_csv, save_json


QUALITY_DIMENSIONS = [
    "completeness",
    "accuracy",
    "temporal_consistency",
    "skewness",
]


@dataclass
class DemandPCAQualityModel:
    """One-component PCA model for client quality aggregation."""

    feature_names: list[str]
    means_: np.ndarray
    stds_: np.ndarray
    component_: np.ndarray
    score_min_: float
    score_max_: float
    explained_variance_ratio_: float
    fallback_to_row_mean_: bool = False

    @classmethod
    def fit(cls, scores_df: pd.DataFrame) -> "DemandPCAQualityModel":
        """Fit PCA on dimension scores."""
        x = scores_df.loc[:, QUALITY_DIMENSIONS].astype(float).copy()
        if x.shape[0] == 0:
            raise ValueError("No rows available for PCA quality scoring.")

        means = x.mean(axis=0).to_numpy(dtype=float)
        stds = x.std(axis=0, ddof=0).to_numpy(dtype=float)
        stds_safe = stds.copy()
        stds_safe[stds_safe == 0.0] = 1.0
        z = (x.to_numpy(dtype=float) - means) / stds_safe
        z[:, stds == 0.0] = 0.0

        row_mean = x.mean(axis=1).to_numpy(dtype=float)
        if x.shape[0] < 2 or np.allclose(z, z[0]):
            component = np.ones(len(QUALITY_DIMENSIONS), dtype=float)
            raw_scores = row_mean.copy()
            explained_variance_ratio = 1.0
            fallback_to_row_mean = True
        else:
            pca = PCA(n_components=1)
            raw_scores = pca.fit_transform(z).reshape(-1)
            component = pca.components_[0].astype(float)
            explained_variance_ratio = float(pca.explained_variance_ratio_[0])
            fallback_to_row_mean = False
            corr = np.corrcoef(raw_scores, row_mean)[0, 1]
            if not np.isnan(corr) and corr < 0:
                raw_scores = -raw_scores
                component = -component

        score_min = float(np.min(raw_scores))
        score_max = float(np.max(raw_scores))
        return cls(
            feature_names=list(QUALITY_DIMENSIONS),
            means_=means,
            stds_=stds_safe,
            component_=component,
            score_min_=score_min,
            score_max_=score_max,
            explained_variance_ratio_=explained_variance_ratio,
            fallback_to_row_mean_=fallback_to_row_mean,
        )

    def transform(self, scores_df: pd.DataFrame) -> np.ndarray:
        """Project scores into a raw PC1 scalar."""
        if self.fallback_to_row_mean_:
            return scores_df.loc[:, self.feature_names].astype(float).mean(axis=1).to_numpy(dtype=float)
        x = scores_df.loc[:, self.feature_names].astype(float).to_numpy(dtype=float)
        z = (x - self.means_) / self.stds_
        return z @ self.component_

    def to_dict(self) -> dict:
        """Serialize the model to a JSON-friendly dict."""
        return {
            "feature_names": self.feature_names,
            "means_": self.means_.tolist(),
            "stds_": self.stds_.tolist(),
            "component_": self.component_.tolist(),
            "score_min_": self.score_min_,
            "score_max_": self.score_max_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "fallback_to_row_mean_": self.fallback_to_row_mean_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DemandPCAQualityModel":
        """Deserialize a PCA model."""
        return cls(
            feature_names=list(payload["feature_names"]),
            means_=np.asarray(payload["means_"], dtype=float),
            stds_=np.asarray(payload["stds_"], dtype=float),
            component_=np.asarray(payload["component_"], dtype=float),
            score_min_=float(payload["score_min_"]),
            score_max_=float(payload["score_max_"]),
            explained_variance_ratio_=float(payload.get("explained_variance_ratio_", 0.0)),
            fallback_to_row_mean_=bool(payload.get("fallback_to_row_mean_", False)),
        )


def compute_pca_quality_scores(
    scores_df: pd.DataFrame,
    config: Optional[dict] = None,
) -> tuple[pd.DataFrame, DemandPCAQualityModel]:
    """Attach PCA-based quality scores and aggregation weights."""
    config = config or {}
    pca_cfg = dict(config.get("pca", {}) or {})
    epsilon = float(pca_cfg.get("epsilon", 1e-9))

    required_cols = ["client_id", "sample_count", *QUALITY_DIMENSIONS]
    missing = [col for col in required_cols if col not in scores_df.columns]
    if missing:
        raise ValueError(f"Missing columns for PCA quality scoring: {missing}")

    model = DemandPCAQualityModel.fit(scores_df)
    raw_scores = model.transform(scores_df)
    row_mean = scores_df.loc[:, QUALITY_DIMENSIONS].mean(axis=1).to_numpy(dtype=float)
    if np.allclose(raw_scores, raw_scores[0]):
        quality_score = row_mean.copy()
    else:
        raw_min = float(raw_scores.min())
        raw_max = float(raw_scores.max())
        if raw_max - raw_min <= epsilon:
            quality_score = row_mean.copy()
        else:
            quality_score = (raw_scores - raw_min) / (raw_max - raw_min)

    sample_count = scores_df["sample_count"].to_numpy(dtype=float)
    weight_mass = quality_score * sample_count
    if np.allclose(weight_mass.sum(), 0.0):
        aggregation_weight = np.full(len(scores_df), 1.0 / max(1, len(scores_df)), dtype=float)
    else:
        aggregation_weight = weight_mass / weight_mass.sum()

    out = scores_df.loc[:, ["client_id", *QUALITY_DIMENSIONS, "sample_count"]].copy()
    out["pca_raw_score"] = raw_scores
    out["quality_score"] = quality_score
    out["aggregation_weight"] = aggregation_weight
    column_order = [
        "client_id",
        "completeness",
        "accuracy",
        "temporal_consistency",
        "skewness",
        "pca_raw_score",
        "quality_score",
        "sample_count",
        "aggregation_weight",
    ]
    out = out.loc[:, column_order]
    return out, model


def compute_pca_quality_scores_from_csv(
    input_path: Path | str,
    output_path: Path | str,
    config: Optional[dict] = None,
    model_output_path: Optional[Path | str] = None,
) -> tuple[pd.DataFrame, DemandPCAQualityModel]:
    """Read dimension scores, compute PCA quality scores, and save outputs."""
    scores_df = pd.read_csv(input_path)
    out_df, model = compute_pca_quality_scores(scores_df, config=config)
    save_csv(out_df, output_path)
    if model_output_path is not None:
        save_json(model.to_dict(), model_output_path)
    return out_df, model
