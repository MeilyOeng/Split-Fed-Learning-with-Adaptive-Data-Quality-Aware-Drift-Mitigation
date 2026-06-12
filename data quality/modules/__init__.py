"""Wide-format demand quality pipeline."""

from .baseline import calibrate_client_baselines, calibrate_demand_baseline_from_csv, load_baseline_artifact
from .features import extract_client_window_features
from .fault_injection import WideDemandFaultInjector, inject_faults_to_csv
from .io import WideDemandDataset, load_wide_demand_csv, load_yaml_config
from .label_dataset import build_demand_label_dataset
from .pca_quality import DemandPCAQualityModel, compute_pca_quality_scores
from .pipeline import run_quality_pipeline
from .predictor import DemandQualityPredictor, predict_demand_quality
from .scoring import WideDemandQualityScorer, score_wide_demand_quality
from .train_xgboost import train_demand_xgboost_models

__all__ = [
    "DemandQualityPredictor",
    "DemandPCAQualityModel",
    "WideDemandDataset",
    "WideDemandFaultInjector",
    "build_demand_label_dataset",
    "calibrate_client_baselines",
    "calibrate_demand_baseline_from_csv",
    "compute_pca_quality_scores",
    "extract_client_window_features",
    "inject_faults_to_csv",
    "load_baseline_artifact",
    "load_wide_demand_csv",
    "load_yaml_config",
    "predict_demand_quality",
    "run_quality_pipeline",
    "score_wide_demand_quality",
    "train_demand_xgboost_models",
    "WideDemandQualityScorer",
]
