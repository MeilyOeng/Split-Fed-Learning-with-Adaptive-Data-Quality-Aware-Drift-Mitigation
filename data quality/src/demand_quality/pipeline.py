"""End-to-end orchestration for the demand quality pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .fault_injection import inject_faults_to_csv
from .io import load_yaml_config
from .pca_quality import compute_pca_quality_scores_from_csv
from .scoring import score_wide_demand_quality


def run_quality_pipeline(
    input_path: Path | str,
    faulty_output_path: Path | str,
    fault_report_path: Path | str,
    score_output_path: Path | str,
    pca_output_path: Path | str,
    fault_config_path: Path | str,
    score_config_path: Path | str,
    timestamp_col: Optional[str] = None,
    seed: int = 42,
    pca_model_output_path: Optional[Path | str] = None,
) -> dict[str, str]:
    """Run fault injection, scoring, PCA scoring, and weight export."""
    fault_config = load_yaml_config(fault_config_path)
    score_config = load_yaml_config(score_config_path)

    inject_faults_to_csv(
        input_path=input_path,
        output_path=faulty_output_path,
        report_path=fault_report_path,
        config=fault_config,
        timestamp_col=timestamp_col,
        seed=seed,
    )

    score_wide_demand_quality(
        current_path=faulty_output_path,
        clean_reference_path=input_path,
        output_path=score_output_path,
        config=score_config,
        timestamp_col=timestamp_col,
    )

    compute_pca_quality_scores_from_csv(
        input_path=score_output_path,
        output_path=pca_output_path,
        config=score_config,
        model_output_path=pca_model_output_path,
    )

    return {
        "faulty_output": str(Path(faulty_output_path)),
        "fault_report": str(Path(fault_report_path)),
        "score_output": str(Path(score_output_path)),
        "pca_output": str(Path(pca_output_path)),
    }
