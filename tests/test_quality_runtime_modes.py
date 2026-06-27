from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_split
from utils.runtime_utils import (
    augment_sequence_inputs_with_quality,
    compute_model_input_size,
    derive_dq_mode,
    quality_scores_needed,
)


class ConfigureModelInputSizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_train_dir = run_split.TRAIN_CLUSTER_DATA_DIR

    def tearDown(self) -> None:
        run_split.TRAIN_CLUSTER_DATA_DIR = self.original_train_dir

    def _write_manifest(self, root: Path) -> None:
        manifest = {
            "static_feature_columns": [
                "meta_num_sum_kwh",
                "meta_num_max",
                "meta_cat_subcategoria_industrial",
            ]
        }
        (root / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_univariate_without_quality_input_sets_model_input_size_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_manifest(root)
            run_split.TRAIN_CLUSTER_DATA_DIR = root
            env = {
                "AUTO_MODEL_INPUT_SIZE": "true",
                "MODEL_INPUT_SIZE": "9",
                "USE_STATIC_FEATURES": "false",
                "USE_QUALITY_WEIGHTS": "false",
                "USE_QUALITY_AS_INPUT": "false",
            }

            meta = run_split.configure_model_input_size(env)

            self.assertEqual(env["MODEL_INPUT_SIZE"], "1")
            self.assertEqual(env["BASE_INPUT_SIZE"], "1")
            self.assertEqual(meta["dq_mode"], "none")
            self.assertEqual(meta["model_input_size"], 1)

    def test_univariate_with_quality_input_adds_one_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_manifest(root)
            run_split.TRAIN_CLUSTER_DATA_DIR = root
            env = {
                "AUTO_MODEL_INPUT_SIZE": "true",
                "MODEL_INPUT_SIZE": "1",
                "USE_STATIC_FEATURES": "false",
                "USE_QUALITY_WEIGHTS": "false",
                "USE_QUALITY_AS_INPUT": "true",
            }

            meta = run_split.configure_model_input_size(env)

            self.assertEqual(env["MODEL_INPUT_SIZE"], "2")
            self.assertEqual(env["BASE_INPUT_SIZE"], "1")
            self.assertEqual(meta["dq_mode"], "input_feature")
            self.assertEqual(meta["model_input_size"], 2)

    def test_static_features_without_quality_input_use_base_feature_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_manifest(root)
            run_split.TRAIN_CLUSTER_DATA_DIR = root
            env = {
                "AUTO_MODEL_INPUT_SIZE": "true",
                "MODEL_INPUT_SIZE": "1",
                "USE_STATIC_FEATURES": "true",
                "STATIC_FEATURE_GROUPS": "numeric,subcategory",
                "STATIC_FEATURE_COLUMNS": "",
                "USE_QUALITY_WEIGHTS": "false",
                "USE_QUALITY_AS_INPUT": "false",
            }

            meta = run_split.configure_model_input_size(env)

            self.assertEqual(env["MODEL_INPUT_SIZE"], "4")
            self.assertEqual(env["BASE_INPUT_SIZE"], "4")
            self.assertEqual(meta["model_input_size"], 4)

    def test_static_features_with_quality_input_add_one_on_top_of_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_manifest(root)
            run_split.TRAIN_CLUSTER_DATA_DIR = root
            env = {
                "AUTO_MODEL_INPUT_SIZE": "true",
                "MODEL_INPUT_SIZE": "1",
                "USE_STATIC_FEATURES": "true",
                "STATIC_FEATURE_GROUPS": "numeric,subcategory",
                "STATIC_FEATURE_COLUMNS": "",
                "USE_QUALITY_WEIGHTS": "true",
                "USE_QUALITY_AS_INPUT": "true",
            }

            meta = run_split.configure_model_input_size(env)

            self.assertEqual(env["MODEL_INPUT_SIZE"], "5")
            self.assertEqual(env["BASE_INPUT_SIZE"], "4")
            self.assertEqual(meta["dq_mode"], "input_feature_and_loss_weight")
            self.assertEqual(meta["model_input_size"], 5)

    def test_auto_model_input_size_false_preserves_manual_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_manifest(root)
            run_split.TRAIN_CLUSTER_DATA_DIR = root
            env = {
                "AUTO_MODEL_INPUT_SIZE": "false",
                "MODEL_INPUT_SIZE": "99",
                "USE_STATIC_FEATURES": "false",
                "USE_QUALITY_WEIGHTS": "false",
                "USE_QUALITY_AS_INPUT": "true",
            }

            meta = run_split.configure_model_input_size(env)

            self.assertEqual(env["MODEL_INPUT_SIZE"], "99")
            self.assertEqual(env["BASE_INPUT_SIZE"], "1")
            self.assertEqual(meta["model_input_size"], 99)


class QualityInputAugmentationTests(unittest.TestCase):
    def test_univariate_input_appends_quality_feature(self) -> None:
        X = torch.zeros(2, 3, 1)
        q = torch.tensor([0.2, 0.8], dtype=torch.float32)

        out = augment_sequence_inputs_with_quality(X, q, enabled=True, mode="train")

        self.assertEqual(tuple(out.shape), (2, 3, 2))
        self.assertTrue(torch.allclose(out[0, :, 1], torch.full((3,), 0.2)))
        self.assertTrue(torch.allclose(out[1, :, 1], torch.full((3,), 0.8)))

    def test_multivariate_input_appends_quality_feature(self) -> None:
        X = torch.zeros(2, 4, 5)
        q = torch.tensor([0.1, 0.9], dtype=torch.float32)

        out = augment_sequence_inputs_with_quality(X, q, enabled=True, mode="train")

        self.assertEqual(tuple(out.shape), (2, 4, 6))

    def test_column_vector_quality_scores_are_accepted(self) -> None:
        X = torch.zeros(2, 3, 2)
        q = torch.tensor([[0.3], [0.7]], dtype=torch.float32)

        out = augment_sequence_inputs_with_quality(X, q, enabled=True, mode="val")

        self.assertEqual(tuple(out.shape), (2, 3, 3))

    def test_missing_quality_scores_raise_when_enabled(self) -> None:
        X = torch.zeros(2, 3, 1)
        with self.assertRaisesRegex(ValueError, "quality_scores missing"):
            augment_sequence_inputs_with_quality(X, None, enabled=True, mode="train")

    def test_wrong_quality_score_length_raises(self) -> None:
        X = torch.zeros(2, 3, 1)
        q = torch.tensor([0.5], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            augment_sequence_inputs_with_quality(X, q, enabled=True, mode="train")

    def test_non_finite_quality_score_raises(self) -> None:
        X = torch.zeros(2, 3, 1)
        q = torch.tensor([0.5, float("nan")], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            augment_sequence_inputs_with_quality(X, q, enabled=True, mode="train")

    def test_out_of_range_quality_score_raises(self) -> None:
        X = torch.zeros(2, 3, 1)
        q = torch.tensor([0.5, 1.2], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            augment_sequence_inputs_with_quality(X, q, enabled=True, mode="test")

    def test_disabled_quality_input_returns_original_tensor(self) -> None:
        X = torch.randn(2, 3, 4)
        q = torch.tensor([0.2, 0.8], dtype=torch.float32)

        out = augment_sequence_inputs_with_quality(X, q, enabled=False, mode="train")

        self.assertTrue(torch.equal(out, X))


class StreamServiceQualityLoadingTests(unittest.TestCase):
    def _write_cluster_payload(self, cluster_dir: Path) -> None:
        timestamps = pd.date_range("2024-01-01", periods=32, freq="h")
        payload = {
            "timeseries": pd.DataFrame(
                {
                    "Fecha_Hora": timestamps,
                    "id": [9901] * len(timestamps),
                    "value": [float(i + 1) for i in range(len(timestamps))],
                }
            ),
            "client_metadata": pd.DataFrame({"id": [9901]}),
            "timestamp_col": "Fecha_Hora",
            "value_col": "value",
            "static_feature_columns": [],
        }
        pd.to_pickle(payload, cluster_dir / "cluster_1.pkl")

    def _write_quality_scores(self, path: Path) -> None:
        pd.DataFrame(
            {
                "client_id": ["9901", "9901", "9901"],
                "mode": ["train", "val", "test"],
                "window_start_index": [0, 0, 0],
                "quality_score": [0.8, 0.9, 1.0],
            }
        ).to_csv(path, index=False)

    def _load_stream_service(self, env: dict[str, str]) -> types.ModuleType:
        module_name = f"stream_service_test_{uuid.uuid4().hex}"
        module_path = PROJECT_ROOT / "data" / "stream_service.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, env, clear=False):
            spec.loader.exec_module(module)
        return module

    def test_quality_mode_helpers_cover_all_four_modes(self) -> None:
        self.assertFalse(quality_scores_needed(False, False))
        self.assertTrue(quality_scores_needed(True, False))
        self.assertTrue(quality_scores_needed(False, True))
        self.assertTrue(quality_scores_needed(True, True))
        self.assertEqual(derive_dq_mode(False, False), "none")
        self.assertEqual(derive_dq_mode(True, False), "loss_weight")
        self.assertEqual(derive_dq_mode(False, True), "input_feature")
        self.assertEqual(derive_dq_mode(True, True), "input_feature_and_loss_weight")
        self.assertEqual(compute_model_input_size(4, True), 5)

    def test_stream_service_loads_quality_scores_when_either_mode_requires_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cluster_dir = root / "clusters"
            cluster_dir.mkdir()
            self._write_cluster_payload(cluster_dir)
            quality_path = root / "quality.csv"
            self._write_quality_scores(quality_path)
            common_env = {
                "CLUSTER_DIR": str(cluster_dir),
                "TRAIN_CLUSTER_DATA_DIR": str(cluster_dir),
                "VAL_CLUSTER_DATA_DIR": str(cluster_dir),
                "TEST_CLUSTER_DATA_DIR": str(cluster_dir),
                "WINDOW_SIZE": "4",
                "FORECAST_HORIZON": "1",
                "TRAIN_RATIO": "0.7",
                "VAL_RATIO": "0.1",
                "TEST_RATIO": "0.2",
                "NORMALIZE": "false",
                "MODEL_INPUT_SIZE": "1",
                "QUALITY_SCORE_PATH": str(quality_path),
            }
            cases = [
                ("false", "false", False, False, "none"),
                ("true", "false", True, True, "loss_weight"),
                ("false", "true", True, True, "input_feature"),
                ("true", "true", True, True, "input_feature_and_loss_weight"),
            ]

            for use_weights, use_input, expected_needed, expected_loaded, dq_mode in cases:
                module = self._load_stream_service(
                    {
                        **common_env,
                        "USE_QUALITY_WEIGHTS": use_weights,
                        "USE_QUALITY_AS_INPUT": use_input,
                    }
                )
                self.assertEqual(module.QUALITY_NEEDED, expected_needed)
                self.assertEqual(module.quality_score_summary["loaded"], expected_loaded)
                self.assertEqual(module.quality_score_summary["dq_mode"], dq_mode)

    def test_stream_service_requires_quality_path_for_input_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cluster_dir = root / "clusters"
            cluster_dir.mkdir()
            self._write_cluster_payload(cluster_dir)

            with self.assertRaisesRegex(ValueError, "QUALITY_SCORE_PATH"):
                self._load_stream_service(
                    {
                        "CLUSTER_DIR": str(cluster_dir),
                        "TRAIN_CLUSTER_DATA_DIR": str(cluster_dir),
                        "VAL_CLUSTER_DATA_DIR": str(cluster_dir),
                        "TEST_CLUSTER_DATA_DIR": str(cluster_dir),
                        "WINDOW_SIZE": "4",
                        "FORECAST_HORIZON": "1",
                        "TRAIN_RATIO": "0.7",
                        "VAL_RATIO": "0.1",
                        "TEST_RATIO": "0.2",
                        "NORMALIZE": "false",
                        "MODEL_INPUT_SIZE": "2",
                        "USE_QUALITY_WEIGHTS": "false",
                        "USE_QUALITY_AS_INPUT": "true",
                        "QUALITY_SCORE_PATH": "",
                    }
                )


if __name__ == "__main__":
    unittest.main()
