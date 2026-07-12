from __future__ import annotations

import importlib.util
import json
import os
import pickle
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
            self.assertEqual(meta["dq_mode"], "input_feature_and_window_loss_weight")
            self.assertEqual(meta["model_input_size"], 5)

    def test_quality_input_columns_add_selected_dimension_count(self) -> None:
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
                "QUALITY_INPUT_COLUMNS": (
                    "dimension_completeness,dimension_validity,dimension_consistency,"
                    "dimension_accuracy,dimension_freshness"
                ),
            }

            meta = run_split.configure_model_input_size(env)

            self.assertEqual(env["MODEL_INPUT_SIZE"], "6")
            self.assertEqual(env["QUALITY_INPUT_DIM"], "5")
            self.assertEqual(meta["quality_input_dim"], 5)
            self.assertEqual(len(meta["quality_input_columns"]), 5)

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

    def test_multi_column_quality_scores_are_appended(self) -> None:
        X = torch.zeros(2, 3, 1)
        q = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=torch.float32)

        out = augment_sequence_inputs_with_quality(X, q, enabled=True, mode="train")

        self.assertEqual(tuple(out.shape), (2, 3, 4))
        self.assertTrue(torch.allclose(out[0, :, 1:], q[0].view(1, 3).expand(3, 3)))
        self.assertTrue(torch.allclose(out[1, :, 1:], q[1].view(1, 3).expand(3, 3)))

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
                "dimension_completeness": [0.1, 0.2, 0.3],
                "dimension_validity": [0.4, 0.5, 0.6],
                "dimension_consistency": [0.7, 0.8, 0.9],
                "dimension_accuracy": [0.11, 0.22, 0.33],
                "dimension_freshness": [0.44, 0.55, 0.66],
            }
        ).to_csv(path, index=False)

    def _write_custom_cluster_payload(
        self,
        cluster_dir: Path,
        *,
        values: list[float],
        drop_timestamp_indices: set[int] | None = None,
    ) -> None:
        timestamps = pd.date_range("2024-01-01", periods=len(values), freq="h")
        frame = pd.DataFrame(
            {
                "Fecha_Hora": timestamps,
                "id": [9901] * len(timestamps),
                "value": values,
            }
        )
        if drop_timestamp_indices:
            frame = frame.drop(index=sorted(drop_timestamp_indices)).reset_index(drop=True)
        payload = {
            "timeseries": frame,
            "client_metadata": pd.DataFrame({"id": [9901]}),
            "timestamp_col": "Fecha_Hora",
            "value_col": "value",
            "static_feature_columns": [],
        }
        pd.to_pickle(payload, cluster_dir / "cluster_1.pkl")

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

    def test_quality_mode_helpers_cover_quality_combinations(self) -> None:
        self.assertFalse(quality_scores_needed(False, False, False))
        self.assertTrue(quality_scores_needed(True, False, False))
        self.assertTrue(quality_scores_needed(False, True, False))
        self.assertTrue(quality_scores_needed(False, False, True))
        self.assertTrue(quality_scores_needed(True, True, True))
        self.assertEqual(derive_dq_mode(False, False, False), "none")
        self.assertEqual(derive_dq_mode(True, False, False), "window_loss_weight")
        self.assertEqual(derive_dq_mode(False, True, False), "input_feature")
        self.assertEqual(derive_dq_mode(False, False, True), "soft_client_gradient_weight")
        self.assertEqual(derive_dq_mode(True, True, False), "input_feature_and_window_loss_weight")
        self.assertEqual(derive_dq_mode(False, True, True), "input_feature_and_soft_client_gradient_weight")
        self.assertEqual(derive_dq_mode(True, False, True), "window_loss_weight_and_soft_client_gradient_weight")
        self.assertEqual(
            derive_dq_mode(True, True, True),
            "input_feature_and_window_loss_weight_and_soft_client_gradient_weight",
        )
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
                ("false", "false", "false", False, False, "none"),
                ("true", "false", "false", True, True, "window_loss_weight"),
                ("false", "true", "false", True, True, "input_feature"),
                ("false", "false", "true", True, True, "soft_client_gradient_weight"),
                ("true", "true", "true", True, True, "input_feature_and_window_loss_weight_and_soft_client_gradient_weight"),
            ]

            for use_weights, use_input, use_agg, expected_needed, expected_loaded, dq_mode in cases:
                module = self._load_stream_service(
                    {
                        **common_env,
                        "USE_QUALITY_WEIGHTS": use_weights,
                        "USE_QUALITY_AS_INPUT": use_input,
                        "USE_QUALITY_AGGREGATION_WEIGHTS": use_agg,
                    }
                )
                self.assertEqual(module.QUALITY_NEEDED, expected_needed)
                self.assertEqual(module.quality_score_summary["loaded"], expected_loaded)
                self.assertEqual(module.quality_score_summary["dq_mode"], dq_mode)

    def test_stream_service_emits_selected_quality_input_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cluster_dir = root / "clusters"
            cluster_dir.mkdir()
            self._write_cluster_payload(cluster_dir)
            quality_path = root / "quality.csv"
            self._write_quality_scores(quality_path)

            module = self._load_stream_service(
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
                    "MODEL_INPUT_SIZE": "6",
                    "USE_QUALITY_WEIGHTS": "false",
                    "USE_QUALITY_AS_INPUT": "true",
                    "QUALITY_SCORE_PATH": str(quality_path),
                    "QUALITY_INPUT_COLUMNS": (
                        "dimension_completeness,dimension_validity,dimension_consistency,"
                        "dimension_accuracy,dimension_freshness"
                    ),
                }
            )

            response = module.app.test_client().post(
                "/stream/next",
                json={"client_id": 9901, "cluster_id": "1", "mode": "train", "batch_size": 1},
            )

            self.assertEqual(response.status_code, 200)
            payload = pickle.loads(response.data)
            self.assertEqual(tuple(payload["quality_scores"].shape), (1, 5))
            self.assertEqual(tuple(payload["quality_weight_scores"].shape), (1,))
            self.assertTrue(torch.allclose(payload["quality_scores"][0], torch.tensor([0.1, 0.4, 0.7, 0.11, 0.44])))
            self.assertTrue(torch.allclose(payload["quality_weight_scores"], torch.tensor([0.8])))
            self.assertEqual(payload["quality_score_metadata"]["quality_input_dim"], 5)

    def test_stream_service_uses_faulty_window_and_clean_timestamp_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "faulty"
            target_dir = root / "clean"
            input_dir.mkdir()
            target_dir.mkdir()
            self._write_custom_cluster_payload(
                input_dir,
                values=[float(100 + i) for i in range(12)],
                drop_timestamp_indices={2},
            )
            self._write_custom_cluster_payload(
                target_dir,
                values=[float(1000 + i) for i in range(12)],
            )

            module = self._load_stream_service(
                {
                    "CLUSTER_DIR": str(input_dir),
                    "TRAIN_CLUSTER_DATA_DIR": str(input_dir),
                    "VAL_CLUSTER_DATA_DIR": str(input_dir),
                    "TEST_CLUSTER_DATA_DIR": str(input_dir),
                    "TRAIN_TARGET_CLUSTER_DATA_DIR": str(target_dir),
                    "VAL_TARGET_CLUSTER_DATA_DIR": str(target_dir),
                    "TEST_TARGET_CLUSTER_DATA_DIR": str(target_dir),
                    "WINDOW_SIZE": "4",
                    "FORECAST_HORIZON": "1",
                    "TRAIN_RATIO": "0.7",
                    "VAL_RATIO": "0.1",
                    "TEST_RATIO": "0.2",
                    "NORMALIZE": "false",
                    "MODEL_INPUT_SIZE": "1",
                    "USE_QUALITY_WEIGHTS": "false",
                    "USE_QUALITY_AS_INPUT": "false",
                    "QUALITY_SCORE_PATH": "",
                }
            )

            response = module.app.test_client().post(
                "/stream/next",
                json={"client_id": 9901, "cluster_id": "1", "mode": "train", "batch_size": 1},
            )

            self.assertEqual(response.status_code, 200)
            payload = pickle.loads(response.data)
            self.assertEqual(payload["window_end_timestamps"][0], "2024-01-01T04:00:00")
            self.assertEqual(payload["target_timestamps"][0], "2024-01-01T05:00:00")
            self.assertTrue(torch.equal(payload["X"][0, :, 0], torch.tensor([100.0, 101.0, 103.0, 104.0])))
            self.assertTrue(torch.equal(payload["y"], torch.tensor([1005.0])))

    def test_stream_service_requires_quality_path_when_any_quality_mode_needs_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cluster_dir = root / "clusters"
            cluster_dir.mkdir()
            self._write_cluster_payload(cluster_dir)

            cases = [
                {"USE_QUALITY_WEIGHTS": "false", "USE_QUALITY_AS_INPUT": "true", "USE_QUALITY_AGGREGATION_WEIGHTS": "false"},
                {"USE_QUALITY_WEIGHTS": "false", "USE_QUALITY_AS_INPUT": "false", "USE_QUALITY_AGGREGATION_WEIGHTS": "true"},
            ]
            for flags in cases:
                with self.subTest(flags=flags):
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
                                "QUALITY_SCORE_PATH": "",
                                **flags,
                            }
                        )


class ServerQualityAggregationTests(unittest.TestCase):
    def _load_server_module(self, env: dict[str, str]) -> types.ModuleType:
        module_name = f"server_quality_agg_test_{uuid.uuid4().hex}"
        module_path = PROJECT_ROOT / "server" / "server.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, env, clear=False):
            spec.loader.exec_module(module)
        return module

    def _server_env(self, output_dir: Path, **overrides: str) -> dict[str, str]:
        env = {
            "OUTPUT_DIR": str(output_dir),
            "CLUSTER_ID": "1",
            "SERVER_LR": "0.01",
            "CHECKPOINT_EVERY_ROUNDS": "0",
            "ACCUM_STEPS_SERVER": "2",
            "MODEL_INPUT_SIZE": "1",
            "MODEL_HIDDEN_SIZE": "4",
            "MODEL_NUM_LAYERS": "2",
            "MODEL_SPLIT_LAYER": "1",
            "MODEL_OUTPUT_SIZE": "1",
            "MODEL_DROPOUT": "0.0",
            "MODEL_BIDIRECTIONAL": "false",
            "USE_QUALITY_WEIGHTS": "false",
            "USE_QUALITY_AS_INPUT": "false",
            "USE_QUALITY_AGGREGATION_WEIGHTS": "true",
            "QUALITY_AGGREGATION_MODE": "soft_client_gradient",
            "QUALITY_AGGREGATION_Q_MIN": "0.5",
            "QUALITY_AGGREGATION_EPS": "1e-8",
            "COMM_LOG_TRAIN": "false",
            "COMM_LOG_EVAL": "false",
        }
        env.update(overrides)
        return env

    def test_soft_client_gradient_aggregation_uses_soft_quality_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module = self._load_server_module(self._server_env(Path(tmpdir)))
            state = module.state
            state.set_round(["c1", "c2"], round_id=1, effective_accum_steps_target=2)

            first = state.train_step(
                client_id="c1",
                activations_cpu=torch.zeros(1, 4),
                targets_cpu=torch.ones(1, 1),
                quality_scores_cpu=torch.tensor([1.0]),
            )
            second = state.train_step(
                client_id="c2",
                activations_cpu=torch.zeros(1, 4),
                targets_cpu=torch.ones(1, 1),
                quality_scores_cpu=torch.tensor([0.0]),
            )

            self.assertFalse(first["window_complete"])
            self.assertTrue(second["window_complete"])
            self.assertAlmostEqual(first["client_soft_quality"], 1.0, places=6)
            self.assertAlmostEqual(second["client_soft_quality"], 0.5, places=6)
            flush_info = second["flush_info"]
            self.assertIsNotNone(flush_info)
            self.assertAlmostEqual(flush_info["flushed_accum_quality_weight_sum"], 1.5, places=6)
            self.assertAlmostEqual(flush_info["effective_accumulation_normalizer"], 1.5 + 1e-8, places=6)
            self.assertEqual(state.optimizer_step_count, 1)

    def test_quality_aggregation_requires_train_quality_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module = self._load_server_module(self._server_env(Path(tmpdir)))
            state = module.state
            state.set_round(["c1"], round_id=1, effective_accum_steps_target=1)

            with self.assertRaisesRegex(ValueError, "quality_scores missing"):
                state.train_step(
                    client_id="c1",
                    activations_cpu=torch.zeros(1, 4),
                    targets_cpu=torch.ones(1, 1),
                    quality_scores_cpu=None,
                )

    def test_quality_aggregation_rejects_out_of_range_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module = self._load_server_module(self._server_env(Path(tmpdir)))
            state = module.state
            state.set_round(["c1"], round_id=1, effective_accum_steps_target=1)

            with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
                state.train_step(
                    client_id="c1",
                    activations_cpu=torch.zeros(1, 4),
                    targets_cpu=torch.ones(1, 1),
                    quality_scores_cpu=torch.tensor([1.2]),
                )


if __name__ == "__main__":
    unittest.main()
