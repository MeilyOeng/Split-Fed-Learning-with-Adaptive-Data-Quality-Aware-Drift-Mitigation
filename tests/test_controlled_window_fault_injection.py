from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DQ_ROOT = PROJECT_ROOT / "data quality"
if str(DQ_ROOT) not in sys.path:
    sys.path.insert(0, str(DQ_ROOT))

from modules.fault_injection import WideDemandFaultInjector, inject_faults_to_csv
from modules.io import load_wide_demand_csv


def _controlled_config() -> dict:
    return {
        "fault_mode": "controlled_window",
        "window_faulting": {
            "window_size": 4,
            "stride": 1,
            "sampling_scope": "per_cluster_per_client",
            "total_faulty_window_fraction": 0.5,
            "dimension_window_fractions": {
                "completeness": 0.125,
                "consistency": 0.125,
                "accuracy": 0.125,
                "validity": 0.125,
            },
            "min_faults_per_faulty_window": 1,
            "max_faults_per_faulty_window": 1,
            "multi_fault_probability": 0.0,
        },
        "cluster_assignment": {
            "client_id_column": "ID",
            "cluster_id_column": "Cluster",
        },
        "faults": {
            "accuracy": {"spike_multiplier": 4.0, "noise_std_fraction": 0.05},
            "consistency": {"lag_choices": [1], "jump_multiplier": 6.0},
            "validity": {"high_multiplier": 3.0},
        },
    }


class ControlledWindowFaultInjectionTests(unittest.TestCase):
    def test_controlled_freshness_masks_window_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"

            pd.DataFrame(
                {
                    "Fecha_Hora": pd.date_range("2024-01-01", periods=24, freq="h"),
                    "9901": [float(i + 1) for i in range(24)],
                    "9902": [100.0 + i for i in range(24)],
                }
            ).to_csv(clean_path, index=False)

            dataset = load_wide_demand_csv(clean_path, timestamp_col="Fecha_Hora")
            config = {
                "fault_mode": "controlled_window",
                "window_faulting": {
                    "window_size": 24,
                    "stride": 1,
                    "sampling_scope": "per_cluster_per_client",
                    "total_faulty_window_fraction": 1.0,
                    "dimension_window_fractions": {"freshness": 1.0},
                    "min_faults_per_faulty_window": 1,
                    "max_faults_per_faulty_window": 1,
                    "multi_fault_probability": 0.0,
                },
                "clients": {
                    "selection_mode": "explicit",
                    "selected": ["9901"],
                },
                "faults": {"freshness": {"delay_steps": 5}},
            }

            injector = WideDemandFaultInjector(config=config)
            faulty, report = injector.inject(dataset, seed=42)

            self.assertTrue(faulty.loc[19:23, "9901"].isna().all())
            self.assertTrue(faulty.loc[0:18, "9901"].notna().all())
            pd.testing.assert_series_equal(
                faulty["9902"],
                dataset.frame["9902"],
                check_names=False,
            )

            event = report.loc[report["fault_type"] == "controlled_delayed_tail"].iloc[0]
            self.assertEqual(int(event["affected_rows"]), 5)
            self.assertEqual(event["dimension"], "freshness")

    def test_controlled_window_manifest_summary_and_seed_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            cluster_path = root / "clusters.csv"
            pd.DataFrame(
                {
                    "Fecha_Hora": pd.date_range("2024-01-01", periods=24, freq="h"),
                    "9901": [float(i + 1) for i in range(24)],
                    "9902": [float((i + 1) * 2) for i in range(24)],
                }
            ).to_csv(clean_path, index=False)
            pd.DataFrame(
                {
                    "ID": ["9901", "9902"],
                    "Cluster": ["1", "2"],
                }
            ).to_csv(cluster_path, index=False)

            dataset = load_wide_demand_csv(clean_path, timestamp_col="Fecha_Hora")
            config = _controlled_config()
            injector_a = WideDemandFaultInjector(config=config)
            faulty_a, report_a = injector_a.inject(dataset, seed=42, cluster_assignment_path=cluster_path)
            injector_b = WideDemandFaultInjector(config=config)
            faulty_b, report_b = injector_b.inject(dataset, seed=42, cluster_assignment_path=cluster_path)

            pd.testing.assert_frame_equal(faulty_a, faulty_b)
            controlled_report = report_a.loc[report_a["scope"] == "controlled_window"]
            self.assertFalse(controlled_report.empty)
            self.assertTrue(controlled_report["window_start_index"].notna().all())
            self.assertTrue(controlled_report["window_end_index"].notna().all())
            self.assertTrue(controlled_report["affected_positions"].notna().all())
            self.assertTrue(controlled_report["affected_timestamps"].notna().all())
            self.assertEqual(set(controlled_report["seed"].dropna().astype(int)), {42})
            self.assertTrue((controlled_report["affected_rows"] >= 1).all())
            self.assertEqual(set(controlled_report["cluster_id"].dropna().astype(str)), {"1", "2"})

            summary = injector_a.last_controlled_window_summary_df
            self.assertEqual(set(summary["client_id"]), {"9901", "9902"})
            self.assertTrue((summary["selected_faulty_windows"] > 0).all())
            self.assertTrue((summary["achieved_faulty_fraction"] > 0).all())
            pd.testing.assert_frame_equal(
                report_a.sort_index(axis=1),
                report_b.sort_index(axis=1),
            )


    def test_per_cluster_random_selection_is_nested_by_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            cluster_path = root / "clusters.csv"
            rows = {"Fecha_Hora": pd.date_range("2024-01-01", periods=16, freq="h")}
            assignments = []
            for cluster_id in ("1", "2"):
                for offset in range(5):
                    client_id = f"99{cluster_id}{offset:03d}"
                    rows[client_id] = [float(offset + 1) for _ in range(16)]
                    assignments.append({"ID": client_id, "Cluster": cluster_id})
            pd.DataFrame(rows).to_csv(clean_path, index=False)
            pd.DataFrame(assignments).to_csv(cluster_path, index=False)

            dataset = load_wide_demand_csv(clean_path, timestamp_col="Fecha_Hora")
            config_one = _controlled_config()
            config_one["clients"] = {
                "selection_mode": "per_cluster_random",
                "random_count": 1,
                "min_count_per_cluster": 1,
                "include_unassigned": False,
            }
            injector_one = WideDemandFaultInjector(config=config_one)
            injector_one.inject(dataset, seed=7, cluster_assignment_path=cluster_path)
            one_df = injector_one.last_controlled_selected_clients_df
            selected_one = set(one_df.loc[one_df["selected"].astype(bool), "client_id"].astype(str))

            config_three = _controlled_config()
            config_three["clients"] = {
                "selection_mode": "per_cluster_random",
                "random_count": 3,
                "min_count_per_cluster": 1,
                "include_unassigned": False,
            }
            injector_three = WideDemandFaultInjector(config=config_three)
            injector_three.inject(dataset, seed=7, cluster_assignment_path=cluster_path)
            three_df = injector_three.last_controlled_selected_clients_df
            selected_three = set(three_df.loc[three_df["selected"].astype(bool), "client_id"].astype(str))

            self.assertEqual(len(selected_one), 2)
            self.assertEqual(len(selected_three), 6)
            self.assertTrue(selected_one.issubset(selected_three))
            self.assertTrue((three_df.groupby("cluster_id")["selection_rank"].min() == 1).all())
            self.assertTrue({"selected_for_1_client", "selected_for_20", "selected_for_40", "selected_for_100"}.issubset(three_df.columns))

    def test_controlled_window_output_keeps_unselected_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            cluster_path = root / "clusters.csv"
            rows = {"Fecha_Hora": pd.date_range("2024-01-01", periods=16, freq="h")}
            assignments = []
            for cluster_id in ("1", "2"):
                for offset in range(2):
                    client_id = f"99{cluster_id}{offset:03d}"
                    rows[client_id] = [float((offset + 1) * (i + 1)) for i in range(16)]
                    assignments.append({"ID": client_id, "Cluster": cluster_id})
            pd.DataFrame(rows).to_csv(clean_path, index=False)
            pd.DataFrame(assignments).to_csv(cluster_path, index=False)

            dataset = load_wide_demand_csv(clean_path, timestamp_col="Fecha_Hora")
            config = _controlled_config()
            config["clients"] = {
                "selection_mode": "per_cluster_random",
                "random_count": 1,
                "min_count_per_cluster": 1,
                "include_unassigned": False,
            }
            injector = WideDemandFaultInjector(config=config)
            faulty, _ = injector.inject(dataset, seed=13, cluster_assignment_path=cluster_path)

            expected_client_cols = [assignment["ID"] for assignment in assignments]
            self.assertTrue(set(expected_client_cols).issubset(faulty.columns))
            self.assertEqual(len(expected_client_cols), len(set(expected_client_cols) & set(faulty.columns)))
            selected = set(
                injector.last_controlled_selected_clients_df.loc[
                    injector.last_controlled_selected_clients_df["selected"].astype(bool), "client_id"
                ].astype(str)
            )
            self.assertEqual(len(selected), 2)
            self.assertTrue(set(expected_client_cols) - selected)


    def test_global_random_selection_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            rows = {"Fecha_Hora": pd.date_range("2024-01-01", periods=16, freq="h")}
            for offset in range(6):
                rows[f"990{offset}"] = [float(offset + 1) for _ in range(16)]
            pd.DataFrame(rows).to_csv(clean_path, index=False)

            dataset = load_wide_demand_csv(clean_path, timestamp_col="Fecha_Hora")
            config = _controlled_config()
            config["clients"] = {"selection_mode": "global_random", "random_count": 2}
            first = WideDemandFaultInjector(config=config)
            first.inject(dataset, seed=11)
            second = WideDemandFaultInjector(config=config)
            second.inject(dataset, seed=11)

            first_selected = first.last_controlled_selected_clients_df.loc[
                first.last_controlled_selected_clients_df["selected"].astype(bool), "client_id"
            ].astype(str).tolist()
            second_selected = second.last_controlled_selected_clients_df.loc[
                second.last_controlled_selected_clients_df["selected"].astype(bool), "client_id"
            ].astype(str).tolist()
            self.assertEqual(first_selected, second_selected)
            self.assertEqual(len(first_selected), 2)

    def test_inject_faults_to_csv_writes_controlled_summary_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean_path = root / "clean.csv"
            output_path = root / "faulty.csv"
            report_path = root / "fault_report.csv"
            pd.DataFrame(
                {
                    "Fecha_Hora": pd.date_range("2024-01-01", periods=12, freq="h"),
                    "9901": [float(i + 1) for i in range(12)],
                }
            ).to_csv(clean_path, index=False)

            inject_faults_to_csv(
                clean_path,
                output_path,
                report_path,
                config=_controlled_config(),
                timestamp_col="Fecha_Hora",
                seed=42,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue((root / "fault_report_summary.csv").exists())
            self.assertTrue((root / "selected_clients.csv").exists())


if __name__ == "__main__":
    unittest.main()
