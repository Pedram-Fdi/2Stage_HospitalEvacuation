"""Unit tests for objective-function sensitivity utilities (no fake optimization data)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ObjectivePolicyWeights import (
    ObjectivePolicyWeights,
    build_factorial_schemes,
    build_oat_schemes,
    default_weight_schemes,
    oat_multipliers_default,
    parse_oat_multipliers_string,
    parse_objective_weights_string,
)
from objective_function_sensitivity_analysis.exceptions import InstanceNotFoundError
from objective_function_sensitivity_analysis.metrics import (
    add_baseline_deltas,
    compute_acf_spatial_metrics,
    validate_weighted_objective,
)
from objective_function_sensitivity_analysis.runner import load_instance_or_raise


class TestWeightConfiguration(unittest.TestCase):
    def test_baseline_defaults(self):
        w = ObjectivePolicyWeights.baseline()
        self.assertEqual(w.as_dict(), {
            "travel_time": 1.0,
            "evacuation_risk": 1.0,
            "unmet_demand": 1.0,
            "threat_risk": 1.0,
        })

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            ObjectivePolicyWeights(travel_time=-1)

    def test_parse_csv_and_named(self):
        a = parse_objective_weights_string("2,1,0.5,1.5")
        b = parse_objective_weights_string(
            "travel_time=2,evacuation_risk=1,unmet_demand=0.5,threat_risk=1.5"
        )
        self.assertEqual(a, b)

    def test_invalid_parse(self):
        with self.assertRaises(ValueError):
            parse_objective_weights_string("1,2,3")

    def test_weight_shares_sum_to_one(self):
        w = ObjectivePolicyWeights(1, 1, 2, 0)
        shares = w.weight_shares()
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_zero_weights_shares(self):
        w = ObjectivePolicyWeights(0, 0, 0, 0)
        shares = w.weight_shares()
        self.assertEqual(sum(shares.values()), 0.0)

    def test_default_schemes_include_baseline(self):
        schemes = default_weight_schemes()
        self.assertIn("Baseline", schemes)
        self.assertEqual(schemes["Baseline"], ObjectivePolicyWeights.baseline())

    def test_oat_grid_shared(self):
        schemes = build_oat_schemes(multipliers=(0.5, 2.0))
        self.assertIn("OAT_travel_time_x0.5", schemes)
        self.assertEqual(schemes["OAT_threat_risk_x2"].threat_risk, 2.0)
        self.assertEqual(schemes["OAT_threat_risk_x2"].travel_time, 1.0)

    def test_oat_grid_intelligent_defaults(self):
        schemes = build_oat_schemes()
        self.assertIn("OAT_travel_time_x10000", schemes)
        self.assertIn("OAT_unmet_demand_x0.001", schemes)
        self.assertIn("OAT_threat_risk_x0.1", schemes)
        self.assertEqual(schemes["OAT_travel_time_x1000"].travel_time, 1000.0)
        self.assertEqual(schemes["OAT_travel_time_x1000"].unmet_demand, 1.0)
        # Travel should NOT use the small unmet-style grid as exclusive set
        self.assertNotIn("OAT_travel_time_x0.05", schemes)

    def test_parse_oat_auto_and_per_component(self):
        auto = parse_oat_multipliers_string("auto")
        self.assertEqual(auto, oat_multipliers_default())
        shared = parse_oat_multipliers_string("0.5,2")
        self.assertEqual(shared["travel_time"], (0.5, 2.0))
        self.assertEqual(shared["unmet_demand"], (0.5, 2.0))
        custom = parse_oat_multipliers_string("travel_time=1,100;unmet_demand=0.05,1")
        self.assertEqual(custom["travel_time"], (1.0, 100.0))
        self.assertEqual(custom["unmet_demand"], (0.05, 1.0))
        # Unspecified components keep intelligent defaults
        self.assertEqual(custom["threat_risk"], oat_multipliers_default()["threat_risk"])

    def test_factorial_schemes_count_and_joint_risk(self):
        schemes = build_factorial_schemes()
        self.assertEqual(len(schemes), 27)
        sample = schemes["FAC_TT-H_RK-L_UD-M"]
        self.assertEqual(sample.travel_time, 10000.0)
        self.assertEqual(sample.evacuation_risk, 0.1)
        self.assertEqual(sample.threat_risk, 0.1)  # joint risk factor
        self.assertEqual(sample.unmet_demand, 0.25)



class TestObjectiveValidation(unittest.TestCase):
    def test_reconstruction_ok(self):
        components = {
            "travel_time_raw": 10.0,
            "evacuation_risk_raw": 20.0,
            "unmet_demand_raw": 30.0,
            "threat_risk_raw": 40.0,
            "first_stage_raw": 5.0,
        }
        weights = ObjectivePolicyWeights(1, 2, 0.5, 1)
        gurobi = 5 + 10 + 40 + 15 + 40
        result = validate_weighted_objective(components, weights, gurobi)
        self.assertTrue(result["solution_valid"])
        self.assertAlmostEqual(result["objective_recalculation_error"], 0.0)

    def test_reconstruction_mismatch(self):
        components = {
            "travel_time_raw": 10.0,
            "evacuation_risk_raw": 0.0,
            "unmet_demand_raw": 0.0,
            "threat_risk_raw": 0.0,
            "first_stage_raw": 0.0,
        }
        result = validate_weighted_objective(
            components, ObjectivePolicyWeights.baseline(), gurobi_objective=999.0
        )
        self.assertFalse(result["solution_valid"])

    def test_zero_denominator_pct_change(self):
        row = {"unmet_demand": 5.0, "total_objective": 10.0}
        base = {"unmet_demand": 0.0, "total_objective": 10.0}
        out = add_baseline_deltas(row, base)
        self.assertIsNone(out["pct_change_unmet_demand"])
        self.assertEqual(out["delta_unmet_demand"], 5.0)


class TestInstanceValidation(unittest.TestCase):
    def test_missing_instance_raises(self):
        with self.assertRaises(InstanceNotFoundError):
            load_instance_or_raise("DOES_NOT_EXIST_XYZ_CRP")

    def test_existing_default_instance_loads(self):
        pickle_path = ROOT / "Instances" / "4_20_5_20_3_1_CRP.pkl"
        if not pickle_path.exists():
            self.skipTest("Default instance pickle not present")
        inst = load_instance_or_raise("4_20_5_20_3_1_CRP")
        self.assertEqual(list(inst.Casualty_Shortage_Cost), [150000.0, 75000.0])
        self.assertEqual(list(inst.EvacuationRiskCost), [100000.0, 50000.0])

    def test_acf_spatial_metrics_on_instance(self):
        pickle_path = ROOT / "Instances" / "4_20_5_20_3_1_CRP.pkl"
        if not pickle_path.exists():
            self.skipTest("Default instance pickle not present")
        inst = load_instance_or_raise("4_20_5_20_3_1_CRP")
        # Open first three candidate ACFs
        acf = [0.0] * inst.NrACFs
        acf[0] = acf[1] = acf[2] = 1.0
        metrics = compute_acf_spatial_metrics(inst, acf)
        self.assertEqual(metrics["acf_locations"], "0,1,2")
        self.assertEqual(metrics["number_of_active_acfs"], 3.0)
        self.assertIsNotNone(metrics["mean_nearest_open_acf_distance_to_demand"])
        self.assertGreater(metrics["mean_nearest_open_acf_distance_to_demand"], 0.0)
        self.assertGreater(metrics["mean_pairwise_open_acf_distance"], 0.0)


class TestCSVGeneration(unittest.TestCase):
    def test_write_minimal_csv(self):
        import pandas as pd
        from objective_function_sensitivity_analysis.runner import write_outputs

        rows = [
            {
                "run_id": "r1",
                "weight_scheme": "Baseline",
                "sensitivity_mode": "baseline",
                "success": True,
                "solution_valid": True,
                "weight_travel_time": 1.0,
                "weight_evacuation_risk": 1.0,
                "weight_unmet_demand": 1.0,
                "weight_threat_risk": 1.0,
                "travel_time_weight_share": 0.25,
                "evacuation_risk_weight_share": 0.25,
                "unmet_demand_weight_share": 0.25,
                "threat_risk_weight_share": 0.25,
                "travel_time_raw": 1.0,
                "evacuation_risk_raw": 2.0,
                "unmet_demand_raw": 3.0,
                "threat_risk_raw": 4.0,
                "travel_time_weighted": 1.0,
                "evacuation_risk_weighted": 2.0,
                "unmet_demand_weighted": 3.0,
                "threat_risk_weighted": 4.0,
                "total_objective": 10.0,
                "unmet_demand": 0.0,
                "non_evacuated_patients": 0.0,
                "_decisions": {"acf": [1]},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(rows, Path(tmp), config={"weight_schemes": {}}, generate_plots=False)
            self.assertTrue(paths["results_csv"].exists())
            df = pd.read_csv(paths["results_csv"])
            self.assertEqual(len(df), 1)
            self.assertNotIn("_decisions", df.columns)

            # Append factorial row without wiping Baseline
            fac_rows = [
                {
                    "run_id": "r2",
                    "weight_scheme": "FAC_TT-H_RK-L_UD-L",
                    "sensitivity_mode": "factorial",
                    "success": True,
                    "solution_valid": True,
                    "weight_travel_time": 10000.0,
                    "weight_evacuation_risk": 0.1,
                    "weight_unmet_demand": 0.001,
                    "weight_threat_risk": 0.1,
                    "number_of_active_acfs": 8,
                    "acf_locations": "0,1,2",
                    "total_objective": 99.0,
                }
            ]
            paths2 = write_outputs(
                fac_rows,
                Path(tmp),
                config={"weight_schemes": {"FAC_TT-H_RK-L_UD-L": {}}},
                generate_plots=False,
                append=True,
                plot_scope="factorial",
            )
            merged = pd.read_csv(paths2["results_csv"])
            self.assertEqual(len(merged), 2)
            self.assertIn("Baseline", set(merged["weight_scheme"]))
            self.assertTrue(paths2["factorial_csv"].exists())
            fac_only = pd.read_csv(paths2["factorial_csv"])
            self.assertEqual(len(fac_only), 1)


if __name__ == "__main__":
    unittest.main()
