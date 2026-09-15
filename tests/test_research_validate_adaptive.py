import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import research_validate_adaptive as validator


class ResearchValidatorContractTest(unittest.TestCase):

    def test_statistics_contract_includes_required_fields(self):
        result = validator.finite_stats([1.0, 2.0, float("nan")])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["nonfinite"], 1)
        for key in ("min", "max", "mean", "std", "p50", "p90", "p95", "p99"):
            self.assertIn(key, result)
            self.assertTrue(math.isfinite(result[key]))

    def test_toy_temporal_round_trip_contract(self):
        result = validator.toy_temporal_round_trip()
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_round_trip_abs_error"], 0.0)
        self.assertTrue(result["fixed_stride_fail_fast"])

    def test_research_ablation_d_requires_explicit_eight_level_ratios(self):
        root = Path(__file__).resolve().parents[1]
        paths = [str(root / "opts" / name) for name in (
            "research_ablation_A_baseline.yaml",
            "research_ablation_B_query_modulation.yaml",
            "research_ablation_C_boundary_importance.yaml",
            "research_ablation_D_adaptive_anchor.yaml",
        )]
        report, _ = validator.validate_ablation_configs(paths)
        self.assertTrue(report["passed"], report["errors"])
        d = report["configs"][3]
        self.assertEqual(d["branch_depth"], 8)
        self.assertEqual(d["keep_ratio_per_level"], [0.5] * 8)
        self.assertEqual(d["acc_mode"], "assignment_acc")

    def test_missing_ego4d_keeps_all_real_gates_blocked(self):
        root = Path(__file__).resolve().parents[1]
        paths = [str(root / "opts" / name) for name in (
            "research_ablation_A_baseline.yaml",
            "research_ablation_B_query_modulation.yaml",
            "research_ablation_C_boundary_importance.yaml",
            "research_ablation_D_adaptive_anchor.yaml",
        )]
        args = SimpleNamespace(
            configs=paths, small_set_size=1, diagnostic_batches=2,
            diagnostic_batch_size=2, gt_queries_per_batch=2, overfit_steps=2,
            smoke_steps=5, compute_warmup=1, compute_iterations=2,
            compute_budget_tolerance=0.1, boundary_neighborhood=3.0, seed=7,
        )
        missing = [
            "data/ego4d/annotations/ego4d_egovlp.json",
            "data/ego4d/egovlp_features/video",
            "data/ego4d/egovlp_features/text/token_768d",
        ]
        with mock.patch.object(validator, "missing_data_paths", return_value=missing):
            report = validator.run(args)
        self.assertEqual(report["status"], "blocked_missing_ego4d")
        self.assertEqual(report["formal_training_gate"], "closed")
        self.assertFalse(report["long_training_started"])
        self.assertFalse(report["substitute_dataset_used"])
        for key in (
            "temporal_diagnostics", "real_gt_round_trip",
            "region_statistics", "query_specific_grouping",
            "regression_distribution", "small_set_overfit",
            "mixed_precision", "ad_compute_comparison",
        ):
            self.assertEqual(report[key]["status"], "blocked_missing_ego4d")


if __name__ == "__main__":
    unittest.main()
