"""Check cached experiment schemas and score-call metadata."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from nab.diffusion import score_evaluation_count


def result(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


class ResultSchemaTests(unittest.TestCase):
    def test_rank_families(self):
        data = result("rank_mi.json")
        comparison = data["rank_comparison"]
        ranks = data["metadata"]["ranks"]
        control = comparison["single_axis_family"]["kl_by_rank"]
        self.assertEqual(len(control), len(ranks))
        self.assertEqual(len(set(control)), 1)
        for row in comparison["product_family"]["theta_results"].values():
            self.assertEqual(row["kl_by_rank"],
                             [k * row["one_dimensional_kl"] for k in ranks])

    def test_training_log(self):
        data = result("wsd_training.json")
        rows = data["training_log"]
        self.assertTrue(rows)
        self.assertEqual([row["epoch"] for row in rows], list(range(1, len(rows) + 1)))
        self.assertIn(data["best_epoch"], [row["epoch"] for row in rows])
        for row in rows:
            self.assertIn("train_nll_per_dim", row)
            self.assertIn("validation_nll_per_dim", row)

    def test_score_call_metadata(self):
        for name in ("covariance_ablation.json", "covariance_nfe8.json"):
            metadata = result(name)["metadata"]
            self.assertEqual(metadata["actual_score_nfe"],
                             score_evaluation_count(metadata["nfe"], 123))
        metadata = result("calibration.json")["metadata"]
        for level in metadata["t_starts"]:
            self.assertEqual(metadata["actual_score_nfe_warm"],
                             score_evaluation_count(metadata["nfe_warm"], level))
        self.assertEqual(metadata["actual_score_nfe_reference"],
                         score_evaluation_count(metadata["nfe_reference"], 255))


if __name__ == "__main__":
    unittest.main()
