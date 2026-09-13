"""Check reported NFE against actual sampler calls, including short grids."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nab.diffusion import Diffusion, score_evaluation_count


class ReportingTests(unittest.TestCase):
    def test_actual_score_calls(self):
        cond = {"nominal": torch.zeros(1, 64, 2),
                "start": torch.zeros(1, 2), "goal": torch.ones(1, 2)}
        diffusion = Diffusion(device="cpu")
        for intervals, start, expected in ((4, 123, 5), (8, 123, 9),
                                            (128, 255, 129), (32, 16, 17),
                                            (128, 123, 124)):
            with self.subTest(intervals=intervals, start=start):
                calls = []

                def model(x, t, cond):
                    calls.append(int(t[0]))
                    return torch.zeros_like(x)

                diffusion.sample(model, cond, intervals, t_start=start,
                                 gen=torch.Generator().manual_seed(1))
                self.assertEqual(len(calls), expected)
                self.assertEqual(score_evaluation_count(intervals, start), len(calls))
                self.assertEqual(calls[-1], 0)


if __name__ == "__main__":
    unittest.main()
