"""Numerical checks for pooled Gaussian source construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from nab.covariance import (  # noqa: E402
    ConditionalDiagonalAnchor,
    ResidualMomentModel,
    condition_features,
)


class CovarianceSourceTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(4)
        transform = torch.tensor(
            [
                [1.8, 0.0, 0.0],
                [0.5, 0.7, 0.0],
                [0.0, 0.4, 0.5],
                [0.0, 0.0, 0.3],
                [0.2, 0.0, 0.0],
                [0.0, 0.1, 0.2],
            ],
            dtype=torch.float64,
        )
        latent = torch.randn((4000, 3), generator=generator, dtype=torch.float64)
        self.residuals = latent @ transform.T
        self.model = ResidualMomentModel.fit(self.residuals)

    def test_sample_covariances(self):
        alpha_bar = 0.63
        for method in ("vp", "diagonal", "lowrank", "full"):
            generator = torch.Generator().manual_seed(10)
            samples = self.model.sample_perturbation(
                method,
                alpha_bar,
                batch=80_000,
                device=torch.device("cpu"),
                generator=generator,
                rank=3,
                dtype=torch.float64,
            ).flatten(1)
            empirical = samples.T @ samples / samples.shape[0]
            target = self.model.source_covariance(method, alpha_bar, rank=3)
            relative_error = torch.linalg.norm(empirical - target) / torch.linalg.norm(target)
            self.assertLess(float(relative_error), 0.02)

    def test_analytic_cross_entropy_matches_monte_carlo(self):
        alpha_bar = 0.47
        heldout = 0.9 * self.residuals[:1000]
        analytic = self.model.heldout_cross_entropy(
            "full", alpha_bar, heldout, rank=3
        )
        generator = torch.Generator().manual_seed(21)
        noise = torch.randn(heldout.shape, generator=generator, dtype=torch.float64)
        forward = alpha_bar**0.5 * heldout + (1.0 - alpha_bar) ** 0.5 * noise
        covariance = self.model.source_covariance("full", alpha_bar, rank=3)
        sign, logdet = torch.linalg.slogdet(covariance)
        self.assertGreater(float(sign), 0.0)
        quadratic = (forward * torch.linalg.solve(covariance, forward.T).T).sum(1)
        monte_carlo = 0.5 * (
            quadratic.mean() + logdet + self.model.dimension * torch.log(torch.tensor(2.0 * torch.pi))
        )
        self.assertLess(abs(float(monte_carlo) - analytic), 0.08)

    def test_conditional_diagonal_source(self):
        batch = 60_000
        conditions = {
            "start": torch.zeros((batch, 2), dtype=torch.float64),
            "goal": torch.ones((batch, 2), dtype=torch.float64),
            "obs": torch.zeros((batch, 3, 3), dtype=torch.float64),
            "mask": torch.zeros((batch, 3), dtype=torch.float64),
        }
        feature_dim = condition_features(conditions).shape[1]
        residual_scale = torch.tensor([0.4, 0.7, 1.1], dtype=torch.float32)
        anchor = ConditionalDiagonalAnchor(
            input_mean=torch.zeros(feature_dim),
            input_scale=torch.ones(feature_dim),
            residual_scale=residual_scale,
            sample_shape=(3,),
            hidden=8,
        ).to(dtype=torch.float64)
        alpha_bar = 0.58
        generator = torch.Generator().manual_seed(31)
        with torch.no_grad():
            samples = anchor.sample_perturbation(
                conditions, alpha_bar, generator=generator
            )
        empirical = samples.square().mean(0)
        expected = (1.0 - alpha_bar) + alpha_bar * residual_scale.double().square()
        self.assertTrue(torch.allclose(empirical, expected, rtol=0.02, atol=0.01))


if __name__ == "__main__":
    unittest.main()
