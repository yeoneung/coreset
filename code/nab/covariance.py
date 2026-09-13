"""Gaussian source models for covariance-aware diffusion warm starts.

The residual model is fitted once from clean-space residuals relative to a
fixed anchor.  We use the second moment E[rr^T], rather than centering by the
pooled residual mean, because the experiment keeps the nominal anchor fixed.
For a source level with a = abar_s, the Gaussian perturbation covariance is

    S_s = (1-a) I + a M,

where M is approximated by a diagonal, a low-rank eigensystem, or the full
pooled residual second moment.  The VP baseline corresponds to M = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn


@dataclass
class ResidualMomentModel:
    """Spectral representation of a residual second moment."""

    sample_shape: Tuple[int, ...]
    diagonal: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    n_fit: int

    @property
    def dimension(self) -> int:
        return int(self.diagonal.numel())

    @property
    def trace(self) -> float:
        return float(self.eigenvalues.sum())

    @classmethod
    def fit(cls, residuals: torch.Tensor) -> "ResidualMomentModel":
        """Fit E[rr^T] from residuals of shape (n, ...)."""
        if residuals.ndim < 2:
            raise ValueError("residuals must have a sample axis and feature axes")
        shape = tuple(residuals.shape[1:])
        flat = residuals.detach().to(device="cpu", dtype=torch.float64).flatten(1)
        moment = flat.T @ flat / flat.shape[0]
        moment = 0.5 * (moment + moment.T)
        values, vectors = torch.linalg.eigh(moment)
        order = torch.argsort(values, descending=True)
        values = values[order].clamp_min(0.0)
        vectors = vectors[:, order]
        return cls(
            sample_shape=shape,
            diagonal=torch.diagonal(moment).clone(),
            eigenvalues=values,
            eigenvectors=vectors,
            n_fit=int(flat.shape[0]),
        )

    def explained_fraction(self, ranks: Iterable[int]) -> Dict[str, float]:
        total = self.eigenvalues.sum().clamp_min(torch.finfo(torch.float64).tiny)
        return {
            str(int(rank)): float(self.eigenvalues[: int(rank)].sum() / total)
            for rank in ranks
        }

    def residual_moment(self, method: str, rank: int = 4) -> torch.Tensor:
        """Return the clean residual second-moment approximation on CPU."""
        d = self.dimension
        if method == "vp":
            return torch.zeros((d, d), dtype=torch.float64)
        if method == "diagonal":
            return torch.diag(self.diagonal)
        if method == "lowrank":
            k = min(int(rank), d)
            u = self.eigenvectors[:, :k]
            return (u * self.eigenvalues[:k]) @ u.T
        if method == "full":
            return (self.eigenvectors * self.eigenvalues) @ self.eigenvectors.T
        raise ValueError(f"unknown covariance method: {method}")

    def source_covariance(self, method: str, alpha_bar: float, rank: int = 4) -> torch.Tensor:
        a = float(alpha_bar)
        eye = torch.eye(self.dimension, dtype=torch.float64)
        return (1.0 - a) * eye + a * self.residual_moment(method, rank=rank)

    def sample_perturbation(
        self,
        method: str,
        alpha_bar: float,
        batch: int,
        device: torch.device,
        generator: torch.Generator,
        rank: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Draw a zero-mean source perturbation with covariance S_s."""
        a = torch.as_tensor(float(alpha_bar), device=device, dtype=dtype)
        base = (1.0 - a).sqrt() * torch.randn(
            (batch, self.dimension), device=device, dtype=dtype, generator=generator
        )
        if method == "vp":
            out = base
        elif method == "diagonal":
            diag = self.diagonal.to(device=device, dtype=dtype)
            out = ((1.0 - a) + a * diag).sqrt() * torch.randn(
                (batch, self.dimension), device=device, dtype=dtype, generator=generator
            )
        elif method in ("lowrank", "full"):
            k = min(int(rank), self.dimension) if method == "lowrank" else self.dimension
            values = self.eigenvalues[:k].to(device=device, dtype=dtype)
            vectors = self.eigenvectors[:, :k].to(device=device, dtype=dtype)
            latent = torch.randn((batch, k), device=device, dtype=dtype, generator=generator)
            out = base + a.sqrt() * ((latent * values.sqrt()) @ vectors.T)
        else:
            raise ValueError(f"unknown covariance method: {method}")
        return out.reshape((batch,) + self.sample_shape)

    def heldout_cross_entropy(
        self,
        method: str,
        alpha_bar: float,
        heldout_residuals: torch.Tensor,
        rank: int = 4,
    ) -> float:
        """Expected Gaussian NLL of a held-out forward marginal.

        The expectation over forward Gaussian noise is evaluated analytically;
        only the held-out clean residual distribution is empirical.
        """
        flat = heldout_residuals.detach().to(device="cpu", dtype=torch.float64).flatten(1)
        heldout_moment = flat.T @ flat / flat.shape[0]
        a = float(alpha_bar)
        target_moment = (1.0 - a) * torch.eye(self.dimension, dtype=torch.float64)
        target_moment = target_moment + a * heldout_moment
        source_cov = self.source_covariance(method, a, rank=rank)
        sign, logdet = torch.linalg.slogdet(source_cov)
        if sign <= 0:
            raise RuntimeError("source covariance is not positive definite")
        trace_term = torch.trace(torch.linalg.solve(source_cov, target_moment))
        return float(0.5 * (trace_term + logdet + self.dimension * log(2.0 * pi)))


def condition_features(conditions: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Vectorize the obstacle-planning context used by the WSD baseline.

    Inactive obstacles are zeroed before flattening, while their mask is kept
    explicitly.  The resulting vector contains start, goal, obstacle triples,
    and the activity mask; it does not use the target trajectory.
    """
    required = ("start", "goal", "obs", "mask")
    missing = [key for key in required if key not in conditions]
    if missing:
        raise KeyError(f"missing condition fields: {missing}")
    masked_obstacles = conditions["obs"] * conditions["mask"][..., None]
    return torch.cat(
        [
            conditions["start"],
            conditions["goal"],
            masked_obstacles.flatten(1),
            conditions["mask"],
        ],
        dim=1,
    )


class ConditionalDiagonalAnchor(nn.Module):
    """WSD-style conditional Gaussian model for nominal residuals.

    The network predicts both a residual mean and marginal residual variance.
    Per-coordinate residual scaling makes the zero-output initialization equal
    to the pooled diagonal Gaussian, so improvements after training come from
    conditioning on the context.  At diffusion level ``s`` the clean residual
    variance ``v(c)`` becomes ``(1-a) I + a diag(v(c))``.
    """

    def __init__(
        self,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        residual_scale: torch.Tensor,
        sample_shape: Tuple[int, ...],
        hidden: int = 192,
        min_log_variance: float = -8.0,
        max_log_variance: float = 5.0,
    ):
        super().__init__()
        input_mean = input_mean.detach().to(dtype=torch.float32).flatten()
        input_scale = input_scale.detach().to(dtype=torch.float32).flatten()
        residual_scale = residual_scale.detach().to(dtype=torch.float32).flatten()
        if input_mean.shape != input_scale.shape:
            raise ValueError("input mean and scale must have the same shape")
        if residual_scale.numel() != int(torch.tensor(sample_shape).prod()):
            raise ValueError("residual scale is incompatible with sample shape")
        self.sample_shape = tuple(int(v) for v in sample_shape)
        self.hidden = int(hidden)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_scale", input_scale.clamp_min(1e-4))
        self.register_buffer("residual_scale", residual_scale.clamp_min(1e-3))
        self.backbone = nn.Sequential(
            nn.Linear(input_mean.numel(), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, residual_scale.numel())
        self.log_variance_head = nn.Linear(hidden, residual_scale.numel())
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

    @property
    def dimension(self) -> int:
        return int(self.residual_scale.numel())

    def forward(self, conditions: Mapping[str, torch.Tensor]):
        features = condition_features(conditions)
        normalized = (features - self.input_mean) / self.input_scale
        hidden = self.backbone(normalized)
        mean = self.mean_head(hidden) * self.residual_scale
        log_variance = self.log_variance_head(hidden).clamp(
            self.min_log_variance, self.max_log_variance
        )
        variance = log_variance.exp() * self.residual_scale.square()
        shape = (features.shape[0],) + self.sample_shape
        return mean.reshape(shape), variance.reshape(shape)

    def clean_nll(
        self,
        conditions: Mapping[str, torch.Tensor],
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        """Mean clean-space Gaussian NLL per residual coordinate."""
        mean, variance = self(conditions)
        return 0.5 * (
            (residuals - mean).square() / variance
            + variance.log()
            + log(2.0 * pi)
        ).mean()

    def source_parameters(
        self,
        conditions: Mapping[str, torch.Tensor],
        alpha_bar: float,
    ):
        """Return the residual mean and diagonal source covariance at a level."""
        mean, variance = self(conditions)
        a = torch.as_tensor(float(alpha_bar), device=mean.device, dtype=mean.dtype)
        source_mean = a.sqrt() * mean
        source_variance = (1.0 - a) + a * variance
        return source_mean, source_variance

    def sample_perturbation(
        self,
        conditions: Mapping[str, torch.Tensor],
        alpha_bar: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Draw the centered part of the WSD conditional Gaussian source."""
        _, source_variance = self.source_parameters(conditions, alpha_bar)
        noise = torch.randn(
            source_variance.shape,
            device=source_variance.device,
            dtype=source_variance.dtype,
            generator=generator,
        )
        return source_variance.sqrt() * noise

    @torch.no_grad()
    def heldout_cross_entropy(
        self,
        conditions: Mapping[str, torch.Tensor],
        alpha_bar: float,
        heldout_residuals: torch.Tensor,
    ) -> float:
        """Analytic source NLL averaged over held-out forward Gaussian noise."""
        mean, variance = self(conditions)
        residuals = heldout_residuals.to(device=mean.device, dtype=mean.dtype)
        a = torch.as_tensor(float(alpha_bar), device=mean.device, dtype=mean.dtype)
        source_variance = (1.0 - a) + a * variance
        quadratic = ((1.0 - a) + a * (residuals - mean).square()) / source_variance
        nll = 0.5 * (quadratic + source_variance.log() + log(2.0 * pi))
        return float(nll.flatten(1).sum(1).mean().cpu())

    def checkpoint(self) -> Dict[str, object]:
        return {
            "state_dict": self.state_dict(),
            "sample_shape": list(self.sample_shape),
            "hidden": self.hidden,
            "min_log_variance": self.min_log_variance,
            "max_log_variance": self.max_log_variance,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, object],
        device: torch.device,
    ) -> "ConditionalDiagonalAnchor":
        state = checkpoint["state_dict"]
        model = cls(
            input_mean=state["input_mean"],
            input_scale=state["input_scale"],
            residual_scale=state["residual_scale"],
            sample_shape=tuple(checkpoint["sample_shape"]),
            hidden=int(checkpoint["hidden"]),
            min_log_variance=float(checkpoint["min_log_variance"]),
            max_log_variance=float(checkpoint["max_log_variance"]),
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        return model
