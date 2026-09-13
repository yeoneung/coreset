"""Controlled effective-rank study of the truncation certificate.

Synthetic conditional data have fixed ambient dimension d and varying latent
rank k:

    X_0 = s * delta * u_0 + U_k z + sigma * xi,   s = +-1 equiprobable.

Only the first coordinate in the U_k basis is non-Gaussian.  All other
coordinates are Gaussian factors shared exactly by q_s and its moment-matched
Gaussian g_s.  Consequently KL(q_s || g_s) is independent of k and d at
fixed signal level.  This script computes that KL by deterministic
one-dimensional quadrature, then obtains mutual information from

    B_s = KL(q_s || g_s) + I(X_0; X_s).

The single-axis mixture serves as a control for certificate conservatism.
A product family places an independent symmetric mixture in
every active direction, for which the actual KL equals k times a
one-dimensional divergence.  The remaining curves show how the sufficient
covariance certificate varies with latent rank.  Outputs results/rank_mi.json.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from nab.diffusion import cosine_alpha_bar  # noqa: E402


def mixture_kl_1d(
    a: float,
    delta: float,
    sigma: float,
    tolerance: float,
) -> tuple[float, float]:
    """KL from the active noisy mixture coordinate to its matched Gaussian."""
    mean = math.sqrt(a) * delta
    component_variance = 1.0 + a * sigma**2
    return symmetric_mixture_kl(mean, component_variance, tolerance)


def symmetric_mixture_kl(
    mean: float,
    component_variance: float,
    tolerance: float,
) -> tuple[float, float]:
    """KL from 0.5 N(-mean,v)+0.5 N(mean,v) to its matched Gaussian."""
    matched_variance = component_variance + mean**2
    log_component_norm = -0.5 * math.log(2.0 * math.pi * component_variance)
    log_matched_norm = -0.5 * math.log(2.0 * math.pi * matched_variance)

    def integrand(x: float) -> float:
        lp_plus = log_component_norm - (x - mean) ** 2 / (2.0 * component_variance)
        lp_minus = log_component_norm - (x + mean) ** 2 / (2.0 * component_variance)
        log_p = float(np.logaddexp(lp_plus, lp_minus) - math.log(2.0))
        log_g = log_matched_norm - x**2 / (2.0 * matched_variance)
        return math.exp(log_p) * (log_p - log_g)

    half_value, half_error = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=tolerance / 2.0,
        epsrel=tolerance,
        limit=300,
    )
    return max(0.0, 2.0 * half_value), 2.0 * half_error


def product_family_kl_1d(
    a: float,
    theta: float,
    active_variance: float,
    sigma: float,
    tolerance: float,
) -> tuple[float, float]:
    """One active-coordinate KL for the product-mixture comparison family."""
    mean = math.sqrt(a * theta * active_variance)
    component_variance = (
        1.0 - a + a * ((1.0 - theta) * active_variance + sigma**2)
    )
    return symmetric_mixture_kl(mean, component_variance, tolerance)


def logdet_certificate(rho: float, eigenvalues: torch.Tensor) -> float:
    return float(0.5 * torch.log1p(rho * eigenvalues).sum())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--grid-step", type=int, default=16)
    parser.add_argument("--quadrature-tolerance", type=float, default=1e-10)
    parser.add_argument("--comparison-signal", type=float, default=0.5)
    parser.add_argument("--product-lambda", type=float, default=1.0)
    parser.add_argument("--product-thetas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.2, 1.0])
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results" / "rank_mi.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    d = args.dimension
    if not args.ranks or min(args.ranks) < 1 or max(args.ranks) > d:
        raise ValueError("ranks must lie between 1 and the ambient dimension")
    abar = cosine_alpha_bar(args.T).double().cpu()
    quadrature_grid = list(range(args.grid_step, args.T, args.grid_step))

    results = {}
    max_quadrature_error = 0.0
    for k in args.ranks:
        # Cov(X_0) has eigenvalues 1+sigma^2+delta^2 in u_0,
        # 1+sigma^2 in the other active directions, and sigma^2 elsewhere.
        covariance_eigenvalues = torch.full((d,), args.sigma**2, dtype=torch.float64)
        covariance_eigenvalues[:k] += 1.0
        covariance_eigenvalues[0] += args.delta**2

        certificate = []
        trace_bound = []
        for t in range(args.T):
            a = float(abar[t])
            rho = a / (1.0 - a)
            certificate.append(logdet_certificate(rho, covariance_eigenvalues))
            trace_bound.append(float(0.5 * rho * covariance_eigenvalues.sum()))

        decomposition = []
        for t in quadrature_grid:
            a = float(abar[t])
            kl, error = mixture_kl_1d(
                a, args.delta, args.sigma, args.quadrature_tolerance
            )
            max_quadrature_error = max(max_quadrature_error, error)
            decomposition.append({
                "t": t,
                "abar": a,
                "B": certificate[t],
                "kl": kl,
                "mi": certificate[t] - kl,
                "kl_quadrature_error": error,
            })

        admissible = {}
        for eps in args.epsilons:
            t_logdet = next((t for t in range(args.T) if certificate[t] <= eps),
                            args.T - 1)
            t_trace = next((t for t in range(args.T) if trace_bound[t] <= eps),
                           args.T - 1)
            admissible[str(eps)] = {"logdet": t_logdet, "trace": t_trace}

        results[str(k)] = {
            "certificate": certificate,
            "trace_bound": trace_bound,
            "decomposition": decomposition,
            "admissible": admissible,
        }
        print(f"k={k} admissible={admissible}", flush=True)

    rank_spread = 0.0
    for row_index in range(len(quadrature_grid)):
        values = [results[str(k)]["decomposition"][row_index]["kl"] for k in args.ranks]
        rank_spread = max(rank_spread, max(values) - min(values))

    comparison_a = args.comparison_signal
    single_axis_kl, single_axis_error = mixture_kl_1d(
        comparison_a, args.delta, args.sigma, args.quadrature_tolerance
    )
    product_comparison = {}
    for theta in args.product_thetas:
        if not 0.0 <= theta <= 1.0:
            raise ValueError("product-family theta must lie in [0, 1]")
        one_dimensional_kl, error = product_family_kl_1d(
            comparison_a,
            theta,
            args.product_lambda,
            args.sigma,
            args.quadrature_tolerance,
        )
        max_quadrature_error = max(max_quadrature_error, error)
        product_comparison[str(theta)] = {
            "one_dimensional_kl": one_dimensional_kl,
            "kl_by_rank": [k * one_dimensional_kl for k in args.ranks],
            "quadrature_error": error,
        }

    rank_comparison = {
        "signal_level": comparison_a,
        "single_axis_family": {
            "kl_by_rank": [single_axis_kl for _ in args.ranks],
            "quadrature_error": single_axis_error,
        },
        "product_family": {
            "active_variance": args.product_lambda,
            "theta_results": product_comparison,
        },
    }

    payload = {
        "metadata": {
            "experiment": "controlled effective-rank certificate conservatism study",
            "kl_method": "deterministic one-dimensional adaptive quadrature",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "dimension": d,
            "ranks": args.ranks,
            "delta": args.delta,
            "sigma": args.sigma,
            "T": args.T,
            "quadrature_tolerance": args.quadrature_tolerance,
            "quadrature_grid": quadrature_grid,
            "comparison_signal": args.comparison_signal,
            "product_lambda": args.product_lambda,
            "product_thetas": args.product_thetas,
            "epsilons": args.epsilons,
            "max_reported_quadrature_error": max_quadrature_error,
            "actual_kl_max_spread_across_ranks": rank_spread,
        },
        "results": results,
        "rank_comparison": rank_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("max quadrature error:", max_quadrature_error)
    print("actual-KL spread across ranks:", rank_spread)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
