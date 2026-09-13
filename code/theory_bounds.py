"""Reproduce the analytic validation figure for the manuscript.

The clean law is a symmetric two-component distribution supported at +/-m.
After VP noising, q_t is an exactly evaluable Gaussian mixture.  We compare
Monte Carlo KL estimates with the trace and covariance-aware log-det bounds
proved in the paper.  No trained model or external data are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from nab.plot_style import finish_figure


def log_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of a full-rank Gaussian for row-wise samples."""
    chol = np.linalg.cholesky(cov)
    centered = x - mean
    solved = np.linalg.solve(chol, centered.T).T
    logdet = 2.0 * np.log(np.diag(chol)).sum()
    d = x.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + (solved**2).sum(1))


def mixture_kl(
    alpha_bar: float,
    mode: np.ndarray,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    base_noise: np.ndarray,
    signs: np.ndarray,
) -> float:
    """Common-random-number estimate of KL(q_t || target Gaussian)."""
    sigma2 = 1.0 - alpha_bar
    means = np.stack([np.sqrt(alpha_bar) * mode, -np.sqrt(alpha_bar) * mode])
    chosen = means[(signs > 0).astype(int)]
    x = chosen + np.sqrt(sigma2) * base_noise
    eye_cov = sigma2 * np.eye(mode.size)
    log_components = np.stack(
        [log_gaussian(x, means[0], eye_cov), log_gaussian(x, means[1], eye_cov)],
        axis=1,
    )
    log_q = logsumexp(log_components, axis=1) - np.log(2.0)
    log_nu = log_gaussian(x, target_mean, target_cov)
    return float(np.mean(log_q - log_nu))


def make_figure(output: Path, seed: int = 7, n_mc: int = 200_000) -> None:
    rng = np.random.default_rng(seed)
    d = 8
    mode = np.zeros(d)
    mode[0] = 2.5
    clean_cov = np.outer(mode, mode)
    base_noise = rng.standard_normal((n_mc, d))
    signs = rng.choice(np.array([-1, 1]), size=n_mc)

    alpha_grid = np.linspace(0.015, 0.94, 32)
    exact_iso, exact_cov, trace_bound, logdet_bound = [], [], [], []
    zero = np.zeros(d)
    for alpha_bar in alpha_grid:
        sigma2 = 1.0 - alpha_bar
        ratio = alpha_bar / sigma2
        cov_t = sigma2 * np.eye(d) + alpha_bar * clean_cov
        exact_iso.append(
            mixture_kl(alpha_bar, mode, zero, sigma2 * np.eye(d), base_noise, signs)
        )
        exact_cov.append(
            mixture_kl(alpha_bar, mode, zero, cov_t, base_noise, signs)
        )
        trace_bound.append(0.5 * ratio * np.trace(clean_cov))
        logdet_bound.append(0.5 * np.linalg.slogdet(np.eye(d) + ratio * clean_cov)[1])

    alpha_bias = 0.55
    sigma2_bias = 1.0 - alpha_bias
    base_bias_kl = mixture_kl(
        alpha_bias,
        mode,
        zero,
        sigma2_bias * np.eye(d),
        base_noise,
        signs,
    )
    biases = np.linspace(0.0, 2.0, 17)
    bias_mc, bias_formula = [], []
    for bias in biases:
        target_mean = np.zeros(d)
        target_mean[0] = np.sqrt(alpha_bias) * bias
        kl = mixture_kl(
            alpha_bias,
            mode,
            target_mean,
            sigma2_bias * np.eye(d),
            base_noise,
            signs,
        )
        bias_mc.append(kl - base_bias_kl)
        bias_formula.append(0.5 * alpha_bias / sigma2_bias * bias**2)

    dims = np.arange(1, 129)
    flat_trace, flat_logdet, decay_trace, decay_logdet = [], [], [], []
    ratio = 1.5
    for dim in dims:
        flat = np.ones(dim)
        decay = 1.0 / np.arange(1, dim + 1) ** 1.5
        flat_trace.append(0.5 * ratio * flat.sum())
        flat_logdet.append(0.5 * np.log1p(ratio * flat).sum())
        decay_trace.append(0.5 * ratio * decay.sum())
        decay_logdet.append(0.5 * np.log1p(ratio * decay).sum())

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))

    ax = axes[0]
    ax.plot(alpha_grid, exact_iso, "o-", ms=3, label="MC KL, diffusion covariance")
    ax.plot(alpha_grid, trace_bound, "--", label="trace upper bound")
    ax.plot(alpha_grid, exact_cov, "o-", ms=3, label="MC KL, matched covariance")
    ax.plot(alpha_grid, logdet_bound, "--", label="log-det upper bound")
    ax.set_yscale("log")
    ax.set_xlabel(r"signal level $\bar\alpha_t$")
    ax.set_ylabel("KL divergence")
    ax.set_title("(a) covariance-aware truncation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8)

    ax = axes[1]
    ax.plot(biases, bias_mc, "o", ms=3.5, label="Monte Carlo excess KL")
    ax.plot(biases, bias_formula, "-", label="exact quadratic identity")
    ax.set_xlabel(r"anchor bias $\|m-m^*\|$")
    ax.set_ylabel("excess KL")
    ax.set_title("(b) misspecified anchor")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(dims, flat_trace, label="trace, flat spectrum")
    ax.plot(dims, flat_logdet, "--", label="log-det, flat spectrum")
    ax.plot(dims, decay_trace, label="trace, decaying spectrum")
    ax.plot(dims, decay_logdet, "--", label="log-det, decaying spectrum")
    ax.set_xlabel("ambient dimension")
    ax.set_ylabel("initialization bound")
    ax.set_title("(c) effective-rank dependence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.8)

    finish_figure(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "theory_bounds.pdf",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=200_000)
    args = parser.parse_args()
    make_figure(args.output, args.seed, args.samples)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
