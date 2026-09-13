"""Shifted diffusion in the equivalence parameterization.

Every shifted process x_t = sqrt(abar_t) x_0 + s_t mu + sqrt(1-abar_t) eps is
handled through the change of variables y_t = x_t - s_t mu, which is a
standard VP diffusion (Theorem 2.1 of the paper).  All variants
use the same y-space sampling updates; they differ in the network input x_t = y_t +
s_t mu, in the prior anchor, and in where sampling starts (truncation).

Shift schedules (as functions of abar):
    zero    s = 0                          (conditional DDPM)
    nab     s = 1 - sqrt(abar)             (nominal-anchored bridge)
    stddpm  s = abar^{1/4} (1 - abar^{1/4})   (ST-DDPM / ShiftDDPMs)
"""

import math
import torch

from . import env


def cosine_alpha_bar(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-6, 1.0)
    return ab[1:].float()  # abar[t] for t = 0..T-1


SHIFTS = {
    "zero": lambda ab: torch.zeros_like(ab),
    "nab": lambda ab: 1.0 - ab.sqrt(),
    "stddpm": lambda ab: ab**0.25 * (1.0 - ab**0.25),
}


def score_evaluation_count(n_steps, t_start):
    """Actual calls for the rounded time grid, including clean reconstruction."""
    return min(int(n_steps) + 1, int(t_start) + 1)


class Diffusion:
    def __init__(self, T=256, shift="zero", device="cuda"):
        self.T = T
        self.shift = shift
        self.abar = cosine_alpha_bar(T).to(device)
        self.s = SHIFTS[shift](self.abar)

    # ---------------- training ----------------
    def loss(self, model, x0, cond):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        ab = self.abar[t][:, None, None]
        eps = torch.randn_like(x0)
        y_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        x_t = y_t + self.s[t][:, None, None] * cond["nominal"]
        pred = model(x_t, t, cond)
        return torch.nn.functional.mse_loss(pred, eps)

    # ---------------- sampling ----------------
    def guidance_grad(self, x0_hat, cond, clip=20.0):
        """Preconditioned plug-in cost gradient.  The raw gradient is
        dominated by the stiff control-energy Hessian (condition number
        ~1e6), so we apply the natural metric M = (H_energy + I)^{-1}: the
        result is a Gauss--Newton-type direction toward the smooth local
        minimizer, with obstacle-penalty components smoothed."""
        if not hasattr(self, "_precond"):
            self._precond = env.energy_precond(x0_hat.device)
        x = x0_hat.detach().requires_grad_(True)
        xp = torch.cat([cond["start"][:, None], x[:, 1:-1], cond["goal"][:, None]], 1)
        # guidance cost: same energy, harder obstacle term (the sampler's J
        # need not equal the data-generating J; feasibility is a constraint)
        J = env.cost(xp, cond["obs"], cond["mask"], margin=0.04, w_obs=4000.0)
        (g,) = torch.autograd.grad(J.sum(), x)
        g_int = torch.einsum("ij,bjd->bid", self._precond, g[:, 1:-1])
        g = torch.cat([torch.zeros_like(g[:, :1]), g_int,
                       torch.zeros_like(g[:, :1])], dim=1)
        n = g.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (clip / n).clamp(max=1.0)
        return g * scale[..., None]

    @torch.no_grad()
    def sample(self, model, cond, n_steps, eta=0.0, t_start=None, anchor="mean",
               guidance_w=0.0, guide_gate=0.3, gen=None,
               pin_idx=None, pin_pts=None, guide_fn=None, anchor_traj=None,
               guide_xt=None, source_perturbation=None):
        """DDIM/ancestral sampling in y-space.

        n_steps: interpolation intervals requested before duplicate time
                 indices are removed; not the number of score evaluations.
                 The final clean reconstruction is an additional score call.
        t_start: last (highest) diffusion index to start from (default T-1).
        anchor:  'mean' -> y init = sqrt(abar) a + sigma eps  (correct anchor)
                 'none' -> y init = sigma eps                 (naive/ST-DDPM ES)
        anchor_traj: x0-guess used for the anchored init (default: nominal).
        pin_idx/pin_pts: known coordinates to inpaint (default: endpoints =
        start/goal, the obstacle-env convention).  guide_fn: callable
        x0_hat -> cost gradient (default: obstacle-env preconditioned cost).
        source_perturbation: optional tensor with the same shape as the nominal.
        When supplied, it replaces sqrt(1-abar_s) times iid Gaussian noise and
        is added to the anchored source mean.  This supports diagonal, low-rank,
        or full covariance warm starts without changing the reverse sampler.
        """
        mu = cond["nominal"]
        B, dev = mu.shape[0], mu.device
        t_start = self.T - 1 if t_start is None else int(t_start)
        ts = torch.linspace(0, t_start, n_steps + 1).round().long().unique().flip(0)
        ab_s = self.abar[ts[0]]
        if source_perturbation is None:
            noise = torch.randn(mu.shape, generator=gen, device=dev)
            y = (1 - ab_s).sqrt() * noise
        else:
            if source_perturbation.shape != mu.shape:
                raise ValueError("source_perturbation must match nominal shape")
            y = source_perturbation.to(device=dev, dtype=mu.dtype)
        if anchor == "mean":
            y = y + ab_s.sqrt() * (mu if anchor_traj is None else anchor_traj)

        if pin_idx is None:
            pin_idx = [0, mu.shape[1] - 1]
            pin_pts = torch.stack([cond["start"], cond["goal"]], 1)
        if guide_fn is None:
            guide_fn = lambda x0h: self.guidance_grad(x0h, cond)
        pin = pin_pts
        for i in range(len(ts) - 1):
            t, t_next = ts[i], ts[i + 1]
            ab_t, ab_n = self.abar[t], self.abar[t_next]
            x_t = y + self.s[t] * mu
            tb = torch.full((B,), t, device=dev, dtype=torch.long)
            eps_hat = model(x_t, tb, cond)
            if guide_xt is not None:
                # amortized value guidance: score shift -grad log h at x_t,
                # valid at every t (no gate, weight baked into the field)
                eps_hat = eps_hat + (1 - ab_t).sqrt() * guide_xt(x_t, tb)
            if guidance_w > 0 and ab_t >= guide_gate:
                # Gate plug-in cost guidance to signal levels at or above
                # guide_gate, near the data end of the reverse chain.
                x0_hat = (y - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
                with torch.enable_grad():
                    g = guide_fn(x0_hat)
                eps_hat = eps_hat + guidance_w * (1 - ab_t).sqrt() * g
            x0_hat = (y - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x0_hat = x0_hat.clamp(-2.5, 2.5)
            sigma = eta * ((1 - ab_n) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_n).sqrt()
            dir_coef = (1 - ab_n - sigma**2).clamp_min(0).sqrt()
            y = ab_n.sqrt() * x0_hat + dir_coef * eps_hat
            if eta > 0:
                y = y + sigma * torch.randn(y.shape, generator=gen, device=dev)
            # replacement inpainting of known coordinates
            rep = ab_n.sqrt() * pin + (1 - ab_n).sqrt() * \
                torch.randn(pin.shape, generator=gen, device=dev)
            for j, idx in enumerate(pin_idx):
                y[:, idx] = rep[:, j]

        # final jump to clean sample
        t = ts[-1]
        ab_t = self.abar[t]
        x_t = y + self.s[t] * mu
        tb = torch.full((B,), t, device=dev, dtype=torch.long)
        eps_hat = model(x_t, tb, cond)
        if guide_xt is not None:
            eps_hat = eps_hat + (1 - ab_t).sqrt() * guide_xt(x_t, tb)
        if guidance_w > 0 and ab_t >= guide_gate:
            x0_hat = (y - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            with torch.enable_grad():
                g = guide_fn(x0_hat)
            eps_hat = eps_hat + guidance_w * (1 - ab_t).sqrt() * g
        x0 = (y - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
        for j, idx in enumerate(pin_idx):
            x0[:, idx] = pin[:, j]
        return x0

    def t_start_rule(self, R, kappa=0.5):
        """Smallest t with sqrt(abar_t) R <= kappa sqrt(1-abar_t): truncation
        point where the anchor-init W2 error drops below kappa x noise scale."""
        ok = (self.abar.sqrt() * R <= kappa * (1 - self.abar).sqrt())
        idx = torch.nonzero(ok)
        return int(idx[0]) if len(idx) else self.T - 1
