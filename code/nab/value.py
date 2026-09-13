"""Amortized value guidance.

The exact guidance drift for the tilted target pi_lambda ~ e^{-J/lambda} p
is g^2 grad log h_t with h_t(x) = E[e^{-J(x_0)/lambda} | x_t = x].  Since a
conditional expectation is an L2 projection, h_t can be learned by plain
regression on forward-diffused data:

    min_psi  E | exp(l_psi(x_t, t, c, w)) - exp(-w (J(x_0) - J*(c))) |^2,

with w = 1/lambda an input (log-uniform at training).  The per-condition
shift J*(c) multiplies h by a constant and leaves grad log h unchanged.
The network outputs l_psi ~ log h; sampling uses grad_x l_psi directly, so
no division by h is needed and no alpha-bar gate is required (the learned
field is valid at every t).

Usage: python -m nab.value --tag pend   (train on pendulum data)
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn

from . import env_pend as ep
from .diffusion import Diffusion
from .model import timestep_embedding


class ValueNet(nn.Module):
    """Scalar log-h estimator: conv encoder over time + FiLM conditioning."""

    def __init__(self, x_dim=1, feat_dim=1, cvec_dim=3, ch=64, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.cond = nn.Sequential(nn.Linear(cvec_dim + emb_dim + 1, emb_dim),
                                  nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.enc = nn.Sequential(
            nn.Conv1d(x_dim + feat_dim + 1, ch, 5, padding=2), nn.SiLU(),
            nn.Conv1d(ch, ch * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(ch * 2, ch * 4, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(ch * 4, ch * 4, 4, stride=2, padding=1), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(ch * 4 + emb_dim, 256), nn.SiLU(),
                                  nn.Linear(256, 256), nn.SiLU(),
                                  nn.Linear(256, 1))

    def forward(self, x_t, t, cond, logw):
        B, L, _ = x_t.shape
        tau = torch.linspace(0, 1, L, device=x_t.device)[None, :, None].expand(B, L, 1)
        z = torch.cat([x_t, cond["feat"], tau], -1).permute(0, 2, 1)
        z = self.enc(z).mean(-1)                                   # (B, ch*4)
        emb = self.cond(torch.cat([cond["cvec"],
                                   timestep_embedding(t, self.emb_dim),
                                   logw[:, None]], -1))
        out = self.head(torch.cat([z, emb], -1))[:, 0]
        return out.clamp(max=2.0)                                  # log h <= 2


def train(tag="pend", steps=25000, batch=256, lr=2e-4, T=256, shift="nab"):
    device = "cuda"
    torch.manual_seed(0)
    d = torch.load(f"data/{tag}_train.pt", weights_only=True)
    d = {k: v.to(device) for k, v in d.items()}
    N = d["traj"].shape[0]
    J_all = ep.cost(d["traj"][..., 0] * torch.pi, d["umax"])
    gap = (J_all - d["best_J"]).clamp_min(0)                       # (N,)

    diff = Diffusion(T=T, shift=shift, device=device)
    net = ValueNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (batch,), device=device)
        x0 = d["traj"][idx]
        cond = {"feat": d["feat"][idx], "cvec": d["cvec"][idx],
                "nominal": d["nominal"][idx]}
        t = torch.randint(0, T, (batch,), device=device)
        ab = diff.abar[t][:, None, None]
        eps = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + diff.s[t][:, None, None] * cond["nominal"] + \
            (1 - ab).sqrt() * eps
        logw = (math.log(0.25) + (math.log(4.0) - math.log(0.25)) *
                torch.rand(batch, device=device))
        y = torch.exp(-logw.exp() * gap[idx])
        pred = net(x_t, t, cond, logw)
        loss = ((pred.exp() - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 5000 == 0 or step == 1:
            print(f"[value/{tag}] {step} loss {loss.item():.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs("runs", exist_ok=True)
    torch.save({"net": net.state_dict(), "T": T, "shift": shift},
               f"runs/{tag}_value.pt")
    print("saved value net")


def make_guide_xt(net, cond, w, clip=20.0):
    """Returns guide_xt(x_t, t) = -grad_x log h  (drift added as
    eps_hat += sqrt(1-abar) * out, so out = -grad log h)."""
    logw = torch.full((cond["cvec"].shape[0],), math.log(w),
                      device=cond["cvec"].device)

    def guide(x_t, t):
        with torch.enable_grad():
            x = x_t.detach().requires_grad_(True)
            l = net(x, t, cond, logw)
            (g,) = torch.autograd.grad(l.sum(), x)
        n = g.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        g = g * (clip / n).clamp(max=1.0)[..., None]
        return -g
    return guide


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pend")
    ap.add_argument("--steps", type=int, default=25000)
    args = ap.parse_args()
    train(tag=args.tag, steps=args.steps)
