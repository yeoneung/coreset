"""Receding-horizon (MPC) experiment on the generalized obstacle world.

Closed loop: replan every EXEC points with a tiny NFE budget; execute with
small disturbances.  Anchor options per replan:
  - full        : full-range strided DDIM (no anchor)
  - nom         : anchored truncation at the nominal (cubic), KL-rule t_s
  - prev@t      : anchored at the previous solution's tail, truncation t
The experiment measures accepted-plan deviations from each anchor and
closed-loop outcomes. These deviations describe output consistency, not
the target-law residual required by the source-error bound.

Usage: python -m nab.mpc
"""

import json
import math
import os
import time

import torch

from . import env, env2
from .diffusion import Diffusion
from .model import UNetG

DEVICE = "cuda"
L = env.L
DT = env.DT
EXEC = 8
N_EP = 128
S_CAND = 4
NFE = 4  # Grid-interval setting; the reported runs use five score calls.
NOISE = 0.005


def load_model(name):
    ck = torch.load(f"runs/{name}.pt", weights_only=True)
    D, F, C = ck["dims"]
    m = UNetG(x_dim=D, feat_dim=F, cvec_dim=C).to(DEVICE)
    m.load_state_dict(ck["ema"])
    m.eval()
    return m, Diffusion(T=ck["T"], shift=ck["shift"], device=DEVICE), ck["R"]


def select_cost(traj, obs, mask):
    return env.control_energy(traj) + \
        4000.0 * (env.penetration(traj, obs, mask, 0.02) ** 2).sum(-1) * DT


def interp_at(plan, idx):
    """Linear interpolation of plan (B,L,2) at float indices idx (n,)."""
    lo = idx.floor().long().clamp(max=L - 2)
    frac = (idx - lo.float())[None, :, None]
    return plan[:, lo] * (1 - frac) + plan[:, lo + 1] * frac


@torch.no_grad()
def replan(model, diff, inst, anchor_traj, t_start, anchor, seed):
    """Sample S_CAND plans and pick the best by hardened cost. (B,L,2)"""
    B = inst["start"].shape[0]
    nom, feat, cvec = env2.featurize(inst)
    cond = {"nominal": nom, "feat": feat, "cvec": cvec}
    condr = {k: v.repeat_interleave(S_CAND, 0) for k, v in cond.items()}
    pin_idx, pin_pts = env2.pin_of(inst)
    pinr = pin_pts.repeat_interleave(S_CAND, 0)
    at = None if anchor_traj is None else anchor_traj.repeat_interleave(S_CAND, 0)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    plans = diff.sample(model, condr, n_steps=NFE, eta=0.0, t_start=t_start,
                        anchor=anchor, gen=gen, pin_idx=pin_idx, pin_pts=pinr,
                        guide_fn=lambda x: torch.zeros_like(x),
                        anchor_traj=at)
    obs_r = inst["obs"].repeat_interleave(S_CAND, 0)
    mask_r = inst["mask"].repeat_interleave(S_CAND, 0)
    J = select_cost(plans, obs_r, mask_r).view(B, S_CAND)
    best = J.argmin(dim=1)
    sel = plans.view(B, S_CAND, L, 2)[torch.arange(B, device=DEVICE), best]
    # cheap network-free polish of the selected plan (standard command
    # smoothing; identical across replanners); first two points pinned to
    # preserve the measured initial state
    sel[:, 1] = pin_pts[:, 1]
    with torch.enable_grad():
        sel = env.polish(sel, inst["obs"], inst["mask"],
                         pin_pts[:, 0], pin_pts[:, 2], k=150, fixed_head=2)
    return sel


@torch.no_grad()
def run_episodes(model, diff, ep_inst, mode, t_prev, ts_nom, seed0=100):
    """Shrinking-horizon MPC on a global clock of L-1 = 63 steps.

    Each replan plans the REMAINING task rescaled to the unit horizon (the
    plan-time velocity is v_global * T_rem); EXEC global steps are executed
    per replan by interpolating the plan at global grid times.  This keeps
    the executed trajectory on the fixed global grid, so energies are
    directly comparable to the open-loop oracle.
    mode: 'full' | 'nom' | 'prev'.
    """
    B = ep_inst["start"].shape[0]
    p = ep_inst["start"].clone()
    v = ep_inst["v0"].clone()                      # global-time velocity
    realized = [p.clone()[:, None]]
    commanded = [p.clone()[:, None]]
    prev_plan, prev_Trem, prev_nexec = None, None, None
    sig_sq = []
    wall, n_plan = 0.0, 0
    gen_d = torch.Generator(device=DEVICE).manual_seed(999)  # shared noise
    k_done, j = 0, 0
    K = L - 1                                      # 63 global steps
    while k_done < K:
        T_rem = 1.0 - k_done / K
        n_exec = min(EXEC, K - k_done)
        inst_j = {"start": p, "v0": v * T_rem, "goal": ep_inst["goal"],
                  "obs": ep_inst["obs"], "mask": ep_inst["mask"]}
        t0 = time.perf_counter()
        if mode in ("oracle", "oracle_warm"):
            Mo = 8
            x_init = None
            if mode == "oracle_warm" and prev_plan is not None:
                # standard MPC warm start: initialize the optimizer at the
                # previous solution's tail (small perturbations), plus two
                # fresh mixed starts for exploration
                idx0 = min(prev_nexec / prev_Trem, L - 1.001)
                grid = torch.linspace(idx0, L - 1, L, device=DEVICE)
                at = interp_at(prev_plan, grid)
                at[:, 0] = p
                at[:, 1] = p + v * T_rem * DT
                x_init = at[:, None].repeat(1, Mo, 1, 1)
                x_init[:, 1:6] += 0.02 * torch.randn(
                    x_init[:, 1:6].shape, device=DEVICE)
                tau_l = torch.linspace(0, 1, L, device=DEVICE)
                bump = torch.sin(torch.pi * tau_l)[None, :, None]
                nomj = env2.nominal(inst_j)
                x_init[:, 6] = nomj + 0.35 * bump
                x_init[:, 7] = nomj - 0.35 * bump
                x_init = x_init.reshape(B * Mo, L, 2)
            with torch.enable_grad():
                tr, e, feas_o, _ = env2.solve_oracle(
                    inst_j, n_inits=Mo, iters=300, gen=None, mixed=True,
                    x_init=x_init)
            Jo = select_cost(tr, inst_j["obs"].repeat_interleave(Mo, 0),
                             inst_j["mask"].repeat_interleave(Mo, 0))
            besto = Jo.view(B, Mo).argmin(dim=1)
            plan = tr.view(B, Mo, L, 2)[torch.arange(B, device=DEVICE), besto]
            if mode == "oracle_warm" and prev_plan is not None:
                sig_sq.append((plan - interp_at(
                    prev_plan, torch.linspace(
                        min(prev_nexec / prev_Trem, L - 1.001), L - 1, L,
                        device=DEVICE))).pow(2).sum(-1).mean().item())
        elif mode == "full":
            plan = replan(model, diff, inst_j, None, None, "none", seed0 + j)
        elif mode == "nom" or prev_plan is None:
            plan = replan(model, diff, inst_j, None, ts_nom, "mean", seed0 + j)
            if mode == "nom":
                nom = env2.nominal(inst_j)
                sig_sq.append((plan - nom).pow(2).sum(-1).mean().item())
        else:
            idx0 = min(prev_nexec / prev_Trem, L - 1.001)
            grid = torch.linspace(idx0, L - 1, L, device=DEVICE)
            at = interp_at(prev_plan, grid)
            at[:, 0] = p
            at[:, 1] = p + v * T_rem * DT
            plan = replan(model, diff, inst_j, at, t_prev, "mean", seed0 + j)
            sig_sq.append((plan - at).pow(2).sum(-1).mean().item())
        torch.cuda.synchronize()
        wall += time.perf_counter() - t0
        n_plan += 1
        # execute n_exec global steps: plan indices k / T_rem, k = 1..n_exec
        k = torch.arange(1, n_exec + 1, device=DEVICE, dtype=torch.float32)
        seg_cmd = interp_at(plan, (k / T_rem).clamp(max=L - 1))
        seg = seg_cmd + NOISE * torch.randn(seg_cmd.shape, generator=gen_d,
                                            device=DEVICE)
        realized.append(seg)
        commanded.append(seg_cmd)
        p = seg[:, -1]
        v_cmd = (seg_cmd[:, -1] - seg_cmd[:, -2]) / DT if n_exec >= 2 else \
            (seg_cmd[:, -1] - commanded[-2][:, -1]) / DT
        v = v_cmd + 0.1 * torch.randn(v_cmd.shape, generator=gen_d, device=DEVICE)
        prev_plan, prev_Trem, prev_nexec = plan, T_rem, n_exec
        k_done += n_exec
        j += 1
    traj = torch.cat(realized, dim=1)
    # commanded energy measured per segment (seams between replans carry
    # the injected disturbances, identical across replanners)
    e_cmd = 0.0
    seg_med = []
    for seg in commanded[1:]:
        if seg.shape[1] >= 3:
            es = env.control_energy(seg)
            e_cmd = e_cmd + es
            seg_med.append(es.median().item())
    pen = env.penetration(traj, ep_inst["obs"], ep_inst["mask"]).amax(-1)
    return {"collision": (pen > 0.01).float().mean().item(),
            "goal_err": (traj[:, -1] - ep_inst["goal"]).norm(dim=-1).median().item(),
            "energy": e_cmd,
            "seg_energies": seg_med,
            "wall_ms": 1000.0 * wall / max(1, n_plan),
            "sigma_pt": float(torch.tensor(sig_sq).mean().sqrt()) if sig_sq else None,
            "traj": traj}


def main():
    os.makedirs("results", exist_ok=True)
    model, diff, R = load_model("w2_nab")
    ts_nom = diff.t_start_rule(R * math.sqrt(L), kappa=0.5)
    gen = torch.Generator(device=DEVICE).manual_seed(30)
    ep_inst = env2.sample_instances(N_EP, DEVICE, gen)
    # open-loop oracle reference (polished to convergence)
    otraj, oe, ofeas, _ = env2.solve_oracle(ep_inst, n_inits=16, iters=1200,
                                            gen=gen, mixed=True)
    M = otraj.shape[0] // N_EP
    rep = lambda v: v.repeat_interleave(M, 0)
    otraj = env.polish(otraj, rep(ep_inst["obs"]), rep(ep_inst["mask"]),
                       rep(ep_inst["start"]), rep(ep_inst["goal"]),
                       k=500, fixed_head=2)
    oe = env.control_energy(otraj)
    ofeas = ofeas & env.feasible(otraj, rep(ep_inst["obs"]), rep(ep_inst["mask"]), tol=1e-2)
    oe = torch.where(ofeas, oe, torch.inf).view(N_EP, M).min(dim=1).values
    keep = torch.isfinite(oe)
    ep_inst = {k: v[keep] for k, v in ep_inst.items()}
    oe = oe[keep]
    print(f"episodes kept {int(keep.sum())}/{N_EP}, ts_nom={ts_nom}", flush=True)

    res = {"t_s_nom": ts_nom, "R": R, "oracle_energy_med": oe.median().item()}
    runs = {"oracle_mpc": ("oracle", None),
            "oracle_warm": ("oracle_warm", None),
            "full": ("full", None), "nom": ("nom", None)}
    for t in [16, 32, 64, 128, ts_nom]:
        runs[f"prev@{t}"] = ("prev", t)
    for name, (mode, t_prev) in runs.items():
        out = run_episodes(model, diff, ep_inst, mode, t_prev, ts_nom)
        ratio = (out["energy"] / oe)
        res[name] = {"collision": out["collision"], "goal_err": out["goal_err"],
                     "energy_ratio_med": ratio.median().item(),
                     "wall_ms": out["wall_ms"],
                     "sigma_pt": out["sigma_pt"]}
        print(name, {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in res[name].items()}, flush=True)

    with open("results/mpc_results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved results/mpc_results.json")


if __name__ == "__main__":
    main()
