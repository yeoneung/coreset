"""Pendulum evaluation: equivalence row, truncation sweep, guidance sweep.

Usage: python -m nab.eval_pend
Writes results/pend_results.json; render the saved results with code/replot.py.
"""

import json
import math
import os

import torch

from . import env_pend as ep
from .diffusion import Diffusion
from .model import UNetG

DEVICE = "cuda"
S = 16
CHUNK = 4096


def load_model(tag_variant):
    ck = torch.load(f"runs/{tag_variant}.pt", weights_only=True)
    D, F, C = ck["dims"]
    m = UNetG(x_dim=D, feat_dim=F, cvec_dim=C).to(DEVICE)
    m.load_state_dict(ck["ema"])
    m.eval()
    return m, Diffusion(T=ck["T"], shift=ck["shift"], device=DEVICE), ck["R"]


def load_instances():
    d = torch.load("data/pend_test.pt", weights_only=True)
    d = {k: v.to(DEVICE) for k, v in d.items()}
    ids = d["inst_id"].tolist()
    first, order = {}, []
    for i, iid in enumerate(ids):
        if iid not in first:
            first[iid] = i
            order.append(i)
    sel = torch.tensor(order, device=DEVICE)
    inst = {k: d[k][sel] for k in ("nominal", "feat", "cvec", "umax",
                                   "best_J", "phi0", "dphi0")}
    modes = {}
    phi = d["traj"][..., 0] * torch.pi
    sig = ep.mode_sign(phi)
    for i, iid in enumerate(ids):
        modes.setdefault(iid, set()).add(int(sig[i]))
    inst["modes"] = [modes[iid] for iid in sorted(first, key=lambda k: first[k])]
    return inst


def rep_cond(inst, s=S):
    keys = ("nominal", "feat", "cvec", "umax", "best_J", "phi0", "dphi0")
    return {k: inst[k].repeat_interleave(s, 0) for k in keys}


def sample_cfg(model, diff, cond, nfe, t_start=None, anchor="mean", w=0.0, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    precond = ep.energy_precond(DEVICE)
    outs = []
    n = cond["cvec"].shape[0]
    for lo in range(0, n, CHUNK):
        sub = {k: v[lo:lo + CHUNK] for k, v in cond.items()}
        p0 = (sub["phi0"] / torch.pi)[:, None]
        p1 = ((sub["phi0"] + sub["dphi0"] * ep.DT) / torch.pi)[:, None]
        pin_pts = torch.stack([p0, p1], 1)                       # (b,2,1)
        guide = ep.make_guide(sub["umax"], precond)
        outs.append(diff.sample(model, sub, n_steps=nfe, eta=0.0,
                                t_start=t_start, anchor=anchor, guidance_w=w,
                                gen=gen, pin_idx=[0, 1], pin_pts=pin_pts,
                                guide_fn=guide))
    return torch.cat(outs)


def metrics(z, inst):
    n_inst = inst["cvec"].shape[0]
    cond = rep_cond(inst)
    phi = z[..., 0] * torch.pi
    ok = ep.success_terminal(phi)
    J = ep.cost(phi, cond["umax"])
    sub = J / cond["best_J"] - 1
    sig = ep.mode_sign(phi)
    ok_i, sub_i = ok.view(n_inst, S), sub.view(n_inst, S)
    sig_i = sig.view(n_inst, S)
    z_i = z.view(n_inst, S, -1)
    div = torch.cdist(z_i, z_i).mean(dim=(1, 2)) * (S / (S - 1))
    rec, rec_mm = [], []
    for i in range(n_inst):
        om = inst["modes"][i]
        sm = {int(sig_i[i, j]) for j in range(S) if ok_i[i, j]}
        r = len(om & sm) / len(om)
        rec.append(r)
        if len(om) > 1:
            rec_mm.append(r)
    sub_ok = sub_i[ok_i]
    return {"success": ok.float().mean().item(),
            "subopt_med": sub_ok.median().item() if len(sub_ok) else float("nan"),
            "mode_recall": sum(rec) / len(rec),
            "mode_recall_mm": sum(rec_mm) / max(1, len(rec_mm)),
            "diversity": div.mean().item()}


def main():
    os.makedirs("results", exist_ok=True)
    inst = load_instances()
    cond = rep_cond(inst)
    res = {}
    bn_m, bn_d, R = load_model("pend_base_nom")
    nab_m, nab_d, _ = load_model("pend_nab")
    res["R"] = R
    L = 64
    ts_kl = nab_d.t_start_rule(R * math.sqrt(L), kappa=0.5)
    ts_dim = nab_d.t_start_rule(R, kappa=0.5)
    res["t_s_kl"], res["t_s_dim"] = ts_kl, ts_dim
    print("R", R, "ts_kl", ts_kl, "ts_dim", ts_dim, flush=True)

    # P1: equivalence at full budget
    res["P1"] = {
        "base_nom": metrics(sample_cfg(bn_m, bn_d, cond, 128, anchor="none"), inst),
        "nab": metrics(sample_cfg(nab_m, nab_d, cond, 128, anchor="none"), inst),
        "nab_ws": metrics(sample_cfg(bn_m, bn_d, cond, 128, t_start=ts_kl,
                                     anchor="mean"), inst),
        "nab_es": metrics(sample_cfg(nab_m, nab_d, cond, 128, t_start=ts_kl,
                                     anchor="mean"), inst),
    }
    print("P1", {k: round(v["success"], 3) for k, v in res["P1"].items()}, flush=True)

    # P2: truncation sweep
    p2 = {}
    for t_s in [16, 32, 64, 96, 128, 160, 192, 224, 255]:
        p2[t_s] = {
            "anchored": metrics(sample_cfg(nab_m, nab_d, cond, 32, t_start=t_s,
                                           anchor="mean"), inst),
            "naive": metrics(sample_cfg(nab_m, nab_d, cond, 32, t_start=t_s,
                                        anchor="none"), inst),
            "abar": nab_d.abar[t_s].item(),
        }
        print("P2", t_s, round(p2[t_s]["anchored"]["success"], 3),
              round(p2[t_s]["naive"]["success"], 3), flush=True)
    res["P2"] = p2

    # P3: guidance sweep
    p3 = {}
    for w in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]:
        p3[w] = metrics(sample_cfg(nab_m, nab_d, cond, 16, t_start=ts_kl,
                                   anchor="mean", w=w), inst)
        print("P3", w, {k: round(v, 3) for k, v in p3[w].items()}, flush=True)
    res["P3"] = p3

    with open("results/pend_results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved results/pend_results.json")


if __name__ == "__main__":
    main()
