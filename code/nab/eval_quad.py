"""Quadrotor evaluation: equivalence, truncation, guidance, with polish.

Usage: python -m nab.eval_quad
Writes results/quad_results.json.
"""

import json
import math
import os

import torch

from . import env_quad as eq
from .diffusion import Diffusion
from .model import UNetG

DEVICE = "cuda"
S = 16
CHUNK = 2048


def load_model(name):
    ck = torch.load(f"runs/{name}.pt", weights_only=True)
    D, F, C = ck["dims"]
    m = UNetG(x_dim=D, feat_dim=F, cvec_dim=C).to(DEVICE)
    m.load_state_dict(ck["ema"])
    m.eval()
    return m, Diffusion(T=ck["T"], shift=ck["shift"], device=DEVICE), ck["R"]


def load_instances():
    d = torch.load("data/quad_test.pt", weights_only=True)
    d = {k: v.to(DEVICE) for k, v in d.items()}
    # strong reference J*: fresh dense multi-start + polish, min against
    # the polished stored modes
    ref_path = "data/quad_test_ref.pt"
    if os.path.exists(ref_path):
        d["best_J"] = torch.load(ref_path, weights_only=True).to(DEVICE)
    else:
        ids_t = d["inst_id"]
        uniq = ids_t.unique()
        first = torch.stack([torch.nonzero(ids_t == i)[0, 0] for i in uniq])
        inst_u = {k: d[k][first] for k in ("start", "goal", "obs", "mask")}
        gen = torch.Generator(device=DEVICE).manual_seed(123)
        tr, J, feas, _ = eq.solve_oracle(inst_u, n_inits=24, iters=2500,
                                         gen=gen, mixed=True)
        M = tr.shape[0] // len(uniq)
        rep = lambda v: v.repeat_interleave(M, 0)
        trp = eq.polish(tr, rep(inst_u["obs"]), rep(inst_u["mask"]),
                        rep(inst_u["start"]), rep(inst_u["goal"]), k=400)
        Jf = eq.cost(trp, rep(inst_u["obs"]), rep(inst_u["mask"]))
        okf = eq.feasible(trp, rep(inst_u["obs"]), rep(inst_u["mask"]),
                          tol=1e-2)
        Jf = torch.where(okf, Jf, torch.inf).view(len(uniq), M).min(1).values
        xp = eq.polish(d["traj"], d["obs"], d["mask"], d["start"], d["goal"],
                       k=400)
        Jp = eq.cost(xp, d["obs"], d["mask"])
        ref = torch.full_like(d["best_J"], torch.inf)
        for j, iid in enumerate(uniq):
            m = ids_t == iid
            ref[m] = torch.minimum(Jp[m].min(), Jf[j])
        torch.save(ref.cpu(), ref_path)
        d["best_J"] = ref
    ids = d["inst_id"].tolist()
    first, order = {}, []
    for i, iid in enumerate(ids):
        if iid not in first:
            first[iid] = i
            order.append(i)
    sel = torch.tensor(order, device=DEVICE)
    inst = {k: d[k][sel] for k in ("nominal", "feat", "cvec", "start",
                                   "goal", "obs", "mask", "best_J")}
    sig = eq.signature(d["traj"], d["obs"], d["mask"])
    modes = {}
    for i, iid in enumerate(ids):
        modes.setdefault(iid, set()).add(tuple(sig[i].tolist()))
    inst["modes"] = [modes[iid] for iid in sorted(first, key=lambda k: first[k])]
    return inst, d


def rep_cond(inst, s=S):
    keys = ("nominal", "feat", "cvec", "start", "goal", "obs", "mask", "best_J")
    return {k: inst[k].repeat_interleave(s, 0) for k in keys}


def sample_cfg(model, diff, cond, nfe, t_start=None, anchor="mean", w=0.0,
               seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    precond = eq.energy_precond(DEVICE)
    outs = []
    n = cond["cvec"].shape[0]
    for lo in range(0, n, CHUNK):
        sub = {k: v[lo:lo + CHUNK] for k, v in cond.items()}
        pin_pts = torch.stack([sub["start"], sub["goal"]], 1)
        guide = eq.make_guide(sub, precond)
        outs.append(diff.sample(model, sub, n_steps=nfe, eta=0.0,
                                t_start=t_start, anchor=anchor, guidance_w=w,
                                gen=gen, pin_idx=[0, eq.L - 1],
                                pin_pts=pin_pts, guide_fn=guide))
    return torch.cat(outs)


def metrics(x, inst):
    n_inst = inst["start"].shape[0]
    cond = rep_cond(inst)
    feas = eq.feasible(x, cond["obs"], cond["mask"], tol=1e-2)
    J = eq.cost(x, cond["obs"], cond["mask"])
    sub = (J - cond["best_J"]) / inst["best_J"].mean()
    sig = eq.signature(x, cond["obs"], cond["mask"])
    feas_i, sub_i = feas.view(n_inst, S), sub.view(n_inst, S)
    sig_i = sig.view(n_inst, S, -1)
    x_i = x.view(n_inst, S, -1)
    div = torch.cdist(x_i, x_i).mean(dim=(1, 2)) * (S / (S - 1))
    rec, rec_mm = [], []
    for i in range(n_inst):
        om = inst["modes"][i]
        sm = {tuple(sig_i[i, j].tolist()) for j in range(S) if feas_i[i, j]}
        r = len(om & sm) / len(om)
        rec.append(r)
        if len(om) > 1:
            rec_mm.append(r)
    sub_ok = sub_i[feas_i]
    return {"feasible": feas.float().mean().item(),
            "subopt_med": sub_ok.median().item() if len(sub_ok) else float("nan"),
            "mode_recall": sum(rec) / len(rec),
            "mode_recall_mm": sum(rec_mm) / max(1, len(rec_mm)),
            "diversity": div.mean().item()}


def pol(x, cond):
    return eq.polish(x, cond["obs"], cond["mask"], cond["start"], cond["goal"])


def main():
    os.makedirs("results", exist_ok=True)
    inst, _ = load_instances()
    cond = rep_cond(inst)
    bn_m, bn_d, R = load_model("quad_base_nom")
    nab_m, nab_d, _ = load_model("quad_nab")
    res = {"R": R}
    ts_kl = nab_d.t_start_rule(R * math.sqrt(eq.L), kappa=0.5)
    res["t_s_kl"] = ts_kl
    print("R", R, "ts_kl", ts_kl, flush=True)

    # pass 1: cache samples (+ polished where reported)
    cache, polc = {}, {}
    q1cfg = {"base_nom": (bn_m, bn_d, None, "none"),
             "nab": (nab_m, nab_d, None, "none"),
             "base_nom_ws": (bn_m, bn_d, ts_kl, "mean"),
             "nab_es": (nab_m, nab_d, ts_kl, "mean")}
    for name, (m, d, t0, anc) in q1cfg.items():
        x = sample_cfg(m, d, cond, 128, t_start=t0, anchor=anc)
        cache[("Q1", name)] = x
        polc[("Q1", name)] = pol(x, cond)
        print("sampled Q1", name, flush=True)
    for t_s in [32, 64, 96, 128, 160, 192, 224, 255]:
        for anc, mode in (("mean", "anchored"), ("none", "naive")):
            x = sample_cfg(nab_m, nab_d, cond, 32, t_start=t_s, anchor=anc)
            cache[("Q2", t_s, mode)] = x
            polc[("Q2", t_s, mode)] = pol(x, cond)
        print("sampled Q2", t_s, flush=True)
    for w in [0.0, 0.5, 1.0, 2.0, 4.0]:
        x = sample_cfg(nab_m, nab_d, cond, 16, t_start=ts_kl, anchor="mean", w=w)
        cache[("Q3", w)] = x
        polc[("Q3", w)] = pol(x, cond)
        print("sampled Q3", w, flush=True)

    # pass 2: best-known reference
    n_inst = inst["start"].shape[0]
    ref_strong = inst["best_J"].clone()
    ref = ref_strong.clone()
    for key, xp in polc.items():
        feas = eq.feasible(xp, cond["obs"], cond["mask"], tol=1e-2)
        J = eq.cost(xp, cond["obs"], cond["mask"])
        J = torch.where(feas, J, torch.inf).view(n_inst, S).min(1).values
        ref = torch.minimum(ref, J)
    res["ref_improved_frac"] = (ref < ref_strong * 0.999).float().mean().item()
    inst["best_J"] = ref
    print(f"best-known ref improves on classical oracle on "
          f"{res['ref_improved_frac']*100:.1f}% of tasks", flush=True)

    res["Q1"] = {n: metrics(cache[("Q1", n)], inst) for n in q1cfg}
    res["Q1p"] = {n: metrics(polc[("Q1", n)], inst) for n in q1cfg}
    res["Q2"] = {t: {m: metrics(cache[("Q2", t, m)], inst)
                     for m in ("anchored", "naive")}
                 for t in [32, 64, 96, 128, 160, 192, 224, 255]}
    res["Q2p"] = {t: {m: metrics(polc[("Q2", t, m)], inst)
                      for m in ("anchored", "naive")}
                  for t in [32, 64, 96, 128, 160, 192, 224, 255]}
    res["Q3"] = {w: metrics(cache[("Q3", w)], inst)
                 for w in [0.0, 0.5, 1.0, 2.0, 4.0]}
    res["Q3p"] = {w: metrics(polc[("Q3", w)], inst)
                  for w in [0.0, 0.5, 1.0, 2.0, 4.0]}
    for w in [0.0, 0.5, 1.0, 2.0, 4.0]:
        print("Q3", w, round(res["Q3"][w]["feasible"], 3), "pol_sub",
              round(res["Q3p"][w]["subopt_med"], 3), flush=True)
    torch.save(ref.cpu(), "data/quad_test_ref_known.pt")

    with open("results/quad_results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved results/quad_results.json")


if __name__ == "__main__":
    main()
