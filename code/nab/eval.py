"""Full evaluation suite: tables and figures for the paper.

Usage: python -m nab.eval
Writes results/results.json; render the saved results with code/replot.py.
"""

import json
import os

import torch

from . import env
from .diffusion import Diffusion
from .model import UNet1D
from .train import VARIANTS

DEVICE = "cuda"
S = 16  # samples per instance
CHUNK = 4096


def load_models(run_dir="runs"):
    models = {}
    for v in VARIANTS:
        ck = torch.load(os.path.join(run_dir, f"{v}.pt"), weights_only=True)
        m = UNet1D(use_nominal=ck["use_nominal"]).to(DEVICE)
        m.load_state_dict(ck["ema"])
        m.eval()
        models[v] = (m, Diffusion(T=ck["T"], shift=ck["shift"], device=DEVICE), ck["R"])
    return models


def load_instances(path="data/test.pt"):
    d = torch.load(path, weights_only=True)
    d = {k: v.to(DEVICE) for k, v in d.items()}
    # strong per-instance reference cost J*: the data-generating oracle is
    # not fully converged (forced-amplitude inits leave hooked local
    # minima), so rebuild the reference with a dense multi-start (small
    # amplitudes included), long optimization, and a polish pass; also take
    # the min against the polished stored modes
    import os
    ref_path = path.replace(".pt", "_ref.pt")
    if os.path.exists(ref_path):
        ref = torch.load(ref_path, weights_only=True).to(DEVICE)
    else:
        ids = d["inst_id"]
        uniq = ids.unique()
        first = torch.stack([torch.nonzero(ids == i)[0, 0] for i in uniq])
        inst_u = {k: d[k][first] for k in ("start", "goal", "obs", "mask")}
        gen = torch.Generator(device=DEVICE).manual_seed(123)
        tr, e, feas, _ = env.solve_oracle(inst_u, n_inits=24, iters=2000,
                                          gen=gen, mixed=True)
        M = tr.shape[0] // len(uniq)
        rep = lambda v: v.repeat_interleave(M, 0)
        trp = env.polish(tr, rep(inst_u["obs"]), rep(inst_u["mask"]),
                         rep(inst_u["start"]), rep(inst_u["goal"]), k=500)
        Jf = env.cost(trp, rep(inst_u["obs"]), rep(inst_u["mask"]))
        okf = env.feasible(trp, rep(inst_u["obs"]), rep(inst_u["mask"]),
                           tol=1e-2)
        Jf = torch.where(okf, Jf, torch.inf).view(len(uniq), M).min(1).values
        # polished stored modes
        xp = env.polish(d["traj"], d["obs"], d["mask"], d["start"],
                        d["goal"], k=500)
        Jp = env.cost(xp, d["obs"], d["mask"])
        ref = torch.full_like(d["best_energy"], torch.inf)
        for j, iid in enumerate(uniq):
            m = ids == iid
            ref[m] = torch.minimum(Jp[m].min(), Jf[j])
        torch.save(ref.cpu(), ref_path)
    d["best_energy"] = ref
    ids = d["inst_id"]
    uniq, first = torch.unique(ids, return_inverse=False), {}
    order = []
    for i, iid in enumerate(ids.tolist()):
        if iid not in first:
            first[iid] = i
            order.append(i)
    sel = torch.tensor(order, device=DEVICE)
    inst = {k: d[k][sel] for k in ("start", "goal", "obs", "mask", "nominal", "best_energy")}
    # oracle mode signatures per instance
    sig_all = env.signature(d["traj"], d["start"], d["goal"], d["obs"], d["mask"])
    modes = {}
    for i, iid in enumerate(ids.tolist()):
        modes.setdefault(iid, set()).add(tuple(sig_all[i].tolist()))
    inst["modes"] = [modes[iid] for iid in sorted(first, key=lambda k: first[k])]
    return inst, d


def rep_cond(inst, s=S):
    keys = ("start", "goal", "obs", "mask", "nominal")
    return {k: inst[k].repeat_interleave(s, 0) for k in keys}


def sample_cfg(model, diff, cond, nfe, t_start=None, anchor="mean", w=0.0, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    outs = []
    n = cond["start"].shape[0]
    for lo in range(0, n, CHUNK):
        sub = {k: v[lo:lo + CHUNK] for k, v in cond.items()}
        outs.append(diff.sample(model, sub, n_steps=nfe, eta=0.0,
                                t_start=t_start, anchor=anchor,
                                guidance_w=w, gen=gen))
    return torch.cat(outs)


def metrics(x, inst):
    """x: (Ninst*S, L, 2) samples.  Suboptimality is measured under the
    full data-generating cost J (energy + soft-margin penalty), so
    obstacle-hugging inside the oracle's safety margin is not free."""
    n_inst = inst["start"].shape[0]
    cond = rep_cond(inst)
    feas = env.feasible(x, cond["obs"], cond["mask"], tol=1e-2)
    e = env.cost(x, cond["obs"], cond["mask"])
    best = inst["best_energy"].repeat_interleave(S, 0)
    # normalized optimality gap: (J - J*) / mean(J*).  A per-task ratio is
    # ill-conditioned here because easy tasks have J* near zero.
    sub = (e - best) / inst["best_energy"].mean()
    sig = env.signature(x, cond["start"], cond["goal"], cond["obs"], cond["mask"])
    feas_i = feas.view(n_inst, S)
    sub_i = sub.view(n_inst, S)
    sig_i = sig.view(n_inst, S, -1)
    x_i = x.view(n_inst, S, -1)
    div = torch.cdist(x_i, x_i).mean(dim=(1, 2)) * (S / (S - 1))
    recall, recall_mm, n_mm = [], [], 0
    for i in range(n_inst):
        om = inst["modes"][i]
        sm = {tuple(sig_i[i, j].tolist()) for j in range(S) if feas_i[i, j]}
        r = len(om & sm) / len(om)
        recall.append(r)
        if len(om) > 1:
            recall_mm.append(r)
            n_mm += 1
    sub_feas = sub_i[feas_i]
    return {
        "feasible": feas.float().mean().item(),
        "subopt_mean": sub_feas.mean().item() if len(sub_feas) else float("nan"),
        "subopt_med": sub_feas.median().item() if len(sub_feas) else float("nan"),
        "mode_recall": float(sum(recall) / len(recall)),
        "mode_recall_mm": float(sum(recall_mm) / max(1, len(recall_mm))),
        "diversity": div.mean().item(),
    }


def main():
    os.makedirs("results", exist_ok=True)
    models = load_models()
    inst, raw = load_instances()
    cond = rep_cond(inst)
    R_pt = models["nab"][2]  # per-point rms residual
    res = {"R": R_pt}
    T = models["nab"][1].T

    # ---------- E1: full budget, 3 seeds ----------
    e1 = {}
    for v, (m, diff, _) in models.items():
        runs = []
        for seed in range(3):
            x = sample_cfg(m, diff, cond, nfe=128, anchor="none", seed=seed)
            runs.append(metrics(x, inst))
        e1[v] = runs
        print("E1", v, runs[0], flush=True)
    res["E1"] = e1

    # ---------- E2: NFE sweep ----------
    nab_m, nab_d, _ = models["nab"]
    bn_m, bn_d, _ = models["base_nom"]
    st_m, st_d, _ = models["stddpm"]
    ts = nab_d.t_start_rule(R_pt, kappa=0.5)
    res["t_s"] = ts
    e2 = {}
    for nfe in [2, 4, 8, 16, 32, 64, 128]:
        row = {}
        row["base_nom_full"] = metrics(sample_cfg(bn_m, bn_d, cond, nfe, anchor="none"), inst)
        row["nab_es"] = metrics(sample_cfg(nab_m, nab_d, cond, nfe, t_start=ts, anchor="mean"), inst)
        row["base_nom_ws"] = metrics(sample_cfg(bn_m, bn_d, cond, nfe, t_start=ts, anchor="mean"), inst)
        row["stddpm_naive"] = metrics(sample_cfg(st_m, st_d, cond, nfe, t_start=ts, anchor="none"), inst)
        e2[nfe] = row
        print("E2 nfe", nfe, {k: round(r['feasible'], 3) for k, r in row.items()}, flush=True)
    res["E2"] = e2

    # ---------- E3: guidance frontier ----------
    e3 = {}
    for w in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        e3[w] = metrics(sample_cfg(nab_m, nab_d, cond, 16, t_start=ts, anchor="mean", w=w), inst)
        print("E3 w", w, e3[w], flush=True)
    res["E3"] = e3

    # ---------- E4: truncation-point sweep (anchored vs naive) ----------
    e4 = {}
    for t_s in [16, 32, 64, 96, 128, 160, 192, 224, T - 1]:
        e4[t_s] = {
            "anchored": metrics(sample_cfg(nab_m, nab_d, cond, 32, t_start=t_s, anchor="mean"), inst),
            "naive": metrics(sample_cfg(nab_m, nab_d, cond, 32, t_start=t_s, anchor="none"), inst),
            "abar": nab_d.abar[t_s].item(),
        }
        print("E4 t_s", t_s, "anch", round(e4[t_s]["anchored"]["feasible"], 3),
              "naive", round(e4[t_s]["naive"]["feasible"], 3), flush=True)
    res["E4"] = e4

    with open("results/results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved results/results.json")


if __name__ == "__main__":
    main()
