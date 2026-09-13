"""Two-pass evaluation with a shared best-known cost reference.

Pass 1 caches every sampler configuration's trajectories (fixed seeds).
Pass 2 builds the best-known reference J* per task --- the minimum of the
strong mixed-init multi-start oracle, the polished stored modes, and the
polished samples of every cached configuration --- and computes all
metrics against it.  Negative suboptimality is impossible by
construction; the fraction of tasks where samples improve on the
classical oracle is reported separately.

Usage: python -m nab.eval_reference
"""

import json

import torch

from . import env
from .eval import (load_models, load_instances, rep_cond, sample_cfg,
                   metrics, S)


def main():
    with open("results/results.json") as f:
        res = json.load(f)
    models = load_models()
    inst, _ = load_instances()   # loads strong-oracle ref into best_energy
    cond = rep_cond(inst)
    ts_dim, ts_kl = res["t_s_dim"], res["t_s_kl"]
    nab_m, nab_d, _ = models["nab"]
    bn_m, bn_d, _ = models["base_nom"]
    st_m, st_d, _ = models["stddpm"]

    # ---------------- pass 1: sample everything ----------------
    cache = {}

    def add(key, x, polished=False):
        cache[key] = (x, polished)
        print("sampled", key, flush=True)

    for v, (m, d, _) in models.items():
        for seed in range(3):
            add(("E1", v, seed), sample_cfg(m, d, cond, 128, anchor="none",
                                            seed=seed))
    for nfe in [2, 4, 8, 16, 32, 64, 128]:
        add(("E2b", nfe, "base_nom_full"), sample_cfg(bn_m, bn_d, cond, nfe, anchor="none"))
        add(("E2b", nfe, "nab_es_dim"), sample_cfg(nab_m, nab_d, cond, nfe, t_start=ts_dim, anchor="mean"))
        add(("E2b", nfe, "nab_es_kl"), sample_cfg(nab_m, nab_d, cond, nfe, t_start=ts_kl, anchor="mean"))
        add(("E2b", nfe, "base_nom_ws_kl"), sample_cfg(bn_m, bn_d, cond, nfe, t_start=ts_kl, anchor="mean"))
        add(("E2b", nfe, "stddpm_naive_kl"), sample_cfg(st_m, st_d, cond, nfe, t_start=ts_kl, anchor="none"))
    for w in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        add(("E3", w), sample_cfg(nab_m, nab_d, cond, 16, t_start=ts_kl,
                                  anchor="mean", w=w))
    T = nab_d.T
    for t_s in [16, 32, 64, 96, 128, 160, 192, 224, T - 1]:
        add(("E4", t_s, "anchored"), sample_cfg(nab_m, nab_d, cond, 32, t_start=t_s, anchor="mean"))
        add(("E4", t_s, "naive"), sample_cfg(nab_m, nab_d, cond, 32, t_start=t_s, anchor="none"))

    # polished companions where reported
    pol = {}
    for key in list(cache):
        if key[0] in ("E1", "E3"):
            x, _ = cache[key]
            pol[key] = env.polish(x, cond["obs"], cond["mask"],
                                  cond["start"], cond["goal"], k=200)
            print("polished", key, flush=True)

    # ---------------- pass 2: best-known reference ----------------
    n_inst = inst["start"].shape[0]
    ref_strong = inst["best_energy"].clone()      # strong classical oracle
    ref = ref_strong.clone()
    for key, xp in pol.items():
        feas = env.feasible(xp, cond["obs"], cond["mask"], tol=1e-2)
        J = env.cost(xp, cond["obs"], cond["mask"])
        J = torch.where(feas, J, torch.inf).view(n_inst, S).min(1).values
        ref = torch.minimum(ref, J)
    improved = (ref < ref_strong * 0.999).float().mean().item()
    res["ref_improved_frac"] = improved
    inst["best_energy"] = ref
    print(f"best-known ref improves on classical oracle on "
          f"{improved*100:.1f}% of tasks", flush=True)

    # ---------------- metrics ----------------
    e1 = {v: [] for v in models}
    e1p = {v: [] for v in models}
    for v in models:
        for seed in range(3):
            e1[v].append(metrics(cache[("E1", v, seed)][0], inst))
            e1p[v].append(metrics(pol[("E1", v, seed)], inst))
    res["E1"], res["E1p"] = e1, e1p
    e2b = {}
    for nfe in [2, 4, 8, 16, 32, 64, 128]:
        e2b[nfe] = {cfgn: metrics(cache[("E2b", nfe, cfgn)][0], inst)
                    for cfgn in ("base_nom_full", "nab_es_dim", "nab_es_kl",
                                 "base_nom_ws_kl", "stddpm_naive_kl")}
    res["E2b"] = e2b
    e3, e3p = {}, {}
    for w in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        e3[w] = metrics(cache[("E3", w)][0], inst)
        e3p[w] = metrics(pol[("E3", w)], inst)
        print("E3", w, round(e3[w]["subopt_med"], 3), "pol",
              round(e3p[w]["subopt_med"], 3), flush=True)
    res["E3e"], res["E3p"] = e3, e3p
    e4 = {}
    for t_s in [16, 32, 64, 96, 128, 160, 192, 224, T - 1]:
        e4[t_s] = {k: metrics(cache[("E4", t_s, k)][0], inst)
                   for k in ("anchored", "naive")}
        e4[t_s]["abar"] = nab_d.abar[t_s].item()
    res["E4"] = e4
    torch.save(ref.cpu(), "data/test_ref_known.pt")

    with open("results/results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
