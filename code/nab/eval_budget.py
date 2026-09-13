"""E2 (budget-aware truncation) and E3 (guidance at the KL-rule
operating point).  Loads results/results.json and updates it in place.

Usage: python -m nab.eval_budget
"""

import json
import math

import torch

from .eval import load_models, load_instances, rep_cond, sample_cfg, metrics


def main():
    with open("results/results.json") as f:
        res = json.load(f)
    models = load_models()
    inst, _ = load_instances()
    cond = rep_cond(inst)
    R_pt = models["nab"][2]
    L = 64
    R_full = R_pt * math.sqrt(L)
    nab_m, nab_d, _ = models["nab"]
    bn_m, bn_d, _ = models["base_nom"]
    st_m, st_d, _ = models["stddpm"]

    ts_dim = nab_d.t_start_rule(R_pt, kappa=0.5)     # per-point RMS heuristic
    ts_kl = nab_d.t_start_rule(R_full, kappa=0.5)    # full-vector KL rule
    res["t_s_dim"], res["t_s_kl"] = ts_dim, ts_kl
    print("t_s per-point:", ts_dim, " t_s KL:", ts_kl, flush=True)

    # ---------- E2b: NFE x truncation-config sweep ----------
    e2b = {}
    for nfe in [2, 4, 8, 16, 32, 64, 128]:
        row = {}
        row["base_nom_full"] = metrics(sample_cfg(bn_m, bn_d, cond, nfe, anchor="none"), inst)
        row["nab_es_dim"] = metrics(sample_cfg(nab_m, nab_d, cond, nfe, t_start=ts_dim, anchor="mean"), inst)
        row["nab_es_kl"] = metrics(sample_cfg(nab_m, nab_d, cond, nfe, t_start=ts_kl, anchor="mean"), inst)
        row["base_nom_ws_kl"] = metrics(sample_cfg(bn_m, bn_d, cond, nfe, t_start=ts_kl, anchor="mean"), inst)
        row["stddpm_naive_kl"] = metrics(sample_cfg(st_m, st_d, cond, nfe, t_start=ts_kl, anchor="none"), inst)
        e2b[nfe] = row
        print("E2b nfe", nfe, {k: round(r["feasible"], 3) for k, r in row.items()}, flush=True)
    res["E2b"] = e2b

    # ---------- E3b: guidance sweep at the KL operating point ----------
    e3b = {}
    for w in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5]:
        e3b[w] = metrics(sample_cfg(nab_m, nab_d, cond, 16, t_start=ts_kl, anchor="mean", w=w), inst)
        print("E3b w", w, {k: round(v, 3) for k, v in e3b[w].items()}, flush=True)
    res["E3b"] = e3b

    with open("results/results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("updated results/results.json")


if __name__ == "__main__":
    main()
