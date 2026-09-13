"""E3c: guidance sweep with alpha-bar gating (guide only where the
x0-estimate is reliable).  Updates results/results.json.

Usage: python -m nab.eval_guidance
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
    nab_m, nab_d, R_pt = models["nab"]
    ts_kl = res["t_s_kl"]

    e3c = {}
    for w in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        e3c[w] = metrics(sample_cfg(nab_m, nab_d, cond, 16, t_start=ts_kl,
                                    anchor="mean", w=w), inst)
        print("E3c w", w, {k: round(v, 3) for k, v in e3c[w].items()}, flush=True)
    res["E3c"] = e3c

    with open("results/results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("updated results/results.json")


if __name__ == "__main__":
    main()
