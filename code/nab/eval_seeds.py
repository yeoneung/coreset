"""Compare translated diffusion variants across training seeds.

Theorem 2.1 concerns equivalent model families; this experiment measures
variation across separately trained models.

For each variant x 3 training seeds x 2 sampling seeds: full-budget E1
metrics, raw and polished, against the frozen best-known reference.
Writes E1t / E1tp into results/results.json.

Usage: python -m nab.eval_seeds
"""

import json

import torch

from . import env
from .diffusion import Diffusion
from .model import UNet1D
from .eval import load_instances, rep_cond, sample_cfg, metrics, DEVICE


def load_ck(name):
    ck = torch.load(f"runs/{name}.pt", weights_only=True)
    m = UNet1D(use_nominal=ck["use_nominal"]).to(DEVICE)
    m.load_state_dict(ck["ema"])
    m.eval()
    return m, Diffusion(T=ck["T"], shift=ck["shift"], device=DEVICE)


def main():
    with open("results/results.json") as f:
        res = json.load(f)
    inst, _ = load_instances()
    inst["best_energy"] = torch.load("data/test_ref_known.pt",
                                     weights_only=True).to(DEVICE)
    cond = rep_cond(inst)

    e1t, e1tp = {}, {}
    for v in ("base_raw", "base_nom", "nab", "stddpm"):
        runs, runsp = [], []
        for sfx in ("", "_s1", "_s2"):
            m, d = load_ck(v + sfx)
            for seed in (0, 1):
                x = sample_cfg(m, d, cond, 128, anchor="none", seed=seed)
                runs.append(metrics(x, inst))
                xp = env.polish(x, cond["obs"], cond["mask"], cond["start"],
                                cond["goal"], k=200)
                runsp.append(metrics(xp, inst))
            print(v + sfx, "feas", round(runs[-1]["feasible"], 3),
                  "gap", round(runs[-1]["subopt_med"], 3), flush=True)
        e1t[v], e1tp[v] = runs, runsp
    res["E1t"], res["E1tp"] = e1t, e1tp
    with open("results/results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved E1t (4 variants x 3 training seeds x 2 sampling seeds)")


if __name__ == "__main__":
    main()
