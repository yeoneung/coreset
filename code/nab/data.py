"""Generate the multimodal near-optimal trajectory dataset on GPU.

For each instance we run multi-start trajectory optimization, keep feasible
solutions, deduplicate by homotopy signature, and retain every mode whose
control energy is within KEEP_FACTOR of the per-instance best.  Each retained
(instance, mode) pair becomes one training sample.

Usage:  python -m nab.data --n_train 6000 --n_test 500
"""

import argparse
import torch

from . import env


KEEP_FACTOR = 2.0


def build_split(n_inst, device, seed, n_inits=8, iters=800, chunk=512):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = {k: [] for k in ("traj", "start", "goal", "obs", "mask", "nominal",
                           "energy", "best_energy", "inst_id")}
    n_done, inst_base = 0, 0
    while n_done < n_inst:
        b = min(chunk, n_inst - n_done)
        inst = env.sample_instances(b, device, gen)
        traj, e, feas, sig = env.solve_oracle(inst, n_inits=n_inits, iters=iters, gen=gen)
        M = n_inits
        traj = traj.view(b, M, env.L, 2)
        e = e.view(b, M)
        feas = feas.view(b, M)
        sig = sig.view(b, M, env.KMAX)
        nom = env.nominal(inst["start"], inst["goal"])
        for i in range(b):
            fi = feas[i]
            if not fi.any():
                continue
            ei = torch.where(fi, e[i], torch.inf)
            best = ei.min()
            # dedupe by signature, keep best per mode, near-optimal only
            seen = {}
            for m in range(M):
                if not fi[m] or ei[m] > best * KEEP_FACTOR:
                    continue
                key = tuple(sig[i, m].tolist())
                if key not in seen or ei[m] < ei[seen[key]]:
                    seen[key] = m
            for m in seen.values():
                out["traj"].append(traj[i, m])
                out["start"].append(inst["start"][i])
                out["goal"].append(inst["goal"][i])
                out["obs"].append(inst["obs"][i])
                out["mask"].append(inst["mask"][i])
                out["nominal"].append(nom[i])
                out["energy"].append(e[i, m])
                out["best_energy"].append(best)
                out["inst_id"].append(torch.tensor(inst_base + i))
        n_done += b
        inst_base += b
        print(f"  {n_done}/{n_inst} instances, {len(out['traj'])} samples", flush=True)
    return {k: torch.stack(v).cpu() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    import os
    os.makedirs(args.out, exist_ok=True)
    print("train split:", flush=True)
    tr = build_split(args.n_train, device, seed=0)
    torch.save(tr, os.path.join(args.out, "train.pt"))
    print("test split:", flush=True)
    te = build_split(args.n_test, device, seed=1, n_inits=16, iters=1200)
    torch.save(te, os.path.join(args.out, "test.pt"))

    res = tr["traj"] - tr["nominal"]
    print(f"train samples {tr['traj'].shape[0]}  test samples {te['traj'].shape[0]}")
    print(f"residual std {res.std():.4f}  mean |residual| {res.norm(dim=-1).mean():.4f}")
    print(f"mean oracle energy {tr['energy'].mean():.3f}")


if __name__ == "__main__":
    main()
