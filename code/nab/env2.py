"""Generalized obstacle world for receding-horizon experiments.

Like env.py but tasks start anywhere in the left region with an initial
velocity: the trajectory pins p_0, p_1 (initial state) and p_{L-1} (goal).
The nominal is the analytic obstacle-free minimum-energy solution with the
given initial state and free terminal velocity:
    mu(t) = p0 + v0 t + (3d/2) t^2 - (d/2) t^3,   d = g - p0 - v0.
"""

import argparse
import os

import torch

from . import env

L = env.L
DT = env.DT
KMAX = env.KMAX


def sample_instances(n, device, gen=None):
    def U(a, b, *shape):
        return a + (b - a) * torch.rand(*shape, generator=gen, device=device)
    start = torch.stack([U(-1.0, 0.7, n), U(-0.4, 0.4, n)], -1)
    v0 = torch.stack([U(-1.5, 1.5, n), U(-1.5, 1.5, n)], -1)
    goal = torch.stack([torch.ones(n, device=device), U(-0.25, 0.25, n)], -1)
    k = torch.randint(1, KMAX + 1, (n,), generator=gen, device=device)
    mask = (torch.arange(KMAX, device=device)[None, :] < k[:, None]).float()
    cx, cy, r = U(-0.5, 0.55, n, KMAX), U(-0.4, 0.4, n, KMAX), U(0.18, 0.32, n, KMAX)
    for _ in range(20):
        c = torch.stack([cx, cy], -1)
        bad = torch.zeros_like(cx, dtype=torch.bool)
        for pt in (start, goal):
            bad |= (c - pt[:, None, :]).norm(dim=-1) < (r + 0.12)
        if not bad.any():
            break
        cx = torch.where(bad, U(-0.5, 0.55, n, KMAX), cx)
        cy = torch.where(bad, U(-0.4, 0.4, n, KMAX), cy)
        r = torch.where(bad, U(0.18, 0.32, n, KMAX), r)
    obs = torch.stack([cx, cy, r], -1)
    return {"start": start, "v0": v0, "goal": goal, "obs": obs, "mask": mask}


def nominal(inst):
    t = torch.linspace(0, 1, L, device=inst["start"].device)[None, :, None]
    p0, v0, g = inst["start"][:, None], inst["v0"][:, None], inst["goal"][:, None]
    d = g - p0 - v0
    return p0 + v0 * t + 1.5 * d * t**2 - 0.5 * d * t**3


def featurize(inst):
    nom = nominal(inst)
    B = nom.shape[0]
    of = torch.cat([inst["obs"], inst["mask"][..., None]], -1) * inst["mask"][..., None]
    of = of.reshape(B, -1)[:, None, :].expand(B, L, KMAX * 4)
    feat = torch.cat([nom, of], -1)                               # (B,L,14)
    cvec = torch.cat([inst["start"], inst["v0"], inst["goal"],
                      (inst["obs"] * inst["mask"][..., None]).reshape(B, -1),
                      inst["mask"]], -1)                          # (B,18)
    return nom, feat, cvec


def pin_of(inst):
    p1 = inst["start"] + inst["v0"] * DT
    return [0, 1, L - 1], torch.stack([inst["start"], p1, inst["goal"]], 1)


def solve_oracle(inst, n_inits=8, iters=800, lr=0.03, gen=None, mixed=False,
                 x_init=None):
    B, dev = inst["start"].shape[0], inst["start"].device
    M = n_inits
    base = nominal(inst)
    if x_init is not None:
        x = x_init
    else:
        tau = torch.linspace(0, 1, L, device=dev)
        modes = torch.stack([torch.sin(torch.pi * m * tau) for m in (1, 2, 3)])
        amp = torch.randn(B, M, 3, 2, generator=gen, device=dev) * \
            torch.tensor([0.45, 0.22, 0.12], device=dev)[None, None, :, None]
        side = torch.where(torch.arange(M, device=dev) % 2 == 0, 1.0, -1.0)
        amp[:, :, 0, 1] = amp[:, :, 0, 1].abs() * side[None, :]
        if mixed:
            amp[:, M // 2:] *= 0.05
        x = (base[:, None] + torch.einsum("bmcd,cl->bmld", amp, modes)
             ).reshape(B * M, L, 2)

    rep = lambda v: v.repeat_interleave(M, 0)
    obs_r, mask_r = rep(inst["obs"]), rep(inst["mask"])
    pin_idx, pin_pts = pin_of(inst)
    pin_r = rep(pin_pts)
    z = x[:, 2:-1].clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        traj = torch.cat([pin_r[:, :2], z, pin_r[:, 2:]], 1)
        env.cost(traj, obs_r, mask_r).sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        traj = torch.cat([pin_r[:, :2], z, pin_r[:, 2:]], 1)
        e = env.control_energy(traj)
        feas = env.feasible(traj, obs_r, mask_r)
        sig = env.signature(traj, rep(inst["start"]), rep(inst["goal"]), obs_r, mask_r)
    return traj.detach(), e, feas, sig


def build_split(n_inst, device, seed, n_inits=8, iters=800, chunk=512):
    gen = torch.Generator(device=device).manual_seed(seed)
    keys = ("traj", "nominal", "feat", "cvec", "start", "v0", "goal", "obs",
            "mask", "energy", "best_energy", "inst_id")
    out = {k: [] for k in keys}
    done, base_id = 0, 0
    while done < n_inst:
        b = min(chunk, n_inst - done)
        inst = sample_instances(b, device, gen)
        traj, e, feas, sig = solve_oracle(inst, n_inits, iters, gen=gen)
        M = traj.shape[0] // b
        traj = traj.view(b, M, L, 2)
        e, feas, sig = e.view(b, M), feas.view(b, M), sig.view(b, M, -1)
        nom, feat, cvec = featurize(inst)
        for i in range(b):
            if not feas[i].any():
                continue
            ei = torch.where(feas[i], e[i], torch.inf)
            best = ei.min()
            seen = {}
            for m in range(M):
                if not feas[i, m] or ei[m] > best * 2.0:
                    continue
                key = tuple(sig[i, m].tolist())
                if key not in seen or ei[m] < ei[seen[key]]:
                    seen[key] = m
            for m in seen.values():
                vals = {"traj": traj[i, m], "nominal": nom[i], "feat": feat[i],
                        "cvec": cvec[i], "start": inst["start"][i],
                        "v0": inst["v0"][i], "goal": inst["goal"][i],
                        "obs": inst["obs"][i], "mask": inst["mask"][i],
                        "energy": e[i, m], "best_energy": best,
                        "inst_id": torch.tensor(base_id + i)}
                for k, v in vals.items():
                    out[k].append(v)
        done += b
        base_id += b
        print(f"  {done}/{n_inst} inst, {len(out['traj'])} samples", flush=True)
    return {k: torch.stack(v).cpu() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=400)
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    print("w2 train:", flush=True)
    tr = build_split(args.n_train, "cuda", seed=20)
    torch.save(tr, "data/w2_train.pt")
    print("w2 test:", flush=True)
    te = build_split(args.n_test, "cuda", seed=21, n_inits=16, iters=1200)
    torch.save(te, "data/w2_test.pt")
    res = tr["traj"] - tr["nominal"]
    print(f"train {tr['traj'].shape[0]} test {te['traj'].shape[0]} "
          f"res_rms {res.pow(2).sum(-1).mean().sqrt():.4f}")


if __name__ == "__main__":
    main()
