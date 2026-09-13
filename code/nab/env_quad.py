"""3D quadrotor corridor environment (minimum-snap, differential flatness).

Trajectories are 3D position sequences p_{0:L-1}, L = 96, on the unit
horizon; by differential flatness the quadrotor inputs are smooth
functions of p and its derivatives, and the standard planning objective is
snap:  J = (w_s/2) sum_k ||D4 p_k / dt^4||^2 dt + w_obs sum pen^2 dt over
K in {3,4,5} random spheres.  The obstacle-free minimizer with pinned
endpoints is the straight line (the nominal).  Homotopy signature: the
quadrant (up/down x left/right) in the plane orthogonal to travel in
which the path passes each obstacle.

Usage: python -m nab.env_quad   (generate dataset)
"""

import argparse
import os

import torch

L = 96
DT = 1.0 / (L - 1)
KMAX = 5
W_SNAP = 5e-3
W_OBS = 2000.0
MARGIN = 0.03


def sample_instances(n, device, gen=None):
    def U(a, b, *shape):
        return a + (b - a) * torch.rand(*shape, generator=gen, device=device)
    start = torch.stack([-torch.ones(n, device=device),
                         U(-0.2, 0.2, n), U(-0.2, 0.2, n)], -1)
    goal = torch.stack([torch.ones(n, device=device),
                        U(-0.2, 0.2, n), U(-0.2, 0.2, n)], -1)
    k = torch.randint(3, KMAX + 1, (n,), generator=gen, device=device)
    mask = (torch.arange(KMAX, device=device)[None, :] < k[:, None]).float()
    cx = U(-0.6, 0.6, n, KMAX)
    cy = U(-0.35, 0.35, n, KMAX)
    cz = U(-0.35, 0.35, n, KMAX)
    r = U(0.16, 0.28, n, KMAX)
    for _ in range(20):
        c = torch.stack([cx, cy, cz], -1)
        bad = torch.zeros_like(cx, dtype=torch.bool)
        for pt in (start, goal):
            bad |= (c - pt[:, None, :]).norm(dim=-1) < (r + 0.12)
        if not bad.any():
            break
        cx = torch.where(bad, U(-0.6, 0.6, n, KMAX), cx)
        cy = torch.where(bad, U(-0.35, 0.35, n, KMAX), cy)
        cz = torch.where(bad, U(-0.35, 0.35, n, KMAX), cz)
        r = torch.where(bad, U(0.16, 0.28, n, KMAX), r)
    obs = torch.stack([cx, cy, cz, r], -1)                       # (n,K,4)
    return {"start": start, "goal": goal, "obs": obs, "mask": mask}


def nominal(inst):
    w = torch.linspace(0, 1, L, device=inst["start"].device)[None, :, None]
    return inst["start"][:, None] * (1 - w) + inst["goal"][:, None] * w


def snap_energy(traj):
    d4 = (traj[:, 4:] - 4 * traj[:, 3:-1] + 6 * traj[:, 2:-2]
          - 4 * traj[:, 1:-3] + traj[:, :-4]) / DT**4
    return 0.5 * W_SNAP * (d4**2).sum(dim=(1, 2)) * DT


def penetration(traj, obs, mask, margin=0.0):
    d = (traj[:, :, None, :] - obs[:, None, :, :3]).norm(dim=-1)
    pen = torch.relu(obs[:, None, :, 3] + margin - d) * mask[:, None, :]
    return pen.sum(dim=-1)


def cost(traj, obs, mask, margin=MARGIN, w_obs=W_OBS):
    return snap_energy(traj) + \
        w_obs * (penetration(traj, obs, mask, margin) ** 2).sum(-1) * DT


def feasible(traj, obs, mask, tol=1e-3):
    return penetration(traj, obs, mask, 0.0).max(dim=-1).values <= tol


def signature(traj, obs, mask):
    """Quadrant of passage per obstacle in the (y,z) plane, proximity
    weighted along the path.  (B,K) in {0..3}, masked entries -1."""
    rel = traj[:, :, None, 1:] - obs[:, None, :, 1:3]            # (B,L,K,2)
    d = (traj[:, :, None, :] - obs[:, None, :, :3]).norm(dim=-1)
    w = torch.exp(-(d - obs[:, None, :, 3]).clamp_min(0) ** 2 / 0.02)
    avg = (rel * w[..., None]).sum(dim=1)                        # (B,K,2)
    quad = (avg[..., 0] > 0).long() + 2 * (avg[..., 1] > 0).long()
    return torch.where(mask.bool(), quad, torch.full_like(quad, -1))


def featurize(inst):
    nom = nominal(inst)
    B = nom.shape[0]
    of = torch.cat([inst["obs"], inst["mask"][..., None]], -1) * \
        inst["mask"][..., None]                                  # (B,K,5)
    of = of.reshape(B, -1)[:, None, :].expand(B, L, KMAX * 5)
    feat = torch.cat([nom, of], -1)                              # (B,L,28)
    cvec = torch.cat([inst["start"], inst["goal"],
                      (inst["obs"] * inst["mask"][..., None]).reshape(B, -1),
                      inst["mask"]], -1)                         # (B,31)
    return nom, feat, cvec


def energy_precond(device, delta=1.0):
    """(H_snap + delta I)^{-1} over interior points 1..L-2."""
    n = L - 2
    D = torch.zeros(L - 4, n, device=device)
    coefs = ((-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0))
    for k in range(2, L - 2):
        for off, cf in coefs:
            j = k + off
            if 1 <= j <= L - 2:
                D[k - 2, j - 1] += cf
    H = W_SNAP * (D.T @ D) / DT**7
    return torch.linalg.inv(H + delta * torch.eye(n, device=device))


def make_guide(cond, precond, clip=20.0):
    def guide(x0_hat):
        x = x0_hat.detach().requires_grad_(True)
        xp = torch.cat([cond["start"][:, None], x[:, 1:-1],
                        cond["goal"][:, None]], 1)
        J = cost(xp, cond["obs"], cond["mask"], margin=0.04, w_obs=4000.0)
        (g,) = torch.autograd.grad(J.sum(), x)
        g_int = torch.einsum("ij,bjd->bid", precond, g[:, 1:-1])
        g = torch.cat([torch.zeros_like(g[:, :1]), g_int,
                       torch.zeros_like(g[:, :1])], 1)
        n = g.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return g * (clip / n).clamp(max=1.0)[..., None]
    return guide


N_MODES = 16
_BASIS = {}


def basis(device):
    """Sine basis S (N_MODES, L) vanishing at both endpoints, and the
    least-squares projector P (N_MODES, L) with a = P @ residual."""
    if device not in _BASIS:
        tau = torch.linspace(0, 1, L, device=device)
        S = torch.stack([torch.sin(torch.pi * (m + 1) * tau)
                         for m in range(N_MODES)])
        P = torch.linalg.pinv(S.T)                               # (n,L)
        _BASIS[device] = (S, P)
    return _BASIS[device]


def traj_from(a, base):
    """a (B, N_MODES, 3), base (B, L, 3) -> grid trajectory."""
    S, _ = basis(a.device)
    return base + torch.einsum("bmd,ml->bld", a, S)


def _refine(a, base, obs, mask, iters, lr, sched=True):
    """Adam on cost over sine-basis coefficients (spectral collocation ---
    the standard smooth-basis treatment of the stiff snap objective)."""
    a = a.clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters,
                                                    eta_min=lr * 0.05) \
        if sched else None
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        cost(traj_from(a, base), obs, mask).sum().backward()
        opt.step()
        if sc:
            sc.step()
    return a.detach()


def polish(traj, obs, mask, start, goal, k=150, lr=0.02):
    """Smoothing projection onto the sine basis followed by basis-space
    refinement of the data-generating cost.  Applied identically to every
    sampler; mirrors the polynomial/spline fit of min-snap practice."""
    inst = {"start": start, "goal": goal}
    base = nominal(inst)
    _, P = basis(traj.device)
    a0 = torch.einsum("ml,bld->bmd", P, traj - base)
    with torch.enable_grad():
        a = _refine(a0, base, obs, mask, iters=k, lr=lr, sched=False)
    return traj_from(a, base)


def solve_oracle(inst, n_inits=8, iters=1200, lr=0.03, gen=None, mixed=False):
    B, dev = inst["start"].shape[0], inst["start"].device
    M = n_inits
    base = nominal(inst)
    amp = torch.randn(B, M, N_MODES, 3, generator=gen, device=dev) * 0.02
    lead = torch.randn(B, M, 3, 3, generator=gen, device=dev) * \
        torch.tensor([0.4, 0.2, 0.1], device=dev)[None, None, :, None]
    amp[:, :, :3] += lead
    # force first-mode (y,z) signs through the four quadrants
    qi = torch.arange(M, device=dev) % 4
    sy = torch.where(qi % 2 == 0, 1.0, -1.0)
    sz = torch.where(qi < 2, 1.0, -1.0)
    amp[:, :, 0, 1] = amp[:, :, 0, 1].abs() * sy[None, :]
    amp[:, :, 0, 2] = amp[:, :, 0, 2].abs() * sz[None, :]
    if mixed:
        amp[:, M // 2:] *= 0.05   # near-straight starts for reference use

    rep = lambda v: v.repeat_interleave(M, 0)
    obs_r, mask_r = rep(inst["obs"]), rep(inst["mask"])
    base_r = rep(base)
    a = _refine(amp.reshape(B * M, N_MODES, 3), base_r, obs_r, mask_r,
                iters=iters, lr=lr)
    with torch.no_grad():
        traj = traj_from(a, base_r)
        J = cost(traj, obs_r, mask_r)
        feas = feasible(traj, obs_r, mask_r)
        sig = signature(traj, obs_r, mask_r)
    return traj.detach(), J, feas, sig


def build_split(n_inst, device, seed, n_inits=8, iters=1200, chunk=384):
    gen = torch.Generator(device=device).manual_seed(seed)
    keys = ("traj", "nominal", "feat", "cvec", "start", "goal", "obs",
            "mask", "best_J", "inst_id")
    out = {k: [] for k in keys}
    done, base_id = 0, 0
    while done < n_inst:
        b = min(chunk, n_inst - done)
        inst = sample_instances(b, device, gen)
        traj, J, feas, sig = solve_oracle(inst, n_inits, iters, gen=gen)
        M = traj.shape[0] // b
        traj = traj.view(b, M, L, 3)
        J, feas, sig = J.view(b, M), feas.view(b, M), sig.view(b, M, -1)
        nom, feat, cvec = featurize(inst)
        for i in range(b):
            if not feas[i].any():
                continue
            Ji = torch.where(feas[i], J[i], torch.inf)
            best = Ji.min()
            seen = {}
            for m in range(M):
                if not feas[i, m] or Ji[m] > best * 2.0:
                    continue
                key = tuple(sig[i, m].tolist())
                if key not in seen or Ji[m] < Ji[seen[key]]:
                    seen[key] = m
            for m in seen.values():
                vals = {"traj": traj[i, m], "nominal": nom[i], "feat": feat[i],
                        "cvec": cvec[i], "start": inst["start"][i],
                        "goal": inst["goal"][i], "obs": inst["obs"][i],
                        "mask": inst["mask"][i], "best_J": best,
                        "inst_id": torch.tensor(base_id + i)}
                for kk, v in vals.items():
                    out[kk].append(v)
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
    print("quad train:", flush=True)
    tr = build_split(args.n_train, "cuda", seed=40)
    torch.save(tr, "data/quad_train.pt")
    print("quad test:", flush=True)
    te = build_split(args.n_test, "cuda", seed=41, n_inits=16, iters=1600)
    torch.save(te, "data/quad_test.pt")
    res = tr["traj"] - tr["nominal"]
    print(f"train {tr['traj'].shape[0]} test {te['traj'].shape[0]} "
          f"res_rms {res.pow(2).sum(-1).mean().sqrt():.4f}")


if __name__ == "__main__":
    main()
