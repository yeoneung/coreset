"""Pendulum swing-up environment (nonlinear dynamics).

Angle phi measured from the downward vertical; dynamics
phi'' = -(g/l) sin(phi) + u.  Trajectories are angle sequences on a 3 s
horizon (L=64); torque is recovered by dynamics inversion
u_k = phi''_k + g sin(phi_k).  Upright targets are {-pi, +pi} (left/right
swing = two homotopy modes).  The model works in normalized units
z = phi / pi.  Cost:

    J = 0.5 sum u^2 dt + w_lim sum relu(|u|-u_max)^2 dt
        + w_T min((phi_T-pi)^2, (phi_T+pi)^2) + w_v phi_T'^2.

The nominal is a cubic Hermite ramp from (phi_0, phi_0') to the heuristic
upright (sign chosen from the initial velocity/offset) --- a deterministic
function of the condition, deliberately imperfect on near-symmetric tasks.
"""

import argparse
import os

import torch

L = 64
T_H = 3.0
DT = T_H / (L - 1)
G = 5.0
W_LIM = 10.0
W_T = 200.0
W_V = 5.0


def sample_instances(n, device, gen=None):
    def U(a, b, *shape):
        return a + (b - a) * torch.rand(*shape, generator=gen, device=device)
    return {"phi0": U(-0.6, 0.6, n), "dphi0": U(-1.5, 1.5, n),
            "umax": U(3.0, 6.0, n)}


def hermite(phi0, dphi0, target, device):
    """Cubic ramp (B,L) from (phi0, dphi0) to (target, 0) over [0, T_H]."""
    t = torch.linspace(0, T_H, L, device=device)[None, :]
    T = T_H
    # h(t) = phi0 + dphi0 t + a2 t^2 + a3 t^3 with h(T)=target, h'(T)=0
    a2 = (3 * (target - phi0) / T**2 - (2 * dphi0) / T)[:, None]
    a3 = (-2 * (target - phi0) / T**3 + dphi0 / T**2)[:, None]
    return phi0[:, None] + dphi0[:, None] * t + a2 * t**2 + a3 * t**3


def nominal_target(inst):
    s = torch.sign(inst["dphi0"])
    s = torch.where(inst["dphi0"].abs() < 0.3, torch.sign(inst["phi0"]), s)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return s * torch.pi


def nominal(inst):
    """(B,L,1) in z units."""
    tgt = nominal_target(inst)
    return (hermite(inst["phi0"], inst["dphi0"], tgt, inst["phi0"].device)
            / torch.pi)[..., None]


def torque(phi):
    """u_k at interior points k=1..L-2.  phi (B,L) -> (B,L-2)."""
    dd = (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, :-2]) / DT**2
    return dd + G * torch.sin(phi[:, 1:-1])


def cost(phi, umax):
    """phi (B,L) radians, umax (B,).  Full task cost (B,)."""
    u = torque(phi)
    J = 0.5 * (u**2).sum(-1) * DT
    J = J + W_LIM * (torch.relu(u.abs() - umax[:, None]) ** 2).sum(-1) * DT
    dT = (phi[:, -1] - phi[:, -2]) / DT
    term = torch.minimum((phi[:, -1] - torch.pi) ** 2, (phi[:, -1] + torch.pi) ** 2)
    return J + W_T * term + W_V * dT**2


def success(phi, umax):
    dT = (phi[:, -1] - phi[:, -2]) / DT
    up = torch.minimum((phi[:, -1] - torch.pi).abs(), (phi[:, -1] + torch.pi).abs())
    viol = torch.relu(torque(phi).abs() - umax[:, None]).amax(-1)
    return (up <= 0.15) & (dT.abs() <= 0.5) & (viol <= 0.1)


def success_terminal(phi):
    """Reaches an upright with near-zero velocity.  Torque feasibility is
    evaluated through the cost: raw diffusion samples carry point-level
    noise that second differences amplify, so a hard torque test measures
    sampler roughness, not plan quality."""
    dT = (phi[:, -1] - phi[:, -2]) / DT
    up = torch.minimum((phi[:, -1] - torch.pi).abs(), (phi[:, -1] + torch.pi).abs())
    return (up <= 0.2) & (dT.abs() <= 0.75)


def mode_sign(phi):
    return (phi[:, -1] > 0).long()


def energy_precond(device, delta=1.0):
    """Regularized inverse energy Hessian over the FREE variables
    phi_2 .. phi_{L-1} (the first two points are pinned, the terminal is
    free and carries the terminal cost).  Row k of D is the acceleration
    a_k = phi_{k+1} - 2 phi_k + phi_{k-1}, k = 1..L-2, differentiated with
    respect to the free variables."""
    n = L - 2                      # free vars j = 2..L-1
    D = torch.zeros(n, n, device=device)
    for k in range(1, L - 1):      # rows r = k-1
        r = k - 1
        for j, coef in ((k + 1, 1.0), (k, -2.0), (k - 1, 1.0)):
            if 2 <= j <= L - 1:
                D[r, j - 2] += coef
    H = D.T @ D / DT**3
    return torch.linalg.inv(H + delta * torch.eye(n, device=device))


def make_guide(umax, precond, clip=20.0):
    """Guidance gradient in z units: preconditioned d(cost)/dz on the free
    variables (interior + terminal); zero on the pinned initial state."""
    def guide(z_hat):
        z = z_hat.detach().requires_grad_(True)
        phi = z[..., 0] * torch.pi
        J = cost(phi, umax)
        (g,) = torch.autograd.grad(J.sum(), z)
        g_free = torch.einsum("ij,bjd->bid", precond, g[:, 2:])
        g = torch.cat([torch.zeros_like(g[:, :2]), g_free], 1)
        n = g.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return g * (clip / n).clamp(max=1.0)[..., None]
    return guide


def solve_oracle(inst, n_inits=8, iters=800, lr=0.02, gen=None):
    phi0, dphi0, umax = inst["phi0"], inst["dphi0"], inst["umax"]
    B, dev = phi0.shape[0], phi0.device
    M = n_inits
    tgt = torch.where(torch.arange(M, device=dev)[None, :] % 2 == 0, torch.pi,
                      -torch.pi).expand(B, M).reshape(-1)
    phi0_r = phi0.repeat_interleave(M)
    dphi0_r = dphi0.repeat_interleave(M)
    umax_r = umax.repeat_interleave(M)
    base = hermite(phi0_r, dphi0_r, tgt, dev)                     # (BM,L)
    t = torch.linspace(0, 1, L, device=dev)
    modes = torch.stack([torch.sin(torch.pi * m * t) for m in (1, 2, 3)])
    amp = torch.randn(B * M, 3, generator=gen, device=dev) * \
        torch.tensor([0.8, 0.4, 0.2], device=dev)[None]
    x = base + torch.einsum("bc,cl->bl", amp, modes)

    pin0, pin1 = phi0_r, phi0_r + dphi0_r * DT
    z = x[:, 2:].clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        traj = torch.cat([pin0[:, None], pin1[:, None], z], 1)
        cost(traj, umax_r).sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        traj = torch.cat([pin0[:, None], pin1[:, None], z], 1)
        J = cost(traj, umax_r)
        ok = success(traj, umax_r)
        sig = mode_sign(traj)
    return traj.detach(), J, ok, sig


def build_split(n_inst, device, seed, n_inits=8, iters=800, chunk=512):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = {k: [] for k in ("traj", "nominal", "feat", "cvec", "umax",
                           "best_J", "inst_id", "phi0", "dphi0")}
    done, base_id = 0, 0
    while done < n_inst:
        b = min(chunk, n_inst - done)
        inst = sample_instances(b, device, gen)
        traj, J, ok, sig = solve_oracle(inst, n_inits, iters, gen=gen)
        M = traj.shape[0] // b
        traj = traj.view(b, M, L)
        J, ok, sig = J.view(b, M), ok.view(b, M), sig.view(b, M)
        nom = nominal(inst)
        cvec = torch.stack([inst["phi0"], inst["dphi0"], inst["umax"] / 5.0], -1)
        for i in range(b):
            if not ok[i].any():
                continue
            Ji = torch.where(ok[i], J[i], torch.inf)
            best = Ji.min()
            seen = {}
            for m in range(M):
                if not ok[i, m] or Ji[m] > best * 2.0:
                    continue
                key = int(sig[i, m])
                if key not in seen or Ji[m] < Ji[seen[key]]:
                    seen[key] = m
            for m in seen.values():
                out["traj"].append((traj[i, m] / torch.pi)[:, None])
                out["nominal"].append(nom[i])
                out["feat"].append(nom[i])
                out["cvec"].append(cvec[i])
                out["umax"].append(inst["umax"][i])
                out["best_J"].append(best)
                out["inst_id"].append(torch.tensor(base_id + i))
                out["phi0"].append(inst["phi0"][i])
                out["dphi0"].append(inst["dphi0"][i])
        done += b
        base_id += b
        print(f"  {done}/{n_inst} inst, {len(out['traj'])} samples", flush=True)
    return {k: torch.stack(v).cpu() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=700)
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    print("pend train:", flush=True)
    tr = build_split(args.n_train, "cuda", seed=10, iters=1000)
    torch.save(tr, "data/pend_train.pt")
    print("pend test:", flush=True)
    te = build_split(args.n_test, "cuda", seed=11, n_inits=16, iters=1200)
    torch.save(te, "data/pend_test.pt")
    res = tr["traj"] - tr["nominal"]
    print(f"train {tr['traj'].shape[0]} test {te['traj'].shape[0]} "
          f"res_rms {res.pow(2).sum(-1).mean().sqrt():.4f}")


if __name__ == "__main__":
    main()
