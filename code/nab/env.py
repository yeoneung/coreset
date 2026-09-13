"""Double-integrator obstacle-avoidance environment.

Trajectories are position sequences p_{0:L-1} in R^2 with L=64 points on the
normalized horizon [0,1] (dt = 1/(L-1)).  Controls are recovered by second
differences a_k = (p_{k+1} - 2 p_k + p_{k-1}) / dt^2.  The task cost is

    J(tau) = 0.5 * sum_k ||a_k||^2 dt  +  w_obs * sum_k pen(p_k)^2 dt

where pen is circular-obstacle penetration.  The obstacle-free minimizer with
pinned endpoints is the straight line (the "nominal" / cluster center).
"""

import torch

L = 64
DT = 1.0 / (L - 1)
KMAX = 3
W_OBS = 400.0
MARGIN = 0.03  # safety margin used during optimization / guidance


def sample_instances(n, device, gen=None):
    """Random instances: start/goal on x = -1 / +1, K in {1,2,3} circles."""
    def U(a, b, *shape):
        return a + (b - a) * torch.rand(*shape, generator=gen, device=device)

    start = torch.stack([-torch.ones(n, device=device), U(-0.25, 0.25, n)], -1)
    goal = torch.stack([torch.ones(n, device=device), U(-0.25, 0.25, n)], -1)
    k = torch.randint(1, KMAX + 1, (n,), generator=gen, device=device)
    mask = (torch.arange(KMAX, device=device)[None, :] < k[:, None]).float()
    cx = U(-0.55, 0.55, n, KMAX)
    cy = U(-0.40, 0.40, n, KMAX)
    r = U(0.18, 0.32, n, KMAX)
    # keep obstacles away from start/goal discs
    for _ in range(20):
        c = torch.stack([cx, cy], -1)                        # (n,K,2)
        bad = torch.zeros_like(cx, dtype=torch.bool)
        for pt in (start, goal):
            d = (c - pt[:, None, :]).norm(dim=-1)
            bad |= d < (r + 0.12)
        if not bad.any():
            break
        cx = torch.where(bad, U(-0.55, 0.55, n, KMAX), cx)
        cy = torch.where(bad, U(-0.40, 0.40, n, KMAX), cy)
        r = torch.where(bad, U(0.18, 0.32, n, KMAX), r)
    obs = torch.stack([cx, cy, r], -1)                       # (n,K,3)
    return {"start": start, "goal": goal, "obs": obs, "mask": mask}


def nominal(start, goal):
    """Straight-line minimum-energy solution ignoring obstacles. (B,L,2)"""
    w = torch.linspace(0, 1, L, device=start.device)[None, :, None]
    return start[:, None, :] * (1 - w) + goal[:, None, :] * w


def control_energy(traj):
    """0.5 * sum ||a_k||^2 dt with a = second difference / dt^2. (B,)"""
    a = (traj[:, 2:] - 2 * traj[:, 1:-1] + traj[:, :-2]) / DT**2
    return 0.5 * (a**2).sum(dim=(1, 2)) * DT


def penetration(traj, obs, mask, margin=0.0):
    """Per-point obstacle penetration depth, masked. (B,L)"""
    d = (traj[:, :, None, :] - obs[:, None, :, :2]).norm(dim=-1)  # (B,L,K)
    pen = torch.relu(obs[:, None, :, 2] + margin - d) * mask[:, None, :]
    return pen.sum(dim=-1)


def cost(traj, obs, mask, margin=MARGIN, w_obs=W_OBS):
    return control_energy(traj) + w_obs * (penetration(traj, obs, mask, margin) ** 2).sum(-1) * DT


def feasible(traj, obs, mask, tol=1e-3):
    return penetration(traj, obs, mask, 0.0).max(dim=-1).values <= tol


def signature(traj, start, goal, obs, mask):
    """Which side of each obstacle the path passes: (B,K) in {-1,+1},
    proximity-weighted cross-product sign along the start->goal axis."""
    u = goal - start
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)     # (B,2)
    rel = traj[:, :, None, :] - obs[:, None, :, :2]          # (B,L,K,2)
    cross = u[:, None, None, 0] * rel[..., 1] - u[:, None, None, 1] * rel[..., 0]
    d = rel.norm(dim=-1)
    w = torch.exp(-(d - obs[:, None, :, 2]).clamp_min(0) ** 2 / 0.02)
    s = torch.sign((cross * w).sum(dim=1))                   # (B,K)
    return (s * mask).long()


def _init_perturbations(base, n_inits, gen=None, mixed=False):
    """base (B,L,2) -> (B*M,L,2) smooth lateral perturbations, both sides.
    mixed=True additionally shrinks the second half of the inits toward the
    base (near-straight starts) --- essential for a reference oracle, since
    side-forced inits can stall in needlessly curved local minima on tasks
    the straight path already solves."""
    B = base.shape[0]
    dev = base.device
    tau = torch.linspace(0, 1, L, device=dev)
    modes = torch.stack([torch.sin(torch.pi * m * tau) for m in (1, 2, 3)])  # (3,L)
    amp = torch.randn(B, n_inits, 3, 2, generator=gen, device=dev) * \
        torch.tensor([0.45, 0.22, 0.12], device=dev)[None, None, :, None]
    # force half the inits to opposite lateral sides for mode diversity
    side = torch.where(torch.arange(n_inits, device=dev) % 2 == 0, 1.0, -1.0)
    amp[:, :, 0, 1] = amp[:, :, 0, 1].abs() * side[None, :]
    if mixed:
        amp[:, n_inits // 2:] *= 0.05
    pert = torch.einsum("bmcd,cl->bmld", amp, modes)          # (B,M,L,2)
    return (base[:, None] + pert).reshape(B * n_inits, L, 2)


def energy_precond(device, delta=1.0):
    """Regularized inverse Hessian of the control energy w.r.t. interior
    points: M = (D2^T D2 / dt^3 + delta I)^{-1}, shape (L-2, L-2).  Acting on
    cost gradients this is a smoothing (quasi-Newton) preconditioner."""
    n = L - 2
    D = torch.zeros(n, n, device=device)
    idx = torch.arange(n, device=device)
    D[idx, idx] = -2.0
    D[idx[:-1], idx[:-1] + 1] = 1.0
    D[idx[1:], idx[1:] - 1] = 1.0
    H = D.T @ D / DT**3
    return torch.linalg.inv(H + delta * torch.eye(n, device=device))


def polish(traj, obs, mask, start, goal, k=60, lr=0.01, fixed_head=1):
    """Fixed cheap post-processing applied identically to every sampler:
    k small Adam steps on the *data-generating* cost (identical objective,
    margin, and penalty weight as the oracle) with pinned endpoints (and
    optionally the first `fixed_head` points, e.g.\\ to preserve an initial
    velocity).  The small steps respect the obstacle barrier, so this is
    basin-preserving local refinement; polished suboptimality measures
    basin quality against the oracle under the identical objective."""
    head = traj[:, :fixed_head].clone()
    head[:, 0] = start
    z = traj[:, fixed_head:-1].clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(k):
        opt.zero_grad(set_to_none=True)
        x = torch.cat([head, z, goal[:, None]], 1)
        cost(x, obs, mask).sum().backward()
        opt.step()
    return torch.cat([head, z, goal[:, None]], 1).detach()


def solve_oracle(inst, n_inits=8, iters=800, lr=0.03, gen=None, mixed=False):
    """Batched multi-start trajectory optimization.  Returns per-candidate
    trajectories (B*M,L,2) with endpoints pinned, plus energies/feasibility."""
    start, goal = inst["start"], inst["goal"]
    obs, mask = inst["obs"], inst["mask"]
    B = start.shape[0]
    base = nominal(start, goal)
    x = _init_perturbations(base, n_inits, gen, mixed=mixed)
    obs_r = obs.repeat_interleave(n_inits, 0)
    mask_r = mask.repeat_interleave(n_inits, 0)
    s_r = start.repeat_interleave(n_inits, 0)
    g_r = goal.repeat_interleave(n_inits, 0)

    z = x[:, 1:-1].clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        traj = torch.cat([s_r[:, None], z, g_r[:, None]], dim=1)
        cost(traj, obs_r, mask_r).sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        traj = torch.cat([s_r[:, None], z, g_r[:, None]], dim=1)
        e = control_energy(traj)
        feas = feasible(traj, obs_r, mask_r)
        sig = signature(traj, s_r, g_r, obs_r, mask_r)
    return traj.detach(), e, feas, sig
