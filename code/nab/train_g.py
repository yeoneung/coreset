"""Generic training for UNetG-based variants on any environment dataset.

Dataset .pt must contain: traj (N,L,D), nominal (N,L,D), feat (N,L,F),
cvec (N,C).  Usage:
    python -m nab.train_g --data data/pend_train.pt --tag pend --variant nab
"""

import argparse
import copy
import os
import time

import torch

from .diffusion import Diffusion
from .model import UNetG

SHIFT = {"base_nom": "zero", "nab": "nab", "stddpm": "stddpm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", required=True, choices=list(SHIFT))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    d = torch.load(args.data, weights_only=True)
    d = {k: v.to(device) for k, v in d.items()}
    N, L, D = d["traj"].shape
    F, C = d["feat"].shape[-1], d["cvec"].shape[-1]
    R = (d["traj"] - d["nominal"]).pow(2).sum(-1).mean().sqrt().item()

    model = UNetG(x_dim=D, feat_dim=F, cvec_dim=C).to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = Diffusion(T=args.T, shift=SHIFT[args.variant], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps,
                                                       eta_min=args.lr * 0.05)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch,), device=device)
        cond = {k: d[k][idx] for k in ("feat", "cvec", "nominal")}
        loss = diff.loss(model, d["traj"][idx], cond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            decay = 0.9995 if step > 1000 else 0.99
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        if step % 5000 == 0 or step == 1:
            print(f"[{args.tag}/{args.variant}] {step} loss {loss.item():.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"ema": ema.state_dict(), "variant": args.variant,
                "shift": SHIFT[args.variant], "T": args.T, "R": R,
                "dims": (D, F, C)},
               os.path.join(args.out, f"{args.tag}_{args.variant}.pt"))
    print(f"saved {args.tag}_{args.variant}  R={R:.4f}")


if __name__ == "__main__":
    main()
