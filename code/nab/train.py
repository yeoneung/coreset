"""Train one diffusion variant.

Variants (process shift / nominal input channels):
    base_raw : shift zero,   no nominal channels
    base_nom : shift zero,   nominal channels
    nab      : shift nab,    nominal channels
    stddpm   : shift stddpm, nominal channels

Usage: python -m nab.train --variant nab --steps 30000
"""

import argparse
import copy
import json
import os
import time

import torch

from .diffusion import Diffusion
from .model import UNet1D

VARIANTS = {
    "base_raw": ("zero", False),
    "base_nom": ("zero", True),
    "nab": ("nab", True),
    "stddpm": ("stddpm", True),
}


def get_cond(d, idx):
    return {k: d[k][idx] for k in ("start", "goal", "obs", "mask", "nominal")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--data", type=str, default="data/train.pt")
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    shift, use_nom = VARIANTS[args.variant]

    d = torch.load(args.data, weights_only=True)
    d = {k: v.to(device) for k, v in d.items()}
    N = d["traj"].shape[0]
    R = (d["traj"] - d["nominal"]).pow(2).sum(-1).mean().sqrt().item()  # rms residual per point

    model = UNet1D(use_nominal=use_nom).to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = Diffusion(T=args.T, shift=shift, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.05)

    os.makedirs(args.out, exist_ok=True)
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch,), device=device)
        loss = diff.loss(model, d["traj"][idx], get_cond(d, idx))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            decay = 0.9995 if step > 1000 else 0.99
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        if step % 1000 == 0 or step == 1:
            log.append({"step": step, "loss": loss.item(), "sec": time.time() - t0})
            print(f"[{args.variant}] step {step} loss {loss.item():.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    ckpt = {"ema": ema.state_dict(), "variant": args.variant, "shift": shift,
            "use_nominal": use_nom, "T": args.T, "R": R, "log": log}
    suffix = f"_s{args.seed}" if args.seed != 0 else ""
    torch.save(ckpt, os.path.join(args.out, f"{args.variant}{suffix}.pt"))
    with open(os.path.join(args.out, f"{args.variant}{suffix}_log.json"), "w") as f:
        json.dump(log, f)
    print(f"saved {args.variant}{suffix}  R={R:.4f}")


if __name__ == "__main__":
    main()
