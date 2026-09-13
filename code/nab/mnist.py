"""Class-conditional MNIST: the shifted-diffusion mechanism outside control.

Anchors are per-class mean images (the literal "cluster centers" of
ST-DDPM).  Two variants share one architecture and conditioning: base
(shift zero) and nab (s_t = 1 - sqrt(abar), residual around the class
mean).  Evaluation uses a small CNN classifier: per-sample quality =
class accuracy; distributional metrics = intra-class feature diversity
and per-class Frechet distance (mean+cov in feature space).

Usage:
  python -m nab.mnist --stage train --variant base_nom
  python -m nab.mnist --stage train --variant nab
  python -m nab.mnist --stage eval
"""

import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn

from .diffusion import Diffusion
from .model import timestep_embedding

DEVICE = "cuda"
RES = 32
D = RES * RES


# ---------------------------------------------------------------- data
def load_mnist():
    from torchvision import datasets
    tr = datasets.MNIST("data/mnist", train=True, download=True)
    x = tr.data.float() / 255.0 * 2 - 1                       # (N,28,28)
    x = torch.nn.functional.pad(x, (2, 2, 2, 2), value=-1.0)  # (N,32,32)
    y = tr.targets
    return x.to(DEVICE), y.to(DEVICE)


def class_means(x, y):
    return torch.stack([x[y == c].mean(0) for c in range(10)])  # (10,32,32)


# ---------------------------------------------------------------- model
class ResBlock2(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(torch.nn.functional.silu(self.n1(x)))
        s, b = self.e(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.c2(torch.nn.functional.silu(self.n2(h) * (1 + s) + b))
        return h + self.skip(x)


class UNet2(nn.Module):
    """Small 2D UNet wrapped to the (B, D, 1) sequence interface of
    Diffusion: cond carries the class index and the class-mean image."""

    def __init__(self, base=32, emb=128):
        super().__init__()
        self.emb = emb
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(),
                                   nn.Linear(emb, emb))
        self.cls = nn.Embedding(10, emb)
        ch = [base, base * 2, base * 4]
        self.inc = nn.Conv2d(2, ch[0], 3, padding=1)
        self.d1 = ResBlock2(ch[0], ch[0], emb)
        self.p1 = nn.Conv2d(ch[0], ch[1], 4, 2, 1)
        self.d2 = ResBlock2(ch[1], ch[1], emb)
        self.p2 = nn.Conv2d(ch[1], ch[2], 4, 2, 1)
        self.m1 = ResBlock2(ch[2], ch[2], emb)
        self.m2 = ResBlock2(ch[2], ch[2], emb)
        self.u2 = nn.ConvTranspose2d(ch[2], ch[1], 4, 2, 1)
        self.b2 = ResBlock2(ch[1] * 2, ch[1], emb)
        self.u1 = nn.ConvTranspose2d(ch[1], ch[0], 4, 2, 1)
        self.b1 = ResBlock2(ch[0] * 2, ch[0], emb)
        self.on = nn.GroupNorm(8, ch[0])
        self.oc = nn.Conv2d(ch[0], 1, 3, padding=1)
        nn.init.zeros_(self.oc.weight)
        nn.init.zeros_(self.oc.bias)

    def forward(self, x_t, t, cond):
        B = x_t.shape[0]
        img = x_t.view(B, 1, RES, RES)
        mu = cond["nominal"].view(B, 1, RES, RES)
        emb = self.t_mlp(timestep_embedding(t, self.emb)) + self.cls(cond["label"])
        h0 = self.inc(torch.cat([img, mu], 1))
        h1 = self.d1(h0, emb)
        h2 = self.d2(self.p1(h1), emb)
        m = self.m2(self.m1(self.p2(h2), emb), emb)
        u2 = self.b2(torch.cat([self.u2(m), h2], 1), emb)
        u1 = self.b1(torch.cat([self.u1(u2), h1], 1), emb)
        out = self.oc(torch.nn.functional.silu(self.on(u1)))
        return out.view(B, D, 1)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(128, 10)

    def forward(self, x, feat=False):
        z = self.f(x)
        return z if feat else self.head(z)


# ---------------------------------------------------------------- train
def train_classifier(x, y):
    net = SmallCNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(3000):
        idx = torch.randint(0, x.shape[0], (256,), device=DEVICE)
        loss = nn.functional.cross_entropy(net(x[idx][:, None]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    torch.save(net.state_dict(), "runs/mnist_cls.pt")
    return net


def train_diffusion(variant, steps=25000, batch=128, T=256):
    torch.manual_seed(0)
    x, y = load_mnist()
    mu = class_means(x, y)
    shift = {"base_nom": "zero", "nab": "nab"}[variant]
    model = UNet2().to(DEVICE)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    diff = Diffusion(T=T, shift=shift, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=2e-5)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, x.shape[0], (batch,), device=DEVICE)
        x0 = x[idx].view(batch, D, 1)
        cond = {"nominal": mu[y[idx]].view(batch, D, 1), "label": y[idx]}
        loss = diff.loss(model, x0, cond)
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
            print(f"[mnist/{variant}] {step} {loss.item():.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    R = (x.view(-1, D) - mu[y].view(-1, D)).pow(2).mean().sqrt().item()
    torch.save({"ema": ema.state_dict(), "shift": shift, "T": T, "R": R},
               f"runs/mnist_{variant}.pt")
    print(f"saved mnist_{variant} R={R:.4f}")


# ---------------------------------------------------------------- eval
@torch.no_grad()
def sample(model, diff, mu, labels, nfe, t_start=None, anchor="mean", seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    cond = {"nominal": mu[labels].view(-1, D, 1), "label": labels}
    out = diff.sample(model, cond, n_steps=nfe, eta=0.0, t_start=t_start,
                      anchor=anchor, gen=gen, pin_idx=[],
                      pin_pts=torch.zeros(labels.shape[0], 0, 1, device=DEVICE))
    return out.view(-1, 1, RES, RES).clamp(-1, 1)


@torch.no_grad()
def metrics(imgs, labels, cls, x, y):
    """Accuracy, intra-class feature diversity, per-class Frechet dist."""
    f_gen = cls(imgs, feat=True)
    acc = (cls.head(f_gen).argmax(1) == labels).float().mean().item()
    fd, div, dref = [], [], []
    for c in range(10):
        fg = f_gen[labels == c]
        fr = cls(x[y == c][:800][:, None], feat=True)
        m1, m2 = fg.mean(0), fr.mean(0)
        c1 = torch.cov(fg.T) + 1e-4 * torch.eye(fg.shape[1], device=DEVICE)
        c2 = torch.cov(fr.T) + 1e-4 * torch.eye(fr.shape[1], device=DEVICE)
        e1, v1 = torch.linalg.eigh(c1)
        s1 = v1 @ torch.diag(e1.clamp_min(0).sqrt()) @ v1.T
        inner = s1 @ c2 @ s1
        ei = torch.linalg.eigvalsh(inner).clamp_min(0)
        fd.append(((m1 - m2).pow(2).sum() + c1.trace() + c2.trace()
                   - 2 * ei.sqrt().sum()).item())
        div.append(torch.cdist(fg, fg).mean().item())
        dref.append(torch.cdist(fr, fr).mean().item())
    return {"acc": acc, "fd": sum(fd) / 10, "div": sum(div) / 10,
            "div_ref": sum(dref) / 10}


def evaluate():
    x, y = load_mnist()
    mu = class_means(x, y)
    cls = SmallCNN().to(DEVICE)
    if os.path.exists("runs/mnist_cls.pt"):
        cls.load_state_dict(torch.load("runs/mnist_cls.pt", weights_only=True))
        cls.eval()
    else:
        cls = train_classifier(x, y)
    labels = torch.arange(10, device=DEVICE).repeat_interleave(200)  # 2000

    res = {}
    models = {}
    for v in ("base_nom", "nab"):
        ck = torch.load(f"runs/mnist_{v}.pt", weights_only=True)
        m = UNet2().to(DEVICE)
        m.load_state_dict(ck["ema"])
        m.eval()
        models[v] = (m, Diffusion(T=ck["T"], shift=ck["shift"], device=DEVICE))
        res["R"] = ck["R"]

    # M1: full-budget equivalence
    m1 = {}
    for v, (m, d) in models.items():
        m1[v] = metrics(sample(m, d, mu, labels, 128, anchor="none"),
                        labels, cls, x, y)
        print("M1", v, {k: round(vv, 3) for k, vv in m1[v].items()}, flush=True)
    res["M1"] = m1

    # M2: truncation sweep (nab model, anchored vs naive)
    m2 = {}
    nab_m, nab_d = models["nab"]
    for t_s in [16, 32, 64, 96, 128, 160, 192, 224, 255]:
        m2[t_s] = {
            "anchored": metrics(sample(nab_m, nab_d, mu, labels, 32,
                                       t_start=t_s, anchor="mean"),
                                labels, cls, x, y),
            "naive": metrics(sample(nab_m, nab_d, mu, labels, 32,
                                    t_start=t_s, anchor="none"),
                             labels, cls, x, y),
        }
        a = m2[t_s]["anchored"]
        print(f"M2 {t_s} acc {a['acc']:.3f} div {a['div']:.2f} "
              f"fd {a['fd']:.1f} | naive acc {m2[t_s]['naive']['acc']:.3f}",
              flush=True)
    res["M2"] = m2

    # montage for the figure: class 3 and 8, deep vs rule truncation
    lab = torch.tensor([3] * 8 + [8] * 8, device=DEVICE)
    for name, ts in (("deep", 32), ("rule", 224)):
        img = sample(nab_m, nab_d, mu, lab, 32, t_start=ts, anchor="mean",
                     seed=2)
        torch.save(img.cpu(), f"results/mnist_montage_{name}.pt")
    torch.save(mu.cpu(), "results/mnist_means.pt")

    with open("results/mnist_results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved results/mnist_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--variant", default="nab")
    ap.add_argument("--steps", type=int, default=25000)
    args = ap.parse_args()
    os.makedirs("runs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    if args.stage == "train":
        train_diffusion(args.variant, steps=args.steps)
    else:
        evaluate()
