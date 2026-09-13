"""Temporal 1D U-Net epsilon-predictor with FiLM time/condition modulation.

Input channels: x_t (2) + nominal (2) + obstacle features (KMAX*4 = 12)
+ normalized time-index channel (1) = 17.  The `use_nominal` flag zeroes the
nominal channels (for the raw-conditioned baseline); every variant otherwise
receives identical conditioning information.
"""

import math
import torch
import torch.nn as nn

from .env import KMAX


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        scale, shift = self.emb(emb)[:, :, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(torch.nn.functional.silu(h))
        return h + self.skip(x)


class UNetG(nn.Module):
    """Generic temporal U-Net: x (B,L,D) + per-step features (B,L,F) +
    global condition vector (B,C).  Environment-agnostic."""

    def __init__(self, x_dim, feat_dim, cvec_dim, base=64, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        self.c_mlp = nn.Sequential(nn.Linear(cvec_dim, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        ch = [base, base * 2, base * 4]
        self.in_conv = nn.Conv1d(x_dim + feat_dim + 1, ch[0], 3, padding=1)
        self.d1 = ResBlock(ch[0], ch[0], emb_dim)
        self.down1 = nn.Conv1d(ch[0], ch[1], 4, stride=2, padding=1)
        self.d2 = ResBlock(ch[1], ch[1], emb_dim)
        self.down2 = nn.Conv1d(ch[1], ch[2], 4, stride=2, padding=1)
        self.mid1 = ResBlock(ch[2], ch[2], emb_dim)
        self.mid2 = ResBlock(ch[2], ch[2], emb_dim)
        self.up2 = nn.ConvTranspose1d(ch[2], ch[1], 4, stride=2, padding=1)
        self.u2 = ResBlock(ch[1] * 2, ch[1], emb_dim)
        self.up1 = nn.ConvTranspose1d(ch[1], ch[0], 4, stride=2, padding=1)
        self.u1 = ResBlock(ch[0] * 2, ch[0], emb_dim)
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_conv = nn.Conv1d(ch[0], x_dim, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t, t, cond):
        feat, cvec = cond["feat"], cond["cvec"]
        B, L, _ = x_t.shape
        emb = self.t_mlp(timestep_embedding(t, self.emb_dim)) + self.c_mlp(cvec)
        tau = torch.linspace(0, 1, L, device=x_t.device)[None, :, None].expand(B, L, 1)
        h0 = self.in_conv(torch.cat([x_t, feat, tau], -1).permute(0, 2, 1))
        h1 = self.d1(h0, emb)
        h2 = self.d2(self.down1(h1), emb)
        m = self.mid2(self.mid1(self.down2(h2), emb), emb)
        u2 = self.u2(torch.cat([self.up2(m), h2], dim=1), emb)
        u1 = self.u1(torch.cat([self.up1(u2), h1], dim=1), emb)
        out = self.out_conv(torch.nn.functional.silu(self.out_norm(u1)))
        return out.permute(0, 2, 1)


class UNet1D(nn.Module):
    def __init__(self, base=64, emb_dim=256, use_nominal=True):
        super().__init__()
        self.use_nominal = use_nominal
        cin = 2 + 2 + KMAX * 4 + 1
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        self.c_mlp = nn.Sequential(nn.Linear(4 + KMAX * 4, emb_dim), nn.SiLU(),
                                   nn.Linear(emb_dim, emb_dim))
        self.emb_dim = emb_dim
        ch = [base, base * 2, base * 4]
        self.in_conv = nn.Conv1d(cin, ch[0], 3, padding=1)
        self.d1 = ResBlock(ch[0], ch[0], emb_dim)
        self.down1 = nn.Conv1d(ch[0], ch[1], 4, stride=2, padding=1)
        self.d2 = ResBlock(ch[1], ch[1], emb_dim)
        self.down2 = nn.Conv1d(ch[1], ch[2], 4, stride=2, padding=1)
        self.mid1 = ResBlock(ch[2], ch[2], emb_dim)
        self.mid2 = ResBlock(ch[2], ch[2], emb_dim)
        self.up2 = nn.ConvTranspose1d(ch[2], ch[1], 4, stride=2, padding=1)
        self.u2 = ResBlock(ch[1] * 2, ch[1], emb_dim)
        self.up1 = nn.ConvTranspose1d(ch[1], ch[0], 4, stride=2, padding=1)
        self.u1 = ResBlock(ch[0] * 2, ch[0], emb_dim)
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_conv = nn.Conv1d(ch[0], 2, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def features(self, x_t, cond):
        """Assemble per-timestep input channels. x_t (B,L,2) -> (B,C,L)."""
        B, L, _ = x_t.shape
        nom = cond["nominal"] if self.use_nominal else torch.zeros_like(x_t)
        obs, mask = cond["obs"], cond["mask"]                     # (B,K,3),(B,K)
        of = torch.cat([obs, mask[..., None]], -1) * mask[..., None]  # (B,K,4)
        of = of.reshape(B, -1)[:, None, :].expand(B, L, KMAX * 4)
        tau = torch.linspace(0, 1, L, device=x_t.device)[None, :, None].expand(B, L, 1)
        feat = torch.cat([x_t, nom, of, tau], dim=-1)             # (B,L,C)
        return feat.permute(0, 2, 1)

    def forward(self, x_t, t, cond):
        emb = self.t_mlp(timestep_embedding(t, self.emb_dim))
        cvec = torch.cat([cond["start"], cond["goal"],
                          (cond["obs"] * cond["mask"][..., None]).reshape(x_t.shape[0], -1),
                          ], dim=-1)
        # pad cvec with mask to fixed width 4+KMAX*4
        cvec = torch.cat([cvec, cond["mask"]], dim=-1)[:, : 4 + KMAX * 4]
        emb = emb + self.c_mlp(cvec)
        h0 = self.in_conv(self.features(x_t, cond))
        h1 = self.d1(h0, emb)
        h2 = self.d2(self.down1(h1), emb)
        m = self.mid2(self.mid1(self.down2(h2), emb), emb)
        u2 = self.u2(torch.cat([self.up2(m), h2], dim=1), emb)
        u1 = self.u1(torch.cat([self.up1(u2), h1], dim=1), emb)
        out = self.out_conv(torch.nn.functional.silu(self.out_norm(u1)))
        return out.permute(0, 2, 1)
