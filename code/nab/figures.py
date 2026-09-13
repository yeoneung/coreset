"""Generate paper figures and tables from results/results.json.

Usage: python -m nab.figures
"""

import json
import math
import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import env
from .eval import load_models, load_instances, rep_cond, sample_cfg
from .diffusion import score_evaluation_count
from .plot_style import finish_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})

NAMES = {"base_raw": r"\textsc{base-raw}", "base_nom": r"\textsc{base-nom}",
         "nab": r"\textsc{nab}", "stddpm": r"\textsc{st-ddpm}"}
PLAIN = {"base_nom_full": "full-range stride",
         "nab_es_dim": "anchored, aggressive $t_s$",
         "nab_es_kl": "anchored, KL-rule $t_s$",
         "base_nom_ws_kl": "warm-started base (equiv.)",
         "stddpm_naive_kl": "naive anchor (ST-DDPM ES)"}
COLORS = {"base_nom_full": "#888888", "nab_es_dim": "#e67e22",
          "nab_es_kl": "#c0392b", "base_nom_ws_kl": "#2980b9",
          "stddpm_naive_kl": "#8e44ad"}


def table_main(res):
    from . import env as _env
    d = torch.load(PROJECT_ROOT / "artifacts/data/test.pt", map_location="cpu", weights_only=True)
    ref_u = torch.load(PROJECT_ROOT / "artifacts/data/test_ref_known.pt", map_location="cpu", weights_only=True)
    ids = d["inst_id"].tolist()
    first = {}
    for iid in ids:
        if iid not in first:
            first[iid] = len(first)
    ref = torch.tensor([ref_u[first[iid]] for iid in ids])
    Jd = _env.cost(d["traj"], d["obs"], d["mask"])
    scale = ref_u.mean().item()
    dgap = ((Jd - ref) / scale).median().item()
    rows = [f"oracle data (teacher) & $100.0$ & ${dgap:.1f}$ & --- & $100.0$ "
            f"& $100.0$ & --- \\\\ \\midrule"]
    src, srcp = ("E1t", "E1tp") if "E1t" in res else ("E1", "E1p")
    for v in ("base_raw", "base_nom", "nab", "stddpm"):
        runs, runsp = res[src][v], res[srcp][v]
        def ms(rr, key, scale=1.0):
            vals = [r[key] * scale for r in rr]
            m = sum(vals) / len(vals)
            s = (sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)) ** 0.5
            return f"${m:.1f}\\pm{s:.1f}$"
        rows.append(f"{NAMES[v]} & {ms(runs, 'feasible', 100)} & "
                    f"{ms(runs, 'subopt_med')} & "
                    f"{ms(runsp, 'subopt_med')} & "
                    f"{ms(runs, 'mode_recall', 100)} & "
                    f"{ms(runs, 'mode_recall_mm', 100)} & "
                    f"{ms(runs, 'diversity')} \\\\")
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Full-budget sampling (DDIM, NFE $129$, $16$ samples per task, "
        "mean $\\pm$ standard deviation over $3$ training seeds and $2$ sampling "
        "seeds).  The reported median gap is the normalized quantity "
        "$(J-J^*)/\\overline{J^*}$ over feasible outputs, with a common mean "
        "best-known reference denominator; it is dimensionless, while feasibility "
        "and mode recall are percentages.  Unmarked and ``+polish'' columns "
        "evaluate raw and identically polished outputs, respectively.  The "
        "teacher row is evaluated against the stronger post hoc reference, "
        "not its data-generation filter.  The three equally conditioned "
        "coordinate parameterizations agree within seed variation, whereas "
        "BASE-RAW omits the nominal input.  This reference differs from the "
        "covariance-ablation reference.}\n"
        "\\label{tab:main}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Variant & Feas.\\ (\\%) & $\\Delta J$ & $\\Delta J$ +polish & "
        "Recall (\\%) & Recall-MM (\\%) & Diversity \\\\\n\\midrule\n"
        + body +
        "\n\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n")
    with open("tables/main.tex", "w") as f:
        f.write(tex)


def fig_nfe(res):
    nfes = sorted(int(k) for k in res["E2b"])
    starts = {"base_nom_full": 255, "nab_es_dim": res["t_s_dim"],
              "nab_es_kl": res["t_s_kl"], "base_nom_ws_kl": res["t_s_kl"],
              "stddpm_naive_kl": res["t_s_kl"]}
    ticks = [score_evaluation_count(n, 255) for n in nfes]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7))
    specs = [("feasible", "Feasibility (%)", 100),
             ("subopt_med", "Median normalized gap $\\Delta J$", 1.0),
             ("mode_recall_mm", "Mode recall, multimodal (%)", 100)]
    for ax, (key, label, sc) in zip(axes, specs):
        for cfg in ("base_nom_full", "nab_es_dim", "nab_es_kl",
                    "base_nom_ws_kl", "stddpm_naive_kl"):
            ys = [res["E2b"][str(n)][cfg][key] * sc for n in nfes]
            actual = [score_evaluation_count(n, starts[cfg]) for n in nfes]
            ax.plot(actual, ys, "o-", ms=3.5, lw=1.4, color=COLORS[cfg], label=PLAIN[cfg])
        ax.set_xscale("log", base=2)
        if key == "subopt_med":
            ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("Score evaluations (NFE)")
        ax.set_ylabel(label)
    finish_figure(fig, shared_legend=True, legend_columns=3)
    fig.savefig("figures/nfe.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_frontier(res):
    ws = [w for w in sorted(float(k) for k in res["E3e"]) if w <= 4.0]
    sub_r = [res["E3e"][str(w)]["subopt_med"] for w in ws]
    sub_p = [res["E3p"][str(w)]["subopt_med"] for w in ws]
    div = [res["E3e"][str(w)]["diversity"] for w in ws]
    feas = [res["E3e"][str(w)]["feasible"] * 100 for w in ws]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    ax = axes[0]
    for sub, c, lbl in ((sub_r, "#c0392b", "raw"), (sub_p, "#1a7a4a", "+polish")):
        ax.scatter(div, sub, c=ws, cmap="viridis", s=30, zorder=3,
                   edgecolors=c, linewidths=1.2)
        ax.plot(div, sub, "-", color=c, lw=1, label=lbl)
    for w, x, y in zip(ws, div, sub_r):
        ax.annotate(f"$w={w:g}$", (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    ax.set_xlabel("Diversity (mean pairwise distance)")
    ax.set_ylabel("Median normalized gap $\\Delta J$")
    ax.legend(fontsize=7, frameon=False)
    ax2 = axes[1]
    ax2.plot(ws, feas, "o-", color="#c0392b", ms=4)
    ax2.set_xlabel("Guidance weight $w$ (effective inverse temperature)")
    ax2.set_ylabel("Feasibility (%)")
    fig.tight_layout()
    fig.savefig("figures/frontier.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_anchor(res):
    tss = sorted(int(k) for k in res["E4"])
    ab = np.array([res["E4"][str(t)]["abar"] for t in tss])
    R = res["R"]
    scaled_rms = np.sqrt(ab) * R  # pooled residual RMS per trajectory point
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for k, lbl, c in (("anchored", "anchored init (mean)", "#c0392b"),
                      ("naive", "naive init ($m=0$)", "#8e44ad")):
        ax.plot(tss, [res["E4"][str(t)][k]["feasible"] * 100 for t in tss],
                "o-", ms=3.5, color=c, label=lbl)
    ax.set_xlabel("truncation point $t_s$")
    ax.set_ylabel("Feasibility (%)")
    for key, lbl in (("t_s_dim", "per-point heuristic"), ("t_s_kl", "KL rule")):
        ax.axvline(res[key], color="0.4", ls="--", lw=1)
        ax.annotate(lbl, (res[key], ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=7, color="0.3",
                    rotation=90)
    ax2 = ax.twinx()
    ax2.plot(tss, scaled_rms, ":", color="0.3", lw=1.2,
             label=r"scaled residual RMS $\sqrt{\bar\alpha_{t_s}}\,R_{\rm pt}$")
    ax2.set_ylabel("Scaled residual RMS (per point)")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="center right")
    finish_figure(fig)
    fig.savefig("figures/anchor.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_qual(res):
    models = load_models()
    inst, raw = load_instances()
    # pick 3 multimodal instances
    idx = [i for i in range(inst["start"].shape[0]) if len(inst["modes"][i]) > 1][:3]
    sel = torch.tensor(idx, device="cuda")
    small = {k: (inst[k][sel] if torch.is_tensor(inst[k]) else [inst[k][i] for i in idx])
             for k in inst}
    cond = rep_cond(small)
    nab_m, nab_d, R = models["nab"]
    ts = res["t_s_kl"]
    cols = {
        "Oracle modes + nominal": None,
        "NAB, NFE 129": sample_cfg(nab_m, nab_d, cond, 128, anchor="none", seed=1),
        f"NAB anchored, NFE {score_evaluation_count(8, ts)}": sample_cfg(nab_m, nab_d, cond, 8, t_start=ts, anchor="mean", seed=1),
        f"NAB + guidance ($w{{=}}2$), NFE {score_evaluation_count(16, ts)}":
            sample_cfg(nab_m, nab_d, cond, 16, t_start=ts, anchor="mean", w=2.0, seed=1),
    }
    fig, axes = plt.subplots(len(idx), 4, figsize=(10.5, 2.6 * len(idx)))
    ids = raw["inst_id"]
    for r, i in enumerate(idx):
        for c, (title, x) in enumerate(cols.items()):
            ax = axes[r, c]
            ax.set_aspect("equal")
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-0.75, 0.75)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
            for k in range(env.KMAX):
                if small["mask"][r, k] > 0:
                    cx, cy, rad = small["obs"][r, k].tolist()
                    ax.add_patch(plt.Circle((cx, cy), rad, color="0.75", zorder=1))
            if x is None:
                # find raw rows belonging to this instance (same start/goal)
                same = ((raw["start"] - small["start"][r]).norm(dim=-1) < 1e-6) & \
                       ((raw["goal"] - small["goal"][r]).norm(dim=-1) < 1e-6)
                for j in torch.nonzero(same).flatten().tolist():
                    tr = raw["traj"][j].cpu()
                    ax.plot(tr[:, 0], tr[:, 1], "-", lw=1.6, color="#1a7a4a", zorder=3)
                nom = small["nominal"][r].cpu()
                ax.plot(nom[:, 0], nom[:, 1], "--", color="#c0392b", lw=1.2, zorder=4)
            else:
                xs = x.view(len(idx), -1, env.L, 2)[r].cpu()
                for j in range(xs.shape[0]):
                    ax.plot(xs[j, :, 0], xs[j, :, 1], "-", lw=0.8, alpha=0.65,
                            color="#2955a3", zorder=3)
            ax.plot(*small["start"][r].cpu(), "ko", ms=4, zorder=5)
            ax.plot(*small["goal"][r].cpu(), "k*", ms=7, zorder=5)
            if r == 0:
                ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/qual.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Regenerate cached obstacle-avoidance reporting assets.")
    parser.add_argument("--qualitative", action="store_true",
                        help="Also resample qualitative trajectories; requires regenerated runs/ and data/.")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    with open("results/results.json") as f:
        res = json.load(f)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    table_main(res)
    fig_nfe(res)
    fig_frontier(res)
    fig_anchor(res)
    if args.qualitative:
        fig_qual(res)
    print("figures + tables written")


if __name__ == "__main__":
    main()
