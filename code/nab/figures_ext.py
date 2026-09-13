"""Figures and tables for the pendulum and MPC experiments.

Usage: python -m nab.figures_ext
"""

import json
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .diffusion import score_evaluation_count
from .plot_style import finish_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]

plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_pend(res):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8))
    tss = sorted(int(k) for k in res["P2"])
    ax = axes[0]
    for k, lbl, c in (("anchored", "anchored init (mean)", "#c0392b"),
                      ("naive", "naive init ($m=0$)", "#8e44ad")):
        ax.plot(tss, [res["P2"][str(t)][k]["success"] * 100 for t in tss],
                "o-", ms=3.5, color=c, label=lbl)
    ax.axvline(res["t_s_kl"], color="0.4", ls="--", lw=1)
    ax.set_xlabel("truncation point $t_s$")
    ax.set_ylabel("Swing-up success (%)")
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_title("(a) swing-up success", fontsize=9.5)

    ax = axes[1]
    ax.plot(tss, [res["P2"][str(t)]["anchored"]["mode_recall_mm"] * 100
                  for t in tss], "o-", ms=3.5, color="#c0392b",
            label="mode recall (bimodal)")
    ax2 = ax.twinx()
    ax2.plot(tss, [res["P2"][str(t)]["anchored"]["diversity"] for t in tss],
             "s--", ms=3, color="#2980b9", label="diversity")
    ax2.spines["right"].set_visible(True)
    ax.axvline(res["t_s_kl"], color="0.4", ls="--", lw=1)
    ax.annotate("KL rule", (res["t_s_kl"], 55), xytext=(-32, 0),
                textcoords="offset points", fontsize=7, color="0.3")
    ax.set_xlabel("truncation point $t_s$")
    ax.set_ylabel("Mode recall, bimodal (%)")
    ax2.set_ylabel("Diversity")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(b) recall and diversity", fontsize=9.5)

    ax = axes[2]
    ws = [w for w in sorted(float(k) for k in res["P3"]) if w <= 1.0]
    sub = [res["P3"][str(w)]["subopt_med"] * 100 for w in ws]
    suc = [res["P3"][str(w)]["success"] * 100 for w in ws]
    ax.plot(ws, sub, "o-", ms=4, color="#c0392b", label="median subopt. (%)")
    ax2 = ax.twinx()
    ax2.plot(ws, suc, "s--", ms=3, color="#2980b9", label="success (%)")
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("Median cost suboptimality (%)")
    ax.set_yscale("log")
    ax2.set_ylabel("Success (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="center left")
    ax.set_title("(c) guidance sensitivity", fontsize=9.5)
    finish_figure(fig)
    fig.savefig("figures/pend.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_value(res):
    """Plug-in vs amortized guidance on the pendulum."""
    wp = [w for w in sorted(float(k) for k in res["P3"]) if 0 < w <= 4.0]
    wa = sorted(float(k) for k in res["P4"])
    base = res["P3"]["0.0"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8))
    specs = [("subopt_med", "Median cost suboptimality (%)", 100, True),
             ("success", "Swing-up success (%)", 100, False)]
    for ax, (key, lbl, sc, logy) in zip(axes, specs):
        ax.plot(wp, [res["P3"][str(w)][key] * sc for w in wp], "o-", ms=4,
                color="#8e44ad", label="plug-in (preconditioned, gated)")
        ax.plot(wa, [res["P4"][str(w)][key] * sc for w in wa], "s-", ms=4,
                color="#1a7a4a", label="amortized $\\nabla_x \\ell_\\psi$")
        ax.axhline(base[key] * sc, color="0.5", ls=":", lw=1.2,
                   label="unguided ($w{=}0$)")
        ax.set_xscale("log")
        ax.set_xticks(wp + [4.0])
        ax.set_xticklabels([f"{w:g}" for w in wp + [4.0]])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("guidance strength")
        ax.set_ylabel(lbl)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    # inset: zoom on the amortized improvement, linear scale
    axin = axes[0].inset_axes([0.52, 0.12, 0.44, 0.36])
    axin.plot(wa, [res["P4"][str(w)]["subopt_med"] * 100 for w in wa],
              "s-", ms=3, color="#1a7a4a")
    axin.axhline(base["subopt_med"] * 100, color="0.5", ls=":", lw=1)
    axin.set_xscale("log")
    axin.set_xticks([0.25, 1, 4])
    axin.set_xticklabels(["0.25", "1", "4"], fontsize=6)
    axin.tick_params(labelsize=6)
    axin.set_title("amortized (zoom)", fontsize=6)
    fig.tight_layout()
    fig.savefig("figures/value.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_mpc(res):
    prev_ts = sorted(int(k.split("@")[1]) for k in res if k.startswith("prev@"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    ax = axes[0]
    ax.plot(prev_ts, [res[f"prev@{t}"]["collision"] * 100 for t in prev_ts],
            "o-", color="#c0392b", ms=4, label="previous-solution anchor")
    ax.axhline(res["nom"]["collision"] * 100, color="#2980b9", ls="--", lw=1.2,
               label=f"nominal anchor ($t_s={res['t_s_nom']}$)")
    ax.axhline(res["full"]["collision"] * 100, color="0.5", ls=":", lw=1.2,
               label="full-range stride")
    ax.set_xlabel("truncation point $t_s$ (per replan)")
    ax.set_ylabel("Closed-loop collision (%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(prev_ts)
    ax.set_xticklabels(prev_ts)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("(a) collisions vs. replan truncation", fontsize=8)

    ax = axes[1]
    ax.plot(prev_ts, [res[f"prev@{t}"]["energy_ratio_med"] for t in prev_ts],
            "o-", color="#c0392b", ms=4)
    ax.axhline(res["nom"]["energy_ratio_med"], color="#2980b9", ls="--", lw=1.2)
    ax.axhline(res["full"]["energy_ratio_med"], color="0.5", ls=":", lw=1.2)
    ax.set_xlabel("truncation point $t_s$ (per replan)")
    ax.set_ylabel("Commanded energy / oracle")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(prev_ts)
    ax.set_xticklabels(prev_ts)
    ax.set_title("(b) closed-loop energy", fontsize=8)
    finish_figure(fig)
    fig.savefig("figures/mpc.pdf", bbox_inches="tight")
    plt.close(fig)


def table_mpc(res):
    prev_ts = sorted(int(k.split("@")[1]) for k in res if k.startswith("prev@"))
    rows = []
    label = {"oracle_mpc": "optimizer, cold start (8-start, 300 it.)",
             "oracle_warm": "optimizer, warm start (prev.\\ solution)",
             "full": "full-range stride",
             "nom": f"nominal anchor, $t_s={res['t_s_nom']}$"}
    order = ["oracle_mpc", "oracle_warm", "full", "nom"] + \
        [f"prev@{t}" for t in prev_ts]
    for k in order:
        r = res[k]
        name = label[k] if k in label else \
            f"prev.\\ solution, $t_s={k.split('@')[1]}$"
        sig = f"{r['sigma_pt']:.3f}" if r.get("sigma_pt") else "---"
        wall = f"{r['wall_ms']:.0f}" if "wall_ms" in r else "---"
        rows.append(f"{name} & {r['collision']*100:.1f} & "
                    f"{r['energy_ratio_med']:.2f} & {wall} & {sig} \\\\")
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Shrinking-horizon control with $5$ score-network evaluations "
        "per replan ($8$ global steps executed per replan, position and "
        "velocity disturbances, $117$ episodes; goal error $0.006$ for all "
        "methods).  Closed-loop collision rate, commanded per-segment "
        "energy relative to the open-loop oracle (median), wall-clock per "
        "replan for the whole $117$-episode batch on one GPU, and the "
        "measured per-point deviation $\\hat\\sigma$ of accepted plans from "
        "the anchor.  The warm optimizer uses eight starts, "
        "including two fresh nominal perturbations.  Collision rates are "
        "descriptive.}\n"
        "\\label{tab:mpc}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Replanner & Coll.\\ (\\%) & Energy ratio & ms/replan & "
        "$\\hat\\sigma$ (per pt) \\\\\n\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    with open("tables/mpc.tex", "w") as f:
        f.write(tex)


def fig_quad(res):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8))
    tss = sorted(int(k) for k in res["Q2p"])
    ax = axes[0]
    for k, lbl, c in (("anchored", "anchored init (mean)", "#c0392b"),
                      ("naive", "naive init ($m=0$)", "#8e44ad")):
        ax.plot(tss, [res["Q2p"][str(t)][k]["feasible"] * 100 for t in tss],
                "o-", ms=3.5, color=c, label=lbl)
    ax.axvline(res["t_s_kl"], color="0.4", ls="--", lw=1)
    ax.set_xlabel("truncation point $t_s$")
    ax.set_ylabel("Feasibility, polished (%)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("(a) truncation and feasibility", fontsize=9.5)

    ax = axes[1]
    ax.plot(tss, [res["Q2p"][str(t)]["anchored"]["mode_recall_mm"] * 100
                  for t in tss], "o-", ms=3.5, color="#c0392b",
            label="mode recall (multimodal)")
    ax2 = ax.twinx()
    ax2.plot(tss, [res["Q2p"][str(t)]["anchored"]["diversity"] for t in tss],
             "s--", ms=3, color="#2980b9", label="diversity")
    ax2.spines["right"].set_visible(True)
    ax.axvline(res["t_s_kl"], color="0.4", ls="--", lw=1)
    ax.set_xlabel("truncation point $t_s$")
    ax.set_ylabel("Mode recall, multimodal (%)")
    ax2.set_ylabel("Diversity")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(b) recall and diversity", fontsize=9.5)

    ax = axes[2]
    import torch as _t
    qsc = _t.load(PROJECT_ROOT / "artifacts/data/quad_test_ref_known.pt", map_location="cpu", weights_only=True).mean().item()
    ws = sorted(float(k) for k in res["Q3p"])
    sub = [res["Q3p"][str(w)]["subopt_med"] * qsc for w in ws]
    ax.plot(ws, sub, "o-", ms=4, color="#8e44ad")
    ax.set_yscale("log")
    ax.set_xlabel("guidance weight $w$")
    ax.set_ylabel("Median polished gap $\\Delta J$")
    ax.set_title("(c) guidance sensitivity", fontsize=9.5)
    finish_figure(fig)
    fig.savefig("figures/quad.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_quad_qual(res):
    import torch
    from . import env_quad as eq
    from .eval_quad import load_model, load_instances, rep_cond, sample_cfg
    models = {"nab": load_model("quad_nab")}
    inst, _ = load_instances()
    idx = [i for i in range(inst["start"].shape[0])
           if len(inst["modes"][i]) > 1][:2]
    sel = torch.tensor(idx, device="cuda")
    small = {k: (inst[k][sel] if torch.is_tensor(inst[k])
                 else [inst[k][i] for i in idx]) for k in inst}
    cond = rep_cond(small)
    nab_m, nab_d, _ = models["nab"]
    ts = res["t_s_kl"]
    x8 = sample_cfg(nab_m, nab_d, cond, 8, t_start=ts, anchor="mean", seed=1)
    cols = {
        "Oracle modes + nominal": None,
        "NAB, NFE 129": sample_cfg(nab_m, nab_d, cond, 128, anchor="none", seed=1),
        f"NAB anchored, NFE {score_evaluation_count(8, ts)}": x8,
        f"NFE {score_evaluation_count(8, ts)} + spectral polish":
            eq.polish(x8, cond["obs"], cond["mask"], cond["start"],
                      cond["goal"]),
    }
    d = torch.load("data/quad_test.pt", weights_only=True)
    fig, axes = plt.subplots(2 * len(idx), 4, figsize=(10.5, 4.6 * len(idx)))
    for r, i in enumerate(idx):
        for proj, (ax_r, dims, dlab) in enumerate(
                [(2 * r, (0, 1), "xy"), (2 * r + 1, (0, 2), "xz")]):
            for c, (title, x) in enumerate(cols.items()):
                ax = axes[ax_r, c]
                ax.set_aspect("equal")
                ax.set_xlim(-1.15, 1.15)
                ax.set_ylim(-0.7, 0.7)
                ax.set_xticks([])
                ax.set_yticks([])
                for k in range(eq.KMAX):
                    if small["mask"][r, k] > 0:
                        o = small["obs"][r, k].tolist()
                        ax.add_patch(plt.Circle((o[dims[0]], o[dims[1]]),
                                                o[3], color="0.8", zorder=1))
                if x is None:
                    same = ((d["start"].cuda() - small["start"][r]).norm(dim=-1) < 1e-6)
                    for j in torch.nonzero(same).flatten().tolist():
                        tr = d["traj"][j]
                        ax.plot(tr[:, dims[0]], tr[:, dims[1]], "-", lw=1.5,
                                color="#1a7a4a", zorder=3)
                    nom = small["nominal"][r].cpu()
                    ax.plot(nom[:, dims[0]], nom[:, dims[1]], "--",
                            color="#c0392b", lw=1.2, zorder=4)
                else:
                    xs = x.view(len(idx), -1, eq.L, 3)[r].cpu()
                    for j in range(xs.shape[0]):
                        ax.plot(xs[j, :, dims[0]], xs[j, :, dims[1]], "-",
                                lw=0.7, alpha=0.6, color="#2955a3", zorder=3)
                s, g = small["start"][r].cpu(), small["goal"][r].cpu()
                ax.plot(s[dims[0]], s[dims[1]], "ko", ms=4, zorder=5)
                ax.plot(g[dims[0]], g[dims[1]], "k*", ms=7, zorder=5)
                if ax_r == 0:
                    ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel(dlab, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/quad_qual.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_mnist(res):
    import torch as _t
    tss = sorted(int(k) for k in res["M2"])
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 2.9))
    ax = axes[0]
    for k, lbl, c in (("anchored", "anchored init (class mean)", "#c0392b"),
                      ("naive", "naive init ($m=0$)", "#8e44ad")):
        ax.plot(tss, [res["M2"][str(t)][k]["acc"] * 100 for t in tss],
                "o-", ms=3.5, color=c, label=lbl)
    ax.set_xlabel("truncation point $t_s$")
    ax.set_ylabel("Classifier accuracy (%)")
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_title("(a) classifier accuracy", fontsize=9.5)

    ax = axes[1]
    ax.plot(tss, [res["M2"][str(t)]["anchored"]["fd"] for t in tss],
            "o-", ms=3.5, color="#c0392b", label="class-Fréchet dist.")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(tss, [res["M2"][str(t)]["anchored"]["div"] for t in tss],
             "s--", ms=3, color="#2980b9", label="intra-class diversity")
    ax2.axhline(res["M1"]["base_nom"]["div_ref"], color="0.5", ls=":",
                lw=1.2, label="data diversity")
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel("truncation point $t_s$")
    ax.set_ylabel("Fréchet distance (log)")
    ax2.set_ylabel("Diversity")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="center right")
    ax.set_title("(b) distributional fidelity", fontsize=9.5)

    ax = axes[2]
    deep = _t.load("results/mnist_montage_deep.pt", weights_only=True)
    rule = _t.load("results/mnist_montage_rule.pt", weights_only=True)
    def grid(imgs, rows, cols):
        g = imgs[: rows * cols, 0].reshape(rows, cols, 32, 32)
        return g.permute(0, 2, 1, 3).reshape(rows * 32, cols * 32)
    top = grid(deep, 2, 8)
    bot = grid(rule, 2, 8)
    canvas = _t.cat([top, _t.full((4, 8 * 32), 1.0), bot], 0)
    ax.imshow(canvas, cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("$t_s{=}224$   $t_s{=}32$", fontsize=7)
    ax.set_title("(c) deep vs. rule truncation", fontsize=9.5)
    finish_figure(fig)
    fig.savefig("figures/mnist.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Regenerate cached pendulum, quadrotor, MPC, and MNIST assets.")
    parser.add_argument("--qualitative", action="store_true",
                        help="Also resample qualitative quadrotor trajectories; requires runs/ and data/.")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    with open("results/pend_results.json") as f:
        pres = json.load(f)
    fig_pend(pres)
    fig_value(pres)
    if os.path.exists("results/quad_results.json"):
        with open("results/quad_results.json") as f:
            qres = json.load(f)
        fig_quad(qres)
        if args.qualitative:
            fig_quad_qual(qres)
    with open("results/mpc_results.json") as f:
        res = json.load(f)
    fig_mpc(res)
    table_mpc(res)
    with open("results/mnist_results.json") as f:
        fig_mnist(json.load(f))
    print("ext figures written")


if __name__ == "__main__":
    main()
