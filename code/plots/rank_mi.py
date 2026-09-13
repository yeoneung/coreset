"""Plot the controlled effective-rank certificate study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from nab.plot_style import finish_figure
RANK_COLORS = plt.cm.viridis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "results" / "rank_mi.json",
    )
    parser.add_argument(
        "--figure", type=Path,
        default=PROJECT_ROOT / "figures" / "rank_mi.pdf",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    ranks = data["metadata"]["ranks"]
    T = data["metadata"]["T"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))

    # (a) actual matched-source KL: single-axis control and product family
    ax = axes[0]
    comparison = data["rank_comparison"]
    ax.plot(
        ranks,
        comparison["single_axis_family"]["kl_by_rank"],
        "o-",
        color="#555555",
        ms=3.5,
        label="one mixture axis (control)",
    )
    theta_results = comparison["product_family"]["theta_results"]
    colors = ["#6a9f58", "#d88324", "#b2432f"]
    for color, (theta, values) in zip(colors, theta_results.items()):
        ax.plot(
            ranks,
            values["kl_by_rank"],
            "s-",
            color=color,
            ms=3.2,
            label=rf"product, $\theta={float(theta):g}$",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(k) for k in ranks])
    ax.set_xlabel(r"active rank $k$")
    ax.set_ylabel(r"actual $\mathrm{KL}(q_s\Vert g_s)$")
    ax.set_title(rf"(a) actual source KL, $a={comparison['signal_level']:g}$")
    ax.legend(fontsize=6.1)
    ax.grid(alpha=0.25)

    # (b) certificate curves across latent ranks
    ax = axes[1]
    for i, k in enumerate(ranks):
        cert = np.array(data["results"][str(k)]["certificate"])
        color = RANK_COLORS(i / max(len(ranks) - 1, 1) * 0.85)
        ax.plot(np.arange(T), cert, color=color, label=f"$k={k}$")
    ax.axhline(0.2, color="black", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    ax.set_xlabel(r"truncation point $t_s$")
    ax.set_ylabel(r"$B_s$ (nats)")
    ax.set_title("(b) sufficient certificate vs rank")
    ax.legend(fontsize=6.4, ncol=2)
    ax.grid(alpha=0.25)

    # (c) deepest admissible start under log-det and trace rules
    ax = axes[2]
    eps_keys = [str(e) for e in data["metadata"]["epsilons"]]
    styles = {"logdet": "-o", "trace": "--s"}
    palette = ["#2a9d55", "#e68613", "#b2432f"]
    for j, eps in enumerate(eps_keys):
        for rule in ("logdet", "trace"):
            values = [
                data["results"][str(k)]["admissible"][eps][rule] for k in ranks
            ]
            label = rf"$\epsilon={eps}$ ({rule})"
            ax.plot(ranks, values, styles[rule], ms=3.5, color=palette[j],
                    label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(k) for k in ranks])
    ax.set_xlabel(r"latent rank $k$ (ambient $d=64$)")
    ax.set_ylabel(r"earliest approved $t_s$")
    ax.set_title("(c) certificate-approved entry")
    ax.legend(fontsize=6.0)
    ax.grid(alpha=0.25)

    finish_figure(fig)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
