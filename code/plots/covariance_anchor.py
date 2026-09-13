"""Plot and tabulate the Gaussian-source warm-start ablation."""

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
from nab.diffusion import score_evaluation_count
from nab.plot_style import finish_figure
METHODS = ("vp", "diagonal", "wsd", "lowrank", "full")
LABELS = {
    "vp": "VP isotropic",
    "diagonal": "pooled diagonal",
    "wsd": "WSD-style cond. diagonal",
    "lowrank": "rank-4",
    "full": "full",
}
COLORS = {
    "vp": "#7f7f7f",
    "diagonal": "#e68613",
    "wsd": "#8c56a8",
    "lowrank": "#2878b5",
    "full": "#2a9d55",
}
MARKERS = {"vp": "o", "diagonal": "s", "wsd": "P", "lowrank": "^", "full": "D"}


def metric_series(data, method, metric):
    ts = sorted(int(t) for t in data["reverse_sampling"])
    means = [
        data["reverse_sampling"][str(t)][method]["summary"][metric]["mean"]
        for t in ts
    ]
    stds = [
        data["reverse_sampling"][str(t)][method]["summary"][metric]["std"]
        for t in ts
    ]
    return np.asarray(ts), np.asarray(means), np.asarray(stds)


def write_table(data, path: Path, selected_t: int):
    actual_nfe = score_evaluation_count(data["metadata"]["nfe"], selected_t)
    rows = []
    block = data["reverse_sampling"][str(selected_t)]
    ce = data["source_cross_entropy"][str(selected_t)]
    for method in METHODS:
        summary = block[method]["summary"]
        fmt = lambda key, scale=1.0, digits=1: (
            f"{scale * summary[key]['mean']:.{digits}f} $\\pm$ "
            f"{scale * summary[key]['std']:.{digits}f}"
        )
        rows.append(
            " & ".join(
                [
                    LABELS[method],
                    f"{ce[method]['excess_vs_full_per_dim']:.3f}",
                    fmt("feasible", 100.0),
                    fmt("subopt_med", digits=2),
                    fmt("mode_recall_mm", 100.0),
                    fmt("diversity", digits=2),
                ]
            )
            + r" \\"
        )
    text = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            (
                r"\caption{Gaussian-source ablation at $t_s="
                + str(selected_t)
                + r"$ and NFE $" + str(actual_nfe) + r"$ (mean $\pm$ standard deviation over three "
                r"sampling seeds).  Source NLL is held-out excess Gaussian "
                r"cross-entropy per trajectory coordinate relative to the full "
                r"pooled covariance.  The WSD-style source predicts a conditional mean and diagonal "
                r"variance; the other learned moments are pooled.  Reverse metrics "
                r"use the same pretrained VP score model.  $\Delta J$ is the "
                r"dimensionless median normalized gap $(J-J^*)/\overline{J^*}$ over "
                r"feasible samples.  It uses the multistart-oracle reference "
                r"in \texttt{artifacts/data/test\_ref.pt}, distinct from the "
                r"post hoc reference in Table~\ref{tab:main}; gap magnitudes "
                r"are not directly comparable across these studies.}"
            ),
            r"\label{tab:covariance}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Source & excess NLL/dim & Feas. (\%) & $\Delta J$ & Recall-MM (\%) & Diversity \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "covariance_ablation.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "figures" / "covariance_ablation.pdf",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=PROJECT_ROOT / "tables" / "covariance.tex",
    )
    parser.add_argument("--table-t", type=int, default=123)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    ts = sorted(int(t) for t in data["source_cross_entropy"])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    ax = axes[0]
    for method in METHODS:
        gap = [
            data["source_cross_entropy"][str(t)][method]["excess_vs_full_per_dim"]
            for t in ts
        ]
        ax.plot(ts, gap, marker=MARKERS[method], color=COLORS[method], label=LABELS[method])
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xlabel(r"truncation point $t_s$")
    ax.set_ylabel("held-out excess NLL / dim")
    ax.set_title("(a) source approximation")
    ax.grid(alpha=0.25)

    for ax, metric, title, ylabel, scale in [
        (axes[1], "feasible", "(b) reverse feasibility", "feasibility (%)", 100.0),
        (axes[2], "mode_recall_mm", "(c) distributional recall", "multimodal recall (%)", 100.0),
    ]:
        for method in METHODS:
            x, mean, std = metric_series(data, method, metric)
            ax.errorbar(
                x,
                scale * mean,
                yerr=scale * std,
                marker=MARKERS[method],
                color=COLORS[method],
                capsize=2,
                label=LABELS[method],
            )
        ax.set_xlabel(r"truncation point $t_s$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    finish_figure(fig, shared_legend=True, legend_columns=3)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    plt.close(fig)
    write_table(data, args.table, args.table_t)
    print(f"wrote {args.figure}")
    print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
