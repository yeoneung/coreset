"""Plot the condition-level certificate calibration."""

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
COLORS = {"64": "#b2432f", "123": "#e68613", "192": "#2878b5", "234": "#2a9d55"}
MARKERS = {"64": "o", "123": "s", "192": "^", "234": "D"}


def binned(values, losses, n_bins=5):
    """Equal-count bins of the certificate; mean and standard error per bin."""
    order = np.argsort(values)
    splits = np.array_split(order, n_bins)
    centers, means, errors = [], [], []
    for idx in splits:
        centers.append(values[idx].mean())
        means.append(losses[idx].mean())
        errors.append(losses[idx].std() / np.sqrt(max(len(idx), 1)))
    return np.array(centers), np.array(means), np.array(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "results" / "calibration.json",
    )
    parser.add_argument(
        "--figure", type=Path,
        default=PROJECT_ROOT / "figures" / "calibration.pdf",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))

    n_modes = np.array(data["per_condition"]["n_modes"])
    reference = np.array(data["per_condition"]["recall_reference_mean"])
    multimodal = n_modes > 1
    t_keys = [str(t) for t in data["metadata"]["t_starts"]]
    ts = np.array([int(k) for k in t_keys])

    def warm_mean(key, source):
        return np.array(
            data["per_condition"]["warm"][key][source]["recall_mean"]
        )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))

    # (a) isotropic source: recall loss against its trace certificate
    ax = axes[0]
    for key in t_keys:
        loss = (reference - warm_mean(key, "isotropic"))[multimodal]
        cert = np.array(data["certificates"][key]["trace"])[multimodal]
        centers, means, errors = binned(cert, loss)
        rho = data["correlations"][key]["isotropic"]["spearman_trace_mm"]
        ax.errorbar(centers, means, yerr=errors, marker=MARKERS[key],
                    color=COLORS[key], capsize=2,
                    label=rf"$t_s={key}$ ($\rho_S={rho:.2f}$)")
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"trace certificate $T_s(c)$ (nats)")
    ax.set_ylabel("recall loss vs full budget")
    ax.set_title("(a) isotropic source vs $T_s(c)$")
    ax.legend(fontsize=6.4)
    ax.grid(alpha=0.25)

    # (b) isotropic recall against truncation point by certificate quartile
    ax = axes[1]
    anchor_key = t_keys[0]
    cert = np.array(data["certificates"][anchor_key]["trace"])
    quartiles = np.quantile(cert[multimodal], [0.25, 0.5, 0.75])
    group_ids = np.digitize(cert, quartiles)
    palette = ["#2a9d55", "#2878b5", "#e68613", "#b2432f"]
    for g in range(4):
        members = multimodal & (group_ids == g)
        series = [warm_mean(k, "isotropic")[members].mean() for k in t_keys]
        ax.plot(ts, 100 * np.array(series), marker="o", color=palette[g],
                label=f"$T$ quartile {g + 1}")
    ax.plot(ts, [100 * reference[multimodal].mean()] * len(ts), ls="--",
            color="black", lw=0.9, label="full budget")
    ax.set_xlabel(r"truncation point $t_s$")
    ax.set_ylabel("multimodal recall (%)")
    ax.set_title("(b) onset by certificate quartile")
    ax.legend(fontsize=6.4)
    ax.grid(alpha=0.25)

    # (c) attribution: isotropic vs per-condition covariance-matched source
    ax = axes[2]
    for source, color, marker, label in [
        ("isotropic", "#b2432f", "o", "isotropic $(1-a)I$"),
        ("matched", "#2878b5", "s", r"matched $(1-a)I+aM_c$"),
    ]:
        series = [warm_mean(k, source)[multimodal].mean() for k in t_keys]
        ax.plot(ts, 100 * np.array(series), marker=marker, color=color,
                label=label)
    ax.plot(ts, [100 * reference[multimodal].mean()] * len(ts), ls="--",
            color="black", lw=0.9, label="full budget")
    ax.set_xlabel(r"truncation point $t_s$")
    ax.set_ylabel("multimodal recall (%)")
    ax.set_title("(c) covariance restoration")
    ax.legend(fontsize=6.4)
    ax.grid(alpha=0.25)

    finish_figure(fig)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
