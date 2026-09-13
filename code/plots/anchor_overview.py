"""Generate the manuscript overview of coordinate and source anchors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "figures" / "anchor_overview.pdf"

BLUE = "#2F6B9A"
ORANGE = "#D77A28"
RED = "#C44E52"
DARK = "#30343B"
MID_GRAY = "#737B84"
LIGHT_GRAY = "#D9DEE4"
PANEL_BG = "#FAFBFC"


def covariance_ellipse(ax, mean, covariance, color, linestyle, label):
    """Draw a constant-Mahalanobis contour for a two-dimensional Gaussian."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    radius = 1.62
    ellipse = Ellipse(
        mean,
        2 * radius * np.sqrt(eigenvalues[0]),
        2 * radius * np.sqrt(eigenvalues[1]),
        angle=angle,
        fill=False,
        linewidth=2.0,
        edgecolor=color,
        linestyle=linestyle,
        label=label,
        zorder=5,
    )
    ax.add_patch(ellipse)


def clean_panel(ax):
    ax.set_facecolor(PANEL_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(LIGHT_GRAY)
        spine.set_linewidth(0.8)


def main():
    rng = np.random.default_rng(12)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.42),
        gridspec_kw={"width_ratios": [1.02, 1.02, 1.36]},
    )

    # (a) One condition can admit multiple outcomes around a simple reference.
    x = np.linspace(-1.72, 1.72, 240)
    phase = np.clip(np.sin(np.pi * (x + 1.72) / 3.44), 0.0, None)
    arch = 0.88 * phase**1.22
    ax = axes[0]
    ax.plot(x, np.zeros_like(x), color=MID_GRAY, linewidth=1.35, linestyle=(0, (4, 3)), zorder=1)
    obstacle = Circle(
        (0.0, 0.0),
        0.43,
        facecolor="#E1E5EA",
        edgecolor="#7B838C",
        linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(obstacle)
    ax.plot(x, arch, color=BLUE, linewidth=2.15, solid_capstyle="round", zorder=4)
    ax.plot(x, -arch, color=ORANGE, linewidth=2.15, solid_capstyle="round", zorder=4)
    # A small white halo keeps the two colored branches from forming a
    # visually merged arrowhead where they meet the endpoint nodes.
    ax.scatter([-1.72, 1.72], [0.0, 0.0], s=66, facecolor="white", edgecolor="none", zorder=5)
    ax.scatter([-1.72, 1.72], [0.0, 0.0], s=31, facecolor="white", edgecolor=DARK, linewidth=1.2, zorder=6)
    ax.text(-1.72, -0.17, "start", ha="center", va="top", color=DARK)
    ax.text(1.72, -0.17, "goal", ha="center", va="top", color=DARK)
    ax.text(0.0, 0.0, "obstacle", ha="center", va="center", fontsize=7.1, color="#555D65", zorder=7)
    ax.annotate(
        r"reference $m(c)$",
        xy=(-1.38, 0.0),
        xytext=(-1.20, 0.86),
        color=MID_GRAY,
        fontsize=7.3,
        arrowprops={"arrowstyle": "-", "color": MID_GRAY, "linewidth": 0.9},
        ha="center",
    )
    ax.text(0.62, 0.97, "valid modes", ha="center", color=DARK)
    ax.set_title("(a) Structured reference", pad=6)
    ax.set_xlim(-1.98, 1.98)
    ax.set_ylim(-1.17, 1.17)
    ax.set_aspect("equal", adjustable="box")
    clean_panel(ax)

    # (b) Subtracting and restoring the reference is a bijection.
    ax = axes[1]
    ax.axhline(0.0, color=MID_GRAY, linewidth=1.25, linestyle=(0, (4, 3)), zorder=1)
    ax.plot(x, arch, color=BLUE, linewidth=2.15, solid_capstyle="round")
    ax.plot(x, -arch, color=ORANGE, linewidth=2.15, solid_capstyle="round")
    ax.scatter([-1.72, 1.72], [0.0, 0.0], s=26, facecolor="white", edgecolor=DARK, linewidth=1.1, zorder=5)
    ax.text(
        0.0,
        1.12,
        r"$r_0=x_0-m(c)$",
        ha="center",
        va="center",
        color=DARK,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": LIGHT_GRAY, "linewidth": 0.8},
    )
    ax.text(0.0, 0.10, "zero reference", ha="center", va="bottom", color=MID_GRAY, fontsize=7.2)
    ax.text(0.0, -0.43, "both modes remain", ha="center", color=DARK, fontsize=7.3)
    ax.text(0.0, -1.18, r"restore: $x_0=r_0+m(c)$", ha="center", color=DARK, fontsize=7.3)
    ax.set_title("(b) Bijective coordinates", pad=6)
    ax.set_xlim(-1.98, 1.98)
    ax.set_ylim(-1.34, 1.34)
    ax.set_aspect("equal", adjustable="box")
    clean_panel(ax)

    # (c) A truncated sampler replaces the true noisy marginal by a source.
    # Repeat the same true marginal in two side-by-side comparisons rather
    # than overlaying both candidate sources in one crowded coordinate frame.
    direction = np.array([1.0, 0.82])
    direction /= np.linalg.norm(direction)
    orthogonal = np.array([-direction[1], direction[0]])
    signs = rng.choice([-1.0, 1.0], size=150)
    local_cloud = (
        signs[:, None] * 0.58 * direction
        + rng.normal(scale=0.13, size=(150, 1)) * direction
        + rng.normal(scale=0.075, size=(150, 1)) * orthogonal
    )
    covariance = np.cov(local_cloud, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    left_center = np.array([-1.12, -0.22])
    right_center = np.array([1.12, -0.22])

    ax = axes[2]
    ax.axvline(0.0, color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    for center in (left_center, right_center):
        shifted = local_cloud + center
        ax.scatter(
            shifted[:, 0],
            shifted[:, 1],
            s=6,
            color="#5F6871",
            alpha=0.24,
            linewidths=0,
            zorder=2,
        )
    covariance_ellipse(ax, left_center, diagonal, RED, "--", "diagonal source")
    covariance_ellipse(ax, right_center, covariance, BLUE, "-", "correlated source")
    ax.text(-1.12, 1.12, r"diagonal $g_s$", ha="center", va="center", color=RED, fontsize=7.4, weight="semibold")
    ax.text(1.12, 1.12, r"correlated $g_s$", ha="center", va="center", color=BLUE, fontsize=7.4, weight="semibold")
    ax.text(0.0, -1.34, r"gray points: same true noisy $q_s$", ha="center", va="center", color=MID_GRAY, fontsize=7.0)
    ax.set_title("(c) Truncated source", pad=6)
    ax.set_xlim(-2.18, 2.18)
    ax.set_ylim(-1.52, 1.52)
    ax.set_aspect("equal", adjustable="box")
    clean_panel(ax)

    fig.subplots_adjust(left=0.018, right=0.992, bottom=0.07, top=0.91, wspace=0.17)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
