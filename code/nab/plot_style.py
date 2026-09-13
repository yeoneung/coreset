"""Readable paper-size labels and legends without changing plotted data."""

import math

from matplotlib.text import Text


def finish_figure(fig, *, shared_legend=False, legend_columns=3):
    """Put legends below the axes while preserving the plotting height."""
    font_size = 9.5
    for text in fig.findobj(Text):
        if text.get_text():
            text.set_fontsize(max(font_size, text.get_fontsize()))

    legends = []
    if shared_legend:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        legends.append((None, handles, labels, legend_columns))
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    else:
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                # Preserve combined legends from paired left/right axes.
                handles = legend.legend_handles
                labels = [text.get_text() for text in legend.get_texts()]
                legends.append((ax, handles, labels, legend._ncols))
                legend.remove()

    rows = max((math.ceil(len(labels) / columns)
                for _, _, labels, columns in legends), default=0)
    width, height = fig.get_size_inches()
    space = 0.20 * rows + 0.18 if rows else 0.0
    fig.set_size_inches(width, height + space)
    fig.tight_layout(rect=(0, space / (height + space), 1, 1))
    for ax, handles, labels, columns in legends:
        center = 0.5 if ax is None else (ax.get_position().x0 + ax.get_position().x1) / 2
        fig.legend(
            handles, labels, loc="upper center", ncol=columns,
            bbox_to_anchor=(center, (space - 0.04) / (height + space)),
            fontsize=font_size, frameon=False, borderaxespad=0,
            handlelength=2.0, columnspacing=1.6, labelspacing=0.4,
        )
