"""Training curves (matplotlib). Maze animations live in mazerl.player."""
import numpy as np
import matplotlib.pyplot as plt

# Categorical colors in fixed order (validated reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

METRICS = [
    ("return", "Average total reward", 1),
    ("success", "Reached the flag (%)", 100),
    ("steps", "Moves per attempt", 1),
    ("new_cells", "New cells visited", 1),
    ("probes", "Wall probes per attempt", 1),
    ("breaks", "Walls broken per attempt", 1),
    ("outside", "Moves spent outside", 1),
    ("bypass", "Went around the maze (%)", 100),
]


def plot_training(runs, labels=None, smooth=10, metrics=None):
    """Small multiples of how training went, one panel per measurement.

    runs: a Run, or a list of Runs to compare (one color each).
    """
    runs = runs if isinstance(runs, (list, tuple)) else [runs]
    if labels is None:
        labels = [getattr(r, "label", None) or f"Run {i + 1}" for i, r in enumerate(runs)]
    chosen = [m for m in METRICS if metrics is None or m[0] in metrics]
    cols = 4 if len(chosen) > 4 else len(chosen)
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), squeeze=False)
    for ax, (key, title, scale) in zip(axes.flat, chosen):
        for i, run in enumerate(runs):
            h = run.history
            x = np.asarray(h["episodes"]) / 1000
            y = _smooth(np.asarray(h[key]) * scale, smooth)
            ax.plot(x, y, color=SERIES[i % len(SERIES)], lw=2, label=labels[i])
        _style(ax, title)
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    for ax in axes[-1]:
        if ax.has_data():
            ax.set_xlabel("Attempts (thousands)", color=INK_2, fontsize=9)
    if len(runs) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center",
                   ncol=min(len(runs), 4), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return fig


def _smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    head = c[1:k] / np.arange(1, k)
    return np.concatenate([head, (c[k:] - c[:-k]) / k])


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
