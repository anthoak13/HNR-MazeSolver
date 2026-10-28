"""Tables and learning curves for the transfer experiments (written by scripts/transfer.py).

    python -m scripts.transfer_summary runs/local/t1/*.json --plot reports/figures/t1.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

THRESHOLDS = {"flag": (0.5, 0.8), "escape": (0.8, 0.9)}


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.array([y[max(0, i - k + 1):i + 1].mean() for i in range(len(y))])


def crossing(history, level):
    """Attempts until smoothed training success first reaches ``level`` (None if never).
    Log points before 1,000 attempts are skipped: at first only the quickest attempts (mostly
    successes) have finished, which inflates early success."""
    s = smooth(history["success"])
    hit = np.flatnonzero((s >= level) & (np.asarray(history["episodes"]) >= 1000))
    return None if len(hit) == 0 else int(history["episodes"][hit[0]])


def label(r):
    if r["start"] == "scratch":
        return "random perception (frozen)" if r["frozen"] else "scratch"
    kind = "pretrained" if "reward-free" in r["start_source"] else "from " + Path(r["start"]).stem
    return kind + (" (frozen)" if r["frozen"] else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    runs = [json.load(open(f)) for f in a.files]
    runs.sort(key=lambda r: (r["task"], label(r), r["seed"]))
    for r in runs:
        t = THRESHOLDS[r["task"]]
        h = r["history"]
        cross = [crossing(h, lv) for lv in t]
        tests = {k: v["reached flag"] for k, v in r["tests"].items()}
        moves = {k: v["avg moves"] for k, v in r["tests"].items()}
        early = smooth(h["success"])[np.searchsorted(h["episodes"], 2000)]
        print(f"{r['task']:6s} {label(r):28s} s{r['seed']} {r['minutes']:5.1f}min "
              f"early(2k) {early:.0%}  "
              + "  ".join(f"to{int(lv * 100)}% {c if c is not None else '-':>6}" for lv, c in zip(t, cross))
              + "  | " + "  ".join(f"{k[:8]} {v:.1%}/{moves[k]:.0f}mv" for k, v in tests.items()))
    if a.plot:
        plot(runs, a.plot)


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mazerl.viz import SERIES, _style
    tasks = sorted({r["task"] for r in runs}, reverse=True)
    labels = sorted({label(r) for r in runs}, key=lambda l: (l != "scratch", l))
    color = {l: SERIES[i % len(SERIES)] for i, l in enumerate(labels)}
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        seen = set()
        for r in runs:
            if r["task"] != task:
                continue
            l = label(r)
            h = r["history"]
            ax.plot(np.asarray(h["episodes"]) / 1000, smooth(h["success"]) * 100, color=color[l], lw=1.8,
                    ls="-" if r["seed"] == 0 else "--", label=l if l not in seen else None)
            seen.add(l)
        _style(ax, f"{task} task: success during training (%)")
        ax.set_xlabel("attempts (thousands)")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.5, -0.02, "solid = seed 0, dashed = seed 1", ha="center", fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
