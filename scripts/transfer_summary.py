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


def blocks(history, size=1000, total=None):
    """Average training success in consecutive blocks of ``size`` attempts (by attempts finished)."""
    e, s = np.asarray(history["episodes"]), np.asarray(history["success"])
    total = total or int(e[-1])
    mids, vals = [], []
    for a in range(0, total, size):
        m = (e > a) & (e <= a + size)
        if m.any():
            mids.append((a + size / 2) / 1000)
            vals.append(s[m].mean() * 100)
    return np.array(mids), np.array(vals)


def plot(runs, path, panels=None):
    """One panel per task; one line per starting point (mean of seeds), band = range across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mazerl.viz import SERIES, _style
    tasks = panels or sorted({r["task"] for r in runs}, reverse=True)
    order = ["scratch", "pretrained", "pretrained (frozen)", "random perception (frozen)"]
    labels = sorted({label(r) for r in runs}, key=lambda l: (order.index(l) if l in order else 9, l))
    color = {l: SERIES[i % len(SERIES)] for i, l in enumerate(labels)}
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.4 * len(tasks), 3.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for l in labels:
            group = [r for r in runs if r["task"] == task and label(r) == l]
            if not group:
                continue
            curves = [blocks(r["history"]) for r in group]
            n = min(len(c[0]) for c in curves)
            x = curves[0][0][:n]
            ys = np.array([c[1][:n] for c in curves])
            ax.fill_between(x, ys.min(0), ys.max(0), color=color[l], alpha=0.18, lw=0)
            ax.plot(x, ys.mean(0), color=color[l], lw=2, label=f"{l} ({len(group)} seeds)")
        _style(ax, f"{task} task: success during training (%)")
        ax.set_xlabel("attempts (thousands)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
