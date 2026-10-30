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
        even = ", even start" if r.get("policy_init_scale", 1.0) != 1.0 else ""
        return ("random perception (frozen)" if r["frozen"] else "scratch") + even
    kind = "pretrained" if "reward-free" in r["start_source"] else "perception from a flag-trained agent"
    even = ", even start" if r.get("policy_init_scale", 1.0) != 1.0 else ""
    return kind + (" (frozen)" if r["frozen"] else "") + even


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--only", default=None, help="comma-separated starting points to plot")
    ap.add_argument("--data", default=None, help="write a compact JSON of results (block curves, tests)")
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
        plot(runs, a.plot, only=a.only.split(",") if a.only else None)
    if a.data:
        compact = [{"name": r["name"], "task": r["task"], "seed": r["seed"], "start": label(r),
                    "episodes": r["episodes"],
                    "success_per_1000_attempts": [round(v, 1) for v in blocks(r["history"])[1]],
                    "tests": r["tests"]} for r in runs]
        Path(a.data).write_text(json.dumps(compact, indent=1))


def blocks(history, size=1000, total=None):
    """Average training success in consecutive blocks of ``size`` attempts (by attempts finished)."""
    e, s = np.asarray(history["episodes"]), np.asarray(history["success"])
    total = total or int(e[-1])
    mids, vals = [], []
    for a in range(0, total, size):
        m = (e > a) & (e <= a + size)
        mids.append((a + size / 2) / 1000)
        vals.append(s[m].mean() * 100 if m.any() else np.nan)
    return np.array(mids), np.array(vals)


SHOW = {"scratch": "from scratch", "pretrained": "pretrained perception",
        "pretrained (frozen)": "pretrained perception, locked",
        "random perception (frozen)": "random perception, locked"}


def plot(runs, path, panels=None, only=None):
    """One panel per task; one line per starting point (mean of seeds), band = range across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mazerl.viz import SERIES, _style
    tasks = panels or sorted({r["task"] for r in runs}, reverse=True)
    order = ["scratch", "pretrained", "perception from a flag-trained agent", "pretrained (frozen)",
             "random perception (frozen)"]
    runs = [r for r in runs if only is None or label(r) in only]
    labels = sorted({label(r) for r in runs}, key=lambda l: (order.index(l) if l in order else 9, l))
    color = {l: SERIES[order.index(l) if l in order else 5] for l in labels}
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.4 * len(tasks), 3.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for l in labels:
            group = [r for r in runs if r["task"] == task and label(r) == l]
            if not group:
                continue
            curves = [blocks(r["history"], total=r["episodes"]) for r in group]
            x = curves[0][0][1:]                       # skip the first 1,000 attempts (see crossing)
            ys = np.array([c[1][1:] for c in curves])
            ax.fill_between(x, np.nanmin(ys, 0), np.nanmax(ys, 0), color=color[l], alpha=0.18, lw=0)
            ax.plot(x, np.nanmean(ys, 0), color=color[l], lw=2, label=f"{SHOW.get(l, l)} ({len(group)} seeds)")
        _style(ax, f"{'Reach the flag' if task == 'flag' else 'Escape the maze'}: success during training (%)")
        ax.set_xlabel("attempts (thousands)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
