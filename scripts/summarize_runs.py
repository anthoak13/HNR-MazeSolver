"""Tabulate experiment results (the .json files written by pretrain.py --evaluate).

Columns: success on new open / interior_break / outside_only mazes; "impossible" = breaks out to finish
an impossible task (reached the flag on the p.6 enclosed maze) and walls broken there; "needless out" =
breaks out when it doesn't need to (left the maze on the p.7 maze, which has an inside route); "detour" =
time to finish: moves to the flag per move of the shortest route, averaged over the three trained kinds
(solved attempts only; needs a suite scored after it was added, see rescore.py).

    python scripts/summarize_runs.py runs/r1/*.json
"""
import json
import sys

COLUMNS = [("open", "reached flag"), ("interior_break", "reached flag"), ("outside_only", "reached flag"),
           ("enclosed", "reached flag"), ("enclosed", "avg walls broken"),
           ("outside_shortcut", "went outside")]
HEAD = ["open", "interior", "outside_only", "detour", "impossible", "impos. breaks", "needless out"]
TRAINED = ("open", "interior_break", "outside_only")


def main(paths):
    rows = [json.load(open(p)) for p in paths]
    print(f"{'run':24s}" + "".join(f"{h:>14s}" for h in HEAD))
    for r in rows:
        cells = []
        for test, key in COLUMNS:
            v = r["tests"][test][key]
            cells.append(f"{v:>14.2f}" if key.startswith("avg") else f"{v:>13.1%} ")
            if test == "outside_only":
                d = [r["tests"][k].get("detour") for k in TRAINED]
                cells.append(f"{sum(d) / 3:>14.2f}" if None not in d else f"{'-':>14s}")
        print(f"{r['name']:24s}" + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
