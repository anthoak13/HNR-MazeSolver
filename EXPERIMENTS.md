# Experiments

Lab notebook for choosing the values that aren't fixed by the article. Every round follows the same
three steps: write the prediction down, run the smallest set of runs that tests it, then compare.

- **Held fixed (the author's numbers):** flag +5, outside −0.01 per move.
- **Free to tune:** the wall penalty (`probe`), the new-cell reward (`new_cell`, plus `explore_outside`),
  and the training mix of maze kinds (`maze_mix`, as open : interior_break : outside_only).

## What the demo needs to show (pedagogical goals)
1. **G1, competence:** the agent reliably solves new mazes of the kinds it trained on.
2. **G2, emergence:** on the article's `enclosed` maze, never seen in training, the agent sometimes
   leaves, breaks back in, and reaches the flag. It should happen often enough to show in class (≳10%).
3. **G3, legible cause and effect:** changing a reward or the training mix changes behavior in a way
   students could predict and explain.

## Method
- Every run is full scale: 126k attempts on 13,148 layouts, 1 A30 GPU, about 5 minutes.
- Scoring uses the standard suite (`mazerl.rollout.standard_report`), 500 attempts per test:
  - new mazes of each kind (open, interior_break, outside_only);
  - `enclosed` (p.6) and `outside_shortcut` (p.7).
- Commands: `sbatch --array=1-N%2 scripts/sweep.sbatch sweeps/<round>.txt`. Results are
  `runs/<round>/*.json` on the cluster.

## Reference run: A (seed 0)
Settings: memory of probed walls, probe −0.01, new cell +0.005 (inside and outside), mix 3:2:2.

| open | interior_break | outside_only | enclosed | shortcut: went outside |
|---|---|---|---|---|
| 99.0% | 89.6% | 97.2% | 15.0% | 8% |

---

## Round 1: How noisy is a single run, and does the breakout need both component skills?

**Question 1: seed noise.** Re-run A's settings with seeds 1 and 2. Any effect smaller than the spread
across these three runs can't be trusted.
- *Prediction:* open 98–100%, interior_break 85–94%, outside_only 95–99%, enclosed **8–20%**.

**Question 2: emergence needs the parts.**
- *Hypothesis:* the article says the agent "combined these separately learned behaviors."
  - The breakout chains two skills: leaving the maze and travelling outside (learned from outside_only
    mazes), and breaking through walls (learned from interior_break mazes).
  - Remove either kind from training and the breakout should mostly disappear, even though nothing else
    changes.
- *Runs:*
  - `mix_no_outside_only` (1:1:0)
  - `mix_no_interior_break` (1:0:1)
- *Predictions:*
  - **No outside_only:**
    - outside_only success falls well below the baseline, under 60%.
    - On `enclosed`, the agent still steps out, because the start pocket only opens outward. But it
      rarely travels and breaks in: **enclosed < 5%**.
  - **No interior_break:**
    - interior_break success falls below 50%.
    - It probes less.
    - **enclosed < 3%**, since that route needs 4 breaks in a row.
  - Open and outside_only (or open alone) stay near the baseline.
- *Why it matters for teaching:* if this holds, it's a clean classroom experiment. "Remove one kind of
  training maze; predict what happens to the breakout."

**Results** (500 attempts per test):

| Run | Open | Interior break | Outside only | Enclosed | Walls broken on enclosed | Shortcut: went outside |
|---|---|---|---|---|---|---|
| A (seed 0) | 99.0% | 89.6% | 97.2% | 15.0% | 3.96 | 8% |
| base, seed 1 | 99.6% | 92.0% | 97.6% | 31.0% | 4.50 | 11% |
| base, seed 2 | 99.2% | 93.0% | 97.6% | 16.2% | 4.25 | 4% |
| no outside_only (1:1:0) | 99.6% | 96.8% | **14.0%** | **0.4%** | 1.75 | 1% |
| no interior_break (1:0:1) | 98.8% | **29.8%** | 97.8% | **0.0%** | 1.39 | 0% |

**Evaluation:**
- **Question 1: noise.** Partly as predicted. Open, interior_break and outside_only are stable across
  seeds, within about 3 points. But enclosed varies far more than predicted: **15–31%**, against the
  predicted 8–20%. Single-run differences on enclosed under about 15 points are noise. Any claim about
  the breakout needs several seeds.
- **Question 2: emergence needs both parts.** Confirmed, strongly. Both predictions held with room to
  spare:
  - **Without outside_only mazes:** the agent can't solve them (14%), and the breakout disappears (0.4%).
    On enclosed it probes 85 times and spends 108 moves outside, wandering and bumping, but it never
    learned to travel to a weak spot and get back in.
  - **Without interior_break mazes:** wall-breaking collapses (30%), and the breakout disappears (0.0%).
    It spends 136 moves outside on enclosed, probing little.
  - **For teaching:** the article's "combined these separately learned behaviors" can be shown directly.
    Take away either kind of training maze and the emergent behavior vanishes. This is a candidate
    in-class experiment.

---

## Round 2: Do the free rewards change behavior in ways students could predict?

Here the training mix stays at 3:2:2, and one reward changes per run. With 500 attempts per test, effects
on open, interior_break and outside_only are measurable to about ±3 points. Effects on enclosed need to be
large (about 20 points or more) to count.

**Wall penalty (`probe`; baseline −0.01).**
- *Hypothesis:* a probe almost never pays off on its own, because most walls are solid. So the wall
  penalty sets how willing the agent is to search walls. At −0.05, a typical 40-probe search costs −2,
  about as much as a distant flag is worth after discounting.
- *Predictions:*
  - **−0.05:** probes per attempt on interior_break at least halved (under 7); interior_break success
    **under 70%**; enclosed **under 5%**.
  - **−0.002:** more probing (at least 18 on interior_break); interior_break success stays about 90%
    (it's near the ceiling); enclosed at or above the baseline band.

**New-cell reward (`new_cell`; baseline +0.005).**
- *Hypothesis:* this is the article's "persistence" incentive. It makes searching pay for itself.
- *Predictions:*
  - **0:** the agent searches less thoroughly; enclosed, the longest search, falls **under 10%**; open and
    outside_only roughly unchanged.
  - **+0.02:** now larger than the outside penalty, so every new outside cell nets +0.01 and the outside
    band becomes worth farming. Moves outside on open mazes at least **3× the baseline** (over 15, vs. 5).
    Open success slightly lower. Enclosed **at or below** the baseline, because the agent harvests the
    outside instead of breaking in. If so, that's reward hacking, a candidate in-class experiment.

**Runs:** `wall_0.05`, `wall_0.002`, `new_0`, `new_0.02`, each at seed 0. Compare against A, seed 0.

**Status:** submitted.
