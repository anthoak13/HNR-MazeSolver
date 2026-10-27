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

**Results** (seed 0; A re-scored on the same suite):

| Run | Open | Interior break | Outside only | Enclosed | Walls broken, enclosed | Shortcut: went outside | Probes, interior_break | Moves outside, open mazes |
|---|---|---|---|---|---|---|---|---|
| A | 99.0% | 90.8% | 97.8% | 17.4% | 4.08 | 8% | 14.4 | 4.7 |
| wall −0.05 | 98.6% | 88.6% | 97.6% | 8.6% | 2.62 | 2% | 9.5 | 4.9 |
| wall −0.002 | 98.6% | 91.2% | 97.0% | 31.0% | 4.44 | 18% | 16.2 | 4.9 |
| new cell 0 | 98.4% | 89.8% | 97.6% | 16.4% | 4.61 | **38%** | 14.4 | 4.7 |
| new cell +0.02 | 99.6% | 91.4% | 97.4% | 5.4% | 3.98 | 22% | 12.4 | **4.9** |

**Evaluation against the predictions:**

- **Wall −0.05: half right.**
  - Probing fell by a third (14.4 → 9.5), short of the predicted halving.
  - **Interior_break success barely moved** (88.6%; predicted under 70%). With memory of probed walls,
    the agent probes in the right places, so a costlier probe makes it economize rather than give up.
  - The breakout fell (8.6%, below the seed band of 16–31%), but not as low as the predicted under 5%.
    It broke only 2.6 walls on enclosed and spent 109 moves wandering outside.
  - **Lesson:** under this cost, the agent still does routine wall-breaking but is less willing to do
    the long, expensive search.
- **Wall −0.002: as predicted** (31% on enclosed, at the top of the seed band). Not a large enough
  effect to show in class.
- **New cell 0: prediction failed.** Enclosed held at 16.4%, inside the seed band. Persistence did not
  depend on the exploration reward. **Surprise:** the agent took the outside shortcut on p.7 in **38%**
  of attempts (seed band 4–11%).
- **New cell +0.02: farming prediction failed.** Moves outside on open mazes didn't change (4.9). But
  the breakout dropped (5.4%), as predicted.
  - **Why no farming:** the flag still dominates. The outside band is worth at most about +1.3 to farm,
    while the flag is worth about 3 at typical distances.
  - **There's also a second barrier:** attempts that end outside are excluded from training, so the agent
    can never be rewarded for farming and then staying out.

---

## Round 3: Two effects worth teaching, made robust

**H5: Exploration rewards decide between the inside route and the outside shortcut (p.7).**
- *Hypothesis:* the p.7 behavior, "taking the outside path although an inside path is available," comes
  from simple arithmetic.
  - The outside route is 2 moves shorter, which raises the discounted flag by about 5 × (1 − 0.99²) ≈
    +0.10, at a cost of 6 outside moves × −0.01 = −0.06. With no exploration reward, outside wins by
    about +0.04.
  - With new cell +0.005, the inside route collects 27 new inside cells (+0.135). The outside route
    collects 25 new cells (+0.125) minus the outside cost, so inside wins by about +0.03.
  - So removing the exploration reward should flip the preference toward the outside.
- *Test:* `new_0`, seeds 1 and 2.
- *Prediction:* shortcut went outside **at least 25%** in both, against the baseline band of 4–11%.

**H6: Reward hacking needs an exploration reward large enough to rival the flag, and the bypass filter
blocks it.**
- *Hypothesis:* farming starts to pay when the total exploration reward approaches the discounted flag.
  At new cell +0.05, the outside band alone is worth 128 × 0.04 ≈ +5.1, and the interior about +5.
  Without the bypass filter, the agent should learn to farm and stay out.
- *Test:* new cell +0.05, filter on and off.
- *Predictions:*
  - **Filter off:** moves outside on open mazes **over 30**; open success **under 85%**.
  - **Filter on:** farming is partly blocked. Moves outside on open mazes rise (**over 10**), but open
    success stays **above 90%**. The agent may learn to farm and then come back in before time runs out,
    which is itself a lesson about gaming a monitor.
- *Why it matters for teaching:* a reward-hacking demo, plus the role of monitoring. That's the article's
  point about "monitoring for unwanted actions."

**Runs:** `new_0_s1`, `new_0_s2`, `new_0.05`, `new_0.05_nofilter`.

**Results:**

| Run | Open | Interior break | Outside only | Enclosed | Shortcut: went outside | Moves outside, open mazes | Score, open mazes |
|---|---|---|---|---|---|---|---|
| new cell 0, seed 0 | 98.4% | 89.8% | 97.6% | 16.4% | **38%** | 4.7 | 4.80 |
| new cell 0, seed 1 | 98.8% | 92.2% | 98.4% | 34.4% | **49%** | 5.3 | 4.81 |
| new cell 0, seed 2 | 99.0% | 93.0% | 96.4% | 31.4% | **30%** | 5.3 | 4.82 |
| new cell +0.05, filter on | 97.8% | 77.8% | 95.0% | 2.0% | 97% | **36.0** | 7.08 |
| new cell +0.05, filter off | 99.4% | 89.0% | 96.4% | 0.0% | 100% | **37.2** | 7.14 |

**Evaluation:**
- **H5: confirmed.** With no exploration reward, the agent takes the p.7 outside shortcut in 30–49% of
  attempts across three seeds, against 4–11% at baseline. Its breakout rate is unaffected (16–34%, the
  baseline band).
  - My back-of-envelope arithmetic for *why* doesn't hold up when redone carefully. The cell counts and
    discounting are too crude to predict the crossover. The effect itself is robust.
- **H6: half right.**
  - The exploration reward of +0.05 produces clear **reward hacking.** The agent tours the outside, about
    36 moves, about 8× the baseline, harvesting exploration points, and only then goes for the flag. Its
    score rises from 4.8 to 7.1.
  - As a side effect, the breakout collapses (0–2%), and interior_break success falls to 78%.
  - **But the filter doesn't matter**, contrary to my prediction. The hacking agent still finishes inside
    the maze, so it never trips a rule that only catches attempts ending outside.
  - For teaching, that's arguably the better lesson: *a monitor that checks only the outcome misses an
    agent that games the process.*
- **The exploration reward has a non-monotonic effect on the p.7 behavior** (seed 0):

  | New cell | 0 | 0.005 | 0.02 | 0.05 |
  |---|---|---|---|---|
  | Shortcut: went outside | 38% | 8% | 22% | 97% |

  The two ends are different mechanisms: taking the outside because it's shorter, vs. farming the
  outside.

---

## Round 4: Choosing the defaults

The article says there *was* a reward for exploring [A], so the base agent can't use new cell 0. The
figures don't imply a maze mix, so the default should be the simplest choice, equal shares (1:1:1),
unless that hurts.

**H7: a small exploration reward keeps both of the article's behaviors.**
- *Hypothesis:* new cell +0.002 sits between 0 (strong shortcut-taking) and 0.005 (little).
- *Predictions:*
  - Shortcut went outside **15–35%**.
  - Breakout in the baseline band (≥ 15%).
  - Open, interior_break and outside_only within 3 points of baseline.
- *Runs:* `new_0.002`, seeds 0 and 1.

**H8: an equal mix (1:1:1) behaves like 3:2:2.**
- *Hypothesis:* a slightly larger share of open mazes isn't what drives anything. Each skill just needs
  enough examples.
- *Predictions:* every test within the seed noise of the baseline: open, interior_break and outside_only
  within 3 points; breakout in 15–31%.
- *Runs:* `mix_1-1-1`, seeds 0 and 1.

**Status:** submitted.

---

## Round 5: Can students change a trained agent's behavior in class time?

In class, students won't train from scratch (126k attempts). They'll fine-tune the shipped base agent
with their own rewards for a few minutes. So the useful questions are whether fine-tuning reproduces
the behavior shifts we saw from scratch, and how many attempts that takes.

**H9: fine-tuning reproduces the from-scratch shifts within a class-sized budget.**
- *Starting point:* agent A (seed 0, current defaults), for every run.
- *Runs and predictions:*
  - **Control** (the same rewards, 20k attempts): **no change**, within seed noise on every test. If
    behavior gets worse, the fine-tuning settings themselves are disruptive. The suspect would be the
    exploration bonus restarting at 0.05.
  - **Hacking** (new cell +0.05): moves outside on open mazes rise from 4.7 to **over 10 at 5k** and
    **over 20 at 20k** attempts. From scratch it was 36.
  - **Shortcut** (new cell 0): shortcut went outside rises from 8% to **10–25% at 5k** and **≥ 25% at
    20k**. From scratch it was 30–49%.
- *Time budget:* these runs go on the cluster's **CPU** partition (`cs`) with 2 threads each, a stand-in
  for a Colab CPU runtime. The run time itself is the measurement.
  - *Prediction:* a fine-tuned agent's attempts are short (about 45–80 moves), so at about 8–10k moves/s
    on 2 threads, 5k attempts should take **about 1 minute** and 20k **about 3–5 minutes**.
  - That's comfortably class-sized. If the predictions hold, students need about 20k attempts for a
    clear effect.

**Status:** submitted (CPU partition, `scripts/sweep_cpu.sbatch`).
