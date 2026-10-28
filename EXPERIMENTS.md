# Experiments

Lab notebook for choosing the values that aren't fixed by the article. Every round follows the same
three steps: write the prediction down, run the smallest set of runs that tests it, then compare.

- **Held fixed (the author's numbers):** flag +5, outside −0.01 per move.
- **Free to tune:** the wall penalty (`probe`), the new-cell reward (`new_cell`, plus `explore_outside`),
  and the training mix of maze kinds (`maze_mix`, as open : interior_break : outside_only).

**Names used from here on** (see `TEACHING.md`):
- **Example 1, "breaks out to finish an impossible task":** success on the p.6 `enclosed` maze. Earlier
  rounds call it "enclosed" or "the breakout."
- **Example 2, "breaks out when it doesn't need to":** the share of attempts that leave the maze on the
  p.7 `outside_shortcut` maze. Earlier rounds call it "shortcut: went outside."
- Round 8's "chain depth" counted interior break-ins. The number of breaks doesn't matter, so it's been
  replaced by three stages: left the maze, broke in from outside, reached the flag (`breakout_stages`).

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

**Results:**

| Run | Open | Interior break | Outside only | Enclosed | Shortcut: went outside |
|---|---|---|---|---|---|
| baseline 3:2:2, new cell 0.005 (seeds 0/1/2) | 99.0–99.6% | 90.8–93.0% | ~97.7% | 16–31% | 4–11% |
| new cell 0.002, seed 0 | 98.8% | 91.0% | 99.2% | 19.2% | 12% |
| new cell 0.002, seed 1 | 98.8% | 91.8% | 97.0% | 31.4% | 4% |
| **mix 1:1:1, seed 0** | 99.4% | 94.0% | 97.6% | **35.8%** | **27%** |
| **mix 1:1:1, seed 1** | 99.6% | 90.8% | 98.2% | **74.2%** | **46%** |

**Evaluation:**
- **H7: failed.** New cell 0.002 behaves like 0.005: shortcut 4–12%, breakout in the baseline band. The
  shortcut effect needs the exploration reward at zero, not merely small.
- **H8: failed, in a useful direction.** The equal mix is *not* like 3:2:2. It raises the breakout from
  16–31% to **36–74%**, and the p.7 outside shortcut from 4–11% to **27–46%**. Open, interior_break and
  outside_only are unchanged.
  - **Likely reason:** fewer open mazes (1/3 vs 3/7) means more practice at the two component skills.
    That fits round 1: removing either component kills the breakout. More of both strengthens it.
- **Decision:** the default mix becomes **1:1:1**. It's the simplest choice (no extrapolation from the
  figures), and it shows both of the article's behaviors at a rate visible in class. The exploration
  reward stays at **+0.005**: the article says there was one, and 0.002 gains nothing.
- **Seed spread on enclosed** under 1:1:1 is very large (36% vs 74%). The shipped `base` should be
  chosen from several seeds and reported with that spread.

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

**Results** (cluster CPU, 2 cores of an AMD EPYC 7502 per run):

| Fine-tune from A | Open | Interior break | Enclosed | Shortcut: went outside | Moves outside, open mazes | Time |
|---|---|---|---|---|---|---|
| A, the starting agent | 99.0% | 90.8% | 17.4% | 8% | 4.7 | — |
| control, same rewards, 20k | 98.6% | 90.2% | 11.2% | **24%** | 5.9 | 12.1 min |
| new cell +0.05, 5k | 98.6% | 90.0% | 13.2% | 13% | 5.9 | 2.8 min |
| new cell +0.05, 20k | 98.2% | 81.0% | 0.4% | 4% | 8.7 | 11.3 min |
| new cell 0, 5k | 98.0% | 91.0% | 20.6% | 13% | 5.8 | 3.0 min |
| new cell 0, 20k | 99.0% | 88.2% | 21.2% | 18% | 5.7 | 11.6 min |

For comparison, training from scratch at +0.05 shows the hacking (61 moves outside per attempt) by
10k attempts.

**Evaluation: H9 failed.**
- **The control isn't stable.** Fine-tuning with unchanged rewards shifts the shortcut rate from 8% to
  24%, and the breakout from 17% to 11%. The fine-tuning process itself perturbs the agent, probably
  because the exploration bonus restarts high.
- **The reward effects are small and slow.** At +0.05, outside moves rise from 4.7 to 8.7 after 20k
  attempts, against 36 from scratch. At new cell 0, the shortcut rate rises to 18%, which is *below* the
  control.
- **Why:** a converged agent has stopped exploring, so it rarely visits the states where the new reward
  would change its choices. From scratch, the agent explores anyway and finds the new incentive at once.
- **Time:** 20k attempts take about 12 minutes on 2 CPU cores. That's already the upper end of a class
  budget, and it buys little.

**Conclusion:** fine-tuning the full base agent is not a good classroom tool. That motivates round 6:
pretrain only perception, with no rewards, then let students' RL build the behavior on top.

---

## Round 6: RL on top of reward-free pretraining

This mirrors the modern LLM recipe. Pretrain a model on prediction tasks with no rewards, then shape its
behavior with RL.

- **What gets pretrained:** the perceiving part of the network, the vision layers plus the first
  reasoning layer (about 220k of the 230k connections).
- **Pretraining tasks** (no rewards), on agents exploring the training layouts:
  1. **Map completion:** predict the true contents of the cells in the 11 × 11 window that the agent
     hasn't seen yet (open, outside, wall, or edge; weak spots count as wall, so no hidden information
     leaks).
  2. **Frontier direction:** which move heads toward the nearest reachable cell it hasn't visited yet.
- **What RL trains:** the deciding part (second reasoning layer, move and value outputs, about 10k
  connections), starting fresh. The pretrained part is either frozen or allowed to keep learning.

**H10: with pretrained perception, RL reaches competence in a class-sized budget.**
- **Runs**, on 2 CPU threads locally (desktop i5-13400F), 20k attempts each:
  - from scratch;
  - pretrained perception, frozen;
  - pretrained perception, still learning.
- **Predictions:**
  - Open-maze success at 20k attempts: from scratch **65–75%** (A's curve passed 69% at 20k); both
    pretrained runs **at least 85%**.
  - Pretraining itself takes **under 10 minutes** on 2 threads.
  - The breakout needs more skill than 20k attempts provide from scratch, so it will be **under 5%** from
    scratch; with pretraining **at least 10%**.

**Results** (desktop i5-13400F, 2 threads per run, three runs side by side; maze mix 1:1:1):

Pretraining took **2.8 min**. On mazes it never trained on, map completion was **86%** accurate on unseen
cells (53% by always guessing the most common cell), and frontier direction **99.2%** (25% by chance).

| Setup (20k attempts) | Time to 50% success | Time to 80% success | Total time | Open | Interior break | Outside only | Enclosed | Shortcut: went outside |
|---|---|---|---|---|---|---|---|---|
| from scratch | 5.6 min | 11.7 min | 13.0 min | 92.6% | 68.2% | 72.0% | 5.2% | 52% |
| pretrained, frozen | 2.3 min | never | 8.8 min | 78.2% | 32.6% | 42.6% | 0.6% | 45% |
| pretrained, still learning | 4.8 min | 8.7 min | 11.8 min | 94.0% | 58.4% | 87.4% | 10.2% | 30% |

The time-to-success columns use training success, averaged over 5 log points.

**Evaluation against the predictions:**

| Prediction | Result |
|---|---|
| From scratch at 20k: open 65–75% | **92.6%.** The 20k schedule finishes annealing within 20k attempts, so a short run does better than the same point on a 126k run's curve |
| Pretrained runs: open ≥ 85% | still learning ✓ (94%); frozen ✗ (78%) |
| Breakout: from scratch < 5%, pretrained ≥ 10% | from scratch 5.2% (just over); still learning ✓ (10.2%); frozen ✗ (0.6%) |
| Pretraining under 10 min | ✓ (2.8 min) |

**Conclusions:**
- **Freezing the pretrained part doesn't work.** The roughly 10k deciding parameters alone can't learn
  the task.
- **Pretraining that keeps learning gives a small speed-up:** 80% success about 3 min sooner. That's
  roughly offset by the 2.8 min the pretraining itself takes, unless the pretrained model is shipped.
- **On 2 CPU threads, nothing reaches good behavior within 3–5 min.** That makes CPU-only class use a
  "predict, start it, come back" activity, not live tuning.
- **Still to measure:** the Colab T4. The A30 does 126k attempts in about 5 min. If a T4 is 2–3× slower,
  20k attempts would take 1–2 min, which would make live tuning feasible. That needs the benchmark
  notebook result.
- **Pretraining's case rests on reuse** (PLAN §10), not speed on the main task.

**Colab T4 timing** (benchmark notebook, early-training rates, so these are upper bounds):

| Colab T4 | Speed | 20k attempts |
|---|---|---|
| RL from scratch | 140 attempts/s | ~2.4 min |
| RL on pretrained perception (frozen) | 229 attempts/s | — |
| Pretraining | 0.006 s/step, so 3,000 steps take ~20 s | — |

- Colab gives PyTorch **1 CPU thread**, so its CPU fallback is slower than the 2-thread measurements
  above.
- **Conclusion:** on a T4, live tuning *is* feasible. A 20k-attempt run takes about 2–3 min, and the
  article's full 126k takes about 15 min. The A30 runs about 2.4× faster than the T4.
- Pretraining's speed-up is irrelevant at these speeds. Its case rests on reuse (PLAN §10).

---

## Round 7: Which classroom experiments show their effect in a class-sized budget?

Students would train **from scratch** on a T4. The budgets are 20k attempts (about 2–3 min) and 40k (about
4–6 min). Every classroom experiment so far was measured at 126k attempts. This round measures each one at
the class budgets, using the current defaults: mix 1:1:1, new cell +0.005, wall −0.01, seed 0.

**Predictions**, at 20k / 40k attempts:

| Run | Prediction |
|---|---|
| baseline | open about 93% / about 97%; breakout about 5% / 10–20%; shortcut went outside *high at 20k* (round 6's 20k run showed 52%: an undertrained agent wanders), lower at 40k (about 25%) |
| new cell +0.05 (hacking) | moves outside on open mazes **≥ 20 at both budgets**. From scratch, the hacking appeared by 10k |
| new cell 0 (shortcut) | indistinguishable from the baseline at 20k, because the baseline already wanders; at 40k, shortcut went outside **at least 15 points above the baseline** |
| no outside_only (1:1:0) | outside_only success **under 30%**, breakout about 0, at both budgets |
| no interior_break (1:0:1) | interior_break success **under 40%**, breakout about 0, at both budgets |
| wall −0.05 | probes on interior_break **at least 30% lower** than the baseline at both budgets; breakout lower than the baseline at 40k |

**Results** (A30 time per run: about 1.1 min at 20k and 1.8 min at 40k, so a T4 takes about 2.6 and
4.3 min):

| Run | Budget | Open | Interior break | Outside only | Enclosed | Shortcut: went outside | Moves outside, open mazes | Probes, interior |
|---|---|---|---|---|---|---|---|---|
| baseline | 20k | 88% | 65% | 53% | 6.6% | 41% | 14.8 | 25.9 |
| baseline | 40k | 96% | 82% | 92% | 8.6% | 36% | 10.5 | 19.1 |
| new cell +0.05 | 20k | 91% | 44% | 76% | 1.6% | 100% | **52.8** | 18.3 |
| new cell +0.05 | 40k | 96% | 69% | 86% | 0.4% | 100% | **46.4** | 11.0 |
| new cell 0 | 20k | 87% | 60% | 53% | 1.4% | 52% | 15.2 | 30.3 |
| new cell 0 | 40k | 97% | 78% | 90% | 8.6% | 36% | 9.4 | 21.5 |
| no outside_only | 20k | 92% | 82% | **0%** | 0.0% | 4% | 0.1 | 23.6 |
| no outside_only | 40k | 96% | 92% | **0%** | 1.2% | 2% | 0.1 | 13.2 |
| no interior_break | 20k | 89% | **39%** | 71% | 0.2% | 25% | 20.2 | 31.9 |
| no interior_break | 40k | 98% | **29%** | 97% | 0.0% | 3% | 12.2 | 18.7 |
| wall −0.05 | 20k | 90% | 64% | 0% | 0.0% | 5% | 0.1 | 11.4 |
| wall −0.05 | 40k | 98% | 75% | 94% | 12.0% | 30% | 7.8 | 13.1 |

**Evaluation:**

| Experiment | Prediction | Result |
|---|---|---|
| Hacking (new cell +0.05) | outside moves on open ≥ 20 | **52.8 / 46.4**, against a baseline of 15 / 10.5; shortcut 100% |
| No outside_only | outside_only < 30%, breakout ~0 | **0%** at both budgets; it never leaves (0.1 moves outside) |
| No interior_break | interior < 40%, breakout ~0 | **39% / 29%**, breakout 0 |
| Shortcut (new cell 0) | +15 points at 40k | **not visible**: 36% against 36%. At 20k both wander (52% vs 41%) |
| Wall −0.05 | probes ≥ 30% lower; breakout lower at 40k | probes 56% / 31% lower ✓; breakout higher ✗ (12% vs 8.6%; single seed, within noise) |
| Baseline | open ~93% / ~97%; breakout ~5% / 10–20% | open 88% / 96%; breakout **6.6% / 8.6%**, below the predicted 10–20% at 40k |

Also: at 20k, a costly wall (−0.05) left the agent unable to solve outside-only mazes (0%). It recovered
by 40k (94%). Costly probing slows early learning broadly.

**Conclusions for class** (T4, 2–5 min per run):
- **Live experiments that work** (large, fast, legible):
  1. **Exploration reward → reward hacking.** The agent tours the outside farming points, and the shortcut
     rate goes to 100%.
  2. **Remove outside-only mazes** → the agent never leaves the maze and can't solve outside-only mazes.
  3. **Remove interior-break mazes** → the agent can't break walls.
- **The breakout itself is too rare at class budgets** (under 10%, against 36–74% at full scale). Show it
  with **pretrained full-scale agents** instead, including the variants. The "remove a skill → the
  breakout vanishes" comparison then also comes from pretrained agents, with the live runs showing the
  missing skill.
- **The shortcut effect (new cell 0) isn't visible at class budgets.** If it's used at all, it would come
  from a pretrained full-scale agent.

---

## Round 8: How do competence and the emergent breakout scale with model size?

> **Superseded by round 10.** These runs used a fixed 126k-attempt budget, and the smaller models had
> not finished learning: xs was still gaining about 5 points per 20k attempts.

The breakout combines skills, and the network has to hold all of them at once. In LLMs, some abilities
appear only above a certain model size. Does anything like that happen here?

**H12: the common skills saturate at small sizes, but the breakout needs capacity.**
- *Sizes*, each trained at full scale (126k attempts, current defaults, mix 1:1:1) on the cluster
  GPUs, seeds 0 and 1:

  | Size | Vision channels | Hidden | Connections |
  |---|---|---|---|
  | tiny | 16/16/16 | 32 | 20k |
  | small | 16/32/32 | 64 | 71k |
  | default | 32/64/64 | 100 | 229k (round 4's `mix_1-1-1` runs, seeds 0 and 1) |
  | large | 64/128/128 | 256 | 1.1M |

- *Predictions:*
  - **open:** at least 95% at every size. It's easy.
  - **interior_break:** rises with size: tiny about 70%, small about 85%, default and large about 92%.
  - **Breakout on enclosed:** rises with size: tiny **under 5%**, small **10–30%**, default 36–74%
    (measured), large **at or above the default**. If the breakout appears only above some size while
    open-maze success is already saturated, that's an "emergence with scale" result for class.
  - **Shortcut:** follows the breakout. Larger models take the outside route more.

**Added measure: chain progress on enclosed** (`mazerl.rollout.chain_progress`).
- Each region of the enclosed maze has a **depth**: the fewest weak spots that must be broken, starting
  from outside, to reach it. The flag is at depth 4. An attempt is scored by the deepest region it
  reaches, by any route.
- Run A (default size, old mix) shows the shape:

  | Left the maze | Depth 1 | Depth 2 | Depth 3 | Depth 4 = flag |
  |---|---|---|---|---|
  | 100% | 99% | 91% | 54% | 15% |

  The agent nearly always starts the chain and loses attempts with each added link.
- *Prediction:* smaller models complete the first links as often as larger ones, but drop off sooner:
  - tiny: depth 3 **under 20%**;
  - small: depth 3 **30–50%**;
  - default and large: depth 3 **at least 50%**.

  "Each skill works on its own, but the chain fails" is the scale result to look for.

**Results** (full scale, two seeds each. Default size = round 4's `mix_1-1-1` runs. Standard suite at 500
attempts; chain progress at 300 attempts):

| Size | Connections | Open | Interior break | Outside only | Enclosed | Shortcut: went outside | Chain: depth 2 | Depth 3 | Depth 4 |
|---|---|---|---|---|---|---|---|---|---|
| tiny | 20k | 97.6–98.0% | 81–87% | ~95% | **15–29%** | 50% | 0.71–0.92 | 0.42–0.60 | 0.14–0.23 |
| small | 71k | 98.2–99.0% | 88–91% | ~97% | **15–22%** | 15–35% | 0.79–0.85 | 0.54–0.57 | 0.13–0.22 |
| default | 230k | 99.4–99.6% | 91–94% | ~98% | **36–74%** | 27–46% | 0.92–0.98 | 0.73–0.93 | 0.31–0.80 |
| large | 1.1M | 99.0–99.2% | 96% | ~98% | **42–56%** | 7% | 0.93–0.98 | 0.78–0.95 | 0.50–0.56 |

All sizes left the maze and reached depth 1 (broke back in from outside) in at least 99% of attempts.

**Evaluation:**
- **Common skills saturate early:** ✓. Open is at least 97.6% at every size, outside_only about 95–98%.
  Interior_break improves gradually (84% → 96%), close to the prediction.
- **The breakout grows with size:** ✓ in trend, ✗ in shape. From small (about 18%) to default and large
  (about 50%) it roughly doubles to triples, but there's no threshold. **Tiny still breaks out 15–29% of
  the time** (predicted under 5%).
- **The chain:** ✓ for the shape, ✗ for the magnitudes. Every size completes the first link. The gap
  opens at the later links. At depth 3, tiny and small reach about 50%, against about 85% for default and
  large; at depth 4, about 18% against about 55%. Smaller models lose more attempts with each added link.
  Predicted depth 3 for tiny was under 20%; it was 42–60%.
- **Large models are more consistent across seeds** (42–56%, against 36–74% for default). They rarely
  take the p.7 outside shortcut (7%).
- **For teaching:** "bigger models chain separately learned skills more reliably" is supported as a
  **gradual** trend, not a sudden emergence threshold. Two seeds per size is thin, especially given the
  default size's spread.

---

## Round 9: Going smaller. Where does chaining fail while single skills survive?

> **Superseded by round 10:** these models were undertrained (see round 8's note).

The tiny model (20k) still solves 98% of open mazes and reaches the enclosed flag 15–29% of the time.
Round 8 never found the capacity limit.

**H13: there's a size range where single-skill mazes still work but the chain collapses.**
- *Sizes*, full scale, seeds 0 and 1:

  | Size | Vision channels | Hidden | Connections |
  |---|---|---|---|
  | xs | 8/8/8 | 16 | 5.5k |
  | xxs | 4/8/8 | 8 | 3.0k |
  | micro | 4/4/4 | 8 | 1.6k |
  | nano | 2/4/4 | 4 | 0.9k |

- *Predictions:*
  - In the **1–5k range:** open at least 90%, outside_only at least 80%, but chain depth 3 **under 10%**
    and the flag about 0%.
  - At **nano**, even open mazes fall below 80%.
  - Interior_break declines before outside_only does, because breaking a wall needs memory of which
    walls were probed.

**Results** (two seeds each; stages from `breakout_stages` at 300 attempts):

| Size | Connections | Open | Interior break | Outside only | Example 1 (impossible task) | Broke in from outside | Example 2 (needless) |
|---|---|---|---|---|---|---|---|
| tiny | 20k | 98% | 81–87% | 95% | 15–29% | 100% | 50% |
| xs | 5.5k | 89–95% | 61–64% | 64–87% | **3–4%** | 79–85% | 18–38% |
| xxs | 3.0k | 85–87% | 55–69% | 1–60% | 1–8% | 53–83% | 1–81% |
| micro | 1.6k | 82–86% | 65–67% | 22–38% | 2–6% | 75–83% | 56–64% |
| nano | 0.9k | 28–74% | 23–53% | 4–25% | 0–9% | 51–75% | 58–91% |

All sizes left the maze in 100% of attempts.

**Evaluation:**
- **Example 1 collapses between 20k and 5.5k connections** (15–29% → 3–4%), while open mazes mostly
  still work (89–95%). That part ✓.
- **But the single skills degrade too,** so "can do each step, can't chain them" is only partly right
  ✗. Interior_break falls from about 84% to about 62%, outside_only drops, and breaking in from outside
  falls to about 80%.
  - **A better description:** *chains amplify small weaknesses.* Each step gets somewhat less reliable,
    and the multi-step task, which needs all of them in sequence, collapses much faster than any single
    skill.
  - **Correction (after discussion):** this does **not** show that bigger models reason or chain better.
    Example 1 tracks interior wall-breaking skill:

    | Size | Interior break | Example 1 |
    |---|---|---|
    | default | 91–94% | 35–75% |
    | tiny | 81–87% | 15–29% |
    | xs | 61–64% | 3–4% |

    The impossible maze needs several interior breaks. A component skill that was learned less well
    explains the drop, and these runs can't separate that from a chaining deficit.
  - **What would separate them:** compare models matched on each component skill, for example by
    training the small model until its interior-break success matches the default's.
  - **For class:** present scaling only as "weak skills compound in multi-step tasks."
- **Below about 3k connections, training is erratic.** The same size gives very different agents (outside_only
  1% vs 60%; open 28% vs 74%). Example 2 numbers there are mostly noise.
- **Nano:** open fell below 80% for one seed (28%), not the other (74%). ✓/✗.


---

## Round 10: Model size, with every model trained to convergence

Rounds 8–9 compared models after the same 126k attempts. Smaller models learn more slowly, so that
compared trained large models against half-trained small ones.

**Protocol.**
- Every size trains for **500k attempts** (4× the earlier budget), with the learning rate annealed over
  the whole run.
- A run counts as **converged** if its training success changed by less than 1 point over its final
  20% of attempts. If it didn't, it's rerun with double the budget.
- Two seeds per size.

| Size | Connections |
|---|---|
| xs | 5.5k |
| tiny | 20k |
| small | 71k |
| default | 230k |
| large | 1.1M |

**H14: at convergence, smaller models match the larger ones on single skills, and on the multi-step
task once their skills match.**
- *Predictions:*
  - Default, large and small change little from round 8: interior_break within 3 points, Example 1
    within the seed spread.
  - Tiny and xs improve: xs interior_break **above 80%**, Example 1 **above 10%**.
  - Plotting Example 1 against interior_break success, all sizes lie on one trend. That would mean
    model size buys learning speed, not separate "reasoning."
  - If smaller models sit clearly *below* that trend at matched skill, that would be evidence of a
    chaining limit.

**Status:** submitted.
