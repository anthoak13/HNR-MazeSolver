# Maze RL activity — build plan

An HNR-1303 (Unit 5) activity built on the NYT article *"A.I.'s Rogue Behavior Starts With Its Training"*
(Saha & Migliozzi, Oct. 26 2026; PDF in repo root). Students train and watch maze-solving agents and see
how reward values shape behavior, including the "go outside and break back in" behavior from the article.
Students don't touch the training code. They change reward values and a few training settings, then look
at plots and animations.

## Sources, in priority order
Every rule below is tagged with where it comes from:

- **[A] The article** — text and figures. This is the basis, and it wins any conflict.
- **[C] The author's comments** on the article:
  - "reaching the flag could fetch the agent +5 points, while stepping outside could result in a penalty
    of −0.01"
  - "it took less than an hour to train the model on my laptop. Once trained, running it takes less than a
    second to compute the solution paths."
- **[P] The paper the article cites**, *AI Safety Gridworlds* (Leike et al. 2017, arXiv:1711.09883). Used
  only to fill gaps the article leaves, for the parts of the method that are its conventions.
- **[D] Our design decision**, where none of the above says anything. Each [D] is a setting in the package,
  not hard-wired.

---

## 1. The world

### Layout (counted from the article's figures, pp. 1–7)
| Item | Rule | Tag |
|---|---|---|
| Arena | 18 × 18 grid of cells | [A] |
| Maze | Middle 14 × 14 block (rows/cols 2–15), with a 2-cell **outside** band all around | [A] |
| Walls | Walls fill whole cells. Wall thickness is 1–2 cells and corridor width is 1–2 cells | [A] |
| Start | Open top-left corner cell of the maze, right next to the outside band | [A] every figure |
| Flag | One cell inside the maze, usually in the lower/right part | [A] position, [D] how it's placed |
| Openings | Besides the corner entrance, the outer wall has **3–6 plain gaps** to the outside | [A] counted in the 7 training figures (5–8 open boundary cells each) |
| Weak spots | **2–6** wall cells per maze, 0–3 of them in the **outer boundary wall**, including in training mazes | [A] pale squares, counted in the figures |
| Layouts | A fixed bank of **13,148** generated layouts | [A] the number, [D] the generator |

**Maze generator [D], matched to the figures:** *recursive division*. Draw a wall across the open
interior with one doorway, then split each side again, and stop early so some chambers stay open rooms.
This reproduces the figures' long walls with single doorways enclosing open rooms. Then:
- open the top-left entrance (3 cells, as in every figure);
- cut 3–6 openings in the outer wall;
- turn 2–6 wall cells with open cells on two opposite sides into weak spots, so each one "opens into a
  passage" [A];
- place the flag at least 12 moves from the start.

The generator was checked by classifying every cell of the 7 training-figure images from the PDF.

**Mazes that require breaking a wall** [A] "They *must* probe walls to discover weak spots that open into
passages": a share of training mazes (`locked_frac`, default 0.5 [D]) turn one doorway into a weak spot, so
the flag's region can only be reached by breaking an *interior* wall. The sealed region never touches an
outer opening or outer weak spot, so the only way in is from inside.

Layouts are *stored* as ASCII art, the same convention as the paper's environments [P]. That covers the
two scenario mazes from pp. 6–7, transcribed from the PDF (§4). ASCII is for definitions and
debugging only (`print(maze)`, test failures). Students never see it. Everything they look at is the
animated article-style view (§4).

### Agent mechanics
| Item | Rule | Tag |
|---|---|---|
| Moves | Up / down / left / right | [A], also [P] |
| Bumping a solid wall | The agent stays put. This counts as a **probe** and gets the probe penalty | [A] probes are penalized, [D] bump = probe |
| Bumping a weak spot | The spot opens and the agent moves into it. This gets the **through-wall** penalty | [A] "weak spots that open into passages," "move through a wall carries a small penalty"; [D] it opens on the first bump |
| What the agent can tell apart | Weak spots look exactly like walls to the agent. Only the human viewer sees them as pale squares | [D] (the article says agents must *probe* to discover them) |
| Vision | 5 × 5 window centered on the agent. Walls block line of sight | [A] |
| Memory | Remembers every cell it has seen during the episode | [A] "remember past observations" |
| Flag direction | Compass direction to the flag, one of 8 (N, NE, …) | [A] "compass direction," [D] 8-way |
| Episode end | Agent reaches the flag, or a **100-step** cap | [P] "maximal episode length 100 steps," goal ends the episode |
| Multiple agents | "One or more agents" share a maze, move independently and never block each other. Mostly used for visuals | [A] figures show several, [D] no interaction |

### Rewards (student-tunable)
| Term | Default | Tag |
|---|---|---|
| Reach flag | **+5** | [C] |
| Each step on an outside cell | **−0.01** | [C] |
| Visit a new cell (inside **or** outside) | +0.02 | [A] exists, [D] value |
| Probe (bump a solid wall) | −0.02 | [A] exists, [D] value |
| Move through a weak spot | −0.05 | [A] exists, [D] value |
| Per-step cost | 0 (available, off by default) | [D]; the article lists no step cost, while [P] uses −1/step vs. +50 goal |

The [D] values are starting points. Milestone 3 tunes them until the base agent reliably solves training
mazes, and the final values get recorded here.

### "Training excluded scenarios where the agent bypassed the maze entirely" [A]
The figures show training layouts with openings near the flag (e.g. p.3), so walking around the outside
was a real route. An attempt counts as a **bypass** [D] if the agent went *around* the maze instead of
through it:
- it **re-entered the maze somewhere other than where it last left**, through another opening or by
  breaking in through an outer weak spot; or
- it **never got more than one cell into the maze**, for example lapping the outside until time runs out.
  This was your reading.

Bypass attempts are dropped from the training batch. They still count toward the 126,000 attempts, and
they're logged. This rule makes the article's statement literally true: "we did not train it … to leave the
area despite the penalties, probe the wall from the outside and break in." Breaking in from outside is
always a bypass, so it is never reinforced.

`exclude_bypass=False` turns the rule off, which is a classroom experiment. A layout-level alternative,
`no_outside_shortcuts=True` (close openings until going around is never shorter), is available but off by
default, because the article's own training layouts have such routes.

---

## 2. The agent

### Observation [D]
An 11 × 11 window of the agent's **memory**, centered on the agent, with one channel per cell type: unseen,
open-inside, outside, wall, flag, visited-by-me, and visible-right-now. Plus the 8-way compass one-hot.
Centering the window on the agent means it doesn't know its absolute position, so what it learns has to
transfer across layouts.

### Network — "vision" + "reasoning" [A], ≈230,000 connections [A]
- **Vision** [A] says it exists, [D] its design: conv 3×3 (7→32) → conv 3×3 (32→64) → 2×2 max-pool →
  conv 3×3 (64→64) → 1,600 features.
- **Reasoning:** concat compass → **two hidden layers of 100 nodes** [P] → move probabilities (4) and
  value estimate (1).
- Measured parameter count of this shape: ≈221–225k. Final channel counts get adjusted to land at ≈230k.

### Learning algorithm: A2C (advantage actor-critic) [P]
The article doesn't name an algorithm. The cited paper's baselines are A2C and Rainbow. A2C is the simpler
of the two and the cheaper per step: one gradient pass per step, against three or more for PPO. That
cheapness is what makes the author's "under an hour on a laptop" realistic (§3).

Hyperparameters taken from the paper's A2C setup [P]:
- discount 0.99
- RMSProp, learning rate 5e-4 annealed to 0
- entropy bonus 0.01–0.1, optionally annealed
- 5-step unroll, baseline loss weight 0.25
- gradient clip 40
- rewards normalized to [−1, 1]

Adaptation for the bypass rule [D]: each environment buffers its current attempt (≤100 steps). When the
attempt ends it is either sent to the update or dropped as a bypass. A pooled attempt may have started a
few updates earlier, so the policy loss uses PPO's clipped probability ratio to stay stable. It is still
one gradient pass per move, the same cost as A2C.

- **Student-facing knobs:** reward values, `learning_rate`, `discount` (how far ahead the agent "cares"),
  `exploration` (entropy bonus), `episodes`, `exclude_bypass`.
- **Everything else is hidden.**

### Engine [D]: our own batched environment, not pycolab
The paper's environments (`google-deepmind/ai-safety-gridworlds`) run on `pycolab`. We tested it, and
neither is usable as a foundation:

- `pycolab` (last commit 2019) crashes on numpy 2.x, which Colab ships. We tested it: `ascii_art_to_game`
  fails in `np.vstack`.
- `ai-safety-gridworlds` is archived (2022) and pins numpy 1.14.
- Both step one game at a time in Python, which is too slow for 126k attempts.

Instead, the environment is batched PyTorch tensor operations: hundreds of mazes step at once, on CPU or
GPU. From pycolab we keep the ASCII-art way of writing mazes.

---

## 3. How long will training take?

**Baseline [C]:** the author trained the full model in **under an hour on a laptop**, and a trained agent
runs in under a second. Our target is to match that: a full article-scale run in **≤1 hour on an ordinary
laptop CPU, with no GPU required**.

### Work in an article-scale run
- **Steps.** 126,000 attempts × at most 100 steps [P cap] gives ≤12.6 M steps. Early attempts hit the cap.
  Trained attempts are much shorter: the shortest paths in the article's figures are ~25 moves. So the
  realistic total is **≈5–9 M steps**.
- **Compute per step (A2C).** One forward pass while playing (≈7 MFLOP) plus one training pass (≈21 MFLOP)
  ≈ 28 MFLOP. A full run is therefore **≈0.15–0.25 PFLOP**.
- **What the author's figure implies.** "Under an hour" means ≥1.5–2.5k steps/s, which takes ≈40–70
  GFLOP/s. A modern laptop CPU can sustain that for a network this size, so the baseline and this design
  are consistent.

### By platform (estimates, until Milestone 2 replaces them with measurements)
| Platform | Full 126k-attempt run | 5-minute class run |
|---|---|---|
| Author's laptop [C] | < 1 h | — |
| Colab **CPU** (2 vCPU, slower cores than a current laptop) | ≈1–3 h | ≈5–10k attempts: enough to watch early learning, not to finish |
| Colab **T4 GPU** | ≈10–20 min | ≈30–60k attempts: enough to retrain the base agent with new rewards |

The Colab numbers are scaled estimates, not measurements. **Milestone 2** is a 2-minute benchmark notebook
that you run once on a Colab CPU runtime and once on a T4. I'll put the measured numbers here and set the
class time budgets from them.

### What this means for class
- **Recommended:** a T4 runtime, as 05-02 already uses. Every activity also works on CPU through the
  pretrained agents.
- Students don't train from scratch at full scale in class. They:
  1. **Retrain from the pretrained base agent** with their own rewards for a few minutes, to watch behavior
     shift. Example: the agent learns to lap the outside when new-cell reward > outside penalty and
     `exclude_bypass` is off.
  2. **Train from scratch for a few minutes**, to watch the start of the learning curve.
  3. **Load precomputed agents** from a reward sweep, so nobody is stuck waiting.
- Pretraining is done once, on the author-style setup (a laptop/desktop CPU, ≤1 h per run), or on a Colab T4.

### Pretrained artifacts (shipped in the package, ~1 MB each)
- `base` — the article's setup: default rewards, 126k attempts, 13,148 layouts, bypass exclusion on.
  Saved partway through at ≈1k, 5k, 20k, 60k and 126k attempts, to reproduce the article's "agents at
  different stages of training" grid.
- **Reward sweep**, ≈6–8 runs at ≤1 h each:
  - outside penalty ∈ {0, −0.01, −0.1, −1}
  - new-cell reward ∈ {0, 0.02, 0.1}
  - probe/through-wall penalties ×{0.2, 1, 5}
  - exclusion on vs. off

---

## 4. What students see (mirrors the article's figures)
Every maze view is an **animation in the NYT style**, not a static plot or text.

**Look**, copied from the figures:
- Faint dotted grid on white, light-gray wall blocks, and pale squares for weak spots (the viewer sees
  them, the agent doesn't).
- Pink tint on the outside band.
- Hollow coral circle for the start, solid red dot for the agent, and a fading red trail behind it.
- Black round flag marker.
- Sparse sans-serif labels (AGENT, GOAL, WALL, OUTSIDE).

**Motion:**
- The agent glides smoothly between cells rather than jumping frame to frame.
- A pulsing halo plus a short callout ("Penalty for stepping outside", "Probe −0.02", "Broke through
  wall") appears when a reward or penalty fires, as on p.3.
- A broken weak spot visibly opens.
- Several agents can move in the same maze at once (pp. 4–5).
- Play / pause / scrub controls and a running score.

**How it's built:**
- The package records each episode as data (positions, events, rewards).
- A small self-contained HTML/SVG/JavaScript player draws it inline in the notebook via
  `IPython.display.HTML`.
- This works in Colab and Jupyter with no external libraries, and SVG stays crisp at any size. matplotlib
  frame-by-frame animation was the alternative; it looks choppy and is heavy.
- The player can also save an episode as a standalone `.html` file, so it can be shared or shown on a
  projector.

**matplotlib is used only for the training curves** (`run.plot()`), which match the 05-01/05-02 plots
students already know.

| Function | Shows | Article figure |
|---|---|---|
| `mz.show_maze(maze)` | One layout | p.1–2 "Agent tasked with finding a flag" |
| `mz.playback(agent, maze)` | Animated attempt with a running score and pop-up labels for penalties/rewards ("Penalty for stepping outside") | p.3 "Playback of a training run" |
| `mz.training_grid(run)` | Several layouts with agents at several training stages | p.4–5 "Agents training on different maze layouts" |
| `mz.scenario("enclosed")` | Flag fully walled in (layout transcribed from p.6) | p.6 "Agent facing a hard problem" |
| `mz.scenario("outside_shortcut")` | An inside path exists, but the outside route is shorter (layout transcribed from p.7) | p.7 "Agent taking outside path…" |
| `mz.agent_view(agent, maze, step)` | What the agent knows: memory, current 5×5 view, compass | Methodology |
| `run.plot()` | Curves over training: average total reward, % reaching flag, steps per attempt, probes, breaks, time outside, % bypass | not in the article. Mirrors the 05-01/05-02 loss/accuracy curves and the paper's "episode return" plots [P] |
| `mz.reward_breakdown(agent, maze)` | Bar chart of where an attempt's points came from | "reward … outweighed the penalties" (p.6) |
| `mz.evaluate(agent, n=500)` | Table of results on *unseen* layouts | training vs. validation, as in 05-02 |

### Student-facing API sketch
```python
!pip install -q git+https://github.com/anthoak13/HNR-MazeSolver@maze-rl
import mazerl as mz

rewards = mz.Rewards(flag=5, outside=-0.01, new_cell=0.02, probe=-0.02, through_wall=-0.05)
base = mz.load_pretrained("base")
run = mz.train(rewards, start_from=base, minutes=5)   # or episodes=…
run.plot()
mz.playback(run.agent, mz.scenario("enclosed"))
mz.evaluate(run.agent, n=500)
```

---

### The article's two test mazes (transcribed)
Both layouts were read cell by cell from the embedded figure images. Every cell's shade falls into a few
clean values, so no cell was ambiguous. They live in `mazerl/scenarios/*.txt`. Two details in the figures
don't match the article's text exactly:

- **p.6 "enclosed":**
  - The start pocket opens *only* to the outside, so the agent must leave the maze first.
  - The flag's room has one weak spot, which isn't directly reachable from outside. The shortest route
    goes outside, then through 4 weak spots.
- **p.7 "outside_shortcut":**
  - The inside path takes 27 moves and the outside route 25.
  - The outside route uses plain *openings*, not a weak spot. The text says the agent exploited "a
    vulnerability in the wall."

We use the figures as drawn.

## 5. Repo layout
```
mazerl/
  __init__.py        public API only
  maze.py            generator, layout bank, ASCII scenarios
  env.py             batched tensor environment (CPU or GPU)
  agent.py           vision + reasoning network
  a2c.py             A2C training, bypass filter, logging
  viz.py             episode recording, training curves (matplotlib)
  player/            self-contained HTML/SVG/JS animation player (NYT style)
  pretrained/        *.pt checkpoints + metadata (rewards, settings, attempt counts)
notebooks/
  00-benchmark.ipynb         Colab speed check (Milestone 2)
  01-maze-rl-in-class.ipynb  student notebook, course style (bold questions)
  01-maze-rl-solution.ipynb  instructor key / notes
scripts/pretrain.py          full-scale runs + sweep
tests/                       pytest
pyproject.toml               pip-installable from GitHub
```

---

## 6. Milestones (I build and debug; you review at each ✔)
1. **World + player:** generator, environment mechanics, and the NYT-style animated player (random agents to start). Tests for: probe/break
   rules, outside penalty, line-of-sight occlusion, compass, bypass detection, every maze solvable.
   ✔ You compare rendered mazes with the PDF.
2. **Benchmark notebook.** ✔ You run it on Colab CPU and on a T4. I update §3 with real numbers and set
   the class time budgets.
3. **A2C training + curves.** Short runs to tune the [D] reward values until the base agent solves unseen
   layouts reliably. Final values recorded in §1.
4. **Full pretraining** (`base` + stage checkpoints). Measure the real run time against the ≤1 h laptop
   target.
5. **Article scenarios.** Evaluate `base` on the pp. 6–7 layouts. ✔ I report honestly whether the
   emergent behaviors show up and how often (over many attempts). If they don't, adjust the *training
   distribution or rewards*, never script the behavior, and document what changed.
6. **Reward sweep** pretrained agents.
7. **Student notebook + instructor key**, in the 05-0x style.
8. **End-to-end Colab test** from a fresh runtime. ✔ Final review.

---

## 7. Risks
- **The emergent "break back in" may not reproduce.** Nothing in training rewards that exact sequence;
  the article's point is that it emerges. Emergence can fail for concrete reasons:
  1. **It's three skills chained together:** leave the maze, travel outside, and break a wall *to get back
     in*. Each piece is practiced in training: the start borders the outside, outer weak spots exist, and
     `locked_frac` mazes require breaking interior walls. Only the combination is new at test time, which
     is the article's point. The risk is that the combination doesn't transfer, not that a piece is missing.
  2. **The view from outside looks different.** In training, the agent probes walls from inside the maze.
     Seen from outside, the same wall sits next to outside cells, a combination it has rarely seen. The
     cited paper's lava-world experiment [P] shows agents like this often fail under exactly this kind of
     shift.
  3. **Reward arithmetic.** The outside route has to fit inside the 100-step cap. And after discounting,
     the +5 flag has to outweigh the outside, probe and through-wall penalties along that route ("the
     reward for completing the task outweighed the penalties," p.6). If the [D] penalty values are too
     harsh, the agent never commits to the long outside route.
  4. **Giving up.** In the enclosed-flag case the agent may keep probing the enclosure from inside, or
     wander once it runs out of new cells. Persistence comes from the new-cell reward, so its value matters.
  5. **Training luck.** A2C can collapse into a near-deterministic policy that never tries leaving. One
     unlucky seed can lose the behavior.
  6. **We don't know how often it happened for the NYT.** The article shows one replay. It may have been
     occasional for them as well.

  Milestone 5 measures it. It's reported with a frequency, for example "in 37% of attempts," not
  cherry-picked. If it's rare or absent, the legitimate levers are the reward values (2–4) and how often
  training mazes have reachable outer weak spots and require wall breaking (1–2). We never script the behavior. A model that only
  sometimes breaks in is also a good discussion point.
- **A2C can be unstable.** The paper reports occasional "entropy collapse" and drops its worst 25% of
  seeds [P]. Mitigations: the entropy floor from [P], fixed seeds for the pretrained agents, and a
  `run.plot()` that makes collapse visible, which is itself a teaching point. If A2C won't train reliably,
  the fallback is PPO, which costs ≈2–3× more compute per step.
- **Colab T4 availability and quotas.** The pretrained agents mean every activity also works on CPU.
- **Our [D] choices aren't the article's.** The student notebook says plainly which rules come from the
  article and which are ours.

## Git
All work goes on the `maze-rl` branch, pushed to `origin` after each milestone (and more often mid-milestone)
so you can switch machines. Merge to `main` after the final review.
