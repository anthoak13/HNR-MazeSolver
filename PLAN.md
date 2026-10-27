# Maze RL activity — build plan

An HNR-1303 (Unit 5) activity built on the NYT article *"A.I.'s Rogue Behavior Starts With Its Training"*
(Saha & Migliozzi, Oct. 26 2026; PDF in repo root). Students train and watch maze-solving agents and see
how reward values shape behavior, including the "go outside and break back in" behavior from the article.
Students don't touch the training code. They change reward values and a few training settings, then look
at plots and animations.

Every rule below is tagged:

- **[A]** stated or shown in the article (text or figures).
- **[D]** our design decision, made where the article is silent. The article doesn't give these details,
  so these are choices, not reconstructions. Each one is a setting in the package, so we can change it
  later without a rewrite.

---

## 1. The world

### Layout (from the article's figures, pp. 1–7)
| Item | Rule | Tag |
|---|---|---|
| Arena | 18 × 18 grid of cells | [A] counted from the figures |
| Maze | Middle 14 × 14 block (rows/cols 2–15), with a 2-cell **outside** band all around | [A] |
| Walls | Walls fill whole cells. Wall thickness is 1–2 cells and corridor width is 1–2 cells | [A] |
| Start | Open top-left corner cell of the maze, right next to the outside band | [A] every figure |
| Flag | One cell inside the maze, usually in the lower/right part | [A] position, [D] how it's placed |
| Weak spots | A few wall cells (≈4–8 per maze), in both interior walls and the **outer boundary wall**, including in training mazes | [A] pale squares in the figures |
| Layouts | A fixed bank of **13,148** generated layouts | [A] the number, [D] the generator |

**Maze generator [D]:** carve a standard 6 × 6-room maze on a 13 × 13 lattice, add a few loops, then
duplicate one random interior row and one random interior column to get 14 × 14. The duplication gives the
mixed 1–2-cell walls and corridors seen in the figures. Then open the top-left entrance and place the flag
(a minimum path distance from the start). Weak spots go on wall cells that have open cells on two opposite
sides, so each one "opens into a passage" [A]. The generator gets checked by eye against the figures
before anything else is built on it.

### Agent mechanics
| Item | Rule | Tag |
|---|---|---|
| Moves | Up / down / left / right | [A] |
| Bumping a solid wall | The agent stays put. This counts as a **probe** and gets the probe penalty | [A] probes are penalized, [D] bump = probe |
| Bumping a weak spot | The spot opens and the agent moves into it. This gets the **through-wall** penalty | [A] "weak spots that open into passages," "move through a wall carries a small penalty"; [D] it opens on the first bump |
| What the agent can tell apart | Weak spots look exactly like walls to the agent. Only the human viewer sees them as pale squares | [D] (the article says agents must *probe* to discover them) |
| Vision | 5 × 5 window centered on the agent. Walls block line of sight | [A] |
| Memory | Remembers every cell it has seen during the episode | [A] "remember past observations" |
| Flag direction | Compass direction to the flag, one of 8 (N, NE, …) | [A] "compass direction," [D] 8-way |
| Episode end | Agent reaches the flag, or a **400-step** cap | [D] |
| Multiple agents | "One or more agents" share a maze, move independently and never block each other. Mostly used for visuals | [A] figures show several, [D] no interaction |

### Rewards (student-tunable)
| Term | Default | Tag |
|---|---|---|
| Reach flag | **+5** | value from the article's notes |
| Each step on an outside cell | **−0.01** | value from the article's notes |
| Visit a new cell (inside **or** outside) | +0.02 | [A] exists, [D] value |
| Probe (bump a solid wall) | −0.02 | [A] exists, [D] value |
| Move through a weak spot | −0.05 | [A] exists, [D] value |
| Per-step cost | 0 (available, off by default) | [D] |

The [D] values are starting points. Milestone 3 tunes them until the base agent reliably solves training
mazes, and the final values get recorded here.

### "Training excluded scenarios where the agent bypassed the maze entirely" [D, your reading]
An episode counts as a **bypass** if the agent never enters the maze interior: it leaves through the
entrance and stays outside until the step cap. Bypass episodes are dropped from the training batch. They
still count toward the 126,000 attempts, and they're logged. This matters because outside cells are new
cells, so if new-cell reward > outside penalty, lapping the outside pays. A setting
(`exclude_bypass=True/False`) turns the rule off, which is itself a classroom experiment: *what does the
agent learn when we don't filter those out?*

---

## 2. The agent

### Observation [D]
An 11 × 11 window of the agent's **memory**, centered on the agent, with one channel per cell type: unseen,
open-inside, outside, wall, flag, visited-by-me, and visible-right-now. Plus the 8-way compass one-hot.
Centering the window on the agent means it doesn't know its absolute position, so what it learns has to
transfer across layouts.

### Network — "vision" + "reasoning" [A], ~230k connections [A]
- **Vision:** conv 3×3 (7→32) → conv 3×3 (32→64) → 2×2 max-pool → conv 3×3 (64→64) → 1,600 features.
- **Reasoning:** concat compass → dense 96 → dense 96 → two outputs: move probabilities (4) and value
  estimate (1).
- Measured parameter count: **221k**. Final tuning of the dense layer width will land it at ≈230k to match
  the article.

### Learning algorithm [D]: PPO (Proximal Policy Optimization)
The article doesn't name one. PPO is a standard choice for a convolutional network policy with discrete
moves, and it's robust to reward changes, which matters when students set strange reward values. It works
on whole episodes: each parallel environment records a full attempt. Completed non-bypass attempts go into
the training pool, and an update runs once the pool holds enough steps. The bypass exclusion is then just a
filter on whole episodes.

- **Student-facing knobs:** reward values, `learning_rate`, `discount` (how far ahead the agent "cares"),
  `exploration` (entropy bonus), `episodes`, `exclude_bypass`.
- **Hidden:** clip 0.2, GAE λ 0.95, 3 epochs per update, minibatch 1024, ~512 parallel environments on GPU.

---

## 3. How long will training take?

### Work per training run
- **Steps per attempt.** An untrained agent runs to the 400-step cap. A trained agent on these mazes needs
  roughly 40–100 steps: the shortest path in the article's figures is ~25 moves, plus some exploring. That
  averages out to an estimated **100–200 steps per attempt** over a whole run.
- **Steps for an article-scale run.** 126,000 attempts × 100–200 steps ≈ **13–25 M environment steps**.
- **Compute per step.** A forward pass is ≈7 MFLOP. PPO does 1 forward pass while playing plus 3 training
  passes (≈21 MFLOP each), so ≈70 MFLOP per step. A full run is therefore **≈1–1.75 PFLOP**.
- **Environment cost.** The environment is written as batched PyTorch tensor operations, so it runs on the
  GPU alongside the network. At 512+ parallel mazes, its cost should be small next to the network.

### Estimates by platform
| Platform | Effective speed (estimate) | Full 126k-attempt run | 5-minute class run |
|---|---|---|---|
| Colab **CPU** (2 vCPU) | 20–60 GFLOP/s → ~300–900 steps/s | **5–24 hours** — not usable | ~2k attempts — too few to learn |
| Colab **T4 GPU** | 1–3 TFLOP/s → ~15–40k steps/s | **≈10–30 min** (+ overhead → plan for ≤1 h) | **≈5–12M steps** — enough to fine-tune or reshape behavior |
| This desktop (i5-13400F, CPU) | measured ≈2.9k steps/s network-only | ≈1.5–3 h | — |

**These are estimates, not measurements**, except the desktop row, where the network part was timed. The
Colab rows come from FLOP counts and typical efficiency for small convolutional networks.
**Milestone 2 replaces them with measured numbers.** That milestone is a short benchmark notebook you run
once on Colab CPU and on a T4 (≈2 minutes each).

### What this means for class
- Students use a **T4 runtime**, as 05-02 already does. A CPU runtime is too slow.
- In class, students don't train from scratch at full scale. They do one of these:
  1. **Retrain from the pretrained base agent** with their own reward values for a few minutes. This is
     enough to visibly change behavior, for example the agent learning to lap the outside when new-cell
     reward > outside penalty.
  2. **Train from scratch for a few minutes**, to watch the start of a learning curve.
  3. **Load precomputed agents** from a reward sweep (below), so nobody gets stuck waiting.
- Pretraining happens once, on a Colab T4 (≤1 h per run) or overnight on the desktop.

### Pretrained artifacts (shipped in the package, ~1 MB each)
- `base` — the article's setup: default rewards, 126k attempts, 13,148 layouts, bypass exclusion on.
  Saved partway through at ≈1k, 5k, 20k, 60k and 126k attempts, to reproduce the article's "agents at
  different stages of training" grid.
- **Reward sweep**, ≈6–8 runs at about ≤1 h each on a T4:
  - outside penalty ∈ {0, −0.01, −0.1, −1}
  - new-cell reward ∈ {0, 0.02, 0.1}
  - probe/through-wall penalties ×{0.2, 1, 5}
  - exclusion on vs. off

---

## 4. What students see (mirrors the article's figures)
Visual style copies the article:
- Dotted grid, light-gray walls, and pale squares for weak spots (the viewer sees them, the agent doesn't).
- Pink tint on the outside band.
- Red agent path, a hollow circle for the start, and a black flag marker.

Plots use matplotlib. Animations use `FuncAnimation` → `to_jshtml()` so they work inline in Colab.

| Function | Shows | Article figure |
|---|---|---|
| `mz.show_maze(maze)` | One layout | p.1–2 "Agent tasked with finding a flag" |
| `mz.playback(agent, maze)` | Animated attempt with a running score and pop-up labels for penalties/rewards ("Penalty for stepping outside") | p.3 "Playback of a training run" |
| `mz.training_grid(run)` | Several layouts with agents at several training stages | p.4–5 "Agents training on different maze layouts" |
| `mz.scenario("enclosed")` | Flag fully walled in; the only way in is through an outer weak spot | p.6 "Agent facing a hard problem" |
| `mz.scenario("outside_shortcut")` | An inside path exists, but the outside route is available | p.7 "Agent taking outside path…" |
| `mz.agent_view(agent, maze, step)` | What the agent knows: memory, current 5×5 view, compass | Methodology |
| `run.plot()` | Curves over training: average total reward, % reaching flag, steps per attempt, probes, breaks, time outside, % bypass | not in the article, but these mirror the loss/accuracy curves from 05-01/05-02 |
| `mz.reward_breakdown(agent, maze)` | Bar chart of where an attempt's points came from | "reward … outweighed the penalties" (p.6) |
| `mz.evaluate(agent, n=500)` | Table of results on *unseen* layouts | training vs. validation, as in 05-02 |

The two scenario layouts get **transcribed by hand from the PDF figures** (pp. 6–7) as ASCII grids, so the
test cases are the article's own mazes.

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

## 5. Repo layout
```
mazerl/
  __init__.py        public API only
  maze.py            generator, layout bank, ASCII scenarios
  env.py             batched tensor environment (CPU or GPU)
  agent.py           vision + reasoning network
  ppo.py             episode-based PPO, bypass filter, logging
  viz.py             article-style rendering, animations, curves
  pretrained/        *.pt checkpoints + metadata (rewards, settings, attempt counts)
notebooks/
  00-benchmark.ipynb       Colab speed check (Milestone 2)
  01-maze-rl-in-class.ipynb  student notebook, course style (bold questions)
  01-maze-rl-solution.ipynb  instructor key / notes
scripts/pretrain.py        full-scale runs + sweep
tests/                     pytest
pyproject.toml             pip-installable from GitHub
```

---

## 6. Milestones (I build and debug; you review at each ✔)
1. **World:** generator, environment mechanics, article-style static rendering. Tests for: probe/break
   rules, outside penalty, line-of-sight occlusion, compass, bypass detection, every maze solvable.
   ✔ You compare rendered mazes with the PDF.
2. **Benchmark notebook.** ✔ You run it on Colab CPU and on a T4. I update §3 with real numbers and set
   the class time budgets.
3. **PPO training + curves.** Short runs to tune default reward values until the base agent solves unseen
   layouts reliably. Final values recorded in §1.
4. **Full pretraining** (`base` + stage checkpoints).
5. **Article scenarios.** Transcribe pp. 6–7 layouts and evaluate `base` on them. ✔ I report honestly
   whether the emergent behaviors show up and how often (over many attempts). If they don't, adjust the
   *training distribution or rewards*, never script the behavior, and document what changed.
6. **Reward sweep** pretrained agents.
7. **Student notebook + instructor key**, in the 05-0x style.
8. **End-to-end Colab test** from a fresh runtime. ✔ Final review.

---

## 7. Risks
- **The emergent "break back in" may not reproduce at our scale and settings.** Milestone 5 measures it.
  It's reported with a frequency, for example "in 37% of attempts," not cherry-picked. A model that
  sometimes does it is also a good discussion point.
- **Colab T4 availability and quotas.** The precomputed sweep means the notebook still works without
  training.
- **Training time may differ from the estimates.** Milestone 2 settles it before any pretraining starts.
- **Our [D] choices aren't the article's.** The student notebook says plainly which rules come from the
  article and which are ours.

## Git
All work goes on the `maze-rl` branch, pushed to `origin` after each milestone (and more often mid-milestone)
so you can switch machines. Merge to `main` after the final review.
