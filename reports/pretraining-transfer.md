# Pretraining and transfer with the maze model

A report on whether this maze model can tell the story "pretrain once with no rewards, then use RL
to make it good at a task (or at several tasks)", the recipe behind LLM agents. It maps what the model
can and can't show, with evidence. It doesn't design the notebook.

All runs here are local (desktop i5-13400F, 2 CPU threads per run, four runs side by side), the 1:1:1
maze mix and the default model size. Runs are short (tens of thousands of attempts) and compared only
at equal length.

## Words used here
- **Attempt:** one try at a maze, up to 200 moves.
- **Perception:** the part of the network that turns what the agent sees and remembers into a summary
  (the vision layers plus the first reasoning layer; about 220k of the 230k connections). The rest,
  about 10k connections, **decides** which move to make.
- **Reward-free pretraining:** training the perception on two prediction quizzes, with no points for
  anything: "what's in the cells you haven't seen yet?" and "which move heads toward ground you
  haven't explored?" (`mazerl/perception.py`). This is the stand-in for an LLM's next-word
  prediction.
- **From scratch:** every connection starts random, and RL trains everything.
- **Pretrained (still learning):** perception starts from pretraining; RL trains the whole network,
  so the perception keeps adjusting. **Frozen:** the pretrained perception is locked; RL trains only
  the deciding 10k connections.
- **Flag task:** the article's task. Start at the top-left entrance, reach the flag (+5).
- **Escape task** (new, built for this report): start on the flag's cell, with no flag on the map
  and no compass. Stepping onto any outside cell ends the attempt and earns the +5. Walls and new
  cells score as usual. Built as agreed in `HANDOFF.md`.
- **Seed:** the random number that fixes a run's starting connections and maze order. The same
  settings with a different seed give a somewhat different agent, so every comparison uses 2 seeds.

---

## Design check before any experiment: how hard is escape?

An agent moving at random, 300 new mazes of each kind, 200 moves:

| Maze kind | Flag task, random | Escape, random | Shortest way out from the flag's cell |
|---|---|---|---|
| open | 1% | **52%** | median 7 moves, no wall breaking needed |
| interior_break | 0% | **24%** | median 11 moves, *must* break a wall (the flag's area is sealed inside) |
| outside_only | 0% | **55%** | median 8 moves, no breaking needed |
| enclosed (p.6) | — | 1% | must break a wall |

**Escape is a much easier task than reaching the flag.** The flag sits 1–5 cells from the outer wall,
and the outer wall has 3–5 openings plus weak spots, so a random walker gets out about half the time.
That shapes what transfer can show: there's less to learn, so less room for pretraining to help.
Interior_break mazes are the hard part of escape (a wall has to be broken), so they're where a
difference would show.

---

## Round T1: does the same pretrained perception help a task it wasn't pretrained for?

**Runs** (20k attempts each, 2 seeds each; seed *s* uses pretraining seed *s*):

| Task | From scratch | Pretrained, still learning | Pretrained, frozen |
|---|---|---|---|
| flag | ✓ | ✓ | (round 6: fails; not rerun) |
| escape | ✓ | ✓ | ✓ |

20k attempts is a class-sized budget (about 2.5 min on a Colab T4). Every run is compared with
others of the same length. The learning rate anneals to zero over each run's own 20k attempts.

**How speed is measured:** training success (share of attempts that succeed, averaged over 5 log
points), and the number of attempts until it first passes a threshold: 50% and 80% for the flag task;
80% and 90% for escape (escape starts at about 45% by chance). Final skill: the test suite on 300 new
mazes of each kind.

**Predictions** (written before running):

- **P1, escape from scratch is quick.** At 20k: open and outside_only **≥ 95%**, interior_break
  **≥ 75%**. Average moves to escape on open mazes **≤ 15** (random: about 135).
- **P2, the transfer test.** Pretrained (still learning) reaches 90% training success on escape in
  **20–40% fewer attempts** than from scratch, on both seeds. At 20k, final success is within 3
  points of from-scratch on open and outside_only (both near the ceiling); interior_break is
  **5+ points higher** with pretraining.
  - *Why:* pretraining taught the network to read walls, openings and the outside band, and to infer
    unseen cells; escape needs exactly that. But escape is easy, so the head start can only save
    part of a short learning curve.
- **P3, relative help: escape about the same as flag.** The flag task speed-up (attempts to 80%)
  is 15–35% (round 6, one seed: 80% reached about 25% sooner). The escape speed-up (attempts to 90%)
  is within 15 points of it. If escape's speed-up is much *larger*, that supports "pretraining is
  general"; if it's much smaller, it says the pretraining fitted the flag task's setting (it was done
  on agents starting at the entrance, with a compass and a flag on the map).
- **P4, frozen works better on escape than on flag.** On the flag task, frozen perception plus a
  10k-connection decider failed (round 6: open 78%, interior_break 33%). Escape is simpler, so frozen
  gets open and outside_only **≥ 90%**, but interior_break stays **≥ 10 points below** still-learning.
  - *Why it matters for the story:* if a locked pretrained model plus a tiny trained "decider" can do
    a new task, that's the clearest form of "pretraining did most of the work."

**Results:** (pending)
