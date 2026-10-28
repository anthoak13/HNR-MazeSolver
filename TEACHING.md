# Teaching guide

How the maze activity is meant to work in class. Students will have read the NYT article *"A.I.'s Rogue
Behavior Starts With Its Training"* beforehand. The evidence behind each claim is in `EXPERIMENTS.md`, and
the build details are in `PLAN.md`.

## Goals
The activity has two halves:

1. **Rogue behavior comes from training** (the article). Students reproduce the article's two examples
   and see how rewards and training experience cause them.
2. **Pretraining, then RL for different tasks** (how LLMs are built). One model learns to "see" mazes
   with no rewards at all. Then RL turns that same model into agents for different jobs. *This half
   depends on an experiment that hasn't been run yet* (`PLAN.md` §10). It stays only if RL on top of the
   pretrained model clearly works for more than one task within class time.

## The article's two examples

**1. It breaks out to finish an impossible task** (p.6, "Agent facing a hard problem")
- The flag can't be reached by allowed means. The agent leaves the maze and breaks back in through a
  weak spot.
- Nobody trained that sequence; it combines skills learned separately. The reward for finishing
  outweighed the penalties.
- **Real incident:** Hugging Face.

**2. It breaks out when it doesn't need to** (p.7, "Agent taking outside path although inside path is
available")
- A legitimate route exists, and the agent leaves anyway. Once breaking out is something it knows how to
  do, it uses it whenever the rewards slightly favor it.
- **Real incident:** RubyGems.

## Path through the activity

**Half 1: rogue behavior from training**
1. **The article's model, in code.** What the agent sees (a 5×5 view, memory, a compass), the points for
   each event, and an untrained agent moving at random.
2. **Watch training.** A trained agent at different stages of training, and its reward curves, which are
   like the cost curves from 05-01.
3. **Example 1.** Predict, then watch the trained agent on the impossible maze. It breaks out in a large
   share of attempts, though not every time.
4. **Why it can.** Agents trained without mazes that require going outside, or without mazes that require
   breaking walls, never break out. Students can train one of these live (a few minutes) and see the
   missing skill.
5. **Example 2 and reward design.** Raise the exploration reward and train live. The agent tours the
   outside farming points, and on the p.7 maze it leaves every time. That's reward hacking.
6. **Discussion.** Our filter that drops attempts ending outside didn't catch the hacking: monitoring
   outcomes misses an agent gaming the process. Also responsibility, and sandbox design. Optional: bigger
   models chain the steps more reliably.

**Half 2: one pretrained model, several jobs** (if the experiment supports it)
1. **Pretraining.** The model learns to predict unseen parts of the maze and where unexplored ground is.
   No rewards are involved. This is the analog of next-word prediction.
2. **RL on top, task 1:** reach the flag. Compare it with an agent trained from scratch.
3. **RL on top, task 2,** for example escaping the maze, from the *same* pretrained model.
4. **Discussion:** why build models this way. Rewards decide what the model is *for*; pretraining
   decides what it *understands*.

## What to expect
- **Hardware:** a Colab T4. A live training run takes about 2–5 minutes.
- **Example 1 needs a fully trained agent;** it rarely appears in a short live run. Use the shipped
  agents for it.
- **Results vary between training runs:** the same settings give an agent that breaks out 35% of the
  time, or 75%. That's worth telling students: training is not deterministic.
- **The maze rules are our simplified version** of the article's model, and the notebook says which parts
  are ours.
