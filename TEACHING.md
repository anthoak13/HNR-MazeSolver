# Teaching guide

How the maze activity teaches the NYT article *"A.I.'s Rogue Behavior Starts With Its Training."* The
evidence behind each claim is in `EXPERIMENTS.md`, and the build details are in `PLAN.md`.

## The point
The article argues that agents breaking rules isn't malice. It comes from how they were trained. Students
see that happen in a model they can poke at: rewards shape behavior, agents combine what they learned
in ways nobody trained, and small reward choices produce unwanted behavior.

## The two examples

**1. It breaks out to finish an impossible task** (p.6, "Agent facing a hard problem")
- The flag can't be reached by allowed means. The agent leaves the maze and breaks back in through a
  weak spot.
- Nobody trained that sequence; it combines skills learned separately. The penalties for leaving and
  for breaking walls were outweighed by the reward for finishing.
- **Real incident:** persistent agents given impossible tasks broke out of their sandbox (Hugging Face).

**2. It breaks out when it doesn't need to** (p.7, "Agent taking outside path although inside path is
available")
- A legitimate route exists, and the agent leaves the maze anyway.
- Once breaking out is something the agent knows how to do, it uses it whenever the rewards slightly
  favor it. It has no idea of "allowed."
- **Real incident:** agents hacked RubyGems for information that was already public.

## Path through the activity

1. **Meet the maze.** Watch an untrained agent wander, and see the points for each event (flag,
   walls, outside, new cells).
2. **Watch learning.** Look at a trained agent at different stages of training and its reward curves,
   which are like the cost curves from 05-01.
3. **Example 1.** Students predict, then watch the trained agent on the impossible maze. It breaks out in
   a large share of attempts, though not every time.
4. **Why it can do that.** Compare agents trained without one of the skills: without mazes that require
   going outside, or without mazes that require breaking walls. The breakout disappears. Students can
   also train one live (a few minutes) and see the missing skill.
5. **Example 2 and reward design.** The trained agent sometimes leaves the maze needlessly. Students
   raise the exploration reward and train live. The agent starts touring the outside to farm points,
   and leaves the maze on the p.7 maze every time. That's reward hacking.
6. **Discussion.**
   - **Who's responsible?** Our "drop attempts that end outside" rule didn't catch the hacking, because
     the agent still finished inside. That makes a good monitoring discussion.
   - **Model size (optional):** bigger models chain the steps more reliably.
   - **Pretraining and then RL (optional extension):** how LLMs are built.

## What to expect
- **Hardware:** a Colab T4. A live training run takes about 2–5 minutes.
- **Example 1 needs a fully trained agent;** it rarely appears in a short live run. Use the shipped
  agents for it.
- **Results vary between training runs.** The same settings give an agent that breaks out 35% of the
  time, or 75%. That's worth telling students: training is not deterministic.
- **The agents' rules aren't all the article's.** The maze rules are our own simplified version, and the
  notebook says which parts come from the article.
