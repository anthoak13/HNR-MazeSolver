# What we've learned

What a small maze-solving RL agent, modeled on the one in the New York Times article *"A.I.'s Rogue
Behavior Starts With Its Training,"* shows, and what that means for a lesson on how RL shapes agents such
as LLMs. The evidence behind each statement is in `EXPERIMENTS.md`.

## The setup
- **The agent:** a CNN plus fully connected layers, the article's size (about 230,000 parameters). It sees
  a 5 × 5 window around itself, remembers everything it has seen during the attempt, and gets an 8-way
  compass direction to the flag.
- **Rewards:**

  | Event | Reward |
  |---|---|
  | Reaching the flag (ends the attempt) | +5 |
  | Each move that ends outside the maze | −0.01 |
  | Walking into a wall | −0.01 |
  | Visiting a square for the first time | +0.005 |

- **Weak spots:** some walls give way when the agent walks into them. They look like solid walls, so the
  agent can only find them by trying.
- **Training:** actor-critic, 126,000 attempts over 13,148 layouts, as in the article. Three kinds of
  layout, in equal numbers:
  - open mazes, solvable without breaking anything;
  - mazes where the only route to the flag goes through a weak spot;
  - mazes where the only route goes out the entrance and back in through another gap.
- **Attempts that end outside the maze are excluded from training,** following the article.

## What the trained agent does
- **It generalizes.** Success on new layouts matches success on training layouts:

  | Kind of maze | Solved |
  |---|---|
  | Open | 99% |
  | Through a weak spot | 91–94% |
  | Outside and back in | 98% |

- **On the article's p.6 maze, where the flag is walled off from the start, it breaks out to finish.** It
  leaves the maze, breaks back in through weak spots, and reaches the flag in 35–75% of attempts.
  - No training layout requires that sequence. It combines two separately learned skills: leaving and
    re-entering the maze, and breaking through weak spots.
  - Both skills are necessary. Training without the layouts that teach either one drops the success rate
    on this maze below 1%.
- **On the article's p.7 maze, which has an inside route, it still leaves the maze in a quarter to a half
  of attempts.**
  - The outside penalty sets the rate. At −0.05 per outside move the agent almost never leaves (0–6%);
    with outside moves free it leaves in up to 99% of attempts. The exploration reward has no clear
    effect on it.
  - That fits the article's account: when the rewards favor the outside route, the agent takes it.

## How the rewards shape behavior
Each reward was varied on its own (exploration 0 or +0.05; wall −0.002 or −0.05; outside 0 or −0.05),
three seeds each.
- **Success on the trained kinds of maze barely moves.** Every change but one stays within a few points
  of the default. The flag reward dominates every other term, so the agent still learns the task.
  Penalties act as prices, not prohibitions.
- **The rewards change how it gets there:** how many walls it tries, how long it spends outside, and how
  direct its route is. A costly wall penalty (−0.05) makes it try a third fewer walls and solve weak-spot
  mazes just as often, but it takes 15% longer to find the way through.
- **Reward hacking appears wherever a step outside nets positive reward.**
  - At +0.05 per new square (10× the default), new outside squares outweigh the outside penalty. The
    agent tours the outside collecting reward before heading to the flag. Its return rises (7.1 vs 5.0
    on open mazes) while its success on weak-spot mazes falls 5–15 points and it takes about 1.5× as
    long to finish them.
  - With outside moves free, the *default* exploration reward is enough. Each new outside square is
    worth +0.005, and the agent tours the outside about as much as at +0.05.
  - So the size of the reward matters less than whether it opens a net-positive loop. The agent finds
    the loop and exploits it.
  - Not every seed finds it: in each of these two settings, one seed of three didn't.
  - Farming agents almost never reach the p.6 flag (0–2%).
- **Excluding attempts that end outside doesn't catch the hacking.** The hacking agent always returns
  inside before the attempt ends. A check on outcomes misses an agent that games the process.

## What doesn't matter much
- **Network size.** A network with one-eleventh the parameters learns more slowly. Given enough training
  it matches the full-size network, both on ordinary mazes and on the p.6 maze. Size changes learning
  speed, not the behavior learned. All work uses the article's size and its 126,000 attempts.

## Variability
- **Identical settings give quite different agents.** Two seeds broke out on the p.6 maze in 36% and 74%
  of attempts. Re-measuring one agent varies by about 1 percentage point on the trained kinds and by up
  to 6 on the p.6 maze. Comparisons between settings need several seeds, and the p.6 rate is too noisy to
  rank most reward settings.
- **The rules are a simplification** of the article's model. The flag reward and the outside penalty come
  from the article's author; the other values are ours.

## Pretraining
Details in `reports/pretraining-transfer.md` (20k-attempt runs on CPU, 2–4 seeds per setting).
- **Reward-free pretraining** of the perception layers, predicting hidden parts of the maze and the
  direction to unexplored ground, is the maze analog of next-token prediction. It predicts unseen cells
  86% of the time (53–55% by always guessing the most common cell) on mazes it never trained on.
- **It builds something RL can use.** With the perception layers frozen and only the last layers (about
  10k of 230k parameters) trained by RL, pretrained perception reaches 97% on open mazes in an escape task
  and 62–75% on open mazes in the flag task. Random perception stays at chance on both.
- **The same pretrained perception serves two tasks:** reaching the flag, and escaping the maze from where
  the flag would be (no compass, first step outside wins). The reward decides which agent RL makes.
- **It doesn't make RL meaningfully faster.** When the whole network trains, pretraining gives escape a
  small, consistent head start (every pretrained run beat every from-scratch run, by 1.5–6.5 points of
  training success averaged over the run) and the same final success. On the flag task it gives no
  reliable gain: about 1 in 4 pretrained runs stalls, because the untrained agent strongly favors one move,
  attempts that leave the maze are excluded from training, and the agent is starved of data.
- **It isn't uniquely general here.** Perception taken from an agent trained by RL on the flag task gives
  escape a bigger head start than reward-free pretraining does.

## What this means for the lesson
- **The agent learns whatever the rewards pay for.** Every behavior, wanted or not, traces back to the
  rewards and the training experience.
- **Separately learned skills combine in situations no one tested.** No one trained the p.6 behavior; it
  follows from two ordinary skills plus a large reward for finishing.
- **Penalties are prices.** A big enough reward makes the agent pay them.
- **A reward that opens a net-positive loop gets gamed,** however small it is, and checking only outcomes
  doesn't reveal it.
- **Rewards shape the route more than the result.** Changing them leaves success nearly unchanged but
  changes how the agent behaves on the way: what it tries, where it goes, how efficient it is.
- **Training outcomes vary.** The same recipe yields agents with noticeably different habits, which is
  part of why testing agents is hard.
