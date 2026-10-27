"""The agent's neural network: a "vision" part and a "reasoning" part (as described in the article)."""
import torch
from torch import nn

from .env import OBS_CHANNELS, CROP, N_COMPASS

N_ACTIONS = 4
PERCEPTION = ("vision.", "reasoning.0.")     # the perceiving part; the rest decides


class Agent(nn.Module):
    """Vision: convolutions over the 11 x 11 window of what the agent remembers.
    Reasoning: two hidden layers of 100 neurons that combine vision with the compass and pick a move.
    Outputs move preferences (4) and an estimate of future reward (the "value")."""

    def __init__(self, hidden=100):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(OBS_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        n_vision = 64 * (CROP // 2) ** 2
        self.reasoning = nn.Sequential(
            nn.Linear(n_vision + N_COMPASS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, N_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def perceive(self, obs, compass):
        """The perceiving part: vision plus the first reasoning layer (100 features)."""
        return self.reasoning[1](self.reasoning[0](torch.cat([self.vision(obs), compass], 1)))

    def forward(self, obs, compass):
        z = self.reasoning[3](self.reasoning[2](self.perceive(obs, compass)))
        return self.policy(z), self.value(z).squeeze(1)

    def perception_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith(PERCEPTION)}

    def load_perception(self, state):
        """Copy in pretrained perception weights; the deciding layers are left as they are."""
        if isinstance(state, Agent):
            state = state.perception_state()
        missing = [k for k in self.state_dict() if k.startswith(PERCEPTION) and k not in state]
        if missing:
            raise ValueError(f"perception weights missing: {missing}")
        self.load_state_dict({**self.state_dict(), **state})

    def freeze_perception(self, frozen=True):
        for name, p in self.named_parameters():
            if name.startswith(PERCEPTION):
                p.requires_grad_(not frozen)

    def act(self, obs, compass, greedy=False):
        logits, value = self(obs, compass)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(1) if greedy else dist.sample()
        return action, dist.log_prob(action), value

    @property
    def n_connections(self):
        return sum(p.numel() for p in self.parameters())
