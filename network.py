from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PolicyNet"]


class PolicyNet(nn.Module):
    """Simple MLP‑based actor‑critic.

    * **Actor**  – categorical distribution over 2 actions.
    * **Critic** – scalar state‑value estimate.
    """

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.policy_head = nn.Linear(hidden, 2)
        self.value_head = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def forward(self, x):
        """Returns (logits, state_value)."""
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        """Sample an action and return (action, log_prob, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs, actions):
        """Return log_probs, entropy, state_values for given *batched* inputs."""
        logits, state_values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, state_values
