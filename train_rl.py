"""Training script

Usage example
-------------
$ python train_rl.py --episodes 200 \
                     --steps-per-update 2048 \
                     --scheduler-path ./fake_scheduler \
                     --save-name a2c_fclk_rl

After each episode the script prints reward statistics and after training
it writes *<save‑name>.pt* (model parameters) and *learning_curve.png*.
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from scheduler_env import SchedulerEnv
from network import PolicyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Rollout buffer helpers
# ----------------------------------------------------------------------
class Buffer:
    def __init__(self):
        self.obs: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def clear(self):
        self.__init__()

    def to_tensor(self):
        obs = torch.stack(self.obs).to(DEVICE)
        actions = torch.stack(self.actions).to(DEVICE)
        logprobs = torch.stack(self.logprobs).to(DEVICE)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=DEVICE)
        values = torch.stack(self.values).to(DEVICE)
        return obs, actions, logprobs, rewards, dones, values


# ----------------------------------------------------------------------
# A2C update
# ----------------------------------------------------------------------

def a2c_update(model: PolicyNet, optimizer: Adam, buffer: Buffer,
               gamma: float = 0.99, vf_coef: float = 0.5,
               ent_coef: float = 0.01):
    obs, actions, old_logprobs, rewards, dones, values = buffer.to_tensor()

    # Compute returns (bootstrapped GAE‑λ with λ=1 → REINFORCE‑with‑baseline)
    returns = torch.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G * (1.0 - dones[t])
        returns[t] = G
    returns = returns.detach()
    advantages = returns - values.squeeze()

    # New log‑probs & state values
    log_probs, entropy, state_values = model.evaluate_actions(obs, actions)

    policy_loss = -(advantages.detach() * log_probs).mean()
    value_loss = nn.functional.mse_loss(state_values.squeeze(), returns)
    entropy_loss = -entropy.mean()

    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train(args):
    env = SchedulerEnv(args.scheduler_path)

    model = PolicyNet(obs_dim=13).to(DEVICE)
    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from, map_location=DEVICE))
        print(f"[INFO] Resumed from {args.resume_from}")

    optimizer = Adam(model.parameters(), lr=3e-4)

    episode_rewards: List[float] = []
    relocation_freq: List[float] = []

    buffer = Buffer()
    global_step = 0

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        relocations = 0

        while not done:
            obs_t = torch.from_numpy(obs).float().to(DEVICE)
            action, log_prob, value = model.act(obs_t)

            next_obs, reward, done, info = env.step(action.item())

            # store
            buffer.obs.append(obs_t.cpu())
            buffer.actions.append(action.cpu())
            buffer.logprobs.append(log_prob.cpu())
            buffer.values.append(value.cpu())
            buffer.rewards.append(reward)
            buffer.dones.append(done)

            ep_reward += reward
            relocations += int(action.item())
            obs = next_obs
            global_step += 1

            # Optimise when enough samples collected or at episode end
            if len(buffer.rewards) >= args.steps_per_update or done:
                a2c_update(model, optimizer, buffer)
                buffer.clear()

        episode_rewards.append(ep_reward)
        relocation_freq.append(relocations)
        print(f"Episode {ep:>4}/{args.episodes} | reward = {ep_reward:8.2f} | reloc = {relocations:4d}")

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    tag = args.save_name or f"model_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = Path(f"{tag}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model parameters saved to {model_path}")

    # Learning curve
    plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(episode_rewards)
    plt.title("Episode return")
    plt.subplot(212)
    plt.plot(relocation_freq)
    plt.title("Relocation count per episode")
    plt.tight_layout()
    curve_path = Path("learning_curve.png")
    plt.savefig(curve_path)
    print(f"[INFO] Learning curve saved to {curve_path}")

    env.close()


# ----------------------------------------------------------------------
# Entry‑point
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train A2C on the scheduler environment")
    p.add_argument("--episodes", type=int, default=100, help="Total training episodes (scheduler runs)")
    p.add_argument("--steps-per-update", type=int, default=2048, help="Batch size before each optimiser step")
    p.add_argument("--scheduler-path", type=str, default="./ecompile", help="Path to scheduler binary")
    p.add_argument("--save-name", type=str, default="", help="Prefix for saved model file (.pt)")
    p.add_argument("--resume-from", type=str, help="Path to .pt file to resume training from", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
