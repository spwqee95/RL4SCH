# net.py
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self._features_dim = 64

    def forward(self, obs):
        return self.net(obs)
