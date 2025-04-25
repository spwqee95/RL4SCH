import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, out_dim=64):
        super().__init__(obs_space, out_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_space.shape[0], 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.ReLU()
        )
    def forward(self, obs):
        return self.net(obs)
