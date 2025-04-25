# train_ppo.py
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import MlpExtractor
from scheduler_env import SchedulerEnv, STATE_DIM
from net import MLPExtractor

def main():
    env = SchedulerEnv()

    policy_kwargs = dict(
        features_extractor_class=MLPExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=[dict(pi=[64, 32], vf=[64, 32])]
    )

    # γ should not too small. Ensure terminal can get the reward. learning_rate can be adjust later
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=1024, batch_size=256,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1, tensorboard_log="./tb"
    )

    model.learn(total_timesteps=200_000)
    model.save("ppo_scheduler")

    env.close()

if __name__ == "__main__":
    main()
