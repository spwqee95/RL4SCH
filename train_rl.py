import argparse, time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from scheduler_env import SchedulerEnv
from network import MLPExtractor

DEBUG = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=100,
                   help='Number of full netlist episodes to train')
    p.add_argument('--steps-per-update', type=int, default=2048,
                   help='Transitions collected per PPO update')
    p.add_argument('--save-name', type=str, default='ppo_box_rl')
    p.add_argument('--resume-from', type=str, default=None,
                   help='Checkpoint path to resume training')
    return p.parse_args()

def main():
    args = parse_args()
    env  = SchedulerEnv()

    # Initialize or resume PPO model
    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
        print(f"Resume from {args.resume_from}")
    else:
        model = PPO(
            "MlpPolicy", env,
            policy_kwargs=dict(features_extractor_class=MLPExtractor),
            n_steps=args.steps_per_update,
            batch_size=min(args.steps_per_update, 64),
            learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95,
            verbose=0
        )

    episode_rewards = []
    final_max_steps = []
    episode_lengths = []

    for ep in range(1, args.episodes + 1):
        print(f"\nEpisode {ep}/{args.episodes}")
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps_in_episode = 0
        buffer = 0
        tic = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(action)

            ep_reward += reward
            steps_in_episode += 1
            buffer += 1

            if buffer >= args.steps_per_update:
                model.learn(total_timesteps=args.steps_per_update,
                            reset_num_timesteps=False)
                buffer = 0
                print(f"🧠 PPO updated at {steps_in_episode} steps")

        # Episode done
        max_step = info.get("final_max_step", -1)
        print(f"Episode {ep} done | Return = {ep_reward:.2f} | MaxStep = {max_step} | Steps = {steps_in_episode}")

        episode_rewards.append(ep_reward)
        final_max_steps.append(max_step)
        episode_lengths.append(steps_in_episode)

    env.close()
    model.save(args.save_name)
    print(f"Saved final model to {args.save_name}")

    # Save plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Episode Reward")
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(final_max_steps, label="Final Max Step")
    plt.xlabel("Episode"); plt.ylabel("Final Max Step")
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.savefig("learning_curve.png")
    print("Saved learning_curve.png")

if __name__ == "__main__":
    main()
