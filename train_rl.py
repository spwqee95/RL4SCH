import argparse, time
from stable_baselines3 import PPO
from scheduler_env import SchedulerEnv
from network import MLPExtractor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps-per-update', type=int, default=2048,
                   help='transitions collected per PPO update')
    p.add_argument('--max-env-steps', type=int, default=10_000_000,
                   help='total environment steps to run')
    p.add_argument('--save-name', type=str, default='ppo_box_rl')
    p.add_argument('--resume-from', type=str, default=None,
                   help='checkpoint path to resume')
    return p.parse_args()

def main():
    args = parse_args()
    env  = SchedulerEnv()

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
        print(f"🔁  Resume from {args.resume_from}")
    else:
        model = PPO(
            "MlpPolicy", env,
            policy_kwargs=dict(features_extractor_class=MLPExtractor),
            n_steps=args.steps_per_update,          # buffer size inside SB3
            batch_size=256, learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95,
            verbose=1
        )

    obs, _ = env.reset()
    buf_cnt = 0
    total_steps = 0
    tic = time.time()

    while total_steps < args.max_env_steps:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, _ = env.step(action)

        buf_cnt   += 1
        total_steps += 1

        # --- mini-batch update ---
        if buf_cnt >= args.steps_per_update:
            model.learn(total_timesteps=args.steps_per_update,
                        reset_num_timesteps=False)
            buf_cnt = 0
            elapsed = time.time() - tic
            print(f"🧠  learn @ {total_steps:,} steps  |  elapsed {elapsed/60:.1f} min")

        # episode boundary (finish or fail)
        if done:
            obs, _ = env.reset()

    env.close()
    model.save(args.save_name)
    print(f"Saved model to {args.save_name}")

if __name__ == "__main__":
    main()
