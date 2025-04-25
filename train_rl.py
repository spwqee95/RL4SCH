import argparse, numpy as np
from stable_baselines3 import PPO
from scheduler_env import SchedulerEnv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steps-per-update', type=int, default=2048,
                   help='transitions per PPO update')
    p.add_argument('--max-env-steps',   type=int, default=10_000_000,
                   help='total env steps to collect')
    p.add_argument('--save-name',       type=str, default='ppo_box_rl')
    p.add_argument('--resume-from',     type=str, default=None,
                   help='load existing model checkpoint')
    return p.parse_args()

def main():
    args = parse_args()
    env  = SchedulerEnv()
    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
        print(f"🔁 resume from {args.resume_from}")
    else:
        model = PPO("MlpPolicy", env,
                    n_steps=args.steps_per_update,
                    batch_size=256, learning_rate=3e-4, verbose=1)

    obs, _ = env.reset()
    step_cnt = buf_cnt = 0

    while step_cnt < args.max_env_steps:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, info = env.step(action)

        step_cnt += 1; buf_cnt += 1

        if buf_cnt >= args.steps_per_update:
            model.learn(total_timesteps=args.steps_per_update,
                        reset_num_timesteps=False)
            buf_cnt = 0
            print(f"PPO update @ {step_cnt} env-steps")

        if done:
            obs, _ = env.reset()

    env.close()
    model.save(args.save_name)
    print(f"model saved as {args.save_name}")

if __name__ == "__main__":
    main()
