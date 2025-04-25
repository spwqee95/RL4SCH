import subprocess, json, numpy as np, gymnasium as gym
from gymnasium import spaces
from typing import Tuple

STATE_DIM = 13
EXEC      = "xxxx"

# Debug message 
DEBUG = False

class SchedulerEnv(gym.Env):
    """Gym wrapper <-> line-based JSON REPL sch"""
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(STATE_DIM,), dtype=np.float32)
        self.proc = None
        self.step_counter = 0
        self.episode_counter = 0

    def _spawn(self):
        self.episode_counter += 1
        if DEBUG:
            print(f"\n[Env] New Episode #{self.episode_counter} → Spawning Sch")
        self.proc = subprocess.Popen(
            [EXEC], text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("[Env] ERROR: Scheduler closed pipe or has no output.")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if isinstance(msg, dict) and "state" in msg:
                    return msg
            except json.JSONDecodeError:
                pass

            if DEBUG:
                print(f"[Env] Ignored non-JSON line: {line}")

    def _send_action(self, a: int):
        self.proc.stdin.write(json.dumps({"action": int(a)}) + "\n")
        self.proc.stdin.flush()
        if DEBUG:
            print(f"[Env] Sent action: {a}")

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        self._spawn()
        self.step_counter = 0
        first = self._read()
        return np.array(first["state"], np.float32), {}

    def step(self, action: int):
        self._send_action(action)
        reply = self._read()
        obs   = np.array(reply["state"], np.float32)
        rew   = float(reply["reward"])
        done  = bool(reply["done"])
        info  = {"final_max_step": reply.get("final_max_step")} if done else {}

        self.step_counter += 1

        if done and DEBUG:
            print(f"[Env] Episode #{self.episode_counter} complete after {self.step_counter} steps | Final MaxStep={info.get('final_max_step', '?'):.2f}")

        return obs, rew, done, False, info

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate(); self.proc.wait()
