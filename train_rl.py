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
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Sch crashed or closed pipe unexpectedly.")
        msg = json.loads(line)

        if DEBUG:
            box_index = int(msg['state'][0]) if len(msg['state']) > 0 else -1
            delta     = msg['state'][8]
            max_step  = msg['state'][10]
            print(f"[Env] BOX {box_index:5d} | ΔStep={delta:+.4f} | MaxStep={max_step:.1f}")

        return msg

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
