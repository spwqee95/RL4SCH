from __future__ import annotations
import subprocess, json, numpy as np, gymnasium as gym
from gymnasium import spaces
from typing import Tuple

STATE_DIM = 13
EXEC      = "xxxx"      # compiled C++ binary

class SchedulerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(STATE_DIM,), dtype=np.float32)
        self.proc = None

    # ---------- I/O ----------
    def _start_proc(self):
        self.proc = subprocess.Popen(
            [EXEC], text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _read_json(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("scheduler crashed")
        return json.loads(line)

    def _send_action(self, a:int):
        self.proc.stdin.write(json.dumps({"action":a}) + "\n")
        self.proc.stdin.flush()

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        self._start_proc()
        first = self._read_json()
        return np.array(first["state"], np.float32), {}

    def step(self, action: int):
        self._send_action(int(action))
        reply = self._read_json()
        done  = bool(reply["done"])
        obs   = np.array(reply["state"], np.float32)
        rew   = float(reply["reward"])
        info  = {}
        if done:
            info["final_max_step"] = reply.get("final_max_step")
        return obs, rew, done, False, info

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate(); self.proc.wait()
