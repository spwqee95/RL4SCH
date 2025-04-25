import subprocess, json, numpy as np, gymnasium as gym
from gymnasium import spaces
from typing import Tuple

STATE_DIM = 13       
EXEC      = "xxx" 

class SchedulerEnv(gym.Env):
    """Gym wrapper  ↔  line-based JSON REPL scheduler"""
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space      = spaces.Discrete(2)          # 0/1 binary
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(STATE_DIM,), dtype=np.float32)
        self.proc = None

    # ---------- IPC ----------
    def _spawn(self):
        self.proc = subprocess.Popen(
            [EXEC], text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Scheduler crashed or closed pipe")
        return json.loads(line)

    def _send_action(self, a: int):
        self.proc.stdin.write(json.dumps({"action": int(a)}) + "\n")
        self.proc.stdin.flush()

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        self._spawn()
        first = self._read()                       # initial state
        return np.array(first["state"], np.float32), {}

    def step(self, action: int):
        self._send_action(action)
        reply = self._read()                       # next state + reward
        obs   = np.array(reply["state"], np.float32)
        rew   = float(reply["reward"])
        done  = bool(reply["done"])
        info  = {"final_max_step": reply.get("final_max_step")} if done else {}
        return obs, rew, done, False, info

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate(); self.proc.wait()
