# env.py
import subprocess, json, numpy as np, gymnasium as gym
from gymnasium import spaces

BIN_PATH = "xxxxxx"

# observations
STATE_DIM = 13

class SchedulerEnv(gym.Env):
    """Gym wrapper around C++ scheduler (one-step-at-a-time REPL)."""
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # action：0=no，1=relocate
        self.action_space = spaces.Discrete(2)
        # continuous vector (-∞, ∞), length = STATE_DIM
        self.observation_space = spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)

        self.proc = None
        self._start_proc()
        self._pending_obs = None              # the latest state from sch

    # ---------- process man ----------
    def _start_proc(self):
        if self.proc and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            [BIN_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1
        )

    def _read_json_line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("scheduler terminated unexpectedly")
        return json.loads(line)

    def _write_action(self, action_int: int):
        msg = {"action": int(action_int)}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        # scheduler will send the first state
        self._start_proc()
        self._pending_obs = self._read_json_line()   # {"state":[...], "delta":..., ...}
        obs = np.array(self._pending_obs["state"], np.float32)
        return obs, {}

    def step(self, action):
        # 1) provide action to sch
        self._write_action(action)

        # 2) read next line's response
        reply = self._read_json_line()
        done   = bool(reply.get("done", False))
        reward = float(reply["reward"])              # r_t = -ΔStep provided by sch
        obs    = np.array(reply["state"], np.float32)

        info = {}
        if done:
            info["terminal_metric"] = reply.get("final_max_step", None)
        return obs, reward, done, False, info

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate(); self.proc.wait()
