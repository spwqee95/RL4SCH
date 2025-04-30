"""Gym‑style environment that wraps the fake scheduler binary.
The scheduler communicates via JSON lines on stdin/stdout.
Each reset() spawns a **new** process so that one complete traversal of the
netlist is exactly one RL episode.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import os

import gym
import numpy as np
from gym import spaces

__all__ = ["SchedulerEnv"]


class SchedulerEnv(gym.Env):
    """OpenAI‑Gym compatible environment.

    Observation  : 13‑dim float32 vector
    Action space : Discrete(2) – 0 = keep, 1 = relocate
    Reward       : shaped (‑Δstep) each decision plus terminal (‑final_max_step)
    Episode ends : scheduler prints a JSON line with "done": true
    """

    metadata: Dict[str, Any] = {"render.modes": []}

    def __init__(self,
                 scheduler_path: str = "xxxx",
                 timeout: float = 2.0,
                 seed: Optional[int] = None):
        super().__init__()
        self.scheduler_path = os.path.abspath(scheduler_path)
        self.timeout = timeout

        # Gym spaces
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(13,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._rng = np.random.RandomState(seed)
        self.proc: Optional[subprocess.Popen[str]] = None
        self._last_state: np.ndarray | None = None
        self._terminal_bonus: float = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> np.ndarray:  # type: ignore[override]
        if self.proc is not None and self.proc.poll() is None:
            # Make sure the previous process is gone.
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._spawn_scheduler()
        data = self._await_state_line()
        assert data is not None, "Scheduler exited prematurely on reset()"
        self._terminal_bonus = 0.0
        state = np.asarray(data["state"], dtype=np.float32)
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:  # type: ignore[override]
        if self.proc is None:
            raise RuntimeError("Environment not yet reset()")
        if self.proc.poll() is not None:
            raise RuntimeError("Scheduler process already exited!")

        # --- send chosen action to scheduler ---
        msg = json.dumps({"action": int(action)}) + "\n"
        try:
            self.proc.stdin.write(msg)  # type: ignore
            self.proc.stdin.flush()  # type: ignore
        except BrokenPipeError as exc:
            raise RuntimeError("Scheduler pipe closed while sending action") from exc

        # --- read next state ---
        data = self._await_state_line()
        if data is None:
            # Scheduler quit without a terminal JSON → treat as done
            done = True
            reward = 0.0
            state = np.zeros(13, dtype=np.float32)
            info = {"abnormal_exit": True}
            return state, reward, done, info

        reward: float = data["reward"] + self._terminal_bonus  # shaped or terminal
        done: bool = data["done"]
        state: np.ndarray = np.asarray(data["state"], dtype=np.float32)
        info: Dict[str, Any] = {}

        if done:
            # make sure we harvest the terminal reward exactly once
            self._terminal_bonus = 0.0
            if "final_max_step" in data:
                reward += -float(data["final_max_step"])
                info["final_max_step"] = float(data["final_max_step"])
            # stop the child process – avoid zombies
            self._graceful_terminate()

        return state, reward, done, info

    def close(self):  # type: ignore[override]
        self._graceful_terminate()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spawn_scheduler(self) -> None:
        print(f"[DEBUG] launching scheduler: {self.scheduler_path}")
        self.proc = subprocess.Popen([self.scheduler_path],
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 text=True,
                                 bufsize=1)

    def _await_state_line(self) -> Optional[Dict[str, Any]]:
        """Read one JSON state line; skip unrelated stdout.  Returns parsed dict."""
        assert self.proc is not None and self.proc.stdout is not None
        start_time = time.time()
        while True:
            if (time.time() - start_time) > self.timeout:
                return None  # timeout – treat as crash
            line = self.proc.stdout.readline()
            if line == "":
                # EOF → scheduler exited
                return None
            line = line.strip()
            if not line:
                continue
            # Normal decision lines are prefixed with "[RL] "
            if line.startswith("[RL]"):
                line = line[4:].lstrip()
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip unrelated output
                continue
            return data

    def _graceful_terminate(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
