"""Gym-style environment that wraps the fake binary.
The sch communicates via JSON lines on stdin/stdout.
Each reset() spawns a **new** process so that one complete traversal of the
netlist is exactly one RL episode.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import os
import shlex
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import gym
import numpy as np
from gym import spaces

__all__ = ["SchedulerEnv"]


class SchedulerEnv(gym.Env):
    """OpenAI-Gym compatible environment.

    Observation  : 13-dim float32 vector
    Action space : Discrete(2) – 0 = keep, 1 = relocate
    Reward       : shaped (-Δstep) each decision plus terminal (-final_max_step)
    Episode ends : scheduler prints a JSON line with "done": true
    """

    metadata: Dict[str, Any] = {"render.modes": []}

    def __init__(self,
                 scheduler_path: str = "./ecompile",
                 timeout: float = 2.0,
                 seed: Optional[int] = None):
        super().__init__()
        self.scheduler_path = scheduler_path  # Now accepts full command
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

        msg = json.dumps({"action": int(action)}) + "\n"
        try:
            self.proc.stdin.write(msg)  # type: ignore
            self.proc.stdin.flush()  # type: ignore
        except BrokenPipeError as exc:
            raise RuntimeError("Scheduler pipe closed while sending action") from exc

        data = self._await_state_line()
        if data is None:
            done = True
            reward = 0.0
            state = np.zeros(13, dtype=np.float32)
            info = {"abnormal_exit": True}
            return state, reward, done, info

        reward: float = data["reward"] + self._terminal_bonus
        done: bool = data["done"]
        state: np.ndarray = np.asarray(data["state"], dtype=np.float32)
        info: Dict[str, Any] = {}

        if done:
            self._terminal_bonus = 0.0
            if "final_max_step" in data:
                reward += -float(data["final_max_step"])
                info["final_max_step"] = float(data["final_max_step"])
            self._graceful_terminate()

        return state, reward, done, info

    def close(self):  # type: ignore[override]
        self._graceful_terminate()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spawn_scheduler(self) -> None:
        cmd = shlex.split(self.scheduler_path)
        print(f"[DEBUG] launching scheduler: {' '.join(cmd)}")
        assert Path(cmd[0]).exists(), f"{cmd[0]} not found"
        assert os.access(cmd[0], os.X_OK), f"{cmd[0]} not executable"
        self.proc = subprocess.Popen(cmd,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     text=True,
                                     bufsize=1)

    def _await_state_line(self) -> Optional[Dict[str, Any]]:
        assert self.proc is not None and self.proc.stdout is not None
        start_time = time.time()
        while True:
            if (time.time() - start_time) > self.timeout:
                return None
            line = self.proc.stdout.readline()
            if line == "":
                return None
            line = line.strip()
            if not line:
                continue
            if line.startswith("[RL]"):
                line = line[4:].lstrip()
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
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
