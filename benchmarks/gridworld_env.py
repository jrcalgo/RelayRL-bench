"""
Python GridWorld matching the Rust gridworld-rl crate exactly:
  - 10x10 grid, obs_dim=100 (one-hot cell encoding)
  - 4 discrete actions: 0=Up(-row), 1=Down(+row), 2=Left(-col), 3=Right(+col)
  - Rewards: wall/bounds collision=-1, goal=+10, step=-0.01
  - Episode ends when actor reaches (9,9) or max_steps reached
  - Walls at: (2,1),(2,2),(2,3),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(2,6),(2,7),(2,8)
  - Actor starts at (0,0)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

WALLS = frozenset([
    (2,1),(2,2),(2,3),(2,4),
    (3,4),(4,4),(5,4),(6,4),(7,4),
    (2,6),(2,7),(2,8),
])
GOAL = (9, 9)
DELTAS = [(-1,0),(1,0),(0,-1),(0,1)]  # Up, Down, Left, Right


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size=10, max_steps=200):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(grid_size * grid_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)
        self._pos = (0, 0)
        self._steps = 0

    def _obs(self):
        obs = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        r, c = self._pos
        obs[r * self.grid_size + c] = 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = (0, 0)
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        r, c = self._pos
        dr, dc = DELTAS[action]
        nr, nc = r + dr, c + dc

        # out of bounds
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            reward = -1.0
            # stay in place
        # wall collision
        elif (nr, nc) in WALLS:
            reward = -1.0
            # stay in place
        else:
            self._pos = (nr, nc)
            if self._pos == GOAL:
                reward = 10.0
            else:
                reward = -0.01

        self._steps += 1
        terminated = self._pos == GOAL
        truncated = self._steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}
