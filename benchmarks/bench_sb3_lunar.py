"""
SB3 PPO benchmark on LunarLander-v3 (discrete).
100 epochs × 1 000 steps = 100 000 total steps.
Hyperparameters mirror the GridWorld benchmark where applicable.
"""
import time, sys, warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

NUM_EPOCHS   = 100
N_STEPS      = 1_000   # steps per rollout (≈5 episodes at avg 200 steps each)
MAX_STEPS    = 1_000   # TimeLimit wrapper default for LunarLander

class EpochLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.ep_lens, self.ep_rets = [], []
        self._ep_len, self._ep_ret = 0, 0.0

    def _on_step(self):
        self._ep_len += 1
        self._ep_ret += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.ep_lens.append(self._ep_len)
            self.ep_rets.append(self._ep_ret)
            self._ep_len, self._ep_ret = 0, 0.0
        return True

    def _on_rollout_end(self):
        self.epoch += 1
        avg_len = sum(self.ep_lens) / len(self.ep_lens) if self.ep_lens else float("nan")
        avg_ret = sum(self.ep_rets) / len(self.ep_rets) if self.ep_rets else float("nan")
        if self.epoch % 10 == 0 or self.epoch == 1:
            print(f"Epoch {self.epoch:3d}/{NUM_EPOCHS}  EpLen={avg_len:.1f}  EpRet={avg_ret:.1f}")
        self.ep_lens.clear()
        self.ep_rets.clear()

def main():
    env = make_vec_env("LunarLander-v3", n_envs=1)
    model = PPO(
        "MlpPolicy", env,
        n_steps=N_STEPS,
        batch_size=N_STEPS,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=0,
        device="cpu",
    )
    cb = EpochLogger()
    total_steps = NUM_EPOCHS * N_STEPS

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)
    wall = time.perf_counter() - t0

    sps = total_steps / wall
    print(f"\n{'='*50}")
    print(f"SB3 PPO — LunarLander-v3, {NUM_EPOCHS} epochs")
    print(f"Wall time  : {wall:.2f}s")
    print(f"Total steps: {total_steps:,}")
    print(f"Steps/sec  : {sps:,.0f}")
    print(f"µs/step    : {1e6/sps:.1f}")

if __name__ == "__main__":
    main()
