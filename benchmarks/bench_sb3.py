"""
SB3 PPO benchmark on Python GridWorld.
Matches RelayRL benchmark: 100 epochs, 8 episodes/epoch, max_steps=200,
policy net [64, 64], same hyperparams where possible.
"""

import time, sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
sys.path.insert(0, "/home/user/RelayRL-end2end/benchmarks")
from gridworld_env import GridWorldEnv

NUM_EPOCHS   = 100
EP_PER_EPOCH = 8         # episodes to collect before each update
MAX_STEPS    = 200
N_STEPS      = EP_PER_EPOCH * MAX_STEPS  # steps per rollout buffer fill

class EpochLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.ep_lens = []
        self.ep_rets = []
        self._ep_len = 0
        self._ep_ret = 0.0

    def _on_step(self):
        self._ep_len += 1
        self._ep_ret += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.ep_lens.append(self._ep_len)
            self.ep_rets.append(self._ep_ret)
            self._ep_len = 0
            self._ep_ret = 0.0
        return True

    def _on_rollout_end(self):
        self.epoch += 1
        if self.ep_lens:
            avg_len = sum(self.ep_lens) / len(self.ep_lens)
            avg_ret = sum(self.ep_rets) / len(self.ep_rets)
        else:
            avg_len = avg_ret = float('nan')
        print(f"Epoch {self.epoch:3d}/{NUM_EPOCHS}  EpLen={avg_len:.1f}  EpRet={avg_ret:.3f}")
        self.ep_lens.clear()
        self.ep_rets.clear()

def main():
    env = make_vec_env(GridWorldEnv, n_envs=1,
                       env_kwargs={"grid_size": 10, "max_steps": MAX_STEPS})

    model = PPO(
        "MlpPolicy", env,
        n_steps=N_STEPS,
        batch_size=N_STEPS,          # single minibatch = full buffer
        n_epochs=10,                  # PPO inner epochs (matches train_pi_iters=80/8)
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
    print(f"SB3 PPO — 1 actor, {NUM_EPOCHS} epochs")
    print(f"Wall time  : {wall:.2f}s")
    print(f"Total steps: {total_steps:,}")
    print(f"Steps/sec  : {sps:,.0f}")
    print(f"µs/step    : {1e6/sps:.1f}")

if __name__ == "__main__":
    main()
