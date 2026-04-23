"""
SB3 PPO benchmark on LunarLander-v3 (discrete).
32 vectorized envs × 10 000 total steps.
"""
import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

N_ENVS       = 32
TOTAL_STEPS  = 10_000
N_STEPS      = 128    # steps per env per rollout → 128 × 32 = 4 096 per update


class EpochLogger(BaseCallback):
    def __init__(self, n_envs):
        super().__init__()
        self.n_envs = n_envs
        self.epoch = 0
        self.ep_lens, self.ep_rets = [], []
        self._ep_len = [0] * n_envs
        self._ep_ret = [0.0] * n_envs

    def _on_step(self):
        for i in range(self.n_envs):
            self._ep_len[i] += 1
            self._ep_ret[i] += float(self.locals["rewards"][i])
            if self.locals["dones"][i]:
                self.ep_lens.append(self._ep_len[i])
                self.ep_rets.append(self._ep_ret[i])
                self._ep_len[i] = 0
                self._ep_ret[i] = 0.0
        return True

    def _on_rollout_end(self):
        self.epoch += 1
        avg_len = np.mean(self.ep_lens) if self.ep_lens else float("nan")
        avg_ret = np.mean(self.ep_rets) if self.ep_rets else float("nan")
        print(f"Rollout {self.epoch}  EpLen={avg_len:.1f}  EpRet={avg_ret:.1f}")
        self.ep_lens.clear()
        self.ep_rets.clear()


def main():
    env = make_vec_env("LunarLander-v3", n_envs=N_ENVS)
    model = PPO(
        "MlpPolicy", env,
        n_steps=N_STEPS,
        batch_size=256,
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
    cb = EpochLogger(N_ENVS)

    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb, progress_bar=False)
    wall = time.perf_counter() - t0

    actual_steps = model.num_timesteps
    sps = actual_steps / wall
    print(f"\n{'='*55}")
    print(f"SB3 PPO — LunarLander-v3, {N_ENVS} envs")
    print(f"Wall time  : {wall:.2f}s")
    print(f"Total steps: {actual_steps:,}")
    print(f"Steps/sec  : {sps:,.0f}")
    print(f"µs/step    : {1e6/sps:.1f}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
