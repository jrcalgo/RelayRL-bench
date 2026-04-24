"""
SB3 PPO benchmark on LunarLander-v3 — 1 env, no batching.
1 000 epochs × 1 000 steps = 1 000 000 total steps.
Matches the RelayRL scalar single-env benchmark step count.
"""
import time, warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

NUM_EPOCHS = 1_000
N_STEPS    = 1_000   # steps per rollout
TOTAL      = NUM_EPOCHS * N_STEPS  # 1 000 000

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
        if self.epoch % 100 == 0 or self.epoch == 1:
            avg_len = sum(self.ep_lens) / len(self.ep_lens) if self.ep_lens else float("nan")
            avg_ret = sum(self.ep_rets) / len(self.ep_rets) if self.ep_rets else float("nan")
            elapsed = time.perf_counter() - self._t0
            sps = (self.epoch * N_STEPS) / elapsed
            print(f"  [{self.epoch * N_STEPS:>8,} steps]  EpLen={avg_len:.1f}  EpRet={avg_ret:.1f}  {sps:,.0f} steps/sec")
            self.ep_lens.clear()
            self.ep_rets.clear()

    def set_t0(self, t):
        self._t0 = t

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

    print("=" * 60)
    print("  SB3 PPO — LunarLander-v3 — 1 env — 1 000 000 steps")
    print("=" * 60)

    cb = EpochLogger()
    t0 = time.perf_counter()
    cb.set_t0(t0)

    model.learn(total_timesteps=TOTAL, callback=cb, progress_bar=False)
    wall = time.perf_counter() - t0

    sps = TOTAL / wall

    # Collect final episode stats from the env's episode info buffer
    ep_rets = []
    ep_lens = []
    obs = env.reset()
    ep_ret, ep_len = 0.0, 0
    for _ in range(10_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_ret += float(reward[0])
        ep_len += 1
        if done[0]:
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            ep_ret, ep_len = 0.0, 0
            obs = env.reset()
        if len(ep_rets) >= 50:
            break

    ep_mean = sum(ep_rets) / len(ep_rets) if ep_rets else float("nan")
    ep_std  = float(np.std(ep_rets)) if len(ep_rets) > 1 else float("nan")
    avg_len = sum(ep_lens) / len(ep_lens) if ep_lens else float("nan")

    print()
    print("=" * 60)
    print("  SB3 PPO — LunarLander-v3 — FINAL RESULTS")
    print("=" * 60)
    print(f"  n_envs                   :          1")
    print(f"  total steps              :  {TOTAL:>10,}")
    print(f"  wall time                :  {wall:>10.2f} s")
    print(f"  steps/sec                :  {sps:>10,.0f}")
    print(f"  µs/step                  :  {1e6/sps:>10.2f}")
    print(f"  episodes/sec             :  {len(ep_rets) / wall * (TOTAL / (10_000 * wall / wall)):>10.1f}  (approx)")
    print(f"  avg steps/episode        :  {avg_len:>10.1f}  (eval rollout)")
    print(f"  episode return mean      :  {ep_mean:>10.3f}  (eval, 50 eps)")
    print(f"  episode return std dev   :  {ep_std:>10.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
