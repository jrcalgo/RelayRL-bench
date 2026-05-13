"""SB3 PPO benchmark on LunarLander-v3 — 1 env, run until mean_ret(100) >= 200."""
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

CONVERGENCE_THRESHOLD = 200.0
WINDOW = 100
MAX_STEPS = 2_000_000

class Tracker(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_returns = []
        self.converged_step = None
        self.converged_time = None
        self.start_time = None
        self.rollout_count = 0

    def _on_training_start(self):
        self.start_time = time.perf_counter()

    def _on_step(self):
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                ep = self.locals['infos'][i].get('episode')
                if ep:
                    self.ep_returns.append(float(ep['r']))
                    if (self.converged_step is None
                            and len(self.ep_returns) >= WINDOW
                            and np.mean(self.ep_returns[-WINDOW:]) >= CONVERGENCE_THRESHOLD):
                        self.converged_step = self.num_timesteps
                        self.converged_time = time.perf_counter() - self.start_time
                        print(f"  *** CONVERGED at step={self.converged_step:,}"
                              f"  t={self.converged_time:.1f}s"
                              f"  mean_ret={np.mean(self.ep_returns[-WINDOW:]):.1f} ***")
                        return False  # stops training
        return True

    def _on_rollout_end(self):
        self.rollout_count += 1
        if self.ep_returns and self.rollout_count % 10 == 0:
            win = self.ep_returns[-min(WINDOW, len(self.ep_returns)):]
            elapsed = time.perf_counter() - self.start_time
            fps = self.num_timesteps / elapsed
            print(f"  step={self.num_timesteps:>8,}  mean_ret(100)={np.mean(win):>8.1f}"
                  f"  fps={fps:>6.0f}  t={elapsed:.1f}s")

env = make_vec_env("LunarLander-v3", n_envs=1)

model = PPO(
    "MlpPolicy", env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    learning_rate=2.5e-4,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=0,
)

print("=" * 60)
print("  SB3 PPO — LunarLander-v3 — 1 env")
print(f"  n_steps=1024  batch=64  epochs=4  lr=2.5e-4")
print(f"  net=[128,128]  gamma=0.999  lam=0.98  ent=0.01")
print("=" * 60)

cb = Tracker()
t0 = time.perf_counter()
model.learn(total_timesteps=MAX_STEPS, callback=cb, progress_bar=False)
wall = time.perf_counter() - t0

total = model.num_timesteps
fps_overall = total / wall
final_mean = float(np.mean(cb.ep_returns[-WINDOW:])) if len(cb.ep_returns) >= WINDOW else float(np.mean(cb.ep_returns)) if cb.ep_returns else 0.0

print()
print("=" * 60)
print("  SB3 RESULTS")
print("=" * 60)
print(f"  total steps      : {total:,}")
print(f"  wall time        : {wall:.1f}s")
print(f"  steps/sec        : {fps_overall:.0f}")
print(f"  total episodes   : {len(cb.ep_returns)}")
print(f"  final mean_ret   : {final_mean:.1f}")
if cb.converged_step:
    fps_conv = cb.converged_step / cb.converged_time
    print(f"  converged at step: {cb.converged_step:,}")
    print(f"  time to converge : {cb.converged_time:.1f}s")
    print(f"  fps at convergence: {fps_conv:.0f}")
else:
    print(f"  did NOT converge within {MAX_STEPS:,} steps")
env.close()
