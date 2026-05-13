"""SB3 PPO benchmark on LunarLander-v3 — 1 env, matching RelayRL hyperparams."""
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# ── Shared hyperparameters (match RelayRL bench_lunar_ppo_scalar1) ───────────
SEED       = 42
GAMMA      = 0.999
LAM        = 0.98
CLIP       = 0.2
LR         = 2.5e-4
ENT_COEF   = 0.05
VF_COEF    = 0.5
MAX_GRAD   = 0.5
N_STEPS    = 1024   # rollout steps per update (≈ RelayRL epoch size at 1 env)
BATCH_SIZE = 64
N_EPOCHS   = 10     # train_pi_iters
TARGET_KL  = 0.05
NET        = [128, 128]
# ─────────────────────────────────────────────────────────────────────────────
CONVERGENCE = 200.0
WINDOW      = 100
MAX_STEPS   = 100_000

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
                            and np.mean(self.ep_returns[-WINDOW:]) >= CONVERGENCE):
                        self.converged_step = self.num_timesteps
                        self.converged_time = time.perf_counter() - self.start_time
                        mean = np.mean(self.ep_returns[-WINDOW:])
                        print(f"  *** CONVERGED step={self.converged_step:,}"
                              f"  t={self.converged_time:.1f}s  mean_ret={mean:.1f} ***")
                        return False
        return True

    def _on_rollout_end(self):
        self.rollout_count += 1
        if self.ep_returns and self.rollout_count % 10 == 0:
            win = self.ep_returns[-min(WINDOW, len(self.ep_returns)):]
            elapsed = time.perf_counter() - self.start_time
            fps = self.num_timesteps / elapsed
            print(f"  step={self.num_timesteps:>8,}  mean_ret(100)={np.mean(win):>8.1f}"
                  f"  fps={fps:>6.0f}  t={elapsed:.1f}s")

env = make_vec_env("LunarLander-v3", n_envs=1, seed=SEED)

model = PPO(
    "MlpPolicy", env,
    learning_rate=LR,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=GAMMA,
    gae_lambda=LAM,
    clip_range=CLIP,
    ent_coef=ENT_COEF,
    vf_coef=VF_COEF,
    max_grad_norm=MAX_GRAD,
    target_kl=TARGET_KL,
    policy_kwargs=dict(net_arch=NET),
    seed=SEED,
    verbose=0,
)

print("=" * 60)
print("  SB3 PPO — LunarLander-v3 — 1 env")
print(f"  lr={LR}  n_steps={N_STEPS}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")
print(f"  gamma={GAMMA}  lam={LAM}  clip={CLIP}  ent={ENT_COEF}"
      f"  vf={VF_COEF}  grad_clip={MAX_GRAD}  target_kl={TARGET_KL}")
print(f"  net={NET}  seed={SEED}")
print("=" * 60)

cb = Tracker()
t0 = time.perf_counter()
model.learn(total_timesteps=MAX_STEPS, callback=cb, progress_bar=False)
wall = time.perf_counter() - t0

total = model.num_timesteps
fps_overall = total / wall
final_mean = (float(np.mean(cb.ep_returns[-WINDOW:])) if len(cb.ep_returns) >= WINDOW
              else float(np.mean(cb.ep_returns)) if cb.ep_returns else 0.0)

print()
print("=" * 60)
print("  SB3 RESULTS")
print("=" * 60)
print(f"  total steps       : {total:,}")
print(f"  wall time         : {wall:.1f}s")
print(f"  steps/sec         : {fps_overall:.0f}")
print(f"  total episodes    : {len(cb.ep_returns)}")
print(f"  final mean_ret    : {final_mean:.1f}")
if cb.converged_step:
    print(f"  converged at step : {cb.converged_step:,}")
    print(f"  time to converge  : {cb.converged_time:.1f}s")
    print(f"  steps/sec (conv.) : {cb.converged_step / cb.converged_time:.0f}")
else:
    print(f"  did NOT converge within {MAX_STEPS:,} steps")
env.close()
