"""RLlib PPO benchmark on LunarLander-v3 — 64 envs (num_env_runners=0), 100k steps."""
import time
import resource
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# ── Shared hyperparameters (match RelayRL bench_lunar_ppo_scalar1) ───────────
SEED       = 42
GAMMA      = 0.999
LAM        = 0.98
CLIP       = 0.2
LR         = 2.5e-4
ENT_COEF   = 0.05
VF_COEF    = 0.5
MAX_GRAD   = 0.5
N_STEPS    = 1024
BATCH_SIZE = 64
N_EPOCHS   = 10
TARGET_KL  = 0.05
NET        = [128, 128]
# ─────────────────────────────────────────────────────────────────────────────
CONVERGENCE = 200.0
WINDOW      = 100
MAX_STEPS_ENV = 100_000

ray.init(num_cpus=2, num_gpus=0, log_to_driver=False, ignore_reinit_error=True,
         _temp_dir="/tmp/ray_bench")

config = (
    PPOConfig()
    .environment("LunarLander-v3")
    .env_runners(num_env_runners=0, num_envs_per_env_runner=64)
    .debugging(seed=SEED)
    .training(
        train_batch_size=N_STEPS,
        minibatch_size=BATCH_SIZE,
        num_epochs=N_EPOCHS,
        gamma=GAMMA,
        lambda_=LAM,
        clip_param=CLIP,
        entropy_coeff=ENT_COEF,
        vf_loss_coeff=VF_COEF,
        grad_clip=MAX_GRAD,
        kl_target=TARGET_KL,
        lr=LR,
        model={"fcnet_hiddens": NET, "fcnet_activation": "relu"},
    )
    .framework("torch")
    .checkpointing(export_native_model_files=False)
)

algo = config.build()

print("=" * 60)
print("  RLlib PPO — LunarLander-v3 — 64 envs (num_env_runners=0)")
print(f"  lr={LR}  n_steps={N_STEPS}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")
print(f"  gamma={GAMMA}  lam={LAM}  clip={CLIP}  ent={ENT_COEF}"
      f"  vf={VF_COEF}  grad_clip={MAX_GRAD}  target_kl={TARGET_KL}")
print(f"  net={NET}  seed={SEED}")
print("=" * 60)

ep_returns = []
converged_step = None
converged_time = None
t0 = time.perf_counter()
total_steps = 0
iteration = 0

while total_steps < MAX_STEPS_ENV:
    result = algo.train()
    iteration += 1
    total_steps = result.get("num_env_steps_sampled_lifetime", total_steps)
    elapsed = time.perf_counter() - t0
    fps = total_steps / elapsed if elapsed > 0 else 0

    # collect episode returns from result
    runners = result.get("env_runners", {})
    hist = runners.get("hist_stats", {})
    new_rets = hist.get("episode_reward", [])
    if not new_rets:
        ep_mean = runners.get("episode_return_mean", None)
        ep_len  = runners.get("episode_len_mean", None)
        # fallback: use mean from result
        if ep_mean is not None:
            ep_returns.append(float(ep_mean))
    else:
        ep_returns.extend([float(r) for r in new_rets])

    if len(ep_returns) >= WINDOW:
        mean_ret = float(np.mean(ep_returns[-WINDOW:]))
    elif ep_returns:
        mean_ret = float(np.mean(ep_returns))
    else:
        mean_ret = float("nan")

    if iteration % 5 == 0 or iteration <= 3:
        print(f"  iter={iteration:>4}  step={total_steps:>8,}  mean_ret(100)={mean_ret:>8.1f}"
              f"  fps={fps:>6.0f}  t={elapsed:.1f}s")

    if (converged_step is None
            and len(ep_returns) >= WINDOW
            and mean_ret >= CONVERGENCE):
        converged_step = total_steps
        converged_time = elapsed
        print(f"  *** CONVERGED at step={converged_step:,}  t={converged_time:.1f}s  mean_ret={mean_ret:.1f} ***")
        break

wall = time.perf_counter() - t0
final_mean = float(np.mean(ep_returns[-WINDOW:])) if len(ep_returns) >= WINDOW else float(np.mean(ep_returns)) if ep_returns else 0.0
fps_overall = total_steps / wall

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss_kb / 1024

print()
print("=" * 60)
print("  RLlib RESULTS")
print("=" * 60)
print(f"  n_envs           : 64")
print(f"  total steps      : {total_steps:,}")
print(f"  wall time        : {wall:.1f}s")
print(f"  steps/sec        : {fps_overall:.0f}")
print(f"  total iterations : {iteration}")
print(f"  final mean_ret   : {final_mean:.1f}")
print(f"  peak RSS (driver): {rss_mb:.0f} MB")
if converged_step:
    print(f"  converged at step: {converged_step:,}")
    print(f"  time to converge : {converged_time:.1f}s")
    print(f"  fps at convergence: {converged_step/converged_time:.0f}")
else:
    print(f"  did NOT converge within {MAX_STEPS_ENV:,} steps")

algo.stop()
ray.shutdown()
