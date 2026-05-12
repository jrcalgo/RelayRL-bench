"""RLlib PPO benchmark on LunarLander-v3 — 1 env (num_env_runners=0), until convergence."""
import time
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig

CONVERGENCE_THRESHOLD = 200.0
WINDOW = 100
MAX_STEPS = 2_000_000

ray.init(num_cpus=2, num_gpus=0, log_to_driver=False, ignore_reinit_error=True,
         _temp_dir="/tmp/ray_bench")

config = (
    PPOConfig()
    .environment("LunarLander-v3")
    .env_runners(num_env_runners=0, num_envs_per_env_runner=1)
    .training(
        train_batch_size=1024,
        minibatch_size=64,
        num_epochs=4,
        gamma=0.999,
        lambda_=0.98,
        entropy_coeff=0.01,
        vf_loss_coeff=0.5,
        grad_clip=0.5,
        lr=2.5e-4,
        model={"fcnet_hiddens": [128, 128], "fcnet_activation": "relu"},
    )
    .framework("torch")
    .checkpointing(export_native_model_files=False)
)

algo = config.build()

print("=" * 60)
print("  RLlib PPO — LunarLander-v3 — 1 env (num_env_runners=0)")
print(f"  train_batch=1024  minibatch=64  epochs=4  lr=2.5e-4")
print(f"  net=[128,128]  gamma=0.999  lam=0.98  ent=0.01")
print("=" * 60)

ep_returns = []
converged_step = None
converged_time = None
t0 = time.perf_counter()
total_steps = 0
iteration = 0

while total_steps < MAX_STEPS:
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
            and mean_ret >= CONVERGENCE_THRESHOLD):
        converged_step = total_steps
        converged_time = elapsed
        print(f"  *** CONVERGED at step={converged_step:,}  t={converged_time:.1f}s  mean_ret={mean_ret:.1f} ***")
        break

wall = time.perf_counter() - t0
final_mean = float(np.mean(ep_returns[-WINDOW:])) if len(ep_returns) >= WINDOW else float(np.mean(ep_returns)) if ep_returns else 0.0
fps_overall = total_steps / wall

print()
print("=" * 60)
print("  RLlib RESULTS")
print("=" * 60)
print(f"  total steps      : {total_steps:,}")
print(f"  wall time        : {wall:.1f}s")
print(f"  steps/sec        : {fps_overall:.0f}")
print(f"  total iterations : {iteration}")
print(f"  final mean_ret   : {final_mean:.1f}")
if converged_step:
    print(f"  converged at step: {converged_step:,}")
    print(f"  time to converge : {converged_time:.1f}s")
    print(f"  fps at convergence: {converged_step/converged_time:.0f}")
else:
    print(f"  did NOT converge within {MAX_STEPS:,} steps")

algo.stop()
ray.shutdown()
