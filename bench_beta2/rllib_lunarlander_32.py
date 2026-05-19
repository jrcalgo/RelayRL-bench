"""RLlib PPO — LunarLander-v3 — 32 parallel environments."""

import time
import ray
from ray.rllib.algorithms.ppo import PPOConfig

NUM_ENVS              = 32
NUM_ENV_RUNNERS       = 4   # one Ray actor per CPU core
ENVS_PER_RUNNER       = NUM_ENVS // NUM_ENV_RUNNERS  # 8 envs per runner
NUM_ITERS             = 50
ENV_ID                = "LunarLander-v3"

ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(ENV_ID)
    .env_runners(
        num_env_runners=NUM_ENV_RUNNERS,
        num_envs_per_env_runner=ENVS_PER_RUNNER,
    )
    .training(
        train_batch_size=4096,
        num_epochs=10,
        lr=3e-4,
    )
    .framework("torch")
    .resources(num_gpus=0)
)

algo = config.build()

print("═" * 67)
print(f"  RLlib PPO — {ENV_ID} — {NUM_ENVS} parallel envs")
print(f"  {NUM_ENV_RUNNERS} env runners × {ENVS_PER_RUNNER} envs each")
print(f"  {NUM_ITERS} training iterations")
print("═" * 67)
print()

t_total = time.time()
for i in range(1, NUM_ITERS + 1):
    t0 = time.time()
    result = algo.train()
    elapsed = time.time() - t0

    ep_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
    ep_len_mean    = result.get("env_runners", {}).get("episode_len_mean",    float("nan"))
    episodes       = result.get("env_runners", {}).get("num_episodes",        0)
    timesteps      = result.get("env_runners", {}).get("num_env_steps_sampled_this_iter", 0)

    print(f"  iter {i:>3}/{NUM_ITERS}  "
          f"reward={ep_reward_mean:>8.2f}  "
          f"ep_len={ep_len_mean:>6.1f}  "
          f"episodes={episodes:>4}  "
          f"steps={timesteps:>6}  "
          f"t={elapsed:.2f}s")

wall = time.time() - t_total

print()
print("═" * 67)
print("  RLlib PPO — FINAL RESULTS")
print("═" * 67)
print(f"  env                : {ENV_ID}")
print(f"  parallel envs      : {NUM_ENVS}  ({NUM_ENV_RUNNERS} runners × {ENVS_PER_RUNNER})")
print(f"  iterations         : {NUM_ITERS}")
print(f"  total wall time    : {wall:.2f} s")
print(f"  avg time / iter    : {wall / NUM_ITERS:.2f} s")
print(f"  final reward mean  : {ep_reward_mean:.2f}")
print("═" * 67)

algo.stop()
ray.shutdown()
