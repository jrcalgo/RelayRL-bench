"""
RLLib PPO benchmark on LunarLander-v3 — 1 env runner, no batching.
1 runner × 1 env × 1 000 steps/batch × 1 000 iterations = ~1 000 000 total steps.
Matches the RelayRL scalar single-env benchmark step count.
"""
import time, os, warnings
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.ppo import PPOConfig

NUM_ITERS   = 1_000
TRAIN_BATCH = 1_000   # steps per iteration
NUM_RUNNERS = 1       # single env runner

ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR",
         num_cpus=2)

cfg = (
    PPOConfig()
    .environment("LunarLander-v3")
    .api_stack(enable_rl_module_and_learner=False,
               enable_env_runner_and_connector_v2=False)
    .env_runners(
        num_env_runners=NUM_RUNNERS,
        rollout_fragment_length=TRAIN_BATCH,
        num_envs_per_env_runner=1,
    )
    .training(
        train_batch_size=TRAIN_BATCH,
        minibatch_size=TRAIN_BATCH,
        num_epochs=10,
        lr=3e-4,
        gamma=0.99,
        lambda_=0.97,
        clip_param=0.2,
        model={"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},
        use_gae=True,
    )
    .framework("torch")
    .resources(num_gpus=0)
)

print("=" * 60)
print("  RLLib PPO — LunarLander-v3 — 1 env — ~1 000 000 steps")
print("=" * 60)

algo  = cfg.build()
t0    = time.perf_counter()
total = 0

for it in range(1, NUM_ITERS + 1):
    r     = algo.train()
    total = r.get("num_env_steps_sampled_lifetime", it * TRAIN_BATCH)
    if it % 100 == 0 or it == 1:
        elapsed = time.perf_counter() - t0
        sps     = total / elapsed
        ep_len  = r.get("episode_len_mean", float("nan"))
        ep_ret  = r.get("episode_reward_mean", float("nan"))
        print(f"  [{total:>8,} steps]  EpLen={ep_len}  EpRet={ep_ret}  {sps:,.0f} steps/sec")

wall = time.perf_counter() - t0
sps  = total / wall

print()
print("=" * 60)
print("  RLLib PPO — LunarLander-v3 — FINAL RESULTS")
print("=" * 60)
print(f"  num_runners              :          {NUM_RUNNERS}")
print(f"  num_envs_per_runner      :          1")
print(f"  total steps              :  {total:>10,}")
print(f"  wall time                :  {wall:>10.2f} s")
print(f"  steps/sec                :  {sps:>10,.0f}")
print(f"  µs/step                  :  {1e6/sps:>10.2f}")
print("=" * 60)

algo.stop()
ray.shutdown()
