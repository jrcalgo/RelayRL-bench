"""
RLLib PPO benchmark on Python GridWorld.
Single worker (no parallelism) to match RelayRL's 1-actor local configuration.
Also runs an 8-worker configuration to match MAPPO-8 throughput test.
"""

import time, sys, os
os.environ["RAY_DEDUP_LOGS"] = "0"

import ray
from ray.rllib.algorithms.ppo import PPOConfig
sys.path.insert(0, "/home/user/RelayRL-end2end/benchmarks")
from gridworld_env import GridWorldEnv

import gymnasium as gym
from ray.tune.registry import register_env

register_env("gridworld", lambda cfg: GridWorldEnv(
    grid_size=cfg.get("grid_size", 10),
    max_steps=cfg.get("max_steps", 200),
))

NUM_EPOCHS   = 100
EP_PER_EPOCH = 8
MAX_STEPS    = 200
# RLLib uses train_batch_size = total transitions per update
TRAIN_BATCH  = EP_PER_EPOCH * MAX_STEPS   # 1,600 steps per epoch


def run_config(num_workers: int, label: str):
    per_worker = max(1, TRAIN_BATCH // max(num_workers, 1))

    cfg = (
        PPOConfig()
        .environment("gridworld", env_config={"grid_size": 10, "max_steps": MAX_STEPS})
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length=per_worker,
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

    algo = cfg.build()

    print(f"\n{label}")
    t0 = time.perf_counter()
    total_steps = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        result = algo.train()
        total_steps = result.get("num_env_steps_sampled_lifetime", epoch * TRAIN_BATCH)
        ep_len = result.get("env_runner_results", {}).get("episode_len_mean", float("nan"))
        ep_ret = result.get("env_runner_results", {}).get("episode_return_mean", float("nan"))
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  EpLen={ep_len:.1f}  EpRet={ep_ret:.3f}  steps={total_steps:,}")

    wall = time.perf_counter() - t0
    sps = total_steps / wall
    print(f"\n{label} summary:")
    print(f"  Wall time  : {wall:.2f}s")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Steps/sec  : {sps:,.0f}")
    print(f"  µs/step    : {1e6/sps:.1f}")
    algo.stop()
    return sps, wall


def main():
    ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")

    sps_1,  w_1  = run_config(1, "RLLib PPO — 1 env-runner (matches RelayRL 1-actor)")
    sps_8,  w_8  = run_config(8, "RLLib PPO — 8 env-runners (matches RelayRL MAPPO-8)")

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'Config':<35} | {'Steps/sec':>10} | {'µs/step':>8}")
    print("-" * 60)
    relayrl = [
        ("RelayRL 1-actor IPPO",    3_260, 306),
        ("RelayRL 8-actor MAPPO",   9_846, 102),
    ]
    for name, sps, mus in relayrl:
        print(f"  {name:<33} | {sps:>10,} | {mus:>8.1f}")
    print(f"  {'RLLib 1 runner':<33} | {sps_1:>10,.0f} | {1e6/sps_1:>8.1f}")
    print(f"  {'RLLib 8 runners':<33} | {sps_8:>10,.0f} | {1e6/sps_8:>8.1f}")

    ray.shutdown()

if __name__ == "__main__":
    main()
