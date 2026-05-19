"""RLLib PPO — 8 runners, 20 epochs, timed."""
import time, sys, os, warnings
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
sys.path.insert(0, "/home/user/RelayRL-end2end/benchmarks")

from ray.tune.registry import register_env
from gridworld_env import GridWorldEnv

register_env("gridworld", lambda cfg: GridWorldEnv(
    grid_size=cfg.get("grid_size", 10),
    max_steps=cfg.get("max_steps", 200),
))

NUM_EPOCHS  = 20
MAX_STEPS   = 200
EP_PER      = 8
TRAIN_BATCH = EP_PER * MAX_STEPS  # 1,600 per worker → 12,800 total for 8 workers

ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR",
         num_cpus=10)

cfg = (
    PPOConfig()
    .environment("gridworld", env_config={"grid_size": 10, "max_steps": MAX_STEPS})
    .api_stack(enable_rl_module_and_learner=False,
               enable_env_runner_and_connector_v2=False)
    .env_runners(
        num_env_runners=8,
        rollout_fragment_length=EP_PER * MAX_STEPS // 8,  # 200 steps/worker
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
t0 = time.perf_counter()
total = 0
for epoch in range(1, NUM_EPOCHS + 1):
    r = algo.train()
    total = r.get("num_env_steps_sampled_lifetime", epoch * TRAIN_BATCH)
    ep_len = r.get("episode_len_mean", float("nan"))
    ep_ret = r.get("episode_reward_mean", float("nan"))
    print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  EpLen={ep_len}  EpRet={ep_ret}  steps={total:,}")

wall = time.perf_counter() - t0
sps  = total / wall
print(f"\nRLLib PPO 8-runners: wall={wall:.1f}s  steps={total:,}  steps/sec={sps:,.0f}  µs/step={1e6/sps:.1f}")
algo.stop()
ray.shutdown()
