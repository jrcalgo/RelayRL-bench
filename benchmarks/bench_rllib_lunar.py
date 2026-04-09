"""RLLib PPO — LunarLander-v3 (discrete), 8 runners, 20 epochs, timed."""
import time, sys, os, warnings
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.ppo import PPOConfig

NUM_EPOCHS  = 20
TRAIN_BATCH = 8_000   # ~8 episodes × ~1 000 steps
NUM_RUNNERS = 8

ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR",
         num_cpus=10)

cfg = (
    PPOConfig()
    .environment("LunarLander-v3")
    .api_stack(enable_rl_module_and_learner=False,
               enable_env_runner_and_connector_v2=False)
    .env_runners(
        num_env_runners=NUM_RUNNERS,
        rollout_fragment_length=TRAIN_BATCH // NUM_RUNNERS,
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
print(f"\nRLLib PPO LunarLander 8-runners: wall={wall:.1f}s  steps={total:,}  steps/sec={sps:,.0f}  µs/step={1e6/sps:.1f}")
algo.stop()
ray.shutdown()
