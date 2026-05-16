"""
Sample Factory APPO benchmark on LunarLander-v3.
Hyperparameters matched to RelayRL PPO conv5 config:
  64 envs, [128,128] net, gamma=0.999, lam=0.98, clip=0.2, ent=0.01,
  lr=2.5e-4, 4 gradient epochs per rollout, CPU.
  rollout=128 → 64*128=8192 steps per update (RelayRL: ~64eps*90steps=5760).
Target: mean episode return >= 250.
"""

import os
import sys
import time

import gymnasium
from sample_factory.envs.env_utils import register_env
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.train import run_rl


def make_lunarlander(full_env_name, cfg=None, env_config=None, render_mode=None):
    return gymnasium.make("LunarLander-v3")


# Register env at module level so child processes see it
register_env("LunarLander-v3", make_lunarlander)


# ── config (matched to RelayRL conv5) ────────────────────────────────────────

N_WORKERS   = 64
ROLLOUT     = 128   # 64 * 128 = 8192 steps per update
BATCH_SIZE  = 8192  # Full-batch gradient step per epoch
N_EPOCHS    = 4     # 4 gradient epochs per update
GAMMA       = 0.999
GAE_LAMBDA  = 0.98
CLIP        = 0.2
ENT_COEF    = 0.01
LR          = 2.5e-4
TRAIN_STEPS = 20_000_000

ARGV = [
    "--env=LunarLander-v3",
    "--algo=APPO",
    f"--num_workers={N_WORKERS}",
    "--num_envs_per_worker=1",
    f"--rollout={ROLLOUT}",
    f"--batch_size={BATCH_SIZE}",
    f"--num_epochs={N_EPOCHS}",
    f"--gamma={GAMMA}",
    f"--gae_lambda={GAE_LAMBDA}",
    f"--ppo_clip_ratio={CLIP}",
    f"--exploration_loss_coeff={ENT_COEF}",
    f"--learning_rate={LR}",
    "--encoder_mlp_layers", "128", "128",
    "--normalize_returns=False",
    f"--train_for_env_steps={TRAIN_STEPS}",
    "--device=cpu",
    "--with_wandb=False",
    "--async_rl=True",
    "--worker_num_splits=1",
    "--env_gpu_observations=False",
    "--seed=42",
    "--experiment=sf_lunarlander_bench",
    "--train_dir=/tmp/sf_bench",
    "--experiment_summaries_interval=5",
    "--save_every_sec=300",
]


if __name__ == "__main__":
    print("=" * 60)
    print("  Sample Factory APPO — LunarLander-v3 — 64 envs — CPU")
    print(f"  lr={LR}  rollout={ROLLOUT}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")
    print(f"  gamma={GAMMA}  lam={GAE_LAMBDA}  clip={CLIP}  ent={ENT_COEF}")
    print(f"  net=[128,128]  seed=42  env_count={N_WORKERS}")
    print(f"  train_for_env_steps={TRAIN_STEPS}")
    print("=" * 60)

    t0 = time.time()

    p, _ = parse_sf_args(ARGV)
    cfg = parse_full_cfg(p, ARGV)
    run_rl(cfg)

    wall = time.time() - t0

    print()
    print("=" * 60)
    print("  Sample Factory RESULTS")
    print("=" * 60)
    print(f"  n_envs       : {N_WORKERS}")
    print(f"  total frames : {TRAIN_STEPS}")
    print(f"  wall time    : {wall:.1f}s")
    print(f"  fps          : {TRAIN_STEPS / wall:.0f}")
    print("=" * 60)
