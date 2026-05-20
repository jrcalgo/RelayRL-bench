"""
Sample Factory APPO benchmark on LunarLander-v3.
Hyperparameters matched to RelayRL PPO conv5:
  64 envs, [128,128] net, gamma=0.999, lam=0.98, clip=0.2, ent=0.01,
  lr=2.5e-4, 4 gradient epochs per rollout, CPU.

Architecture notes:
  - use_rnn=False: pure MLP (matches RelayRL's 128x128 feed-forward policy)
  - nonlinearity=relu + policy_initialization=orthogonal: match RelayRL defaults
  - batched_sampling=True: single vectorized-env process (mirrors RelayRL's
    single-process env loop rather than 64 separate Python subprocesses)
  - normalize_returns=True: SF default for training stability; RelayRL doesn't
    normalize, but this is an impl detail, not a core hyperparameter
  - rollout=90 → 64*90=5760 steps/update ≈ RelayRL's 64eps*~90steps=5760
  - batch_size=5760 (full-batch), num_epochs=4 → 4 full-batch gradient steps
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium
from sample_factory.envs.env_utils import register_env
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.train import run_rl


def make_lunarlander(full_env_name, cfg=None, env_config=None, render_mode=None):
    return gymnasium.make("LunarLander-v2")


register_env("LunarLander-v3", make_lunarlander)


# ── config ────────────────────────────────────────────────────────────────────
N_WORKERS           = 1
N_ENVS_PER_WORKER   = 64   # 64 parallel envs in one process
ROLLOUT             = 90   # 64 * 90 = 5760 steps/update ≈ RelayRL's ~5760
BATCH_SIZE          = 5760 # full-batch per gradient step
N_BATCHES_PER_EPOCH = 1
N_EPOCHS            = 4    # 4 gradient passes per update
GAMMA               = 0.999
GAE_LAMBDA          = 0.98
CLIP                = 0.2
ENT_COEF            = 0.01
LR                  = 2.5e-4
TRAIN_STEPS         = 20_000_000

ARGV = [
    "--env=LunarLander-v3",
    "--algo=APPO",
    f"--num_workers={N_WORKERS}",
    f"--num_envs_per_worker={N_ENVS_PER_WORKER}",
    f"--rollout={ROLLOUT}",
    f"--batch_size={BATCH_SIZE}",
    f"--num_batches_per_epoch={N_BATCHES_PER_EPOCH}",
    f"--num_epochs={N_EPOCHS}",
    f"--gamma={GAMMA}",
    f"--gae_lambda={GAE_LAMBDA}",
    f"--ppo_clip_ratio={CLIP}",
    f"--exploration_loss_coeff={ENT_COEF}",
    f"--learning_rate={LR}",
    "--encoder_mlp_layers", "128", "128",
    "--use_rnn=False",
    "--nonlinearity=relu",
    "--policy_initialization=orthogonal",
    "--normalize_returns=True",
    f"--train_for_env_steps={TRAIN_STEPS}",
    "--device=cpu",
    "--with_wandb=False",
    "--batched_sampling=True",
    "--async_rl=False",
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
    print(f"  net=[128,128]  relu  orthogonal_init  no_rnn  seed=42")
    print(f"  env_count={N_WORKERS*N_ENVS_PER_WORKER}  batched_sampling=True  normalize_returns=True")
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
    print(f"  n_envs       : {N_WORKERS * N_ENVS_PER_WORKER}")
    print(f"  total frames : {TRAIN_STEPS}")
    print(f"  wall time    : {wall:.1f}s")
    print(f"  fps          : {TRAIN_STEPS / wall:.0f}")
    print("=" * 60)

