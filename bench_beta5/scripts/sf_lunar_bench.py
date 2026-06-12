#!/usr/bin/env python3
"""sf_lunar_bench.py — Sample Factory APPO on LunarLander-v2, hyperparameters
matched to bench_lunar_ppo_tch (RelayRL beta.5 PPO / LibTorch, 64 envs).

Mapping from bench_lunar_ppo_tch.rs constants -> SF APPO CLI args:
  GAMMA=0.999          -> --gamma=0.999
  LAM=0.98             -> --gae_lambda=0.98
  CLIP_RATIO=0.2       -> --ppo_clip_ratio=0.2
  PI_LR=VF_LR=2.5e-4   -> --learning_rate=2.5e-4 --lr_schedule=constant
  VF_COEF=1.0          -> --value_loss_coeff=1.0
  TRAIN_PI/VF_ITERS=4  -> --num_epochs=4
  ENT_COEF=0.01        -> --exploration_loss_coeff=0.01
  NORMALIZE_RETURNS    -> --normalize_returns=True (SF default)
  MINI_BATCH=5760      -> --batch_size=5760 --rollout=90 (5760 = 64 envs x 90)
  ENV_COUNT=64         -> --num_workers x --num_envs_per_worker = 64
  MAX_STEPS=500        -> TimeLimit(LunarLander-v2, max_episode_steps=500)
  [128,128] ReLU,      -> --encoder_mlp_layers 128 128 --nonlinearity=relu
  separate pi/vf nets  -> --use_rnn=False --actor_critic_share_weights=False
  TOTAL_STEPS=600_000  -> --train_for_env_steps=38_400_000 (600_000 * 64)

Run:
  python3 scripts/sf_lunar_bench.py --experiment=lunar_sf_run1
  python3 scripts/sf_lunar_bench.py --experiment=lunar_sf_smoke --train_for_env_steps=200000
"""

import sys
from typing import Optional

import gymnasium as gym

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

ENV_NAME = "lunar_bench_v2"
MAX_STEPS = 500


def make_lunar_bench_env(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    return gym.make("LunarLander-v2", max_episode_steps=MAX_STEPS, render_mode=render_mode)


def register_custom_components():
    register_env(ENV_NAME, make_lunar_bench_env)


# Matches bench_lunar_ppo_tch.rs hyperparameters (see module docstring for mapping).
DEFAULT_ARGS = [
    f"--env={ENV_NAME}",
    "--algo=APPO",
    "--gamma=0.999",
    "--gae_lambda=0.98",
    "--ppo_clip_ratio=0.2",
    "--value_loss_coeff=1.0",
    "--exploration_loss_coeff=0.01",
    "--num_epochs=4",
    "--num_batches_per_epoch=1",
    "--rollout=90",
    "--batch_size=5760",
    "--learning_rate=2.5e-4",
    "--lr_schedule=constant",
    "--normalize_returns=True",
    "--normalize_input=True",
    "--max_grad_norm=4.0",
    "--actor_critic_share_weights=False",
    "--policy_initialization=orthogonal",
    "--encoder_mlp_layers", "128", "128",
    "--nonlinearity=relu",
    "--use_rnn=False",
    "--with_vtrace=False",
    "--num_workers=4",
    "--num_envs_per_worker=16",
    "--worker_num_splits=2",
    "--device=cpu",
    "--seed=1",
    "--train_for_env_steps=38400000",
    "--with_wandb=False",
    "--stats_avg=100",
    "--experiment_summaries_interval=10",
]


def parse_custom_args(argv=None, evaluation=False):
    # User-supplied argv overrides come after DEFAULT_ARGS, so later
    # occurrences of the same flag win (argparse keeps the last value).
    full_argv = DEFAULT_ARGS + (argv if argv is not None else sys.argv[1:])
    parser, cfg = parse_sf_args(argv=full_argv, evaluation=evaluation)
    cfg = parse_full_cfg(parser, full_argv)
    return cfg


def main():
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
