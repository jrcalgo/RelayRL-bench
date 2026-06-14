#!/usr/bin/env python3
"""sf_lunar_bench.py — Sample Factory APPO on EnvPool LunarLander-v2, hyperparameters
matched to bench_lunar_ppo_tch (RelayRL beta.5 PPO / LibTorch, 512 envs via EnvPool).

Mapping from bench_lunar_ppo_tch.rs constants -> SF APPO CLI args:
  GAMMA=0.999          -> --gamma=0.999
  LAM=0.98             -> --gae_lambda=0.98
  CLIP_RATIO=0.2       -> --ppo_clip_ratio=0.2
  PI_LR=VF_LR=2.5e-4   -> --learning_rate=2.5e-4 --lr_schedule=constant
  VF_COEF=1.0          -> --value_loss_coeff=1.0
  TRAIN_PI/VF_ITERS=4  -> --num_epochs=4
  ENT_COEF=0.01        -> --exploration_loss_coeff=0.01
  NORMALIZE_RETURNS    -> --normalize_returns=True (SF default)
  MINI_BATCH=46080     -> --batch_size=46080 --rollout=90 (46080 = 512 envs x 90)
  ENV_COUNT=512        -> one envpool instance, num_envs=512 (num_workers=1,
                          num_envs_per_worker=1, worker_num_splits=1, batched_sampling=True)
  MAX_STEPS=500        -> envpool.make(..., max_episode_steps=500)
  [128,128] ReLU,      -> --encoder_mlp_layers 128 128 --nonlinearity=relu
  separate pi/vf nets  -> --use_rnn=False --actor_critic_share_weights=False
  TOTAL_STEPS=75_000   -> --train_for_env_steps=38_400_000 (75_000 * 512, same
                          total budget as the 64-env config's 600_000 * 64)

Run:
  python3 scripts/sf_lunar_bench.py --experiment=lunar_sf_run1
  python3 scripts/sf_lunar_bench.py --experiment=lunar_sf_smoke --train_for_env_steps=200000
"""

import os
import sys
from typing import Optional

import envpool

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

ENV_NAME = "lunar_bench_v2"
MAX_STEPS = 500
NUM_ENVS = 512


def make_lunar_bench_env(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    # Single envpool instance holding all NUM_ENVS sub-envs — releases the GIL
    # during step() and steps all envs on a C++ thread pool, matching the Rust
    # side's single EnvPoolVecEnv(make_sf_matched_envpool_lunar_lander_vec).
    # EnvPool auto-resets terminated/truncated sub-envs internally.
    env = envpool.make(
        "LunarLander-v2",
        env_type="gymnasium",
        num_envs=NUM_ENVS,
        seed=1,
        max_episode_steps=MAX_STEPS,
        num_threads=os.cpu_count(),
    )
    env.num_agents = NUM_ENVS  # required by SF's batched sampler
    # envpool's GymnasiumEnvPool doesn't run gymnasium.VectorEnv.__init__, so
    # these attrs are missing; gymnasium's VectorEnv.close() (called by SF's
    # spawn_tmp_env_and_get_info) needs them.
    env.closed = False
    env.viewer = None
    return env


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
    "--batch_size=46080",
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
    "--num_workers=1",
    "--num_envs_per_worker=1",
    "--worker_num_splits=1",
    "--batched_sampling=True",
    "--serial_mode=False",
    "--async_rl=True",
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
