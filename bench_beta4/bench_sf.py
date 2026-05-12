"""Sample Factory APPO benchmark on LunarLander-v3 — 1 worker, 1 env, until convergence."""
import os, sys, time
os.environ["SF_DISABLE_TENSORBOARD"] = "1"
os.environ["SF_SAVE_EVERY_SEC"] = "99999"

import numpy as np
import gymnasium as gym

from sample_factory.train import run_rl
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace

CONVERGENCE_THRESHOLD = 200.0
MAX_STEPS = 2_000_000

def make_lunarlander(full_env_name, cfg=None, env_config=None, render_mode=None):
    return gym.make("LunarLander-v3")

register_env("lunarlander_v3", make_lunarlander)

argv = [
    "--algo=APPO",
    "--env=lunarlander_v3",
    "--experiment=bench_sf_lunar",
    "--train_dir=/tmp/sf_bench",
    "--num_workers=1",
    "--num_envs_per_worker=1",
    "--worker_num_splits=1",
    "--rollout=1024",
    "--batch_size=1024",
    "--num_epochs=4",
    "--gamma=0.999",
    "--gae_lambda=0.98",
    "--exploration_loss_coeff=0.01",
    "--learning_rate=0.00025",
    "--train_for_env_steps=" + str(MAX_STEPS),
    "--save_every_sec=99999",
    "--keep_checkpoints=0",
    "--serial_mode=True",
    "--async_rl=False",
    "--encoder_type=mlp",
    "--encoder_subtype=mlp_mujoco",
    "--hidden_size=128",
    "--nonlinearity=relu",
    "--policy_initialization=orthogonal",
    "--with_wandb=False",
    "--wandb_project=disabled",
]

print("=" * 60)
print("  Sample Factory APPO — LunarLander-v3 — 1 worker/1 env")
print(f"  rollout=1024  batch=1024  epochs=4  lr=2.5e-4")
print(f"  hidden=128  gamma=0.999  lam=0.98  ent=0.01  serial_mode")
print("=" * 60)

cfg = parse_full_cfg(parse_sf_args(argv))
t0 = time.perf_counter()
status = run_rl(cfg)
wall = time.perf_counter() - t0
print(f"\nSample Factory finished in {wall:.1f}s  status={status}")
