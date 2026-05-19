"""Sample Factory sync-PPO (APPO serial_mode) benchmark on LunarLander-v3 — 1 worker, 64 envs.

Note: async_rl=True crashes with numpy>=2.0 (SF 2.1.1 bug). serial_mode=True is synchronous PPO.
"""
import os, sys, time, resource
os.environ["SF_DISABLE_TENSORBOARD"] = "1"
os.environ["SF_SAVE_EVERY_SEC"] = "99999"

import numpy as np
import gymnasium as gym

from sample_factory.train import run_rl
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env

# ── Shared hyperparameters (match RelayRL bench_lunar_ppo_scalar1) ───────────
SEED       = 42
GAMMA      = 0.999
LAM        = 0.98
LR         = 2.5e-4
ENT_COEF   = 0.05
N_STEPS    = 1024
BATCH_SIZE = 64
N_EPOCHS   = 10
HIDDEN     = 128
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEPS = 100_000

def make_lunarlander(full_env_name, cfg=None, env_config=None, render_mode=None):
    return gym.make("LunarLander-v3")

register_env("lunarlander_v3", make_lunarlander)

if __name__ == "__main__":
    argv = [
        "--algo=APPO",
        "--env=lunarlander_v3",
        "--experiment=bench_sf_lunar_64env",
        "--train_dir=/tmp/sf_bench_64",
        "--num_workers=1",
        "--num_envs_per_worker=64",
        "--worker_num_splits=1",
        f"--rollout={N_STEPS}",
        f"--batch_size={N_STEPS}",
        f"--num_batches_per_epoch=64",
        f"--num_epochs={N_EPOCHS}",
        f"--gamma={GAMMA}",
        f"--gae_lambda={LAM}",
        f"--exploration_loss_coeff={ENT_COEF}",
        f"--learning_rate={LR}",
        "--train_for_env_steps=" + str(MAX_STEPS),
        "--seed=" + str(SEED),
        "--save_every_sec=99999",
        "--keep_checkpoints=0",
        "--async_rl=False",
        "--serial_mode=True",
        "--batched_sampling=True",
        "--device=cpu",
        "--encoder_mlp_layers", str(HIDDEN), str(HIDDEN),
        "--use_rnn=False",
        "--nonlinearity=relu",
        "--policy_initialization=orthogonal",
        "--with_wandb=False",
        "--wandb_project=disabled",
    ]

    print("=" * 60)
    print("  Sample Factory APPO serial — LunarLander-v3 — 1 worker/64 envs")
    print(f"  lr={LR}  rollout={N_STEPS}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")
    print(f"  gamma={GAMMA}  lam={LAM}  ent={ENT_COEF}  hidden={HIDDEN}  serial_mode  seed={SEED}")
    print("=" * 60)

    parser, _ = parse_sf_args(argv)
    cfg = parse_full_cfg(parser, argv)
    t0 = time.perf_counter()
    status = run_rl(cfg)
    wall = time.perf_counter() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024
    print(f"\nSample Factory finished in {wall:.1f}s  status={status}")
    print(f"  peak RSS (driver) : {rss_mb:.0f} MB")
