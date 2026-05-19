"""RLlib PPO — LunarLander-v3 — vectorized env throughput benchmark.

Usage:
    python rllib_lunarlander_vec.py --num-envs 1024
    python rllib_lunarlander_vec.py --num-envs 4096
    python rllib_lunarlander_vec.py --num-envs 8192

Primary throughput metric: env transitions / second (sampled, not trained),
measured over the timed portion after a warm-up iteration.

train_batch_size is scaled as num_envs × STEPS_PER_ENV_PER_ITER so each env
contributes STEPS_PER_ENV_PER_ITER transitions per training iteration.
This makes per-iteration step counts meaningful at any env scale.
"""

import argparse
import time

import ray
from ray.rllib.algorithms.ppo import PPOConfig

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs",             type=int, default=1024)
parser.add_argument("--num-runners",          type=int, default=4)
parser.add_argument("--num-iters",            type=int, default=20)
parser.add_argument("--warmup-iters",         type=int, default=2)
parser.add_argument("--steps-per-env-per-iter", type=int, default=10,
                    help="How many steps each env contributes per training iter.")
args = parser.parse_args()

NUM_ENVS          = args.num_envs
NUM_RUNNERS       = args.num_runners
ENVS_PER_RUNNER   = NUM_ENVS // NUM_RUNNERS
NUM_ITERS         = args.num_iters
WARMUP_ITERS      = args.warmup_iters
STEPS_PER_ENV     = args.steps_per_env_per_iter
TRAIN_BATCH_SIZE  = NUM_ENVS * STEPS_PER_ENV
ENV_ID            = "LunarLander-v3"

# ── RLlib config ──────────────────────────────────────────────────────────────

ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(ENV_ID)
    .env_runners(
        num_env_runners=NUM_RUNNERS,
        num_envs_per_env_runner=ENVS_PER_RUNNER,
    )
    .training(
        train_batch_size=TRAIN_BATCH_SIZE,
        num_epochs=4,
        lr=3e-4,
    )
    .framework("torch")
    .resources(num_gpus=0)
)

algo = config.build()

# ── Header ────────────────────────────────────────────────────────────────────

print("═" * 67)
print(f"  RLlib PPO — {ENV_ID} — vectorized throughput benchmark")
print(f"  {NUM_ENVS} total envs  ({NUM_RUNNERS} runners × {ENVS_PER_RUNNER} envs each)")
print(f"  {STEPS_PER_ENV} steps/env/iter  →  train_batch_size = {TRAIN_BATCH_SIZE:,}")
print(f"  warm-up: {WARMUP_ITERS} iters   timed: {NUM_ITERS} iters")
print("═" * 67)
print()


def parse_int(v):
    try:
        return int(float(str(v)))
    except (TypeError, ValueError):
        return 0


def parse_float(v, default=float("nan")):
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return default


# ── Warm-up ───────────────────────────────────────────────────────────────────

print(f"Warming up ({WARMUP_ITERS} iters)…")
for _ in range(WARMUP_ITERS):
    algo.train()
print("Warm-up done. Starting timed run…\n")

# ── Timed run ─────────────────────────────────────────────────────────────────

total_steps        = 0
total_sample_time  = 0.0
total_update_time  = 0.0
ep_reward_mean     = float("nan")
ep_len_mean        = float("nan")

t_wall_start = time.perf_counter()

for i in range(1, NUM_ITERS + 1):
    t0     = time.perf_counter()
    result = algo.train()
    elapsed = time.perf_counter() - t0

    er  = result.get("env_runners", {})
    tr  = result.get("timers",      {})

    steps_this_iter = parse_int(er.get("num_env_steps_sampled", 0))
    ep_reward_mean  = parse_float(er.get("episode_return_mean"), float("nan"))
    ep_len_mean     = parse_float(er.get("episode_len_mean"),    float("nan"))
    episodes        = parse_int(er.get("num_episodes", 0))

    sample_t = tr.get("env_runner_sampling_timer", 0.0)
    update_t = tr.get("learner_update_timer",      0.0)

    total_steps       += steps_this_iter
    total_sample_time += sample_t
    total_update_time += update_t

    tps = steps_this_iter / elapsed if elapsed > 0 else 0.0
    print(f"  iter {i:>3}/{NUM_ITERS}  "
          f"reward={ep_reward_mean:>8.2f}  "
          f"ep_len={ep_len_mean:>6.1f}  "
          f"episodes={episodes:>5}  "
          f"steps={steps_this_iter:>7}  "
          f"t={elapsed:.2f}s  "
          f"(sample={sample_t:.3f}s update={update_t:.3f}s)  "
          f"t/s={tps:>10,.0f}")

wall          = time.perf_counter() - t_wall_start
avg_tps       = total_steps / wall if wall > 0 else 0.0
sample_tps    = total_steps / total_sample_time if total_sample_time > 0 else 0.0

# ── Results ───────────────────────────────────────────────────────────────────

print()
print("═" * 67)
print(f"  RLlib PPO — {ENV_ID} — FINAL RESULTS  ({NUM_ENVS} envs)")
print("═" * 67)
print()
print("─── Config ──────────────────────────────────────────────────────────")
print(f"  env                      : {ENV_ID}")
print(f"  total parallel envs      : {NUM_ENVS}")
print(f"  env runners              : {NUM_RUNNERS}")
print(f"  envs per runner          : {ENVS_PER_RUNNER}")
print(f"  train_batch_size         : {TRAIN_BATCH_SIZE:,}")
print(f"  timed iterations         : {NUM_ITERS}")
print(f"  total env steps sampled  : {total_steps:,}")
print()
print("─── Throughput (wall time incl. PPO update) ─────────────────────────")
print(f"  wall time (timed run)    : {wall:>10.2f} s")
print(f"  avg time / iter          : {wall / NUM_ITERS:>10.3f} s")
print(f"  env transitions / sec    : {avg_tps:>10,.0f}")
print(f"  µs / env transition      : {1e6 / avg_tps if avg_tps > 0 else float('inf'):>10.3f}")
print()
print("─── Throughput (sampling only, env step + inference) ────────────────")
print(f"  total sample time        : {total_sample_time:>10.2f} s")
print(f"  avg sample time / iter   : {total_sample_time / NUM_ITERS:>10.3f} s")
print(f"  env transitions / sec    : {sample_tps:>10,.0f}")
print(f"  µs / env transition      : {1e6 / sample_tps if sample_tps > 0 else float('inf'):>10.3f}")
print()
print("─── Training quality ────────────────────────────────────────────────")
print(f"  final episode reward mean: {ep_reward_mean:>10.2f}")
print(f"  final episode len mean   : {ep_len_mean:>10.1f}")
print("═" * 67)

algo.stop()
ray.shutdown()
