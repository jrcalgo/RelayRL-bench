"""SB3 vectorized LunarLander-v3 — raw env-stepping throughput benchmark.

No algorithm — measures pure env stepping throughput with random actions.

Two modes:
  DummyVecEnv   — N envs in a single process, stepped sequentially.
                  Represents the single-process SB3 baseline.
  SubprocVecEnv — NUM_WORKERS subprocesses (one per core), each holding
                  one env.  Steps scaled to same total transitions as
                  DummyVecEnv run so IPC overhead is directly visible.

Note: SB3 SubprocVecEnv requires exactly one Gymnasium env per worker;
nested VecEnvs are rejected.  At large N the practical SB3 approach is
DummyVecEnv (sequential) or a custom multi-env subprocess not provided
by SB3 out of the box.

Usage:
    python sb3_lunarlander_vec.py --num-envs 1024
    python sb3_lunarlander_vec.py --num-envs 4096
    python sb3_lunarlander_vec.py --num-envs 8192
"""

import argparse
import time
import multiprocessing
import numpy as np
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# ── CLI — parsed with parse_known_args so subprocess re-imports are harmless ──

_parser = argparse.ArgumentParser()
_parser.add_argument("--num-envs",     type=int, default=1024)
_parser.add_argument("--num-steps",    type=int, default=500,
                     help="step() calls for DummyVecEnv run.")
_parser.add_argument("--warmup-steps", type=int, default=50)
_parser.add_argument("--num-workers",  type=int, default=4,
                     help="Subprocess workers (one env each) for SubprocVecEnv run.")
_args, _ = _parser.parse_known_args()

NUM_ENVS     = _args.num_envs
NUM_STEPS    = _args.num_steps
WARMUP_STEPS = _args.warmup_steps
NUM_WORKERS  = _args.num_workers
ENV_ID       = "LunarLander-v3"
NCORES       = multiprocessing.cpu_count()

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env(seed: int):
    def _init():
        e = gym.make(ENV_ID)
        e.reset(seed=seed)
        return e
    return _init


def os_stats():
    rss = vol = nvol = 0
    try:
        for line in open("/proc/self/status"):
            k, _, v = line.partition(":")
            v = v.strip()
            if k == "VmRSS":                         rss  = int(v.split()[0])
            elif k == "voluntary_ctxt_switches":     vol  = int(v)
            elif k == "nonvoluntary_ctxt_switches":  nvol = int(v)
    except OSError:
        pass
    return rss, vol, nvol


def bench(vec_env, label, n_envs, n_steps, warmup):
    act   = vec_env.action_space
    total = n_envs * n_steps
    vec_env.reset()

    for _ in range(warmup):
        vec_env.step(np.array([act.sample() for _ in range(n_envs)]))

    t0 = time.perf_counter()
    for _ in range(n_steps):
        vec_env.step(np.array([act.sample() for _ in range(n_envs)]))
    wall = time.perf_counter() - t0

    rss, vol, nvol = os_stats()
    tps = total / wall

    print(f"\n{'═'*67}")
    print(f"  {label}")
    print(f"{'═'*67}\n")
    print(f"─── Throughput ──────────────────────────────────────────────────────")
    print(f"  num_envs                 : {n_envs:>10,}")
    print(f"  step() calls             : {n_steps:>10,}")
    print(f"  total env transitions    : {total:>10,}")
    print(f"  wall time                : {wall:>10.3f} s")
    print(f"  env transitions / sec    : {tps:>10,.0f}")
    print(f"  µs / env transition      : {1e6/tps:>10.3f}")
    print(f"  ms / step() call         : {1000*wall/n_steps:>10.3f}")
    print()
    print(f"─── OS ──────────────────────────────────────────────────────────────")
    print(f"  RSS                      : {rss/1024:>7.1f} MB")
    print(f"  context switches (vol)   : {vol:>10,}")
    print(f"  context switches (nonvol): {nvol:>10,}")
    print(f"  context switches (total) : {vol+nvol:>10,}")
    print(f"{'═'*67}")
    return tps


if __name__ == "__main__":
    # ── Header ────────────────────────────────────────────────────────────────

    TOTAL_TRANS     = NUM_ENVS * NUM_STEPS
    # SubprocVecEnv uses NUM_WORKERS envs; scale steps to same total transitions
    SUBPROC_STEPS   = TOTAL_TRANS // NUM_WORKERS

    print("═" * 67)
    print(f"  SB3 VecEnv — {ENV_ID} — no-algorithm throughput benchmark")
    print(f"  target: {NUM_ENVS} envs × {NUM_STEPS} steps = {TOTAL_TRANS:,} transitions")
    print(f"  warm-up: {WARMUP_STEPS} steps  ·  {NCORES} logical cores")
    print("═" * 67)

    # ── 1. DummyVecEnv (N envs, single process) ───────────────────────────────

    print(f"\nBuilding DummyVecEnv ({NUM_ENVS} envs, single process)…")
    t0 = time.perf_counter()
    dummy = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])
    print(f"  built in {time.perf_counter()-t0:.2f}s")

    tps_dummy = bench(
        dummy,
        f"SB3 DummyVecEnv — {NUM_ENVS} envs (single-process sequential)",
        NUM_ENVS, NUM_STEPS, WARMUP_STEPS,
    )
    dummy.close()

    # ── 2. SubprocVecEnv (NUM_WORKERS subprocs, 1 env each) ───────────────────

    print(f"\nBuilding SubprocVecEnv ({NUM_WORKERS} subprocesses, 1 env each)…")
    print(f"  steps scaled to {SUBPROC_STEPS:,} to match {TOTAL_TRANS:,} total transitions")
    t0 = time.perf_counter()
    subproc = SubprocVecEnv([make_env(i) for i in range(NUM_WORKERS)])
    print(f"  built in {time.perf_counter()-t0:.2f}s")

    tps_subproc = bench(
        subproc,
        f"SB3 SubprocVecEnv — {NUM_WORKERS} subprocesses × 1 env each",
        NUM_WORKERS, SUBPROC_STEPS, WARMUP_STEPS,
    )
    subproc.close()

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n{'═'*67}")
    print(f"  SUMMARY — {ENV_ID} — {TOTAL_TRANS:,} total transitions")
    print(f"{'═'*67}")
    print(f"  DummyVecEnv  ({NUM_ENVS:>5} envs, 1 proc)    : {tps_dummy:>10,.0f} t/s")
    print(f"  SubprocVecEnv ({NUM_WORKERS:>3} envs, {NUM_WORKERS} procs)   : {tps_subproc:>10,.0f} t/s")
    if tps_dummy > 0:
        print(f"  Subprocess IPC overhead             : {tps_subproc/tps_dummy:>10.3f}× vs DummyVecEnv")
    print(f"{'═'*67}")
