"""SB3 SubprocVecEnv — LunarLander-v3 — 32 envs — pure step throughput.

No algorithm, no training.  Creates 32 parallel subprocess environments and
steps them with fixed zero-actions in a tight loop, measuring raw env-step
throughput — the same thing RelayRL's vec benchmarks measure.

Also runs a DummyVecEnv (single-process) pass for comparison.

Run:
  python3 sb3_lunarlander_32.py
"""

import time
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

NUM_ENVS      = 32
TARGET_STEPS  = 50_000   # loop iterations (each steps all NUM_ENVS at once)
WARMUP_STEPS  = 500
ENV_ID        = "LunarLander-v3"
ACT_DIM       = 1        # discrete: single int per env

def make_env(seed: int):
    def _init():
        env = gym.make(ENV_ID)
        env.reset(seed=seed)
        return env
    return _init


def run_bench(vec_env, label: str, warmup: int, steps: int) -> dict:
    obs = vec_env.reset()
    # Fixed action: always 0 (do nothing) — matches RelayRL's bootstrap model
    actions = np.zeros(NUM_ENVS, dtype=np.int32)

    print(f"  [{label}] warming up ({warmup} iters × {NUM_ENVS} envs)…")
    for _ in range(warmup):
        vec_env.step(actions)

    print(f"  [{label}] running {steps} iters × {NUM_ENVS} envs…")
    t0 = time.perf_counter()
    for _ in range(steps):
        vec_env.step(actions)
    wall = time.perf_counter() - t0

    total_transitions   = steps * NUM_ENVS
    transitions_per_sec = total_transitions / wall
    loop_iters_per_sec  = steps / wall
    us_per_transition   = 1_000_000.0 / transitions_per_sec

    return dict(
        wall=wall,
        total_transitions=total_transitions,
        transitions_per_sec=transitions_per_sec,
        loop_iters_per_sec=loop_iters_per_sec,
        us_per_transition=us_per_transition,
    )


def print_result(label: str, r: dict):
    print(f"\n─── {label} ─────────────────────────────────────────────────────")
    print(f"  total env transitions    : {r['total_transitions']:>12,}")
    print(f"  wall time                : {r['wall']:>12.2f} s")
    print(f"  loop iters/sec           : {r['loop_iters_per_sec']:>12.0f}")
    print(f"  env transitions/sec      : {r['transitions_per_sec']:>12.0f}")
    print(f"  µs / env transition      : {r['us_per_transition']:>12.3f}")


if __name__ == "__main__":
    import os
    num_cores = os.cpu_count() or 1

    print("═" * 67)
    print(f"  SB3 VecEnv — {ENV_ID} — pure step throughput")
    print(f"  {NUM_ENVS} envs · {TARGET_STEPS} loop iters · {num_cores} logical cores")
    print(f"  no algorithm — fixed action=0, measure env-step rate only")
    print("═" * 67)

    # ── SubprocVecEnv (N subprocesses, one env each) ──────────────────────────
    print(f"\nBuilding SubprocVecEnv ({NUM_ENVS} workers)…")
    subproc_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    subproc_result = run_bench(subproc_env, "SubprocVecEnv", WARMUP_STEPS, TARGET_STEPS)
    subproc_env.close()

    # ── DummyVecEnv (single process, sequential) ──────────────────────────────
    print(f"\nBuilding DummyVecEnv ({NUM_ENVS} envs, single process)…")
    dummy_env = DummyVecEnv([make_env(i) for i in range(NUM_ENVS)])
    dummy_result = run_bench(dummy_env, "DummyVecEnv", WARMUP_STEPS, TARGET_STEPS)
    dummy_env.close()

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 67}")
    print(f"  SB3 VecEnv — {ENV_ID} — FINAL RESULTS")
    print(f"{'═' * 67}")
    print(f"  envs                     : {NUM_ENVS}")
    print(f"  loop iterations          : {TARGET_STEPS:,}")
    print(f"  logical cores            : {num_cores}")

    print_result(f"SubprocVecEnv  ({NUM_ENVS} subprocesses)", subproc_result)
    print_result(f"DummyVecEnv    (1 process, sequential)", dummy_result)

    speedup = subproc_result['transitions_per_sec'] / dummy_result['transitions_per_sec']
    print(f"\n  SubprocVecEnv speedup vs DummyVecEnv : {speedup:.2f}×")
    print("═" * 67)
