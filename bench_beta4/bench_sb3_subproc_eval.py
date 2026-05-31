#!/usr/bin/env python3
"""
bench_sb3_subproc_eval.py — Pure env-stepping benchmark using SB3 SubprocVecEnv.

1024 parallel LunarLander-v3 envs, random discrete actions (no model/algorithm),
500-step warmup + 5000-step timed run.  Reports identical metrics to the Rust
bench_lunar_eval_py binary so results can be compared directly.

Run:
  /usr/bin/time -v python3 bench_sb3_subproc_eval.py
"""

import os
import time

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym

ENV_COUNT   = 1024
WARMUP_STEPS = 500
TIMED_STEPS  = 5_000
ACT_N        = 4   # LunarLander-v3 discrete action count


def _make_env(rank: int):
    def _init():
        env = gym.make("LunarLander-v3")
        env.reset(seed=rank)
        return env
    return _init


def _read_proc_stats() -> dict:
    s = {"rss_kb": 0, "vol_ctx": 0, "nvol_ctx": 0, "minor_faults": 0, "major_faults": 0}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                v = v.strip()
                if   k == "VmRSS":                        s["rss_kb"]   = int(v.split()[0])
                elif k == "voluntary_ctxt_switches":       s["vol_ctx"]  = int(v)
                elif k == "nonvoluntary_ctxt_switches":    s["nvol_ctx"] = int(v)
    except OSError:
        pass
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().split()
            s["minor_faults"] = int(fields[9])
            s["major_faults"] = int(fields[11])
    except OSError:
        pass
    return s


def main() -> None:
    num_cores = os.cpu_count() or 1

    print("══════════════════════════════════════════════════════════════════")
    print(f"  SB3 SubprocVecEnv — eval — LunarLander-v3 — {ENV_COUNT} envs")
    print(f"  backend : random discrete actions (no model inference)")
    print(f"  warmup  : {WARMUP_STEPS} steps × {ENV_COUNT} envs = {WARMUP_STEPS * ENV_COUNT} transitions")
    print(f"  timed   : {TIMED_STEPS} steps × {ENV_COUNT} envs = {TIMED_STEPS * ENV_COUNT} transitions")
    print(f"  cores   : {num_cores} logical")
    print("══════════════════════════════════════════════════════════════════\n")

    print(f"Creating {ENV_COUNT} SubprocVecEnv workers…")
    env = SubprocVecEnv([_make_env(i) for i in range(ENV_COUNT)], start_method="fork")
    env.reset()
    print(f"SubprocVecEnv OK — {ENV_COUNT} LunarLander-v3 sub-envs registered\n")

    rng = np.random.default_rng(0)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"Warming up ({WARMUP_STEPS} steps × {ENV_COUNT} envs)…")
    t_warmup = time.perf_counter()
    for _ in range(WARMUP_STEPS):
        actions = rng.integers(0, ACT_N, size=ENV_COUNT, dtype=np.int32)
        env.step(actions)
    warmup_wall = time.perf_counter() - t_warmup
    warmup_trans = WARMUP_STEPS * ENV_COUNT
    print(f"Warm-up done in {warmup_wall:.2f}s  ({warmup_trans / warmup_wall:.0f} env transitions/sec)\n")

    # ── Baseline /proc snapshot ───────────────────────────────────────────────
    before = _read_proc_stats()

    # ── Timed run ─────────────────────────────────────────────────────────────
    print(f"Starting timed run ({TIMED_STEPS} steps × {ENV_COUNT} envs)…")
    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        actions = rng.integers(0, ACT_N, size=ENV_COUNT, dtype=np.int32)
        env.step(actions)
    wall = time.perf_counter() - t0

    # ── Post-run /proc snapshot ───────────────────────────────────────────────
    after = _read_proc_stats()

    env.close()

    # ── Derived metrics ───────────────────────────────────────────────────────
    total_transitions = TIMED_STEPS * ENV_COUNT
    steps_per_sec     = TIMED_STEPS / wall
    transitions_sec   = total_transitions / wall
    us_per_step       = 1_000_000.0 / steps_per_sec
    us_per_transition = 1_000_000.0 / transitions_sec

    vol_delta   = max(0, after["vol_ctx"]   - before["vol_ctx"])
    nvol_delta  = max(0, after["nvol_ctx"]  - before["nvol_ctx"])
    total_ctx   = vol_delta + nvol_delta
    minor_delta = max(0, after["minor_faults"] - before["minor_faults"])
    major_delta = max(0, after["major_faults"] - before["major_faults"])

    print()
    print("══════════════════════════════════════════════════════════════════")
    print(f"  RESULTS — LunarLander-v3 eval — SB3 SubprocVecEnv — {ENV_COUNT} envs")
    print("══════════════════════════════════════════════════════════════════")

    print()
    print("─── Throughput ────────────────────────────────────────────────────")
    print(f"  env count              : {ENV_COUNT:>10}")
    print(f"  loop steps (timed)     : {TIMED_STEPS:>10}")
    print(f"  total env transitions  : {total_transitions:>10}")
    print(f"  wall time              : {wall:>10.3f} s")
    print(f"  steps / sec            : {steps_per_sec:>10.1f}")
    print(f"  env transitions / sec  : {transitions_sec:>10.1f}")
    print(f"  µs / step              : {us_per_step:>10.3f}")
    print(f"  µs / env transition    : {us_per_transition:>10.3f}")

    print()
    print("─── Memory ────────────────────────────────────────────────────────")
    print(f"  RSS (after run)        : {after['rss_kb'] / 1024.0:>8.1f} MB")
    print(f"  minor page faults (Δ) : {minor_delta:>10}")
    print(f"  major page faults (Δ) : {major_delta:>10}")

    print()
    print("─── OS scheduling ─────────────────────────────────────────────────")
    print(f"  vol ctx switches  (Δ) : {vol_delta:>10}")
    print(f"  nvol ctx switches (Δ) : {nvol_delta:>10}")
    print(f"  total ctx switches(Δ) : {total_ctx:>10}")
    print(f"  ctx switches / step   : {total_ctx / TIMED_STEPS:>10.4f}")
    print(f"  logical cores         : {num_cores:>10}")

    print()
    print("─── Timing breakdown ──────────────────────────────────────────────")
    print(f"  warmup  ({WARMUP_STEPS:>5} steps): {warmup_wall:>8.2f} s  ({warmup_trans / warmup_wall:.0f} transitions/sec)")
    print(f"  timed   ({TIMED_STEPS:>5} steps): {wall:>8.2f} s  ({transitions_sec:.0f} transitions/sec)")
    print("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
