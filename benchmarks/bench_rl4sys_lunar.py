#!/usr/bin/env python3
"""
RL4Sys client benchmark on gymnasium LunarLander-v3.

Measures the same 40+ metrics as the Rust bench_lunar binary:
throughput, per-step latency percentiles (P50/P95/P99/P99.9), inference vs
env-step breakdown, scheduling overhead, RSS memory, and CPU/OS counters.

No training server is required.  The benchmark instantiates RLActorCritic
directly (the same model class used internally by RL4SysAgent) and manages
trajectory objects locally, so it measures the pure client-side pipeline cost
in the same way that bench_relayrl_gymnasium.py and bench_lunar.rs run fully
locally without a remote server.

Timing breakdown (mirrors bench_lunar.rs):
  infer_ns  = model.step() only              (policy forward pass)
  env_ns    = env.step() only                (physics simulation)
  step_ns   = full loop iteration            (obs tensor, infer, action
               decode, env step, RL4SysAction alloc, add_to_trajectory,
               reward update, episode bookkeeping)
  overhead  = step_ns - infer_ns - env_ns    (Python dispatch + trajectory
               object construction + GIL contention)

Usage:
    python benchmarks/bench_rl4sys_lunar.py
    python benchmarks/bench_rl4sys_lunar.py --target-steps 200000

Requirements:
    pip install gymnasium[box2d] torch
    pip install git+https://github.com/DIR-LAB/RL4Sys.git
"""

import argparse
import math
import os
import threading
import time
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from rl4sys.algorithms.PPO.kernel import RLActorCritic
from rl4sys.common.action import RL4SysAction
from rl4sys.common.trajectory import RL4SysTrajectory

# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL4Sys LunarLander comprehensive benchmark (server-free)"
    )
    p.add_argument(
        "--target-steps", type=int, default=100_000,
        help="Total env steps to collect (default: 100 000)",
    )
    p.add_argument(
        "--warmup-steps", type=int, default=200,
        help="Steps to discard as warm-up before timing starts (default: 200)",
    )
    p.add_argument(
        "--max-ep-steps", type=int, default=500,
        help="Maximum steps per episode (default: 500)",
    )
    return p.parse_args()


# ─── /proc helpers ───────────────────────────────────────────────────────────

class _ProcSample:
    __slots__ = (
        "rss_kb", "vol_ctx_sw", "nonvol_ctx_sw",
        "utime_ticks", "stime_ticks", "threads", "runq",
    )

    def __init__(self) -> None:
        self.rss_kb        = 0
        self.vol_ctx_sw    = 0
        self.nonvol_ctx_sw = 0
        self.utime_ticks   = 0
        self.stime_ticks   = 0
        self.threads       = 0
        self.runq: float   = 0.0


def _sample_proc() -> _ProcSample:
    s = _ProcSample()
    try:
        for line in open("/proc/self/status").read().splitlines():
            key, _, val = line.partition(":")
            val = val.strip()
            k   = key.strip()
            if k == "VmRSS":
                s.rss_kb = int(val.split()[0])
            elif k == "voluntary_ctxt_switches":
                s.vol_ctx_sw = int(val)
            elif k == "nonvoluntary_ctxt_switches":
                s.nonvol_ctx_sw = int(val)
            elif k == "Threads":
                s.threads = int(val)
    except Exception:
        pass
    try:
        fields = open("/proc/self/stat").read().split()
        s.utime_ticks = int(fields[13])
        s.stime_ticks = int(fields[14])
    except Exception:
        pass
    try:
        s.runq = float(open("/proc/loadavg").read().split()[0])
    except Exception:
        pass
    return s


def _start_proc_sampler(interval: float = 0.2):
    """Spawn a daemon thread that appends _ProcSample objects every *interval* s.

    Returns (samples_list, stop_event).
    """
    samples: List[_ProcSample] = []
    stop_evt = threading.Event()

    def _worker() -> None:
        while not stop_evt.is_set():
            samples.append(_sample_proc())
            time.sleep(interval)

    threading.Thread(target=_worker, daemon=True).start()
    return samples, stop_evt


# ─── Statistics helpers ───────────────────────────────────────────────────────

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _stddev(vals, m=None):
    if len(vals) < 2:
        return 0.0
    if m is None:
        m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _pct(sorted_vals, pct: float):
    """Return the *pct*-th percentile of a pre-sorted list."""
    if not sorted_vals:
        return 0
    idx = int(round(pct / 100.0 * (len(sorted_vals) - 1)))
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


# ─── Trajectory helpers (mirrors RL4SysAgent internals, no gRPC) ─────────────

def _new_traj(version: int = 0) -> RL4SysTrajectory:
    return RL4SysTrajectory(version=version)


def _make_action(obs_t, action_nd, data_dict, version: int = 0) -> RL4SysAction:
    return RL4SysAction(
        obs=obs_t,
        action=action_nd,
        reward=-1,
        done=False,
        mask=None,
        data=data_dict,
        version=version,
    )


def _end_episode(traj: RL4SysTrajectory, action: RL4SysAction) -> None:
    """Mirrors agent.mark_end_of_trajectory — no server send."""
    action.done = True
    traj.mark_completed()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args        = _parse_args()
    target_steps = args.target_steps
    warmup_steps = args.warmup_steps
    max_ep_steps = args.max_ep_steps
    num_cores    = os.cpu_count() or 1

    print("═" * 67)
    print("  RL4Sys — LunarLander comprehensive benchmark (server-free)")
    print(f"  1 actor · {target_steps:,} steps · {num_cores} logical cores")
    print("═" * 67)
    print()

    # ── Model + env ──────────────────────────────────────────────────────────
    # Instantiate the same RLActorCritic that RL4SysAgent uses internally.
    # actor_type='mlp' matches luna_conf.json (MLP policy, 8-dim obs, 4 actions).
    model = RLActorCritic(input_size=8, act_dim=4, actor_type="mlp")
    model.eval()

    env = gym.make("LunarLander-v3")

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"Warming up ({warmup_steps} steps)…")
    obs, _ = env.reset()
    traj   = _new_traj()
    action: Optional[RL4SysAction] = None
    warm   = 0

    while warm < warmup_steps:
        obs_t             = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_nd, ddict  = model.step(obs_t)
        act_val           = int(action_nd.flat[0]) if isinstance(action_nd, np.ndarray) \
                            else int(action_nd.item())
        action            = _make_action(obs_t, action_nd, ddict)
        obs, reward, terminated, truncated, _ = env.step(act_val)
        action.rew        = float(reward)
        traj.add_action(action)
        warm += 1
        if terminated or truncated or warm % max_ep_steps == 0:
            _end_episode(traj, action)
            obs, _ = env.reset()
            traj   = _new_traj()

    if not traj.is_completed() and action is not None:
        _end_episode(traj, action)
    obs, _ = env.reset()
    traj   = _new_traj()
    action = None
    print("Warm-up done. Starting benchmark…\n")

    # ── Storage (pre-allocate to reduce GC pressure during timing) ────────────
    infer_times_ns: List[int]   = []
    env_times_ns:   List[int]   = []
    step_times_ns:  List[int]   = []
    ep_returns:     List[float] = []
    ep_lengths:     List[int]   = []

    cur_return: float = 0.0
    cur_len:    int   = 0
    total_steps: int  = 0
    last_print:  int  = 0

    # ── Start /proc background sampler (200 ms, same as bench_lunar.rs) ──────
    proc_samples, stop_proc = _start_proc_sampler(interval=0.2)

    # ── Main collection loop ──────────────────────────────────────────────────
    t_start = time.perf_counter()

    while total_steps < target_steps:
        step_start = time.perf_counter_ns()

        # ── Build observation tensor ────────────────────────────────────────
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        # ── Inference: model.step() (mirrors request_for_action internals) ──
        infer_start = time.perf_counter_ns()
        action_nd, data_dict = model.step(obs_t)
        infer_ns = time.perf_counter_ns() - infer_start

        # ── Decode action ────────────────────────────────────────────────────
        act_val = int(action_nd.flat[0]) if isinstance(action_nd, np.ndarray) \
                  else int(action_nd.item())

        # ── Env step ────────────────────────────────────────────────────────
        env_start = time.perf_counter_ns()
        obs, reward, terminated, truncated, _ = env.step(act_val)
        env_ns = time.perf_counter_ns() - env_start

        # ── Trajectory bookkeeping (client-side overhead, no gRPC) ──────────
        action = _make_action(obs_t, action_nd, data_dict)
        action.rew = float(reward)
        traj.add_action(action)

        cur_return  += float(reward)
        cur_len     += 1
        total_steps += 1

        done = terminated or truncated or (cur_len >= max_ep_steps)
        if done:
            _end_episode(traj, action)
            ep_returns.append(cur_return)
            ep_lengths.append(cur_len)
            cur_return = 0.0
            cur_len    = 0
            obs, _     = env.reset()
            traj       = _new_traj()
            action     = None

        step_ns = time.perf_counter_ns() - step_start
        infer_times_ns.append(infer_ns)
        env_times_ns.append(env_ns)
        step_times_ns.append(step_ns)

        # ── Progress ────────────────────────────────────────────────────────
        if total_steps - last_print >= 10_000:
            elapsed = time.perf_counter() - t_start
            print(f"  [{total_steps:>7,} steps]  {total_steps / elapsed:.0f} steps/sec")
            last_print = total_steps

    elapsed_sec = time.perf_counter() - t_start
    stop_proc.set()

    # ── Finalize any open episode ─────────────────────────────────────────────
    if action is not None and not traj.is_completed() and cur_len > 0:
        _end_episode(traj, action)
        ep_returns.append(cur_return)
        ep_lengths.append(cur_len)

    # ─────────────────────────── Compute metrics ──────────────────────────────

    infer_sorted = sorted(infer_times_ns)
    env_sorted   = sorted(env_times_ns)
    step_sorted  = sorted(step_times_ns)

    # Inference
    infer_mean  = _mean(infer_times_ns)
    infer_std   = _stddev(infer_times_ns, infer_mean)
    infer_p50   = _pct(infer_sorted, 50.0)
    infer_p95   = _pct(infer_sorted, 95.0)
    infer_p99   = _pct(infer_sorted, 99.0)
    infer_p999  = _pct(infer_sorted, 99.9)

    # Env step
    env_mean    = _mean(env_times_ns)
    env_std     = _stddev(env_times_ns, env_mean)
    env_p50     = _pct(env_sorted, 50.0)
    env_p99     = _pct(env_sorted, 99.0)

    # Full step (tensor build + infer + env + trajectory ops)
    step_mean   = _mean(step_times_ns)
    step_std    = _stddev(step_times_ns, step_mean)
    step_p50    = _pct(step_sorted, 50.0)
    step_p95    = _pct(step_sorted, 95.0)
    step_p99    = _pct(step_sorted, 99.0)
    step_p999   = _pct(step_sorted, 99.9)
    jitter_ns   = step_p99 - step_p50

    # Overhead = step - infer - env  (Python dispatch, tensor alloc, RL4SysAction/Traj ops)
    overhead_mean  = max(0.0, step_mean - infer_mean - env_mean)
    overhead_ratio = overhead_mean / step_mean if step_mean > 0 else 0.0

    # Throughput
    steps_per_sec  = total_steps / elapsed_sec
    steps_per_core = steps_per_sec / num_cores

    # Episode statistics
    total_eps   = len(ep_returns)
    eps_per_sec = total_eps / elapsed_sec
    avg_ep_len  = _mean(ep_lengths)
    ep_ret_mean = _mean(ep_returns)
    ep_ret_std  = _stddev(ep_returns, ep_ret_mean)
    ep_ret_var  = ep_ret_std ** 2

    # /proc metrics
    samples      = proc_samples
    rss_vals     = [s.rss_kb for s in samples]
    rss_init_kb  = samples[0].rss_kb  if samples else 0
    rss_final_kb = samples[-1].rss_kb if samples else 0
    rss_peak_kb  = max(rss_vals)      if rss_vals else 0
    rss_mean_kb  = int(_mean(rss_vals)) if rss_vals else 0
    alloc_rate   = (rss_final_kb - rss_init_kb) / elapsed_sec if elapsed_sec else 0.0

    ctx_first    = (samples[0].vol_ctx_sw  + samples[0].nonvol_ctx_sw)  if samples else 0
    ctx_last     = (samples[-1].vol_ctx_sw + samples[-1].nonvol_ctx_sw) if samples else 0
    total_ctx    = max(0, ctx_last - ctx_first)
    ctx_per_sec  = total_ctx / elapsed_sec
    ctx_per_step = total_ctx / total_steps if total_steps else 0.0

    cpu_first    = (samples[0].utime_ticks  + samples[0].stime_ticks)  if samples else 0
    cpu_last     = (samples[-1].utime_ticks + samples[-1].stime_ticks) if samples else 0
    cpu_ticks    = max(0, cpu_last - cpu_first)
    cpu_util     = (cpu_ticks / 100.0) / elapsed_sec * 100.0
    cpu_per_core = cpu_util / num_cores

    thread_mean  = _mean([s.threads      for s in samples]) if samples else 0.0
    runq_mean    = _mean([float(s.runq)  for s in samples]) if samples else 0.0

    rss_mean_gb  = rss_mean_kb / (1024.0 * 1024.0)
    sps_per_gb   = steps_per_sec / rss_mean_gb if rss_mean_gb > 0 else 0.0

    # Relative scalability vs the Rust RelayRL 1-actor baseline
    RELAYRL_1A_BASELINE_SPS: float = 19_443.0
    scalability = steps_per_sec / RELAYRL_1A_BASELINE_SPS

    # ─────────────────────────── Print report ─────────────────────────────────
    print()
    print("═" * 67)
    print("  RL4Sys LunarLander — FINAL RESULTS  (1 actor)")
    print("═" * 67)
    print()

    print("─── Throughput ──────────────────────────────────────────────────────")
    print(f"  steps/sec (global)           : {steps_per_sec:>10.1f}")
    print(f"  steps/sec per actor          : {steps_per_sec:>10.1f}")
    print(f"  steps/sec per logical core   : {steps_per_core:>10.1f}")
    print(f"  episodes/sec                 : {eps_per_sec:>10.3f}")
    print(f"  total steps (all actors)     : {total_steps:>10}")
    print(f"  steps per actor              : {total_steps:>10}")
    print(f"  total episodes               : {total_eps:>10}")
    print(f"  wall time                    : {elapsed_sec:>10.2f}s")
    print(f"  logical cores                : {num_cores:>10}")
    print()

    print("─── Episode Statistics ───────────────────────────────────────────────")
    print(f"  avg steps per episode        : {avg_ep_len:>10.1f}")
    print(f"  episode return mean          : {ep_ret_mean:>10.3f}")
    print(f"  episode return std dev       : {ep_ret_std:>10.3f}")
    print(f"  episode completion variance  : {ep_ret_var:>10.3f}")
    print()

    print("─── Per-Step Timing (µs) ─────────────────────────────────────────────")
    print(f"  step mean (infer+env)        : {step_mean  / 1_000:>10.3f} µs")
    print(f"  step P50  (round/N)          : {step_p50   / 1_000:>10.3f} µs")
    print(f"  step P95  (round/N)          : {step_p95   / 1_000:>10.3f} µs")
    print(f"  step P99  (round/N)          : {step_p99   / 1_000:>10.3f} µs")
    print(f"  step P99.9                   : {step_p999  / 1_000:>10.3f} µs")
    print(f"  jitter (P99−P50)             : {jitter_ns  / 1_000:>10.3f} µs")
    print(f"  step std dev (infer)         : {step_std   / 1_000:>10.3f} µs")
    print()

    print("─── Inference Timing (µs) ────────────────────────────────────────────")
    print(f"  inference mean               : {infer_mean  / 1_000:>10.3f} µs")
    print(f"  inference std dev            : {infer_std   / 1_000:>10.3f} µs")
    print(f"  inference P50               : {infer_p50  / 1_000:>10.3f} µs")
    print(f"  inference P95               : {infer_p95  / 1_000:>10.3f} µs")
    print(f"  inference P99               : {infer_p99  / 1_000:>10.3f} µs")
    print(f"  inference P99.9             : {infer_p999 / 1_000:>10.3f} µs")
    print(f"  actor dispatch latency      ≈ {infer_p50  / 1_000:>10.3f} µs  (P50 inference)")
    print(f"  inference / step ratio       : {infer_mean / step_mean if step_mean else 0:>10.3f}")
    print()

    print("─── Env Step Timing (µs) ─────────────────────────────────────────────")
    print(f"  env step mean                : {env_mean / 1_000:>10.3f} µs")
    print(f"  env step std dev             : {env_std  / 1_000:>10.3f} µs")
    print(f"  env step P50                 : {env_p50  / 1_000:>10.3f} µs")
    print(f"  env step P99                 : {env_p99  / 1_000:>10.3f} µs")
    print(f"  env step / step ratio        : {env_mean / step_mean if step_mean else 0:>10.3f}")
    print()

    print("─── Scheduling / Overhead ────────────────────────────────────────────")
    print(f"  overhead per round           : {overhead_mean  / 1_000:>10.3f} µs")
    print(f"  overhead ratio               : {overhead_ratio:>10.3f}")
    print(f"  deadtime per actor           : {overhead_ratio:>10.3f}")
    print(f"  round P50                    : {step_p50  / 1_000:>10.3f} µs")
    print(f"  round P99                    : {step_p99  / 1_000:>10.3f} µs")
    print(f"  round std dev                : {step_std  / 1_000:>10.3f} µs")
    print(f"  action serialization         :   included in inference timing")
    print(f"  state update / buffer write  :   included in overhead (add_action)")
    print(f"  dropped/late updates         : {0:>10}")
    print()

    print("─── Memory ───────────────────────────────────────────────────────────")
    print(f"  RSS init                     : {rss_init_kb  / 1024:>7.1f} MB")
    print(f"  RSS peak                     : {rss_peak_kb  / 1024:>7.1f} MB")
    print(f"  RSS mean                     : {rss_mean_kb  / 1024:>7.1f} MB")
    print(f"  RSS final                    : {rss_final_kb / 1024:>7.1f} MB")
    print(f"  allocation rate (RSS Δ)      : {alloc_rate:>7.3f} KB/s")
    print(f"  /proc samples                : {len(samples):>10}")
    print()

    print("─── CPU / OS ─────────────────────────────────────────────────────────")
    print(f"  CPU utilisation (1 core %)   : {cpu_util:>10.2f}%")
    print(f"  CPU util / logical core      : {cpu_per_core:>10.2f}%")
    print(f"  mean threads                 : {thread_mean:>10.1f}")
    print(f"  mean run-queue (1-min avg)   : {runq_mean:>10.3f}")
    print(f"  context switches total       : {total_ctx:>10}")
    print(f"  context switches/sec         : {ctx_per_sec:>10.1f}")
    print(f"  context switches/step        : {ctx_per_step:>10.6f}")
    print()

    print("─── Efficiency Ratios ────────────────────────────────────────────────")
    print(f"  steps/sec / logical core     : {steps_per_core:>10.1f}")
    print(f"  steps/sec / GB RSS (proxy)   : {sps_per_gb:>10.1f}")
    print(f"  S(n) vs RelayRL 1-actor base : {scalability:>10.3f}")
    print(f"  overhead ratio               : {overhead_ratio:>10.3f}")
    print()

    print("─── Notes (hardware counters require perf) ───────────────────────────")
    print("  cache misses (L1/L2/L3)      : perf stat -e cache-misses,LLC-load-misses")
    print("  IPC                          : perf stat -e cycles,instructions")
    print("  memory bandwidth             : perf stat -e cache-references")
    print("  queue backlog / contention   : N/A — server-free local benchmark")
    print("  inter-thread msg latency     : N/A — no gRPC in this benchmark")
    print("  sync wait time               : included in overhead per round")
    print("  steps/sec / watt             : requires external power measurement")
    print("  allocator contention         : requires py-spy / tracemalloc profiling")
    print()
    print("═" * 67)

    env.close()


if __name__ == "__main__":
    main()
