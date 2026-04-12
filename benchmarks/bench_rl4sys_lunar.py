#!/usr/bin/env python3
"""
RL4Sys client benchmark on gymnasium LunarLander-v3.

Measures the same 40+ metrics as bench_lunar.rs: throughput, per-step
latency percentiles (P50/P95/P99/P99.9), inference vs env-step breakdown,
scheduling overhead, RSS memory, and CPU/OS counters.

No server required.  RLActorCritic, RL4SysAction, and RL4SysTrajectory are
inlined verbatim from the RL4Sys source (rl4sys/algorithms/PPO/kernel.py and
rl4sys/common/{action,trajectory}.py) so the benchmark requires only
torch + gymnasium[box2d] — same as every other benchmark in this repo.

Usage:
    python benchmarks/bench_rl4sys_lunar.py
    python benchmarks/bench_rl4sys_lunar.py --target-steps 200000
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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# ═══════════════════════════════════════════════════════════════════════════════
#  RL4Sys internals — inlined verbatim from:
#    rl4sys/algorithms/PPO/kernel.py   (RLActorCritic / RLActor / RLCritic)
#    rl4sys/common/action.py           (RL4SysAction)
#    rl4sys/common/trajectory.py       (RL4SysTrajectory)
#  Source: https://github.com/DIR-LAB/RL4Sys  (commit 95255c8)
# ═══════════════════════════════════════════════════════════════════════════════

def _mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class _RLActor(nn.Module):
    def __init__(self, input_size, act_dim,
                 mlp_hidden_sizes=(32, 16, 8),
                 activation=nn.ReLU, actor_type="mlp", job_features=8):
        super().__init__()
        self.input_size  = input_size
        self.act_dim     = act_dim
        self.job_features = job_features
        self.actor_type  = actor_type
        if actor_type == "attn":
            self.att_q   = nn.Linear(job_features, 32)
            self.att_k   = nn.Linear(job_features, 32)
            self.att_v   = nn.Linear(job_features, 32)
            self.att_fc16 = nn.Linear(32, 16)
            self.att_fc8  = nn.Linear(16, 8)
            self.att_out  = nn.Linear(8, 1)
        elif actor_type == "kernel":
            self.pol_fc1      = nn.Linear(job_features, 32)
            self.pol_fc2      = nn.Linear(32, 16)
            self.pol_fc3      = nn.Linear(16, 8)
            self.pol_fc_logits = nn.Linear(8, 1)
        else:  # "mlp"
            self.pi = _mlp([input_size] + list(mlp_hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, mask=None):
        if self.actor_type == "attn":
            obs_r  = obs.view(obs.shape[0], self.act_dim, self.job_features)
            logits = self._attention_logits(obs_r)
        elif self.actor_type == "kernel":
            obs_r  = obs.view(obs.shape[0], self.act_dim, self.job_features)
            logits = self._rl_kernel(obs_r)
        else:
            logits = self.pi(obs)
        if mask is not None:
            logits = logits + (mask - 1) * 1e6
        return Categorical(logits=logits)

    def _rl_kernel(self, obs):
        x = F.relu(self.pol_fc1(obs))
        x = F.relu(self.pol_fc2(x))
        x = F.relu(self.pol_fc3(x))
        return self.pol_fc_logits(x).squeeze(-1)

    def _attention_logits(self, obs_seq):
        q = F.relu(self.att_q(obs_seq))
        k = F.relu(self.att_k(obs_seq))
        v = F.relu(self.att_v(obs_seq))
        score   = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        attn    = torch.matmul(score, v)
        x = F.relu(self.att_fc16(attn))
        x = F.relu(self.att_fc8(x))
        return self.att_out(x).squeeze(-1)

    def forward(self, obs, act=None, mask=None):
        pi    = self._distribution(obs, mask)
        logp  = pi.log_prob(act) if act is not None else None
        return pi, logp


class _RLCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(128, 128), activation=nn.ReLU):
        super().__init__()
        self.v_net = _mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class RLActorCritic(nn.Module):
    """RL4Sys PPO actor-critic (verbatim from rl4sys/algorithms/PPO/kernel.py)."""
    def __init__(self, input_size: int, act_dim: int, actor_type: str = "kernel"):
        super().__init__()
        self.pi = _RLActor(input_size, act_dim, actor_type=actor_type)
        self.v  = _RLCritic(input_size)

    def step(self, obs, mask=None):
        with torch.no_grad():
            pi     = self.pi._distribution(obs, mask)
            a      = pi.sample()
            logp_a = pi.log_prob(a)
        return a.detach().cpu().numpy(), {"logp_a": logp_a.detach().cpu().numpy()}

    def get_model_name(self):
        return "PPO RLActorCritic"


class RL4SysAction:
    """RL4Sys action container (verbatim from rl4sys/common/action.py)."""
    def __init__(self, obs=None, action=None, reward=None,
                 done=None, mask=None, data=None, version=0):
        self.obs     = obs
        self.act     = action
        self.rew     = reward
        self.done    = done
        self.mask    = mask
        self.data    = data or {}
        self.version = version

    def update_reward(self, reward):
        self.rew = reward

    def set_done(self, done: bool):
        self.done = done


class RL4SysTrajectory:
    """RL4Sys trajectory container (verbatim from rl4sys/common/trajectory.py)."""
    def __init__(self, version: int = 0):
        self.actions       = []
        self.version       = version
        self.invalid_mixed = False
        self.completed     = False

    def add_action(self, action: RL4SysAction):
        if action.version != self.version:
            self.invalid_mixed = True
        self.actions.append(action)

    def mark_completed(self):
        self.completed = True

    def is_completed(self) -> bool:
        return self.completed

    def is_valid(self) -> bool:
        return not self.invalid_mixed

    def clear(self):
        self.actions       = []
        self.invalid_mixed = False


# ═══════════════════════════════════════════════════════════════════════════════
#  /proc helpers (mirrors bench_lunar.rs ProcSample + background sampler)
# ═══════════════════════════════════════════════════════════════════════════════

class _ProcSample:
    __slots__ = ("rss_kb", "vol_ctx_sw", "nonvol_ctx_sw",
                 "utime_ticks", "stime_ticks", "threads", "runq")
    def __init__(self):
        self.rss_kb = self.vol_ctx_sw = self.nonvol_ctx_sw = 0
        self.utime_ticks = self.stime_ticks = self.threads = 0
        self.runq: float = 0.0


def _sample_proc() -> _ProcSample:
    s = _ProcSample()
    try:
        for line in open("/proc/self/status").read().splitlines():
            k, _, v = line.partition(":")
            v = v.strip()
            if   k == "VmRSS":                    s.rss_kb        = int(v.split()[0])
            elif k == "voluntary_ctxt_switches":   s.vol_ctx_sw    = int(v)
            elif k == "nonvoluntary_ctxt_switches": s.nonvol_ctx_sw = int(v)
            elif k == "Threads":                   s.threads       = int(v)
    except Exception: pass
    try:
        f = open("/proc/self/stat").read().split()
        s.utime_ticks, s.stime_ticks = int(f[13]), int(f[14])
    except Exception: pass
    try:    s.runq = float(open("/proc/loadavg").read().split()[0])
    except Exception: pass
    return s


def _start_proc_sampler(interval=0.2):
    samples, stop = [], threading.Event()
    def _w():
        while not stop.is_set():
            samples.append(_sample_proc())
            time.sleep(interval)
    threading.Thread(target=_w, daemon=True).start()
    return samples, stop


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _mean(v):   return sum(v) / len(v) if v else 0.0
def _var(v, m): return sum((x-m)**2 for x in v) / (len(v)-1) if len(v) > 1 else 0.0
def _std(v, m=None):
    if len(v) < 2: return 0.0
    m = m if m is not None else _mean(v)
    return math.sqrt(_var(v, m))
def _pct(sv, p):
    if not sv: return 0
    return sv[min(int(round(p/100*(len(sv)-1))), len(sv)-1)]


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _args():
    p = argparse.ArgumentParser(description="RL4Sys LunarLander benchmark (server-free)")
    p.add_argument("--target-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-ep-steps", type=int, default=500)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args         = _args()
    target_steps = args.target_steps
    warmup_steps = args.warmup_steps
    max_ep_steps = args.max_ep_steps
    num_cores    = os.cpu_count() or 1

    print("═"*67)
    print("  RL4Sys — LunarLander comprehensive benchmark (server-free)")
    print(f"  1 actor · {target_steps:,} steps · {num_cores} logical cores")
    print("═"*67); print()

    # model: RLActorCritic(8, 4, actor_type="mlp") — matches luna_conf.json
    model = RLActorCritic(input_size=8, act_dim=4, actor_type="mlp")
    model.eval()
    env = gym.make("LunarLander-v3")

    # ── warm-up ──────────────────────────────────────────────────────────────
    print(f"Warming up ({warmup_steps} steps)…")
    obs, _ = env.reset()
    traj   = RL4SysTrajectory()
    action = None
    for w in range(warmup_steps):
        obs_t           = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        act_nd, ddict   = model.step(obs_t)
        act_val         = int(act_nd.flat[0])
        action          = RL4SysAction(obs_t, act_nd, reward=-1, done=False, data=ddict)
        obs, rew, term, trunc, _ = env.step(act_val)
        action.rew = float(rew)
        traj.add_action(action)
        if term or trunc or (w+1) % max_ep_steps == 0:
            action.done = True; traj.mark_completed()
            obs, _ = env.reset(); traj = RL4SysTrajectory()
    if not traj.is_completed() and action:
        action.done = True; traj.mark_completed()
    obs, _ = env.reset()
    traj   = RL4SysTrajectory()
    print("Warm-up done. Starting benchmark…\n")

    # ── storage ───────────────────────────────────────────────────────────────
    infer_ns_list: List[int] = []
    env_ns_list:   List[int] = []
    step_ns_list:  List[int] = []
    ep_returns:    List[float] = []
    ep_lengths:    List[int]  = []
    cur_ret, cur_len = 0.0, 0
    total_steps, last_print = 0, 0

    proc_samples, stop_proc = _start_proc_sampler(0.2)
    t_start = time.perf_counter()

    # ── main loop ─────────────────────────────────────────────────────────────
    while total_steps < target_steps:
        step_t0 = time.perf_counter_ns()

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        t0 = time.perf_counter_ns()
        act_nd, ddict = model.step(obs_t)
        infer_ns = time.perf_counter_ns() - t0

        act_val = int(act_nd.flat[0])

        t0 = time.perf_counter_ns()
        obs, rew, term, trunc, _ = env.step(act_val)
        env_ns = time.perf_counter_ns() - t0

        # trajectory bookkeeping (client-side overhead, no gRPC)
        action = RL4SysAction(obs_t, act_nd, reward=float(rew), done=False, data=ddict)
        traj.add_action(action)

        cur_ret  += float(rew)
        cur_len  += 1
        total_steps += 1

        done = term or trunc or (cur_len >= max_ep_steps)
        if done:
            action.done = True; traj.mark_completed()
            ep_returns.append(cur_ret); ep_lengths.append(cur_len)
            cur_ret, cur_len = 0.0, 0
            obs, _ = env.reset(); traj = RL4SysTrajectory()

        step_ns = time.perf_counter_ns() - step_t0
        infer_ns_list.append(infer_ns)
        env_ns_list.append(env_ns)
        step_ns_list.append(step_ns)

        if total_steps - last_print >= 10_000:
            el = time.perf_counter() - t_start
            print(f"  [{total_steps:>7,} steps]  {total_steps/el:.0f} steps/sec")
            last_print = total_steps

    elapsed = time.perf_counter() - t_start
    stop_proc.set()

    # ── metrics ───────────────────────────────────────────────────────────────
    si = sorted(infer_ns_list); se = sorted(env_ns_list); ss = sorted(step_ns_list)

    im = _mean(infer_ns_list); ist = _std(infer_ns_list, im)
    em = _mean(env_ns_list);   est = _std(env_ns_list, em)
    sm = _mean(step_ns_list);  sst = _std(step_ns_list, sm)

    ip50  = _pct(si, 50);  ip95 = _pct(si, 95);  ip99 = _pct(si, 99);  ip999 = _pct(si, 99.9)
    ep50  = _pct(se, 50);  ep99 = _pct(se, 99)
    sp50  = _pct(ss, 50);  sp95 = _pct(ss, 95);  sp99 = _pct(ss, 99);  sp999 = _pct(ss, 99.9)
    jitter = sp99 - sp50

    overhead_m = max(0.0, sm - im - em)
    overhead_r = overhead_m / sm if sm else 0.0

    sps  = total_steps / elapsed
    spsc = sps / num_cores

    tot_eps = len(ep_returns); eps_s = tot_eps / elapsed
    avg_el  = _mean(ep_lengths)
    erm     = _mean(ep_returns); ers = _std(ep_returns, erm); erv = ers**2

    smp = proc_samples
    rss = [s.rss_kb for s in smp]
    ri = smp[0].rss_kb if smp else 0; rf = smp[-1].rss_kb if smp else 0
    rp = max(rss) if rss else 0;      rm = int(_mean(rss)) if rss else 0
    ar = (rf - ri) / elapsed if elapsed else 0.0

    ctx0 = (smp[0].vol_ctx_sw  + smp[0].nonvol_ctx_sw)  if smp else 0
    ctx1 = (smp[-1].vol_ctx_sw + smp[-1].nonvol_ctx_sw) if smp else 0
    tc   = max(0, ctx1 - ctx0); cps = tc / elapsed; cpst = tc / total_steps if total_steps else 0

    cpu0 = (smp[0].utime_ticks  + smp[0].stime_ticks)  if smp else 0
    cpu1 = (smp[-1].utime_ticks + smp[-1].stime_ticks) if smp else 0
    cput = max(0, cpu1 - cpu0)
    cu   = (cput / 100.0) / elapsed * 100.0; cuc = cu / num_cores

    thm  = _mean([s.threads    for s in smp]) if smp else 0.0
    rqm  = _mean([float(s.runq) for s in smp]) if smp else 0.0
    rmgb = rm / (1024.0*1024.0); spsgb = sps / rmgb if rmgb else 0.0
    scal = sps / 19_443.0   # vs RelayRL 1-actor Rust baseline

    # ── report ────────────────────────────────────────────────────────────────
    print(); print("═"*67)
    print("  RL4Sys LunarLander — FINAL RESULTS  (1 actor)")
    print("═"*67); print()

    print("─── Throughput ──────────────────────────────────────────────────────")
    print(f"  steps/sec (global)           : {sps:>10.1f}")
    print(f"  steps/sec per actor          : {sps:>10.1f}")
    print(f"  steps/sec per logical core   : {spsc:>10.1f}")
    print(f"  episodes/sec                 : {eps_s:>10.3f}")
    print(f"  total steps (all actors)     : {total_steps:>10}")
    print(f"  steps per actor              : {total_steps:>10}")
    print(f"  total episodes               : {tot_eps:>10}")
    print(f"  wall time                    : {elapsed:>10.2f}s")
    print(f"  logical cores                : {num_cores:>10}")
    print()

    print("─── Episode Statistics ───────────────────────────────────────────────")
    print(f"  avg steps per episode        : {avg_el:>10.1f}")
    print(f"  episode return mean          : {erm:>10.3f}")
    print(f"  episode return std dev       : {ers:>10.3f}")
    print(f"  episode completion variance  : {erv:>10.3f}")
    print()

    print("─── Per-Step Timing (µs) ─────────────────────────────────────────────")
    print(f"  step mean (infer+env)        : {sm/1e3:>10.3f} µs")
    print(f"  step P50  (round/N)          : {sp50/1e3:>10.3f} µs")
    print(f"  step P95  (round/N)          : {sp95/1e3:>10.3f} µs")
    print(f"  step P99  (round/N)          : {sp99/1e3:>10.3f} µs")
    print(f"  step P99.9                   : {sp999/1e3:>10.3f} µs")
    print(f"  jitter (P99−P50)             : {jitter/1e3:>10.3f} µs")
    print(f"  step std dev (infer)         : {sst/1e3:>10.3f} µs")
    print()

    print("─── Inference Timing (µs) ────────────────────────────────────────────")
    print(f"  inference mean               : {im/1e3:>10.3f} µs")
    print(f"  inference std dev            : {ist/1e3:>10.3f} µs")
    print(f"  inference P50               : {ip50/1e3:>10.3f} µs")
    print(f"  inference P95               : {ip95/1e3:>10.3f} µs")
    print(f"  inference P99               : {ip99/1e3:>10.3f} µs")
    print(f"  inference P99.9             : {ip999/1e3:>10.3f} µs")
    print(f"  actor dispatch latency      ≈ {ip50/1e3:>10.3f} µs  (P50 inference)")
    print(f"  inference / step ratio       : {im/sm if sm else 0:>10.3f}")
    print()

    print("─── Env Step Timing (µs) ─────────────────────────────────────────────")
    print(f"  env step mean                : {em/1e3:>10.3f} µs")
    print(f"  env step std dev             : {est/1e3:>10.3f} µs")
    print(f"  env step P50                 : {ep50/1e3:>10.3f} µs")
    print(f"  env step P99                 : {ep99/1e3:>10.3f} µs")
    print(f"  env step / step ratio        : {em/sm if sm else 0:>10.3f}")
    print()

    print("─── Scheduling / Overhead ────────────────────────────────────────────")
    print(f"  overhead per round           : {overhead_m/1e3:>10.3f} µs")
    print(f"  overhead ratio               : {overhead_r:>10.3f}")
    print(f"  deadtime per actor           : {overhead_r:>10.3f}")
    print(f"  round P50                    : {sp50/1e3:>10.3f} µs")
    print(f"  round P99                    : {sp99/1e3:>10.3f} µs")
    print(f"  round std dev                : {sst/1e3:>10.3f} µs")
    print(f"  action serialization         :   included in inference timing")
    print(f"  state update / buffer write  :   included in overhead (add_action)")
    print(f"  dropped/late updates         : {0:>10}")
    print()

    print("─── Memory ───────────────────────────────────────────────────────────")
    print(f"  RSS init                     : {ri/1024:>7.1f} MB")
    print(f"  RSS peak                     : {rp/1024:>7.1f} MB")
    print(f"  RSS mean                     : {rm/1024:>7.1f} MB")
    print(f"  RSS final                    : {rf/1024:>7.1f} MB")
    print(f"  allocation rate (RSS Δ)      : {ar:>7.3f} KB/s")
    print(f"  /proc samples                : {len(smp):>10}")
    print()

    print("─── CPU / OS ─────────────────────────────────────────────────────────")
    print(f"  CPU utilisation (1 core %)   : {cu:>10.2f}%")
    print(f"  CPU util / logical core      : {cuc:>10.2f}%")
    print(f"  mean threads                 : {thm:>10.1f}")
    print(f"  mean run-queue (1-min avg)   : {rqm:>10.3f}")
    print(f"  context switches total       : {tc:>10}")
    print(f"  context switches/sec         : {cps:>10.1f}")
    print(f"  context switches/step        : {cpst:>10.6f}")
    print()

    print("─── Efficiency Ratios ────────────────────────────────────────────────")
    print(f"  steps/sec / logical core     : {spsc:>10.1f}")
    print(f"  steps/sec / GB RSS (proxy)   : {spsgb:>10.1f}")
    print(f"  S(n) vs RelayRL 1-actor base : {scal:>10.3f}")
    print(f"  overhead ratio               : {overhead_r:>10.3f}")
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
    print(); print("═"*67)

    env.close()


if __name__ == "__main__":
    main()
