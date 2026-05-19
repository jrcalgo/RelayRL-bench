"""RelayRL Python bindings — scalar LunarLander-v3 — 32 envs — 50k steps.

Uses PyScalarEnv: the framework clones a gymnasium env ×32 via the factory
callable, running each one sequentially through the Rust ScalarVecEnv path.

Run:
  ORT_DYLIB_PATH=... python3 bench_beta2/relayrl_scalar_py_32.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bench_beta2/target/release"))

import gymnasium as gym
import relayrl_pyo3 as rl

NUM_ENVS     = 32
TARGET_STEPS = 50_000
WARMUP_STEPS = 500

print("═" * 67)
print(f"  RelayRL (Python) — scalar LunarLander-v3 — {NUM_ENVS} envs")
print(f"  {TARGET_STEPS} loop iters · {NUM_ENVS * TARGET_STEPS:,} total transitions")
print(f"  path: PyScalarEnv → ScalarVecEnv (sequential)")
print("═" * 67)

agent = rl.RelayRLAgent(obs_dim=8, act_dim=4, actor_count=1)
ids   = agent.get_actor_ids()

factory = lambda: gym.make("LunarLander-v3")
agent.set_scalar_env(ids[0], factory=factory,
                     obs_dim=8, act_dim=4, discrete=True, count=NUM_ENVS)

print(f"set_scalar_env OK — {NUM_ENVS} envs registered to actor {ids[0]}")

print(f"Warming up ({WARMUP_STEPS} iters)…")
agent.run_env(ids[0], WARMUP_STEPS)
print("Warm-up done. Starting timed run…\n")

t0   = time.perf_counter()
agent.run_env(ids[0], TARGET_STEPS)
wall = time.perf_counter() - t0

total_transitions   = TARGET_STEPS * NUM_ENVS
transitions_per_sec = total_transitions / wall
loop_iters_per_sec  = TARGET_STEPS / wall
us_per_transition   = 1_000_000.0 / transitions_per_sec

agent.shutdown()

SB3_DUMMY_SPS    = 23_222.0
SB3_SUBPROC_SPS  = 13_130.0
RELAYRL_RUST_SPS = 1_403_081.0

print("═" * 67)
print(f"  RelayRL (Python) — scalar — FINAL RESULTS  ({NUM_ENVS} envs)")
print("═" * 67)
print(f"  total env transitions    : {total_transitions:>12,}")
print(f"  wall time                : {wall:>12.2f} s")
print(f"  loop iters/sec           : {loop_iters_per_sec:>12.0f}")
print(f"  env transitions/sec      : {transitions_per_sec:>12.0f}")
print(f"  µs / env transition      : {us_per_transition:>12.3f}")
print()
print(f"─── vs baselines ({'─'*44}")
print(f"  SB3 DummyVecEnv (Python) : {SB3_DUMMY_SPS:>12.0f}  t/s")
print(f"  SB3 SubprocVecEnv (Py)   : {SB3_SUBPROC_SPS:>12.0f}  t/s")
print(f"  RelayRL scalar (Rust)    : {RELAYRL_RUST_SPS:>12.0f}  t/s")
print(f"  RelayRL scalar (Python)  : {transitions_per_sec:>12.0f}  t/s  ← this run")
print(f"  speedup vs SB3 Dummy     : {transitions_per_sec/SB3_DUMMY_SPS:>12.2f}×")
print(f"  speedup vs SB3 Subproc   : {transitions_per_sec/SB3_SUBPROC_SPS:>12.2f}×")
print(f"  vs RelayRL Rust scalar   : {transitions_per_sec/RELAYRL_RUST_SPS:>12.3f}×  (GIL cost)")
print("═" * 67)
