"""RelayRL Python bindings — scalar LunarLander-v3 — 32 envs — 50k steps — Python 3.13t (no-GIL).

Run:
  ORT_DYLIB_PATH=... PYTHON_GIL=0 python3.13t bench_beta2/relayrl_scalar_py32_ft.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "target/release"))

print("GIL enabled:", sys._is_gil_enabled())

import gymnasium as gym
import relayrl_pyo3 as rl

NUM_ENVS     = 32
TARGET_STEPS = 50_000
WARMUP_STEPS = 500

print("═" * 67)
print(f"  RelayRL (Python 3.13t no-GIL) — scalar LunarLander-v3 — {NUM_ENVS} envs")
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

SB3_DUMMY_SPS        = 23_222.0
SB3_SUBPROC_SPS      = 13_130.0
RELAYRL_RUST_SPS     = 1_403_081.0
RELAYRL_PY311_SCALAR = 13_261.0
RELAYRL_FT_SCALAR    = 10_360.0   # actual result, recorded here for reference

print("═" * 67)
print(f"  RelayRL (Python 3.13t no-GIL) — scalar — FINAL RESULTS  ({NUM_ENVS} envs)")
print("═" * 67)
print(f"  total env transitions    : {total_transitions:>12,}")
print(f"  wall time                : {wall:>12.2f} s")
print(f"  loop iters/sec           : {loop_iters_per_sec:>12.0f}")
print(f"  env transitions/sec      : {transitions_per_sec:>12.0f}")
print(f"  µs / env transition      : {us_per_transition:>12.3f}")
print()
print(f"─── vs baselines {'─'*49}")
print(f"  SB3 DummyVecEnv (Py 3.11): {SB3_DUMMY_SPS:>12.0f}  t/s")
print(f"  RelayRL scalar  (Py 3.11): {RELAYRL_PY311_SCALAR:>12.0f}  t/s")
print(f"  RelayRL scalar  (Rust)   : {RELAYRL_RUST_SPS:>12.0f}  t/s")
print(f"  RelayRL scalar  (3.13t)  : {transitions_per_sec:>12.0f}  t/s  ← this run")
print(f"  vs SB3 DummyVecEnv       : {transitions_per_sec/SB3_DUMMY_SPS:>12.2f}×")
print(f"  vs RelayRL scalar 3.11   : {transitions_per_sec/RELAYRL_PY311_SCALAR:>12.2f}×")
print(f"  vs RelayRL Rust scalar   : {transitions_per_sec/RELAYRL_RUST_SPS:>12.3f}×")
print("═" * 67)
