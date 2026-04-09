#!/usr/bin/env python3
"""
RelayRL IPPO benchmark using Python gymnasium LunarLander-v3.

The Rust RelayRL agent handles inference (forward pass) and training (PPO
gradient updates) while Python drives the gymnasium environment loop.

Usage:
    python benchmarks/bench_relayrl_gymnasium.py

Requirements:
    pip install gymnasium[box2d] relayrl_pyo3
"""
import os
import sys
import time
import shutil

import gymnasium as gym
import relayrl_pyo3

# ─── Config ──────────────────────────────────────────────────────────────────

EPOCHS         = 20
TRAJ_PER_EPOCH = 8
MAX_STEPS      = 1000
MODEL_PATH     = "./model_lunar"       # must exist (created by relayrl-e2e)
TRAJ_DIR       = "./trajectories_lunar_pyo3"
SAVE_DIR       = "./trained_model_lunar_pyo3"

# ─── Init ─────────────────────────────────────────────────────────────────────

# Clear stale trajectory files from prior runs.
if os.path.isdir(TRAJ_DIR):
    shutil.rmtree(TRAJ_DIR)
os.makedirs(TRAJ_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

agent = relayrl_pyo3.RelayRLPPOAgent(
    obs_dim=8,
    act_dim=4,
    model_path=MODEL_PATH,
    traj_dir=TRAJ_DIR,
    save_model_dir=SAVE_DIR,
    traj_per_epoch=TRAJ_PER_EPOCH,
)

env = gym.make("LunarLander-v3")

t0          = time.perf_counter()
total_steps = 0
ep_rets     = []

# ─── Training loop ────────────────────────────────────────────────────────────

for epoch in range(EPOCHS):
    ep_count = 0
    while ep_count < TRAJ_PER_EPOCH:
        obs, _  = env.reset()
        ep_ret  = 0.0
        ep_len  = 0
        for _ in range(MAX_STEPS):
            action = agent.get_action(obs.tolist())
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_ret      += float(reward)
            total_steps += 1
            ep_len      += 1
            if terminated or truncated:
                agent.end_episode(float(reward))
                ep_rets.append(ep_ret)
                ep_count += 1
                break

    avg_ret = sum(ep_rets[-TRAJ_PER_EPOCH:]) / TRAJ_PER_EPOCH
    print(f"Epoch {epoch + 1:3d}/{EPOCHS}  AvgRet={avg_ret:8.1f}  Steps={total_steps}")

# ─── Summary ──────────────────────────────────────────────────────────────────

elapsed = time.perf_counter() - t0
agent.shutdown()
env.close()

print()
print("=" * 55)
print(f"RelayRL IPPO (gymnasium LunarLander-v3)")
print(f"  Total steps : {total_steps}")
print(f"  Wall time   : {elapsed:.2f}s")
print(f"  Steps/sec   : {total_steps / elapsed:.0f}")
print("=" * 55)
