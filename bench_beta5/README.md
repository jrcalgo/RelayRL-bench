# bench_beta5 — RelayRL Benchmark Suite

Benchmarks for **RelayRL 0.5.0-beta.5** covering latency, throughput scaling, and PPO training
convergence on LunarLander-v3 and GridWorld environments. All binaries target the
`bench-beta5` Cargo package and are compiled in `--release` mode.

---

## Quick reference

| Binary | Category | Env | Backend | Scale |
|---|---|---|---|---|
| [bench_lunar_direct_scalar1](#bench_lunar_direct_scalar1) | Latency | Scalar | NdArray+ONNX | 1 env · 100 k iters |
| [bench_request_action_latency](#bench_request_action_latency) | Latency | None | NdArray+ONNX | synthetic · 100 k calls |
| [bench_start_latency](#bench_start_latency) | Latency | None | LibTorch | single run |
| [bench_lunar_direct_scalar8192](#bench_lunar_direct_scalar8192) | Throughput · Direct | Scalar×8192 | NdArray+ONNX | 8 192 envs · 5 k steps |
| [bench_lunar_set_env_scalar1](#bench_lunar_set_env_scalar1) | Throughput · set_env | Scalar | NdArray+ONNX | 1 env · 100 k steps |
| [bench_lunar_set_env_scalar8192](#bench_lunar_set_env_scalar8192) | Throughput · set_env | Scalar×8192 | NdArray+ONNX | 8 192 envs · 5 k steps |
| [bench_lunar_set_env_scalar16384](#bench_lunar_set_env_scalar16384) | Throughput · set_env | Scalar×16384 | NdArray+ONNX | 16 384 envs · 5 k steps |
| [bench_lunar_set_env_vec32](#bench_lunar_set_env_vec32) | Throughput · VecEnv | VecEnv×32 | NdArray+ONNX | 32 envs · 500 k steps |
| [bench_lunar_set_env_vec1024](#bench_lunar_set_env_vec1024) | Throughput · VecEnv | VecEnv×1024 | NdArray+ONNX | 1 024 envs · 50 k steps |
| [bench_lunar_set_env_vec4096](#bench_lunar_set_env_vec4096) | Throughput · VecEnv | VecEnv×4096 | NdArray+ONNX | 4 096 envs · 10 k steps |
| [bench_lunar_set_env_vec8192](#bench_lunar_set_env_vec8192) | Throughput · VecEnv | VecEnv×8192 | NdArray+ONNX | 8 192 envs · 5 k steps |
| [bench_lunar_eval_py](#bench_lunar_eval_py) | Throughput · Python | gymnasium | NdArray+ONNX | 1 024 envs · 5 k steps |
| [bench_lunar_eval_envpool](#bench_lunar_eval_envpool) | Throughput · EnvPool | EnvPool | NdArray+ONNX | 1 024 envs (CLI) · 5 k steps |
| [bench_lunar_eval_envpool_tch](#bench_lunar_eval_envpool_tch) | Throughput · EnvPool | EnvPool | LibTorch+TorchScript | 1 024 envs · 5 k steps |
| [bench_lunar_ppo_1env](#bench_lunar_ppo_1env) | Training · PPO | Scalar | NdArray+ONNX | 1 env · 100 k steps |
| [bench_lunar_ppo_64env](#bench_lunar_ppo_64env) | Training · PPO | Scalar×64 | NdArray+ONNX | 64 envs · ~100 k frames |
| [bench_lunar_ppo_scalar1](#bench_lunar_ppo_scalar1) | Training · PPO | Scalar×64 | NdArray+ONNX | 64 envs · 1.5 M frames |
| [bench_lunar_ppo_tch](#bench_lunar_ppo_tch) | Training · PPO | Scalar×64 | LibTorch | 64 envs · 600 k steps |
| [bench_lunar_ppo_py](#bench_lunar_ppo_py) | Training · PPO | gymnasium×64 | LibTorch | 64 envs · 600 k steps |
| [bench_lunar_sfppo_py](#bench_lunar_sfppo_py) | Training · SFPPO | gymnasium×64 | LibTorch | 64 envs · 600 k steps |
| [bench_grid_ppo_scalar1](#bench_grid_ppo_scalar1) | Training · PPO | GridWorld×64 | NdArray | 64 envs · ~500 k frames |

---

## Latency micro-benchmarks

### bench_lunar_direct_scalar1

Measures the **single-actor request-action round-trip latency** through the coordinator dispatch
path. One scalar `LunarLanderEnv` is stepped in a tight loop of 100 000 iterations (10 000
warm-up): each iteration calls `agent.request_action()`, receives a discrete action via argmax,
applies it to the environment, and calls `flag_last_action()`. There is no training; the policy
comes from a pre-loaded ONNX model (8→64→64→4). Reports **iters/sec**, **µs/iter**, **ns/iter**,
RSS, and context switches per iteration. This is the canonical baseline for the coordinator hot
path with a single live environment.

---

### bench_request_action_latency

Isolates the **coordinator `request_action()` dispatch overhead** without any environment
stepping or observation generation. A fixed synthetic observation tensor of shape `[1, 8]` is
reused across 100 000 calls (10 000 warm-up) so that allocation cost is excluded. Reports
**calls/sec**, **µs/call**, **ns/call**, RSS, and context switches per call. This is the tightest
possible microbench for measuring RwLock or HotPathChannels dispatch latency in isolation.

---

### bench_start_latency

Measures the **agent framework cold-start cost** — specifically the wall time for
`agent.build()`, `agent.start()`, and `agent.shutdown()` in sequence, with no environment or
inference loop involved. Uses the LibTorch backend. Reports each phase in milliseconds plus the
total. Useful for quantifying how much framework initialization contributes to startup latency
in production deployments.

---

## Throughput — direct integration path

### bench_lunar_direct_scalar8192

Benchmarks **bulk rayon-parallel inference via the direct integration path** with 8 192
independent scalar `LunarLanderEnv` instances. Each step fans out via Rayon: observations are
collected in a zero-copy parallel pass, dispatched to `agent.request_action()` as a single
batched `[8192×8]` ONNX inference call, and actions are unpacked via manual argmax before being
applied to their respective environments in a second parallel pass. 500 warm-up + 5 000 timed
steps. Reports **env transitions/sec**, **µs/transition**, RSS, and context switches per step.
Establishes the ceiling for the direct hot path at large environment counts.

---

## Throughput — set_env / run_env path

These benchmarks use the framework-owned `set_env` + `run_env_eval` loop where the framework
drives the step-inference cycle. They are the reference path for production deployments.

### bench_lunar_set_env_scalar1

**Apples-to-apples comparison** against `bench_lunar_direct_scalar1` for the framework-internal
`set_env`/`run_env_eval` path with a single scalar environment. 10 000 warm-up + 100 000 timed
iterations. Reports iters/sec, env transitions/sec, µs/iter, RSS, and context switches per
iteration. The gap between this and `direct_scalar1` quantifies the framework overhead of the
`set_env` code path versus calling `request_action` directly.

---

### bench_lunar_set_env_scalar8192

Exercises the `set_env`/`run_env_eval` path with **8 192 cloned scalar environments** stepped in
parallel via Rayon inside `ScalarVecEnv`. Observations are collected into a contiguous flat
buffer, batched ONNX inference produces `[8192×4]` logits, and discrete actions are decoded via
argmax before parallel `step` calls. 500 warm-up + 5 000 timed steps. Reports env
transitions/sec, µs/transition, RSS, and context switches. Directly comparable to
`bench_lunar_direct_scalar8192` to measure set_env path overhead at scale.

---

### bench_lunar_set_env_scalar16384

Same path as `set_env_scalar8192` but pushed to **16 384 cloned scalar environments** to probe
the Rayon thread-pool saturation point and batched ONNX inference cost at `[16384×8]` input
size. 500 warm-up + 5 000 timed steps. Reports the same metrics. Identifies where parallel
dispatch overhead or ONNX batch cost becomes the bottleneck.

---

## Throughput — vector env (SyncLunarVectorEnvFramework)

These benchmarks replace the scalar-clone approach with `SyncLunarVectorEnvFramework`, a true
`VectorEnvironment` whose sub-envs are stepped in a single Rayon-parallel call and whose
observations are maintained in a contiguous flat buffer with a zero-copy fill path. All four
variants use the `set_env`/`run_env_eval` framework loop.

### bench_lunar_set_env_vec32

**Small-scale sustained throughput**: 32 rayon-parallel sub-envs, 500 warm-up + **500 000 timed
steps**. The long step count makes this suitable for detecting throughput drift, memory growth,
or GC pauses over an extended run. Reports env transitions/sec, µs/transition, RSS, and context
switches per iteration.

---

### bench_lunar_set_env_vec1024

**Mid-scale vector env throughput**: 1 024 rayon-parallel sub-envs, 500 warm-up + 50 000 timed
steps. This scale sits in the typical EnvPool-competitive range and is directly comparable to
`bench_lunar_eval_envpool` (same env count, same step budget). Reports the same set of metrics.

---

### bench_lunar_set_env_vec4096

**Large-scale vector env throughput**: 4 096 rayon-parallel sub-envs, 500 warm-up + 10 000
timed steps. Measures the Rayon thread-pool scaling behaviour and batched ONNX cost at
`[4096×8]` input. Reports env transitions/sec, µs/transition, RSS, and context switches.

---

### bench_lunar_set_env_vec8192

**Maximum-scale vector env throughput**: 8 192 rayon-parallel sub-envs, 500 warm-up + 5 000
timed steps. At this scale ONNX batch cost, Rayon scheduling overhead, and memory bandwidth all
become relevant. Reports the same metrics as the other vec variants. Comparable to
`bench_lunar_set_env_scalar8192` to isolate the effect of the VectorEnvironment flat-buffer path
versus scalar cloning.

---

## Throughput — Python gymnasium

### bench_lunar_eval_py

Measures **pure inference throughput against Python gymnasium's `SyncVectorEnv`** with 1 024
sub-environments. A pre-loaded ONNX policy (8→64→64→4) drives inference; the gymnasium step
loop runs via PyO3 with GIL acquisition managed around each call. 500 warm-up + 5 000 timed
steps. Reports steps/sec, transitions/sec, µs/transition, RSS, page faults, and context
switches. Baseline for Python-backed environments before switching to EnvPool.

---

## Throughput — EnvPool

### bench_lunar_eval_envpool

Benchmarks **EnvPool's C++ thread-pool-backed LunarLander-v3** with a configurable number of
environments (default 1 024, override with `--envs N`). Inference uses the NdArray+ONNX path
(zero-copy observation buffer, argmax action decode). 500 warm-up + 5 000 timed steps. Reports
steps/sec, transitions/sec, µs/transition, RSS, minor/major page faults, and context switches.
This is the primary scaling benchmark: run at 4 096, 8 192, and 12 288 envs to produce the
EnvPool throughput curve.

---

### bench_lunar_eval_envpool_tch

Same EnvPool backend as `bench_lunar_eval_envpool` but with **LibTorch/TorchScript inference**
(`.pt` model) instead of ONNX, at a fixed 1 024 environments. Measures whether switching from
ORT to the TorchScript runtime changes throughput or memory footprint when the env-stepping cost
is held constant by EnvPool. Reports the same full set of throughput and OS-level metrics.

---

## Training — PPO (Rust environments)

### bench_lunar_ppo_1env

**Single-environment PPO convergence** using 1 scalar `LunarLanderEnv` (seed 42) over 100 000
training steps with NdArray+ONNX inference. Hyperparameters are fixed to match SB3 / RLlib /
Sample Factory cross-framework benchmarks (γ=0.999, λ=0.98, clip=0.2, π-LR=2.5e-4, 10 π/V
epochs, KL threshold 0.05). No LR schedule. Reports steps/sec and wall time. The reference
point for comparing RelayRL's single-env PPO throughput against competing frameworks.

---

### bench_lunar_ppo_64env

**Short 64-environment PPO convergence run** (~100 k total frames, 1 563 loop steps). Uses the
same cross-framework hyperparameters as `bench_lunar_ppo_1env` (γ=0.999, λ=0.98, clip=0.2,
π-LR=2.5e-4, mini-batch 64, KL threshold 0.05, entropy 0.05). Reports env frames/sec and wall
time. Designed as a quick end-to-end PPO sanity check — fast enough to run in CI while still
exercising the full training path with multiple environments.

---

### bench_lunar_ppo_scalar1

**Full-convergence 64-environment PPO benchmark** running for ~1.5 M environment frames
(23 438 loop steps, 128 trajectories per epoch, mini-batch 64, buffer 100 k). Logs
**per-epoch mean return** and last episode return so convergence progress is visible. Includes a
linear LR decay schedule over 200 k steps (matching SB3 Zoo style) and higher entropy
coefficient (0.05) to prevent premature entropy collapse. This is the primary PPO convergence
validation benchmark for the NdArray+ONNX backend.

---

### bench_lunar_ppo_tch

**600 k-step PPO training with the LibTorch backend**, aligned to Sample Factory APPO defaults:
64 environments, mini-batch size 5 760 (64 envs × 90 steps), 4 π/V gradient steps per epoch,
KL target effectively disabled (1.0), return normalization enabled, entropy coefficient 0.01.
Reports env frames/sec and wall time. Used to compare LibTorch training throughput against
NdArray+ONNX and to replicate Sample Factory published results from a Rust runtime.

---

## Training — PPO with Python / LibTorch backends

### bench_lunar_ppo_py

**600 k-step PPO training** with 64 parallel `gymnasium` LunarLander-v3 environments accessed
via PyO3. Uses LibTorch (`burn_tch`) for gradient computation. Hyperparameters match
`bench_lunar_ppo_tch` exactly (SF APPO settings: mini-batch 5 760, 4 gradient steps, return
normalization). GIL acquisition is managed before async awaits to prevent deadlocks. Reports env
frames/sec and wall time. Isolates the cost of driving Python gymnasium from a Rust training loop
versus using the native Rust environment.

---

### bench_lunar_sfppo_py

**SFPPO (Sample Factory APPO-aligned) variant** of the Python gymnasium benchmark. Same
infrastructure as `bench_lunar_ppo_py` but with Sample Factory's canonical hyperparameters:
rollout length 32, mini-batch 2 048 (64×32), single training epoch per rollout, entropy 0.01,
clip ratio 0.1, return normalization on. These settings match Sample Factory's own published
LunarLander experiment exactly and are intended for direct framework-to-framework throughput and
sample-efficiency comparison. Reports env frames/sec and wall time over 600 k total steps.

---

## Training — GridWorld

### bench_grid_ppo_scalar1

**PPO convergence on a 5×5 GridWorld** environment (observation: 25-dimensional one-hot
encoding; 4 discrete actions; episode length 100 steps) with 64 scalar environments and ~500 k
total frames (7 813 loop steps, 64 trajectories per epoch, buffer 10 k). Uses a smaller
`[64×64]` MLP (vs. `[128×128]` for LunarLander). Full-batch training with KL-gated early exit
on π updates. Logs per-epoch convergence metrics. Serves as a **lightweight PPO correctness
check** on a simpler environment that should converge in 10–20 epochs, validating the training
loop independently of LunarLander's physics complexity.

---

## Running the benchmarks

Use the interactive launcher:

```bash
bash scripts/bench.sh
```

Or run a pre-built binary directly (example):

```bash
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
  ./target/release/bench_lunar_eval_envpool --envs 4096
```

Build a specific binary:

```bash
LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
  cargo build --release -p bench-beta5 --bin bench_lunar_eval_envpool
```
