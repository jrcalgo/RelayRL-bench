# RelayRL vs Sample Factory — direct 1:1 benchmark (LunarLander-v2, EnvPool, 512 envs)

Sequential, single-seed, same-machine head-to-head comparison between RelayRL's PPO (LibTorch
backend, current accepted baseline: **H24-lite** — `sync_epoch_boundary` + `normalize_obs`,
`gamma=0.999`, `lam=0.98`) and Sample Factory's APPO, with hyperparameters matched as closely as
the two frameworks' APIs allow. Both runs use `PPO_SEED`/`--seed=1`, the same EnvPool LunarLander-v2
environment (512 envs, `max_episode_steps=500`), the same network shape ([128,128] ReLU, separate
pi/vf nets), and the same training budget (~38.3-38.5M env frames). Both runs were profiled with
`perf stat` (software events only — this sandbox's `perf_event_paranoid=2` and lack of PMU
passthrough mean hardware counters like `cycles`/`instructions` report `<not supported>`; the
underlying `/usr/lib/linux-tools-6.8.0-106/perf` binary was used directly, bypassing the
`/usr/bin/perf` kernel-version dispatch wrapper which otherwise refuses to run on this container's
6.18.5 kernel since no matching `linux-tools-6.18.5` package exists upstream).

## Matched config

| Hyperparameter | RelayRL (`bench_lunar_ppo_tch.rs`) | Sample Factory (`scripts/sf_lunar_bench.py`) |
|---|---|---|
| gamma | 0.999 | `--gamma=0.999` |
| GAE lambda | 0.98 | `--gae_lambda=0.98` |
| clip ratio | 0.2 | `--ppo_clip_ratio=0.2` |
| pi_lr / vf_lr | 3.5e-4 | `--learning_rate=3.5e-4 --lr_schedule=constant` |
| vf_coef | 1.0 | `--value_loss_coeff=1.0` |
| SGD iters/epoch | 6 | `--num_epochs=6` |
| entropy coef | 0.01 | `--exploration_loss_coeff=0.01` |
| batch size | 46080 (512 envs x 90-step rollout) | `--batch_size=46080 --rollout=90` |
| envs | 512 | `--num_envs_per_worker=1` x envpool(num_envs=512) |
| network | [128,128] ReLU, separate pi/vf | `--encoder_mlp_layers 128 128 --nonlinearity=relu --actor_critic_share_weights=False` |
| obs normalization | `normalize_obs=true` (Welford running stats) | `--normalize_input=True` |
| grad clip | max_norm=4.0 | `--max_grad_norm=4.0` |
| seed | `PPO_SEED=1` | `--seed=1` |
| training budget | 831 epochs x 46080 = 38,288,480 frames | `--train_for_env_steps=38400000` (actual: 38,384,640) |

**Note**: RelayRL's `sync_epoch_boundary` (a hard collect/train barrier within each epoch, the
single strongest lever found across this entire hypothesis-testing log — see H22/H24/H29) has no
direct SF equivalent; SF's APPO is natively asynchronous (separate rollout/inference/learner
processes overlapping continuously), which is closer to RelayRL's *default* (`sync_epoch_boundary
=false`) behavior. SF is run with its own native async architecture (`--async_rl=True`,
`--serial_mode=False`) rather than forced into a synchronous mode, since there is no SF flag that
reproduces RelayRL's specific per-epoch barrier — this benchmark compares each framework as it is
actually meant to run, not a forced architectural match.

## Results

| Metric | RelayRL (H24-lite, seed=1) | Sample Factory (seed=1) | Delta (SF vs RelayRL) |
|---|---|---|---|
| final MeanReturn | 164.70 | 181.86 | **+10.4%** |
| AUC (10-point fractional sample, matching this log's standard method) | 140.12 | 179.16 | **+27.9%** |
| total env frames | 38,288,480 | 38,384,640 | (~equal, +0.3%) |
| wall time | 1003.72s | 1604.56s | RelayRL **1.60x faster** |
| env-frames/sec (throughput) | 38,420 | 23,918 | RelayRL **1.61x faster** |
| `perf stat` task-clock | 2,739,475.71 ms (2.73 CPUs utilized) | 2,828,043.37 ms (1.76 CPUs utilized) | RelayRL uses more parallel CPU-time but finishes faster |
| `perf stat` context-switches | 32,394,819 (11,825/sec) | 1,233,253 (436/sec) | RelayRL has **26.3x more** context-switches |
| `perf stat` cpu-migrations | 478,315 (174.6/sec) | 48,057 (17.0/sec) | RelayRL has **9.9x more** migrations |
| `perf stat` page-faults | 62,001,454 (22,633/sec) | 232,550,785 (82,230/sec) | SF has **3.75x more** page-faults |
| `perf stat` major-faults | 11 | 58 | both negligible (no real I/O-driven swapping) |

## Analysis

**Sample efficiency**: this single-seed result reproduces the gap this entire log has been trying
to close — SF reaches a higher final return (+10.4%) and, more importantly, a substantially higher
AUC (+27.9%), meaning SF's policy converges faster per env-frame throughout training, not just at
the end. This is consistent with the H24-lite multi-seed average (final 157.24, AUC 138.78) — this
single seed=1 RelayRL run (164.70/140.12) sits close to that multi-seed average, confirming it's a
representative draw, not an outlier run skewing the comparison.

**Wall-clock throughput**: RelayRL is 1.6x *faster* in wall-clock terms on this 4-core sandboxed
machine despite training on the LibTorch backend through Rust/Burn — RelayRL's synchronous,
single-process design (rollout collection and SGD interleaved in one process via `rayon`) avoids
the multi-process IPC/serialization overhead inherent to SF's async architecture (separate
rollout-worker, inference-worker, and learner processes communicating via queues/shared memory).
This shows up directly in the `perf stat` counters: SF's page-fault rate is 3.75x RelayRL's
(232.55M vs 62.00M total) — consistent with repeated allocation/mapping of IPC buffers and
per-process memory regions across SF's multiple OS processes, whereas RelayRL's context-switch
count is 26x higher (32.4M vs 1.2M) — consistent with `rayon`'s fine-grained thread-pool work
distribution for vectorized env stepping (many short-lived parallel tasks per epoch) rather than
a few long-lived dedicated processes.

**The throughput/sample-efficiency tradeoff**: RelayRL trains faster per wall-clock second but
needs more env-frames to reach the same return — i.e. the two frameworks sit at different points
of a throughput-vs-sample-efficiency tradeoff curve, not a strict dominance relationship. For a
fixed wall-clock training budget rather than a fixed frame budget, RelayRL's throughput advantage
(1.61x) partly offsets its sample-efficiency deficit (-27.9% AUC) — a fixed-time comparison would
narrow the practical gap from what the fixed-frame numbers above suggest in isolation, though it
would not close it (1.61x throughput vs 1.28x AUC deficit still favors SF's per-frame learning
curve overall after rescaling to wall-clock-equivalent frame budgets, roughly 1.61/1.279 ≈ 1.26x
net edge to SF even after accounting for RelayRL's speed).

## Reproduction

```bash
# Sample Factory (run from bench_beta5/)
/usr/lib/linux-tools-6.8.0-106/perf stat -e task-clock,context-switches,cpu-migrations,page-faults,major-faults,minor-faults \
  -o /tmp/sf_perf.txt -- python3 scripts/sf_lunar_bench.py --experiment=<name>

# RelayRL (run from bench_beta5/, after `cargo build --release -p bench-beta5 --bin bench_lunar_ppo_tch`)
rm -rf envs/lunar_ppo_tch models/lunar_ppo_tch
/usr/lib/linux-tools-6.8.0-106/perf stat -e task-clock,context-switches,cpu-migrations,page-faults,major-faults,minor-faults \
  -o /tmp/relayrl_perf.txt -- env LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
  LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH \
  RAYON_NUM_THREADS=4 PPO_SEED=1 ./target/release/bench_lunar_ppo_tch
```

Logs and raw `perf` output for this comparison: `/tmp/sf_h24lite_run1.log`,
`/tmp/sf_h24lite_run1_perf.txt`, `/tmp/relayrl_h24lite_run1.log`,
`/tmp/relayrl_h24lite_run1_perf.txt` (not committed — ephemeral container paths).

## Follow-up: can RelayRL's context-switch count be reduced?

The 26x context-switch multiplier above traces to LibTorch's intra-op thread pool, not `rayon`
directly: with `sync_epoch_boundary=true`, collection and SGD training already run on disjoint
phases, but LibTorch's CPU thread pool is left unconstrained (default: one thread per core,
contending with rayon's own worker threads and EnvPool's C++ thread pool for the same 4 physical
cores). Sample Factory avoids this by deliberately pinning `torch.set_num_threads(1)` in every
rollout/inference worker process (`sample_factory/algo/utils/torch_utils.py`,
`rollout_worker.py:65`).

Two variants were implemented and benchmarked (`PPO_SEED=1`, same `perf stat` software-event set,
same machine):

| Variant | wall time | env-frames/sec | context-switches |
|---|---|---|---|
| Baseline (H24-lite, unconstrained LibTorch threads) | 999.5s | 38,420 | 32,394,819 |
| Global `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` (naive, matches SF's blanket approach) | 1308.1s | 29,355 | 8,076,385 |
| Phase-toggled (`tch::set_num_threads(1)` during collection/inference, `set_num_threads(num_cores)` only inside `start_epoch_training()`) | 1190.3s | 32,261 | 34,171,668 |

**Verdict: neither variant is an improvement; the phase-toggled approach is rejected.** The naive
global cap does cut context-switches by 75%, confirming the diagnosis that LibTorch's thread pool
is the dominant source — but it costs 24% throughput, because LibTorch's SGD passes (the actual
matmul-heavy compute, run on a dedicated tokio worker thread concurrently with the main collection
loop) are themselves multi-threaded in the baseline and lose real parallelism when capped to 1.

The phase-toggled approach was implemented specifically to keep that training-time parallelism
(toggling `tch::set_num_threads()` to 1 before `start_epoch_training()` and back up to
`num_cores` once a training task is pending, at all 3 call sites in
`training/mod.rs::train_ppo`) while still constraining threads during collection. It does **not**
work: context-switches went *up* slightly versus the baseline (34.17M vs 32.39M, +5.5%) rather
than down, and throughput dropped 16% (32,261 vs 38,420 env-frames/sec) — worse than doing
nothing, though still better than the naive blanket cap. The most likely explanation is that
`set_num_threads()` is not a cheap atomic toggle: each call reconfigures (and on this LibTorch
build, appears to tear down/recreate) the intra-op worker thread pool, and at 832 epochs x 2
toggles/epoch this adds on the order of 1,600 thread-pool reconfigurations over the run — the
teardown/recreation overhead itself generates additional context-switches and stalls that outweigh
any savings from running single-threaded during the (already comparatively cheap) collection
phase.

**Conclusion**: the framework's elevated context-switch count is a true artifact of LibTorch's
unconstrained thread pool, but neither a blanket cap nor a phase-toggled cap actually increases
throughput in this configuration — both regress it (24% and 16% respectively) for a join-the-dots
reduction in context-switches that doesn't translate into wall-clock benefit on this 4-core
machine. The H24-lite baseline (default unconstrained LibTorch threading) remains the better
choice and stands as-is; no thread-toggling change is recommended for adoption. A higher-core-count
machine, where LibTorch's training-phase parallelism has more headroom to actually pay for itself,
might change this calculus — untested here due to the 4-core sandbox constraint.

Logs and raw `perf` output for this follow-up: `/tmp/relayrl_h24lite_1thread_run1.log`,
`/tmp/relayrl_h24lite_1thread_perf.txt`, `/tmp/relayrl_phasetoggled_run1.log`,
`/tmp/relayrl_phasetoggled_perf.txt` (not committed — ephemeral container paths).
