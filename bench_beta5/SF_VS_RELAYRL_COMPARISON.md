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
