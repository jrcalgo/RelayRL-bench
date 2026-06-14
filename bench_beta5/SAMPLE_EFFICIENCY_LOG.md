# PPO sample-efficiency optimization log

Tracks the ongoing hypothesis loop comparing `relayrl_algorithms`' PPO against Sample
Factory's APPO on the matched EnvPool `LunarLander-v2` setup (512 envs, see
`scripts/sf_lunar_bench.py` and `src/bin/bench_lunar_ppo_tch.rs`). Each entry records a
hypothesis, the change, results (averaged over N runs), and accept/reject.

Metric definitions:
- **final**: last printed `MeanReturn` (RelayRL) / `Avg episode reward` (SF) of the run.
- **AUC**: average of the metric sampled at 10 proportional points
  `[480,720,960,1200,1440,1680,1920,2160,2400,2640] / 2832` of the run's total epoch count
  (fractions ≈ 0.169–0.932), i.e. a proxy for "area under the learning curve" /
  time-to-convergence over the bulk of training.

## Baseline (EnvPool 512-env, both frameworks)

Both RelayRL and SF use a single shared `envpool.make("LunarLander-v2", num_envs=512,
max_episode_steps=500, seed=1)` instance; hyperparameters matched per
`scripts/sf_lunar_bench.py`'s module docstring.

| | final (avg of 5) | AUC (avg of 5) | env-frames/sec | wall/run |
|---|---|---|---|---|
| RelayRL (independent PPO, kernel.rs) | 141.02 | 93.54 | ≈39,664 | ≈972.6 s |
| Sample Factory (APPO) | 185.88 | 178.64 | ≈27,430 | ≈1400.2 s |

RelayRL is ~1.44x faster in raw env-frame throughput, but SF reaches ~155-166 reward within
the first ~17% of the budget — already exceeding RelayRL's *final* return after the full
budget. This per-frame sample-efficiency gap is the target of the loop.

## Hypothesis 1: rollout-chunked GAE with bootstrap (REJECTED, rolled back)

**Idea**: `replay_buffer.rs::insert_trajectory` only pushes an `episode_boundaries` entry when
`action.get_done()` is true (true env episode end). Rollout-length cutoffs (`rollout_len=90`,
`trajectory.set_truncated()` called in `training/mod.rs` but with the *last action's*
`done=false`) never create a boundary — those ~90-step chunks sit in the buffer with zero
advantage/return until the *true* episode (up to 500 steps) eventually ends, at which point
ONE large `episode_boundaries` entry covers the whole true episode (bootstrap=0, natural
termination). This differs from SF, which computes bootstrapped GAE over fixed 90-step
rollout chunks every epoch.

**Change** (`replay_buffer.rs`):
1. `insert_trajectory`: push an `episode_boundaries` entry on the *last* action of a
   trajectory when `action.get_done() || trajectory.is_truncated` (covers rollout-length
   cutoffs too), not just `action.get_done()`.
2. Fixed a latent bug this exposed: bootstrap value was read as `values[end-1]`
   (duplicating the chunk's last value) instead of `values[end]` (the true next-state
   value `V(s_end)`), in `finalize_and_drain_blocking`, `finalize_and_drain_first_n_blocking`,
   and `finalize_gae_blocking`.
3. Fixed dead code: `insert_trajectory` read `map.get("value")` but the key pushed by
   `training/mod.rs` is `"val"` — `buffers.values` was always 0.0 at insert time (inert for
   the `[0,cut_step)` range since `finalize_and_drain_first_n_blocking` overwrites it with
   fresh `value_forward` output, but left `values[cut_step]`, the last boundary's bootstrap
   target, stale at 0.0).

**Results**:
- Naive version (#1 only): final=154.3, AUC=49.55 (n=1) — AUC well below the entire baseline
  range (73.2–108.5). The newly-frequent bootstrap (`values[end-1]`) bias dominated.
- Corrected version (#1+#2+#3, n=3): final = [141.3, 123.5, 139.1] avg **134.63**;
  AUC = [80.65, 97.76, 79.30] avg **85.90**. Both slightly below baseline (141.02 / 93.54),
  within baseline's run-to-run noise band but with no improvement. Throughput dropped
  ~14% (≈34,000 vs ≈39,664 env-frames/sec) from the extra per-chunk GAE bookkeeping
  (`episode_boundaries` now has ~512 entries/epoch instead of a handful).

**Verdict**: REJECTED, rolled back via `git checkout`. No net diff vs baseline.

**Takeaway for future hypotheses**: RelayRL's existing full-true-episode GAE (bootstrap=0 at
natural termination, no mid-episode chunking bias) is *not* the bottleneck — it may in fact be
a more accurate advantage estimate than SF's chunked-bootstrap approach. The ~2x sample-
efficiency gap vs SF likely lies elsewhere: minibatch/epoch cadence semantics
(`episodes_needed_for_steps` vs SF's fixed 90-step rollout), advantage/return normalization
scope (per-batch vs SF's running normalizer), value-loss weighting/clipping differences,
policy/value network initialization, or LR schedule interaction with `train_pi_iters=4`
sequential (unshuffled) minibatch passes.

## Hypothesis 2: PPO2-style value-function clipping (REJECTED, rolled back)

**Idea**: SF's `_value_loss` (`sample_factory/algo/learning/learner.py:438-456`) uses PPO2-style
value clipping: `value_clipped = old_values + clamp(new_values - old_values, -clip_value,
clip_value)`, `loss = max((new_values-target)^2, (value_clipped-target)^2)`, with
`--ppo_clip_value` defaulting to 1.0. RelayRL's `train_step_discrete` (kernel.rs) instead used
plain MSE: `(v_pred - ret).powf_scalar(2.0).mean()`, with no clipping. (Gradient clipping was
separately confirmed already present and matching SF's `max_grad_norm=4.0`, ruling that out as a
lead.)

**Change** (`kernel.rs`, `independent/mod.rs`):
1. Added `PPOKernelOps::normalize_with_current_stats(&self, vals: &[f32]) -> Vec<f32>` — a
   read-only counterpart to `normalize_persistent_returns` that z-scores `vals` using the
   CURRENT (pre-mutation) `returns_mean`/`returns_variance`/`returns_count`, clamped to [-5,5].
2. In `run_ppo_sgd_flat`, computed `old_val_normalized = kernel.normalize_with_current_stats(&batch.val)`
   BEFORE calling `normalize_persistent_returns(&batch.ret)` (which mutates the running stats),
   to bring `PPOBatch.val` (reward-scale, from the previous epoch's `value_forward`) onto the
   same normalized scale as `v_pred`/`ret_normalized`.
3. Threaded `old_val` through `train_step`/`train_step_discrete`. In `train_step_discrete`,
   implemented `value_clipped = old_val + clamp(v_pred - old_val, -1.0, 1.0)`,
   `vf_loss = max((v_pred-ret)^2, (value_clipped-ret)^2).mean()` (clip_value=1.0, matching SF's
   default), using Burn's `max_pair` (analogous to the existing `min_pair` used for the PPO
   policy-clip objective).

**Results** (n=3):
- final = [77.9, 132.6, 120.6], avg **110.37** (baseline avg 141.02, range 131.8-162.3 — all 3
  runs below baseline's lowest individual run).
- AUC = [79.54, 60.19, 83.50], avg **74.41** (baseline avg 93.54, range 73.22-108.50 — below
  baseline's lowest individual run).
- Throughput: ≈34,720 / 34,821 / 33,803 env-frames/sec (~12-15% below baseline's ≈39,664) from
  the extra value-clipping tensor ops.
- Instability signals: AUC sample points included severe dips (-45.1 in run2, -3.7 and -0.7 in
  runs 1/3 — MeanReturn briefly going strongly negative mid-training, never seen in the
  baseline). Run 3's final epoch printed `ClipFrac=0.9783` (vs baseline's consistent `0.0000`),
  i.e. nearly all policy-ratio samples were being clipped — a strong indicator the value-clipping
  change destabilized the policy update too, likely via the combined-loss backward pass sharing
  gradients between `pi_loss` and `vf_loss`.

**Verdict**: REJECTED, reverted via `git revert` (commit 25b28e5). All 3 metrics (final, AUC,
throughput) regressed vs baseline, with clear instability signatures.

**Takeaway for future hypotheses**: the `normalize_with_current_stats` approximation for
`old_val` may itself be miscalibrated (it re-z-scores an already-denormalized `batch.val` using
running return stats, which is only an approximate inverse of `value_forward`'s denormalization
— see kernel.rs's `value_forward`/`normalize_persistent_returns`/`set_return_denorm_stats`
chain). Even if value clipping itself were beneficial, a poorly-scaled `old_val` would make the
clip bounds meaningless or harmful, which is consistent with the observed instability. If value
clipping is revisited, it would need a more careful scale-matched `old_val` (e.g. caching the
network's raw `v_norm` output at rollout time, before any denormalization, rather than
reconstructing it from `batch.val`). Absent that, focus shifts to other candidates: minibatch/
epoch cadence, normalization scope, network initialization, or LR schedule.

## Hypothesis 3: enable `normalize_obs` to match SF's `--normalize_input=True` (REJECTED, rolled back)

**Idea**: `scripts/sf_lunar_bench.py`'s `DEFAULT_ARGS` sets `--normalize_input=True` (SF's
`RunningMeanStd`-based observation normalization). RelayRL's framework already implements the
equivalent — `ObsNormalizer` (Welford running per-feature mean/variance) in
`relayrl_framework/.../training/mod.rs`, gated by `IPPOParams.normalize_obs: bool` (default
`false`, i.e. previously unused by `bench_lunar_ppo_tch`). This was the one major hyperparameter
not yet matched between the two frameworks.

**Change** (`bench_lunar_ppo_tch.rs`, benchmark-binary config only, no algorithm/framework edits):
1. Set `normalize_obs: true` in the `IPPOParams { ... }` literal.
2. Updated the printed config banner to reflect the new setting.

**Results** (n=3):
- final = [148.70, 156.00, 143.70], avg **149.47** (baseline avg 141.02, range 131.8-162.3 — a
  modest +6% bump, but within baseline's run-to-run range).
- AUC = [103.83, 87.54, 88.36], avg **93.24** (baseline avg 93.54, range 73.22-108.50 —
  essentially identical to baseline, no improvement in the convergence-speed proxy).
- Throughput: 33215 / 33720 / 33321 env-frames/sec, avg **≈33419** (~15.7% below baseline's
  ≈39664), from the extra per-step Welford `ObsNormalizer` update/normalize on all 512 envs.
- No instability signals: `ClipFrac=0.0000` throughout all 3 runs (consistent with baseline).

**Verdict**: REJECTED, rolled back (removed `normalize_obs: true` from `bench_lunar_ppo_tch.rs`).
AUC — the primary time-to-convergence proxy — showed no improvement (93.24 vs 93.54), while final
return's modest bump was within baseline noise. Combined with a real ~16% throughput regression,
this hyperparameter-matching change offers no net benefit.

**Takeaway for future hypotheses**: observation normalization is NOT the source of the
sample-efficiency gap — RelayRL's un-normalized obs (raw LunarLander-v2 state, already small-
magnitude/well-scaled features) perform on par with SF's running-normalized obs. With H1
(rollout-chunked GAE), H2 (PPO2 value clipping), and H3 (obs normalization) all ruled out, the
remaining candidates are: minibatch/epoch cadence (`episodes_needed_for_steps` vs SF's fixed
90-step rollout — see H1's takeaway), network initialization scheme details (orthogonal init
*gains* per layer, not just orthogonality), LR schedule (SF may anneal LR/clip-ratio over
training; RelayRL's `bench_lunar_ppo_tch` uses fixed `PI_LR`/`VF_LR`/`CLIP_RATIO`), or
entropy-coefficient/KL-target interaction with `train_pi_iters=4` early-stopping (`StopIter`
values, `target_kl` behavior) across the two implementations.
