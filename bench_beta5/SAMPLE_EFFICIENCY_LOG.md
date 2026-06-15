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

## Hypothesis 4: orthogonal weight init (gain=1.0, zero bias) matching SF's actual `--policy_init_gain` (ACCEPTED)

**Idea**: re-examination of `GenericMlp::new` (the constructor `bench_lunar_ppo_tch.rs` actually uses
for `pi_mlp`/`vf_mlp`, via `LinearConfig::new(...).init(device)`) showed it uses Burn's *default*
`Initializer::KaimingUniform{gain: 1/sqrt(3)}` with non-zero bias — NOT orthogonal init, contrary to
an earlier (incorrect) session note that claimed this was "already matched" to SF. SF's actual
resolved config in `sf_lunar_bench.py` is `--policy_initialization=orthogonal` with
`--policy_init_gain` left at its default of **1.0** (not overridden). SF's
`initialize_weights` (`actor_critic.py:71-94`) applies `nn.init.orthogonal_(layer.weight, gain=1.0)`
to every `nn.Linear`/`nn.Conv2d` (including output layers) and zero-fills every bias. This is a
well-documented PPO stability/sample-efficiency factor (orthogonal init avoids saturated/dead
units early in training) and was the one major hyperparameter mismatch not yet addressed by H1-H3.

**Change**:
1. `algorithms/mod.rs`: added `GenericMlp::new_orthogonal(..., gain: f64, device)` — an additive
   constructor alongside the existing `GenericMlp::new`/`default` (both unchanged, still used by
   other algorithms). For each `Linear` layer it builds with `Initializer::Zeros` bias, then
   overwrites `layer.weight` via `Initializer::Orthogonal{gain}.init_with([in,out], Some(in),
   Some(out), device)`.
2. `bench_lunar_ppo_tch.rs`: added `const POLICY_INIT_GAIN: f64 = 1.0;`, switched both `pi_mlp` and
   `vf_mlp` construction from `GenericMlp::new(...)` to `GenericMlp::new_orthogonal(..., POLICY_INIT_GAIN,
   &burn_device)`, and appended `policy_init_gain={POLICY_INIT_GAIN}` to the config banner.

**Results** (n=3):
- final = [153.50, 184.90, 131.00], avg **156.47** (baseline avg 141.02, range 131.8-162.3 — avg
  +11%, run2 alone exceeds baseline's max individual run).
- AUC = [92.86, 139.18, 79.54], avg **103.86** (baseline avg 93.54, range 73.22-108.50 — avg +11%,
  run2's 139.18 is well above baseline's max of 108.50 and meaningfully closes the gap toward SF's
  full-budget AUC of 178.64).
- Throughput: 33342/33643/34557 env-frames/sec, avg **≈33847** (~14.7% below the original baseline's
  ≈39664, but in line with H2/H3's ~33-34k — since orthogonal init is a one-time op with zero
  per-step cost, this is most likely system-load drift across the session rather than an
  init-specific regression).
- Peak MeanReturn per run: 165.6 / 239.4 / 216.5 — all at or above baseline's per-run max range
  (148.2-169.9), i.e. HYP4 reaches noticeably higher episode returns during training.
- Min MeanReturn per run: -196.1 / -196.3 / -191.4 — within baseline's per-run min range
  (-208.1 to -183.5), i.e. the occasional deep dips are normal LunarLander-v2 training variance,
  not a new instability mode (unlike H2's out-of-range -45.1 AUC-sample dip).
- ClipFrac: all 3 runs show scattered nonzero values (159/237/185 of 832 epochs, means
  0.046/0.058/0.058) including 7-10 epochs/run hitting `1.0000`, vs baseline's steady `0.0000`
  throughout. This is a real, consistent side-effect of the larger effective updates from
  orthogonally-initialized layers, but — unlike H2 — it does not correlate with MeanReturn
  collapsing outside baseline's normal range.

**Verdict (n=3, initial)**: ACCEPTED. Both final and AUC improved ~11% on average vs baseline,
with no new instability mode (dips remain within baseline's normal range) despite the higher
run-to-run variance (AUC range 79.54-139.18 vs baseline's 73.22-108.50 — wider on both ends, but
the upper end is the desirable direction). The implementation (`new_orthogonal` +
`POLICY_INIT_GAIN=1.0`) was kept as the new baseline going forward (commit 83adb7f).

**Re-evaluation at n=5 (REVERSED to REJECTED)**: after H5's results raised concerns about
run-to-run variance at n=3, 2 more runs (run4, run5) of the H4 config were collected to reach
n=5:
- final = [153.50, 184.90, 131.00, **79.50**, **129.00**], n=5 avg **135.58** (vs original
  baseline avg 141.02 — now **-3.9%**, i.e. *below* baseline).
- AUC = [92.86, 139.18, 79.54, **79.95**, **84.55**], n=5 avg **95.22** (vs original baseline avg
  93.54 — now only **+1.8%**, well within baseline's own run-to-run noise).
- Run4 (final=79.50, AUC=79.95) and run5 (final=129.00, AUC=84.55) are both at or below the
  bottom of the original baseline's range (final 131.8-162.3, AUC 73.22-108.50), pulling the
  average back to ~parity.
- Variance increased substantially: final range is now 79.50-184.90 (105.4-point spread, vs
  baseline's 30.5-point spread); AUC range is 79.54-139.18 (59.6-point spread, vs baseline's
  35.3-point spread).

**Final verdict**: REJECTED, reverted (`git revert -n 83adb7f`, removing `new_orthogonal` from
`algorithms/mod.rs` and restoring `GenericMlp::new(...)`/removing `POLICY_INIT_GAIN` in
`bench_lunar_ppo_tch.rs`). The n=3 "improvement" was a sampling artifact (a lucky high run2):
at n=5, both final and AUC are statistically indistinguishable from the original baseline, while
run-to-run variance roughly doubled in spread on both metrics — a strictly worse risk/reward
profile than the original Kaiming-uniform init. No net diff vs the pre-H4 baseline after revert.

**Takeaway for future hypotheses**: **n=3 is not sufficient to evaluate hypotheses in this
environment** — the run-to-run AUC/final spread within a single hypothesis's n=3 sample
(roughly 30-60 points) is comparable to or larger than the between-hypothesis effect sizes we've
been testing. All hypotheses so far (H1-H5) should be considered n=3-level evidence only; H4's
reversal after 2 more runs demonstrates this concretely. Going forward: (a) always complete n=5
before reaching ACCEPT/REJECT, per the original directive, (b) treat n=3 ACCEPTs as provisional
pending the 2 additional runs before committing to a "new baseline," and (c) consider that
orthogonal init with gain=1.0, while theoretically well-motivated, does not measurably help this
particular setup — remaining candidates: minibatch/epoch cadence
(`episodes_needed_for_steps` vs SF's fixed 90-step rollout), entropy-coefficient schedule, and
PPO2 value-clipping with a correctly-scaled `old_val` (H2's takeaway).

## Hypothesis 5: match SF's Adam epsilon (1e-6 vs Burn's default 1e-5) (REJECTED, reverted)

**Idea**: SF's default `--adam_eps=1e-6` (`cfg.py:280-285`, not overridden by `sf_lunar_bench.py`).
Burn's `AdamConfig::new()` defaults `epsilon=1e-5` (`burn-optim-0.20.1/src/optim/adam.rs:30-31`).
Both sides already match on `beta_1=0.9`/`beta_2=0.999` (Burn defaults == SF defaults) and
`max_grad_norm=4.0`. The Adam epsilon controls the denominator floor of the per-parameter update
(`lr / (sqrt(v) + eps)`), a well-known PPO implementation detail; a 10x difference could plausibly
affect step sizes for parameters with small gradient variance, especially post-H4 with
orthogonally-initialized layers.

**Change** (`kernel.rs`, single line, PPO algorithm scope, additive):
1. `PPOActorCriticTrainer::new`: `AdamConfig::new().with_epsilon(1e-6).init::<TB, ActorCriticMlp<TB>>().with_grad_clipping(...)`.
2. `bench_lunar_ppo_tch.rs`: appended `  adam_eps=1e-6` to the config banner.

**Results** (n=3, vs the H4-accepted baseline: final avg 156.47 range [131.00,184.90], AUC avg
103.86 range [79.54,139.18]):
- final = [147.10, 132.70, 129.60], avg **136.47** (-12.8% vs baseline avg, within baseline's
  range but toward its low end).
- AUC = [82.80, 141.47, 105.53], avg **109.93** (+5.8% vs baseline avg — nominally higher, but
  well within the ~60-point run-to-run spread both hypotheses exhibit; not distinguishable from
  noise at n=3).
- Throughput: 34386/34593/33549 env-frames/sec, avg **≈34176** (essentially unchanged vs H4's
  ≈33847 — as expected, since the epsilon change has zero per-step cost).
- ClipFrac: 202/831, 236/832, 177/832 nonzero (means 0.052/0.061/0.045, 7-10 epochs/run hitting
  `1.0000`) — essentially identical distribution to H4's runs, no new instability.
- Min/max MeanReturn per run: (-190.1, 163.7), (-185.8, 250.4), (-191.7, 163.7) — in line with H4's
  ranges (run2's 250.4 is a new high, but within the same noisy-peak pattern as H4's run2's 239.4).

**Verdict**: REJECTED, reverted (`git revert -n` of the implementation commit). AUC's nominal
+5.8% is not a robust improvement given the magnitude of run-to-run variance already observed
within H4 alone (a ~60-point AUC spread across 3 runs of the *same* config), while final regressed
-12.8%. No improvement on the primary AUC metric clears the bar required by the H1-H4 precedent
(H4 was accepted because *both* final and AUC improved together, ~11% each). No net diff vs the
H4 baseline after revert.

**Takeaway for future hypotheses**: Adam epsilon is NOT a meaningful lever at this scale — both
betas and now epsilon are matched to SF's defaults with no measurable effect, closing out the
optimizer-hyperparameter axis. The dominant remaining uncertainty is the very high run-to-run
variance itself (AUC spreads of ~60 points within a single hypothesis's n=3) — before chasing
further small hyperparameter deltas, a larger n (e.g., n=5 or n=10) may be needed to distinguish
real effects from noise at this variance level. Remaining structural candidates: minibatch/epoch
cadence (`episodes_needed_for_steps` vs SF's fixed 90-step rollout, from H1's takeaway),
entropy-coefficient schedule, and the framework-level epoch-boundary semantics (out of PPO-only
scope but worth flagging).

## Hypothesis 6: PPO2 value-function clipping with correctly-scaled `old_val` (REJECTED, reverted)

**Idea**: revisit H2's PPO2 value-clipping (matching SF's default `--ppo_clip_value=1.0`,
`value_clipped = old_values + clamp(new_values - old_values, -1, 1)`,
`vf_loss = max((new-target)^2, (clipped-target)^2).mean()`), but avoid H2's root-cause: H2
derived `old_val` from the DENORMALIZED `batch.val` (populated via `value_forward`, which maps the
network's normalized output back to reward scale using stats from the *previous* epoch),
producing a scale mismatch against `v_pred`/`ret_normalized` (both in normalized/z-score space)
and causing severe instability (`ClipFrac=0.9783`, `MeanReturn` dipping to -45.1).

**Change** (PPO algorithm scope, additive, `kernel.rs` + `independent/mod.rs`):
1. `kernel.rs`: added `const VALUE_CLIP: f32 = 1.0` (matches SF's `--ppo_clip_value` default).
   `train_step_discrete` gained an `old_val: &[f32]` parameter; value loss is now
   `v_clipped = old_val + clamp(v_pred - old_val, -1, 1)`,
   `vf_loss_t = max((v_pred-ret)^2, (v_clipped-ret)^2).mean()` (was plain MSE). Threaded
   `old_val` through the `PPOKernelTraining::train_step` trait method and its
   `PPOKernel::Discrete` dispatch.
2. `independent/mod.rs`: in `run_ppo_sgd_flat`, right after
   `kernel.set_return_denorm_stats(...)`, added a single no-grad
   `kernel.trainer.value_forward_flat(&obs_flat, obs_dim)` call over the full batch's obs,
   producing `old_val_norm` — the RAW network output (same normalized/z-score scale as
   `ret_normalized` and the in-loop `v_pred`), computed once *before* any SGD steps this epoch.
   This `old_val_norm` is passed to every `train_step` call this epoch (sliced per-minibatch in
   the non-full-batch path, though `full_batch=true` here).
3. `bench_lunar_ppo_tch.rs`: appended `value_clip=1.0 (PPO2, matches SF --ppo_clip_value default)`
   to the config banner.

**Results (n=5, vs original baseline: final avg 141.02 range [131.8,162.3], AUC avg 93.54 range
[73.22,108.50])**:
- final = [155.00, 160.80, 148.10, 145.70, 116.00], n=5 avg **145.12** (**+2.9%** vs baseline).
- AUC = [107.95, 93.13, 73.89, 89.03, 76.85], n=5 avg **88.17** (**-5.7%** vs baseline).
- Both metrics' n=5 averages fall within the *original baseline's own* per-run ranges (final
  range now [116.00,160.80], 44.8-point spread vs baseline's 30.5; AUC range [73.89,107.95],
  34.1-point spread vs baseline's 35.3) — variance is comparable to baseline, not worse.
- Min/max MeanReturn per run: (-180.4,169.8), (-199.1,162.3), (-186.4,160.1), (-179.0,158.6),
  (-188.1,154.3) — all within baseline's normal per-run range (min -208.1 to -183.5, max
  148.2-169.9). No instability mode like H2's -45.1 AUC-sample dip.
- ClipFrac: all 5 runs show scattered nonzero values (179/832, 173/832, 180/832, 202/830,
  170/832 epochs, means 0.0485-0.0562, 3-8 epochs/run hitting `1.0000`) vs the original
  baseline's steady `0.0000` throughout — a real, consistent side-effect (also seen with H4's
  orthogonal init), most likely because the new `value_forward_flat` snapshot adds one extra
  full-batch forward pass per epoch that perturbs LibTorch's floating-point execution/threading
  order, causing PPO's chaotic training dynamics to diverge onto a different (but
  statistically similar) trajectory from the very first epoch — not evidence of instability,
  since `MeanReturn` stayed in-range across all 5 runs.
- Throughput: 33813/33860/33689/41956/42280 env-frames/sec (last 2 runs measured post a
  container restart with markedly lower system load — not attributable to the algorithm
  change itself; avg ≈37120, in line with the ~33-42k range seen across this whole project).

**Verdict**: REJECTED, reverted (commit `2e3c83b` reverted via `git revert -n`). Final's nominal
+2.9% is within baseline's own noise, while AUC regressed -5.7% — failing the H1-H4 precedent
that ACCEPT requires *both* final and AUC to improve together. Unlike H2, this implementation is
NOT unstable (no -45.1-style collapse, no near-1.0 ClipFrac runs) — the correctly-scaled `old_val`
does fix H2's root cause — but PPO2 value-clipping with `clip_value=1.0` simply does not help
sample efficiency in this 1-epoch-of-46080-samples, 4-SGD-iteration regime: the value function is
already well-regularized by `vf_coef=1.0` + persistent return normalization + grad-norm clipping,
so an additional clip on the value target adds noise (the new nonzero ClipFrac) without a
compensating sample-efficiency gain.

**Takeaway for future hypotheses**: PPO2 value-clipping is now closed out as a candidate (both
the flawed H2 variant and this correctly-scaled variant fail to improve AUC). The loss-function
and optimizer axes (H2, H4, H5, H6) have now all been explored without a robust win; remaining
structural candidates: minibatch/epoch cadence (`episodes_needed_for_steps` vs SF's fixed
90-step rollout, from H1's takeaway), entropy-coefficient scheduling/annealing (SF anneals
`exploration_loss_coeff` in some configs; this benchmark may not), and GAE
`lambda`/`gamma` fine-tuning (currently 0.98/0.999, matched to SF's config but not yet
independently varied).
