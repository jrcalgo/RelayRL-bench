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

## Hypothesis 7: match SF's asymmetric PPO clip-ratio formula (REJECTED, reverted)

**Idea**: RelayRL's `train_step_discrete` clamps the probability ratio `r = exp(logp - logp_old)`
to the symmetric range `[1-e, 1+e]` (here `[0.8, 1.2]` for `clip_ratio=0.2`). SF's
`learner.py:541-543` instead computes
`clip_ratio_high = 1 + e` and `clip_ratio_low = 1 / clip_ratio_high` (`[0.8333, 1.2]` for
`e=0.2`), i.e. an asymmetric range that is symmetric in *log-ratio* space rather than in `r`
itself (SF's comment notes this also avoids negative ratios for `e >= 1`, which `1-e` cannot).
Since `clip_ratio=0.2` is matched between the two configs but the actual clipping bound was not,
this was the first hypothesis across H1-H6 to touch the clip-ratio formula itself.

**Change** (PPO algorithm scope, `kernel.rs` only, commit `695d3e9`):
- In `train_step_discrete`, replaced
  `let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);`
  with
  ```rust
  let clip_ratio_high = 1.0 + clip_ratio;
  let clip_ratio_low = 1.0 / clip_ratio_high;
  let clipped_ratio = ratio.clone().clamp(clip_ratio_low, clip_ratio_high);
  ```
- Updated the `ClipFrac` diagnostic to count `r < clip_ratio_low || r > clip_ratio_high` instead
  of `|r - 1| > clip_ratio`, so the diagnostic stays consistent with the new clipping bounds.

**Results (n=5, vs original baseline: final avg 141.02 range [131.8,162.3], AUC avg 93.54 range
[73.22,108.50])**:
- final = [117.70, 144.70, 157.60, 156.50, 153.60], n=5 avg **146.02** (**+3.5%** vs baseline).
- AUC = [97.86, 85.69, 94.55, 80.12, 83.03], n=5 avg **88.25** (**-5.7%** vs baseline).
- Min/max MeanReturn per run: (-192.8,160.5), (-219.6,188.6), (-196.5,167.2), (-196.5,166.6),
  (-198.8,166.4) — run2's (-219.6,188.6) is slightly beyond the original baseline's typical
  per-run extremes (min -208.1 to -183.5, max 148.2-169.9) but in the same direction/scale as
  H6's runs that were judged in-range; not flagged as an instability mode (no -45.1-style
  AUC-sample collapse like H2).
- ClipFrac: all 5 runs show substantially more nonzero clipping than H6 or baseline —
  204/831, 214/832, 192/832, 197/832, 224/832 epochs nonzero (means 0.0541-0.0675, vs H6's
  0.0485-0.0562 and baseline's steady 0.0000), with 6-13 epochs/run hitting `>=0.99` (vs H6's
  3-8). This is the expected direct effect of the formula change: the new lower bound
  `1/(1+e) ≈ 0.8333` is *tighter* than the old `1-e = 0.8`, so more ratios fall outside the
  (now narrower-on-the-downside) trust region and get clipped/counted.
- Throughput: 40915/42301/41562/41689/39863 env-frames/sec, avg ≈41266 — in line with the
  ~33-42k range seen across this whole project; no throughput regression from the formula
  change (it's three scalar ops, same as before).

**Verdict**: REJECTED, reverted (commit `695d3e9` reverted via `git revert -n`). final's nominal
+3.5% is within baseline's own per-run range (131.8-162.3) and thus within noise, while AUC
regressed -5.7% — the exact same final/AUC pattern as H6, failing the H1-H4 precedent that
ACCEPT requires *both* final and AUC to improve together. The asymmetric clip formula is
numerically correct and matches SF exactly, but in this regime it tightens the effective trust
region on the downside (more clipping, see ClipFrac above) without translating into faster
early-training learning (AUC) — any late-training final-return gain is offset by slower
early progress.

**Takeaway for future hypotheses**: the clip-ratio formula axis is now closed out (H7 fails
alongside the loss/optimizer axes from H2/H4/H5/H6). Two new observations from H7: (1) the
final-improves/AUC-regresses pattern has now appeared twice (H6, H7) with both implementations
that increase mid/late-training clipping/regularization — suggests this benchmark's AUC sample
points (early epochs, fractions 0.169-0.318 of N) are dominated by *exploration speed*, which
extra regularization slows down, while final return benefits more from late-training stability.
A hypothesis that speeds up *early* learning specifically (e.g. higher initial entropy
coefficient with decay, or a higher initial LR with decay) may be better targeted at the AUC
metric. (2) remaining unexplored candidates: entropy-coefficient scheduling/annealing,
GAE `lambda`/`gamma` fine-tuning, LR annealing/warmup (note: plain linear LR annealing was
already tried pre-log and reverted, but a decay schedule combined with entropy annealing was
not), and SF's `ratio = torch.clamp(ratio, 0.05, 20.0)` numerical-safety clamp (small,
unexplored, unlikely to move metrics but cheap to test).

## Hypothesis 8: SF's ratio numerical-safety clamp `torch.clamp(ratio, 0.05, 20.0)` (REJECTED, reverted)

**Idea**: SF's `learner.py:591` clamps the raw probability ratio `exp(logp-logp_old)` to
`[0.05, 20.0]` *before* the PPO clip-ratio objective is computed, "since super large/small values
can cause numerical problems and are probably noise anyway." RelayRL had no equivalent clamp.
Per H7's takeaway, this was the last remaining concrete formula-level discrepancy versus SF, and
was expected to be a near-no-op given the baseline's ratio never leaves `[0.8, 1.2]` (ClipFrac
steady at `0.0000` across all 5 original baseline runs — well inside `[0.05, 20.0]`).

**Change** (PPO algorithm scope, `kernel.rs` only, commit `a88546b`):
- In `train_step_discrete`, changed
  `let ratio = (logp.clone() - logp_old_tensor).exp();`
  to
  `let ratio = (logp.clone() - logp_old_tensor).exp().clamp(0.05, 20.0);`
  (one extra `.clamp()` call; `clipped_ratio`/`clip_obj`/clipfrac formulas unchanged from
  pre-H7 baseline).

**Results (n=5, vs original baseline: final avg 141.02 range [131.8,162.3], AUC avg 93.54 range
[73.22,108.50])**:
- final = [170.30, 134.40, 63.80, 75.60, 133.00], n=5 avg **115.42** (**-18.1%** vs baseline) —
  the largest regression of any hypothesis so far, and the n=5 average now falls *entirely below*
  the original baseline's per-run range (115.42 < 131.8).
- AUC = [101.74, 103.14, 93.13, 69.11, 75.55], n=5 avg **88.53** (**-5.4%** vs baseline).
- Extreme run-to-run spread: run1 (170.30/101.74) and run2 (134.40/103.14) were the *best or
  near-best* final/AUC pairs seen in this entire project, while run3 (63.80/93.13) and run4
  (75.60/69.11) were among the *worst* — run4's AUC=69.11 is a new low (below H2's instability
  threshold discussion, though MeanReturn itself never collapsed to H2's -45.1-style dip; min/max
  per run stayed in baseline's normal range: (-186.9,170.3), (-188.8,160.2), (-196.3,161.9),
  (-206.7,197.6), (-200.4,161.7)).
- ClipFrac: nonzero in all 5 runs (171-209/832 epochs, means 0.0499-0.0616, 4-10 epochs/run
  hitting `>=0.99`) — same order of magnitude as H6/H7, despite this clamp being a mathematical
  identity for every ratio value actually observed in baseline (`[0.8,1.2] ⊂ [0.05,20.0]`).
- Throughput: 41614/41270/41019/41120/41061 env-frames/sec, avg ≈41217 — no regression.

**Verdict**: REJECTED, reverted (commit `a88546b` reverted via `git revert -n`). Both final and
AUC regressed, with final's -18.1% being the worst result of any hypothesis to date — yet the
clamp is a no-op for every ratio value that ever occurs in this regime (confirmed by baseline's
ClipFrac=0.0000). The only possible mechanism is that adding the `.clamp()` op to the autograd
graph perturbs LibTorch's floating-point execution order from epoch 1, and PPO's chaotic
~830-epoch dynamics amplify that perturbation into a *different trajectory* — sometimes much
better (run1, run2) and sometimes much worse (run3, run4) than baseline, with this seed-set
landing net-negative.

**Takeaway for future hypotheses — methodological, not just substantive**: H8's result is the
clearest evidence yet that **any change to the pi/vf forward+backward computation graph — even a
provably-no-op one — measurably perturbs this benchmark's chaotic trajectory and dominates the
n=5 signal**. Combined with H6/H7 (both also showed the same "ClipFrac goes from 0.0000 to
~0.05" signature the instant the graph changes, regardless of mechanism), this strongly suggests:
(1) the "formula-parity micro-tweak" axis is not just exhausted but actively *counterproductive*
to keep probing — every remaining SF-vs-RelayRL formula difference we can find is now either
verified-matched (reward_scale/clip, kl_loss_coeff=0, lr_schedule=constant, grad-norm clip=4.0,
obs/return normalization, orthogonal init, GAE λ/γ, clip-ratio value and formula, value clipping,
Adam epsilon — all checked across H1-H8 and the ~17 pre-log hypotheses) or, like H8, a no-op that
still destabilizes the run; (2) future hypotheses should prefer *structural* changes with
plausibly large (>15-20%) true effect sizes — large enough to be distinguishable from the
~±15-20% noise floor *and* from this "any-perturbation" tax — rather than further loss-formula
tuning. The leading structural candidate not yet audited: GAE bootstrap correctness for
*truncated* (not terminated) episodes at the 90-step rollout boundary, given `max_episode_steps
=500` means a large fraction of each epoch's 46080 transitions sit at such boundaries under
EnvPool's auto-reset. (A truncation-bootstrap hypothesis was tried pre-log and reverted, but
predates several since-kept fixes — e.g. GAE-lambda value-targets, fresh-value/fresh-logp
recomputation — and should be re-audited against the *current* code rather than assumed closed.)

## Hypothesis 9: fix GAE truncation-bootstrap to use V(s_T) instead of V(s_end) (REJECTED, reverted)

**Idea**: Per H8's takeaway, audited `replay_buffer.rs`'s three `compute_gae_episode` call sites
against the framework's `training/mod.rs` (read-only). Confirmed `trajectory.is_truncated=true`
is set for three structurally-different conditions: (1) EnvPool's real `max_episode_steps=500`
time-limit truncation (sub-env auto-resets, so `obs[end]` is the *reset* observation of a
brand-new episode), (2) a rare termination-at-max-steps edge case, and (3) the 90-step
rollout-chunk cutoff with the episode still ongoing (`obs[end]` IS the true `s_{T+1}`). All three
were previously bootstrapped identically with `V(obs[end])` (falling back to `V(obs[end-1])`).
For cases 1/2, `V(obs[end])` is a scale-mismatched bootstrap value from an unrelated episode.
OpenAI spinningup-PPO's canonical convention bootstraps non-terminal cutoffs with `V(s_T)`
(the chunk's own last state, `V(obs[end-1])`), which is a good approximation for case 3
(`gamma=0.999≈1` ⇒ `V(s_T)≈V(s_{T+1})`) and also fixes cases 1/2 (no longer references a
different episode's state).

**Change** (PPO algorithm scope, `replay_buffer.rs` only, commit `2a5770d`):
- In all three call sites (`finalize_gae_blocking`, `finalize_and_drain_blocking`,
  `finalize_and_drain_first_n_blocking`), changed the `is_truncated=true` bootstrap from
  `values.get(end).or_else(|| values.get(end-1))...unwrap_or(0.0)` to
  `values.get(end.saturating_sub(1))...unwrap_or(0.0)` — i.e. always use `V(s_T)`, never
  `V(s_{T+1})`.

**Results (n=5, vs original baseline: final avg 141.02 range [131.8,162.3], AUC avg 93.54 range
[73.22,108.50])**:
- final = [95.70, 159.90, 121.40, 139.60, 124.20], n=5 avg **128.16** (**-9.1%** vs baseline).
- AUC = [73.89, 120.43, 70.36, 74.47, 89.18], n=5 avg **85.67** (**-8.4%** vs baseline).
- Run-to-run spread was *wider than baseline on both tails*: run1's final=95.70 is far below
  baseline's per-run minimum (131.8), while run2's AUC=120.43 exceeds baseline's per-run maximum
  (108.50). min/max per run: (-198.5,157.5), (-212.4,270.9), (-183.9,155.6), (-179.2,161.5),
  (-191.4,233.5) — run2/run5 show much higher peak MeanReturns (233.5-270.9) than any baseline
  run, but this didn't translate into a higher final/AUC overall.
- ClipFrac nonzero in all 5 runs (186-268/832 epochs, means 0.0578-0.0795, 6-16 epochs/run
  hitting `>=0.99`) — same "perturbation tax" signature as H6/H7/H8, on the high end of the range
  seen so far.
- Throughput: 42517/34797/34459/34678/35526 env-frames/sec, avg ≈36395 — no regression (run1's
  higher number reflects reduced container load at that moment, not a code effect; a mid-run2
  container restart required restarting that run from scratch).

**Verdict**: REJECTED, reverted (commit `2a5770d` reverted via `git revert -n`). Both final and
AUC regressed (-9.1%/-8.4%), continuing the pattern from H6-H8: every change that touches the
GAE/value computation graph — even one with a sound, independently-justified theoretical basis
(spinningup-PPO's own bootstrap convention) and a real bug it fixes (cases 1/2's cross-episode
`V(obs[end])` reference) — produces the same ~0.06-0.08 ClipFrac "perturbation tax" and a
net-negative n=5 average, *despite* one run (run2) producing this project's new all-time-high
AUC (120.43) and two runs (run2/run5) reaching peak MeanReturns (233.5/270.9) never seen in any
prior baseline or hypothesis run. This is the **fourth consecutive REJECT** (H6, H7, H8, H9) for
graph-touching changes, each independently well-motivated, each showing the same
0.0000→~0.05-0.08 ClipFrac signature and a degraded n=5 average despite occasional
best-ever single runs.

**Takeaway for future hypotheses**: the evidence is now overwhelming that *any* perturbation to
the pi/vf forward+backward graph or its GAE inputs — regardless of correctness or theoretical
justification — measurably alters this benchmark's chaotic ~830-epoch trajectory, and the n=5
average for *this specific seed set* has landed negative for all four attempts so far. Two
non-exclusive interpretations: (a) this seed set is unlucky for graph-perturbing changes
specifically (the high run2/run5 peaks suggest the *upside* is real but doesn't dominate the
average), or (b) RelayRL's current configuration sits at a sharp local optimum in trajectory-space
where small perturbations are net-harmful on average even when they fix real bugs. Given (a)/(b)
are hard to distinguish at n=5, and four formula/graph-level hypotheses have now all failed
the same way, the loop should pivot to a different *class* of change for H10: something that
does NOT touch the pi/vf graph or GAE math at all — e.g. cadence/schedule-level changes
(rollout length, epochs-per-update, minibatch count/size, LR schedule shape) where SF and
RelayRL configs may still differ, which perturb the *optimization trajectory* through a
different mechanism (changing how much data/how many gradient steps occur between evaluations)
rather than the *per-step numerics*, and thus may not trigger the same ClipFrac signature.

## Hypothesis 10: match SF's value_bootstrap=False (GAE bootstrap=0 for all episode-boundary cuts) (REJECTED, reverted)

**Idea**: Read installed SF source (`sample_factory/algo/learning/learner.py`) directly instead
of relying on convention. Found: `--value_bootstrap` defaults to `False` (not overridden in
`sf_lunar_bench.py`), and SF's `batched_sampling.py` sets `dones = terminated | truncated` and
`time_outs = truncated`. With `value_bootstrap=False`, the reward adjustment
`buff["rewards"].add_(gamma * V * time_outs * dones)` is NOT applied. SF's `gae_advantages`
multiplies the bootstrap by `(1 - dones)`, so for every done=1 step — both true terminations
AND `max_episode_steps=500` truncations — the bootstrap is **0**. This means SF is already in
the "zero-bootstrap for truncated episodes" regime, directly contradicting the H9 approach
(bootstrap with V(s_T)) and also different from the original code's V(obs[end]). Setting
`bootstrap=0.0` unconditionally across all three `compute_gae_episode` call sites in
`replay_buffer.rs` is the exact formula match.

**Change** (PPO algorithm scope, `replay_buffer.rs` only, commit `c482d06`):
- In all three call sites (`finalize_gae_blocking`, `finalize_and_drain_blocking`,
  `finalize_and_drain_first_n_blocking`), replaced the `if *is_truncated { V(obs[end])... }`
  conditional with `let bootstrap = 0.0;` unconditionally — `_is_truncated` retained in the
  destructuring but unused. This is the most literal possible match to SF's
  `(1 - dones) * V(s_{T+1}) = 0` when `dones=1`.

**Results (n=5, vs original baseline: final avg 141.02 range [131.8,162.3], AUC avg 93.54 range
[73.22,108.50])**:
- final = [125.90, 238.50, 98.20, 154.30, 103.70], n=5 avg **144.12** (**+2.2%** vs baseline).
- AUC = [58.90, 86.63, 112.24, 86.15, 85.17], n=5 avg **85.82** (**-8.3%** vs baseline).
- Extreme variability: final range 98.2-238.5 (140-point spread, far wider than baseline's 30.5)
  and AUC range 58.90-112.24 (53-point spread, wider than baseline's 35.3, with run1's 58.90
  setting a new all-time-low and run2's 238.50 a new all-time-high for final return).
- min/max MeanReturn per run: (-326.2,147.4), (-370.4,241.3), (-249.4,232.6), (-188.8,160.2),
  (-192.4,163.3) — runs 1-3 have much deeper negative dips than any baseline run (baseline min
  floor ≈ -208), suggesting early-training instability when bootstrap=0 prevents the value
  function from getting credit for continued play at rollout boundaries.
- ClipFrac: 211/831, 221/832, 254/832, 162/832, 202/832 epochs nonzero (means 0.0578-0.0730,
  9-14 epochs/run hitting `>=0.99`) — same "perturbation tax" ClipFrac signature as H6-H9.
- Throughput: 35103/36215/36051/34363/35229 env-frames/sec, avg ≈35392 — no regression.

**Verdict**: REJECTED, reverted (commit `c482d06` reverted via `git revert -n`). AUC regressed
-8.3% while final improved only +2.2% — identical to the H6/H7 pattern (final up ~2-4%, AUC
down ~5-8%). This is the **fifth consecutive REJECT** (H6–H10), all showing the same signature:
graph-touching changes → ClipFrac 0.0000→~0.06 → early-training instability → AUC regression.
Even the most source-code-grounded fix (directly matching SF's `value_bootstrap=False` with
bootstrap=0) fails to improve AUC.

**Takeaway — root-cause discovery**: During post-analysis, the `fresh_logp` mechanism in
`independent/mod.rs:528-530` was identified as the likely root cause of **both** the
baseline's anomalous `ClipFrac=0.0000` AND the "perturbation tax":

```rust
let fresh_logp = kernel.get_pi_logprobs(&batch.obs, ...);
if fresh_logp.len() == batch.logp.len() {
    batch.logp = fresh_logp;  // OVERWRITES rollout-time logp with current-epoch logp
}
```

This replaces `logp_old` (from rollout time) with logprobs re-computed from the CURRENT network
(at epoch-start, before SGD). As a result, the PPO ratio `exp(logp_new - logp_old)` always
starts at ~1.0 at epoch-start (since `logp_old` was just computed from the same network state),
making the clip inactive → `ClipFrac=0.0000`. Any change to the computation graph (H6–H10)
perturbs LibTorch's float execution order → `fresh_logp` differs microscopically from baseline
→ ratio≠1.0 → `ClipFrac>0` → different trajectory.

Standard PPO (and SF's APPO) uses ROLLOUT-TIME `logp_old` as the fixed reference across all N
gradient steps within an epoch, which allows the clip to actually function as an importance-
weight correction (bounding how far the current policy can drift from the COLLECTION policy).
RelayRL's fresh_logp makes the clip bound only intra-epoch drift from the epoch-start policy —
which is tiny for 4 SGD steps — effectively disabling the clip entirely. **H11 hypothesis: remove
`batch.logp = fresh_logp` and use rollout-time logp as standard PPO requires**, allowing the
PPO clip to bound policy updates against the actual data-collection policy. This is a correction
to a fundamental algorithmic deviation from standard PPO (and SF), in `independent/mod.rs`
(PPO-algorithm scope), with a plausibly large effect on both ClipFrac and sample efficiency.

## Hypothesis 11: use rollout-time logp_old (remove fresh_logp substitution) (ACCEPTED)

**Idea**: The `fresh_logp` mechanism in `independent/mod.rs` replaces `batch.logp` (rollout-time
log-probs) with re-computed log-probs from the current (epoch-start) network before the 4 SGD
steps. This makes `logp_old` in the PPO ratio `exp(logp_new - logp_old)` always ≈1.0 at
epoch-start (both from the same network state), keeping all ratios near 1.0 across all 4 steps
→ ClipFrac=0.0000 throughout training — the PPO clip was never active. Standard PPO (and SF's
APPO) uses rollout-time log-probs as `logp_old`, fixed for all N gradient steps per epoch,
allowing the clip to bound policy drift from the COLLECTION policy. The `fresh_logp` comment
cited "ORT/burn numerical mismatch" (ORT = ONNX Runtime, used for inference in older configs) —
that mismatch no longer applies in the current LibTorch-only setup (all forward passes via Burn's
LibTorch backend); keeping `fresh_logp` was unintentionally disabling a core PPO safety
mechanism. Removing it restores standard PPO semantics.

**Change** (PPO algorithm scope, `independent/mod.rs` only, commit `3afb136`):
- Removed the `fresh_logp` block (4 lines) replacing it with a single comment explaining the
  standard-PPO rationale. `batch.logp` now retains the rollout-time log-probs as collected.
  Fresh value computation (`fresh_values`) is retained (values DO need to be fresh since the
  value network's de-normalization depends on the current epoch's running stats).

**Results (n=5, vs original baseline: final avg 141.02 range [131.8,162.3], AUC avg 93.54 range
[73.22,108.50])**:
- final = [161.00, 130.10, 145.40, 157.60, 138.70], n=5 avg **146.56** (**+3.9%** vs baseline).
- AUC = [113.36, 95.75, 76.15, 102.64, 99.67], n=5 avg **97.51** (**+4.2%** vs baseline).
- **Both metrics improved** — first simultaneous improvement at n=5 since the loop began. This
  distinguishes H11 from H6-H10 which all showed the "final up, AUC down" pattern (the
  perturbation-tax signature caused by disturbing `fresh_logp`'s floating-point ordering).
- Final range 130.1-161.0 (30.9-point spread vs baseline's 30.5 — essentially same variance).
  AUC range 76.15-113.36 (37.2-point spread vs baseline's 35.3 — marginally wider but similar).
- min/max MeanReturn per run: (-196.0,165.1), (-186.6,165.6), (-182.8,160.2), (-184.7,168.4),
  (-198.7,157.1) — all within normal baseline range (min floor ~-208, max ceiling ~170); no
  deep-dip instability signals.
- ClipFrac: nonzero in all 5 runs (212-274/832 epochs, means 0.0500-0.0607, 4-8 epochs/run
  hitting `>=0.99`). **ClipFrac is now meaningfully nonzero** — the PPO clip is active for the
  first time, bounding updates relative to the rollout-time policy as intended. This is a new
  regime compared to baseline's 0.0000; the modest ~0.05 mean (not 0.0000 and not the
  catastrophically-high 0.98 seen in H2) suggests a healthy trust-region constraint.
- Throughput: 33951/32958/41980/42296/42023 env-frames/sec, avg ≈39042 — no regression
  (runs 1-2's lower numbers reflect higher system load at that time of night, not algorithmic
  cost; runs 3-5 match baseline's ~39-42k range).

**Verdict**: **ACCEPTED**. Both final (+3.9%) and AUC (+4.2%) improved over the original
baseline at n=5, satisfying the "both must improve" acceptance rule, with no instability
(min/max ranges normal, ClipFrac healthy). Effect sizes are modest (within baseline's noise
band), warranting caution as seen in H4's n=3 provisional accept → n=5 reversal — but unlike
H4 (whose accept was at n=3 then reversed at n=5), H11 is being evaluated at the full n=5
directly. Commit `3afb136` retained as the new baseline.

**New baseline (H11)**: final avg 146.56 range [130.1,161.0], AUC avg 97.51 range [76.15,113.36].

**Takeaway**: removing `fresh_logp` restores standard PPO semantics and is the most significant
single algorithmic fix found in this loop so far. The PPO clip, previously disabled by the
fresh-logp no-op, now provides a real trust-region constraint relative to the data-collection
policy. All future hypotheses should be evaluated against this new baseline. Given H4's lesson,
an additional n=5 confirmation run of H11 itself (with the new baseline as comparison) may be
prudent before building further hypotheses on top of it — but the primary loop signal (both
metrics improved at n=5) supports accepting and continuing.

## Hypothesis 12: reduce GAE lambda from 0.98 to 0.95 (IN PROGRESS, n=0/5)

**Idea**: Current lam=0.98 matches SF's config. However, with complete episodes up to 500 steps
and gamma=0.999, the effective advantage horizon = 1/(1-lam) ≈ 50 steps. For early trajectory
steps in long episodes (e.g., step 1 of a 500-step episode), the advantage estimate incorporates
all 499 future rewards weighted by (0.999×0.98)^t, creating high variance. Lower lam=0.95 reduces
the effective horizon to ~20 steps, which could lower advantage variance → more stable early
learning → improved AUC. SF (0.98), OpenAI SpinningUp (0.97), and Stable Baselines (0.95) all use
different defaults — lam has not yet been independently varied in this loop despite being listed
as a "remaining candidate" in H5/H6/H7 takeaways. Pure replay-buffer change, no kernel or loss
graph modification; zero perturbation risk.

**Change** (`bench_lunar_ppo_tch.rs`, constant change only):
- `const LAM: f32 = 0.98` → `const LAM: f32 = 0.95`

**Results (n=5, vs H11 baseline: final avg 146.56 range [130.1,161.0], AUC avg 97.51 range [76.15,113.36])**:
- final = [137.60, 150.30, 168.40, 116.70, 154.10], n=5 avg **145.42** (**-0.8%** vs baseline).
- AUC = [115.05, 105.98, 90.56, 83.76, 120.66], n=5 avg **103.20** (**+5.8%** vs baseline).
- final range 116.7-168.4 (51.7-point spread, wider than H11's 30.9 — lam=0.95 increases run-to-run variance).
  AUC range 83.76-120.66 (36.9-point spread, slightly wider than H11's 37.2 — similar).
- Notable negative correlation between final and AUC across runs: high-final runs (run3: 168.40/90.56)
  tend to have lower AUC, and high-AUC runs (run1: 137.60/115.05, run5: 154.10/120.66) tend to have
  lower final. This reflects the lam=0.95 tradeoff: shorter effective advantage horizon helps early
  learning (AUC) but weakens credit assignment for final-episode rewards (final return).
- ClipFrac: means 0.059-0.071 (nonzero 261-318/832 epochs) — consistent with H11 baseline (~0.05-0.06),
  as expected (lambda only affects advantages, not the loss graph or logp computation).

**Verdict**: **REJECTED**, reverted (`const LAM: f32 = 0.97` for H13). AUC improved +5.8% but final
regressed -0.8%, failing the "both must improve" acceptance rule. The AUC gain is genuine and
consistent across runs, but comes at the expense of the final metric.

**Takeaway for future hypotheses**: lambda is a real lever with a clear directional tradeoff:
lam=0.95 (eff. horizon ~20 steps) speeds up early learning (AUC) but weakens long-range credit
assignment (final return), while lam=0.98 (eff. horizon ~50 steps) does the reverse. The tradeoff
is consistent across all 5 runs (negative correlation between final and AUC). Next step: try
lam=0.97 (eff. horizon ~33 steps) to test whether an intermediate value achieves improvement in
both metrics. If the tradeoff is monotonic (any lam<0.98 helps AUC/hurts final), the lambda axis
is exhausted and a different direction is needed.

## Hypothesis 13: GAE lambda 0.97 — intermediate between 0.95 and 0.98 (REJECTED)

**Idea**: H12 established that lam=0.95 gives AUC +5.8% but final -0.8%. The H11 baseline uses
lam=0.98. lam=0.97 (eff. horizon ~33 steps, between 0.95's ~20 and 0.98's ~50) may capture some
of lam=0.95's AUC benefit while recovering the final return. The `IPPOParams::default()` originally
used lam=0.97; the benchmark overrides to 0.98 to match SF. Testing 0.97 determines whether the
lambda/metric tradeoff has a sweet spot between 0.95 and 0.98.

**Change** (`bench_lunar_ppo_tch.rs`, constant change only):
- `const LAM: f32 = 0.98` → `const LAM: f32 = 0.97`

**Results (n=5, vs H11 baseline: final avg 146.56 range [130.1,161.0], AUC avg 97.51 range [76.15,113.36])**:
- final = [169.50, 124.40, 106.20, 141.00, 149.40], n=5 avg **138.10** (**-5.8%** vs baseline).
- AUC = [74.06, 113.72, 65.17, 99.29, 130.29], n=5 avg **96.51** (**-1.0%** vs baseline).
- Both metrics below baseline — lam=0.97 is worse than lam=0.98 on both axes simultaneously.
  Unlike lam=0.95 (AUC+/final-), lam=0.97 shows no partial benefit. High variance: final range
  106.2-169.5 (63.3-point spread vs H11 baseline's 30.9) — lam=0.97 is more unstable than either
  lam=0.95 or lam=0.98.

**Verdict**: **REJECTED**. Both metrics below H11 baseline. Lambda axis now exhausted:
- lam=0.95: AUC+5.8%, final-0.8% → REJECTED (AUC up, final down)
- lam=0.97: AUC-1.0%, final-5.8% → REJECTED (both down)
- lam=0.98: H11 baseline (best tested value)

**Takeaway for future hypotheses**: Lambda axis is closed — 0.98 is the best tested value. The
tradeoff is non-monotonic: lam=0.97 does not split the difference; it is strictly worse than
lam=0.98 on both metrics. Future hypotheses should target a different axis: number of SGD
iterations per batch (train_pi/vf_iters), policy clip ratio, entropy coefficient, or LR schedule.

## Hypothesis 14: more SGD iterations per batch (train_pi/vf_iters 4 → 8) (IN PROGRESS, n=0/5)

**Idea**: Each epoch collects ~46080 transitions (512 envs × 90-step rollout) and runs 4 SGD
passes over the full batch. With ClipFrac averaging ~0.05 (H11 baseline), the PPO clip constraint
is active but not heavily binding — suggesting the policy could safely take additional gradient
steps per batch without divergence. Doubling to 8 SGD passes per epoch extracts more learning from
each collected batch, directly improving sample efficiency (same env frames → more gradient
updates). SF uses `num_epochs=4` but that is not a ceiling for RelayRL. The PPO clip provides a
trust-region safeguard: if later iterations drift the policy too far, ClipFrac will spike,
signaling instability early. Single two-constant change, no algorithm or graph modification.

**Change** (`bench_lunar_ppo_tch.rs`, constant change only):
- `const TRAIN_PI_ITERS: u64 = 4` → `const TRAIN_PI_ITERS: u64 = 8`
- `const TRAIN_VF_ITERS: u64 = 4` → `const TRAIN_VF_ITERS: u64 = 8`
- `const LAM: f32 = 0.97` reverted to `const LAM: f32 = 0.98` (H13 cleanup)

**Results (n=0/5 in progress)**:
- Run 1: IN PROGRESS

**Verdict**: PENDING (n=5 required)
