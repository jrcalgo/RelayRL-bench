# Sample Factory APPO vs RelayRL PPO on LunarLander-v3

## 0. The data: same hyperparameters, very different outcomes

Both runs use 64 envs, lr=2.5e-4, gamma=0.999, lam=0.98, clip=0.2, ent=0.01,
[128,128] MLP, ReLU, orthogonal init, seed=42 — wired up by the matched
benchmark scripts in this repo:

- `bench_beta4/bench_sf_lunarlander.py:38-82`
- `bench_beta4/src/bin/bench_lunar_ppo_64env.rs:22-44`

| Run | Wall | Total frames | FPS | Final 100-ep reward |
|---|---|---|---|---|
| Sample Factory APPO | 2407 s | 20.0 M | **8,307** | **~290** (solved: >200) |
| RelayRL PPO (LibTorch) | 485 s | 38.4 M | **79,097** | **~56** (plateaued) |

Sources: `bench_beta4/results_sf_lunarlander.txt` (Avg episode reward 293.631 at
20M frames) and `bench_beta4/results_sf_matched.txt` (`mean_ret(100) = 57.9` at
epoch 4790, ~38M frames).

RelayRL is **9.5× faster** but ends **~234 reward points lower**, even after
seeing nearly **2× more environment frames**. The question is: where does
APPO's training-side recipe convert frames into return that RelayRL's recipe
doesn't?

---

## 1. Sample Factory APPO: system architecture

APPO decouples sampling from learning into four process roles connected by an
event-loop / signal-slot graph and shared-memory tensors.

**Wiring** — `sample_factory/algo/runners/runner.py:626-678`
```python
sampler.connect_on_new_trajectories(policy_id, batcher.on_new_trajectories)
batcher.training_batches_available.connect(learner_worker.on_new_training_batch)
learner_worker.training_batch_released.connect(batcher.on_training_batch_released)
sampler.connect_stop_experience_collection(batcher.stop_experience_collection)
sampler.connect_resume_experience_collection(batcher.resume_experience_collection)
```

The roles:

1. **RolloutWorker** owns the env vector, sends obs to the InferenceWorker,
   receives actions, accumulates trajectories. With `worker_num_splits=2`
   (default), each rollout worker double-buffers its envs so half is being
   inferred-on while the other half is stepping — `rollout_worker.py:97-126`.

2. **InferenceWorker** runs the policy forward pass on batched requests.
   Weights live in shared memory; it polls a shared `policy_versions` tensor
   to know when to reload — `sample_factory/algo/utils/model_sharing.py:128-143`:
   ```python
   def ensure_weights_updated(self):
       server_policy_version = self._get_server_policy_version()
       if self.latest_policy_version < server_policy_version and ...:
           with self.timing.time_avg("weight_update"), self._policy_lock:
               self._actor_critic.load_state_dict(self._shared_model_weights)
   ```
   This is "asynchronous" — no message, just a polled integer in
   `cfg.num_policies`-sized SHM tensor (`shared_buffers.py:237-239`).

3. **Batcher** stitches incoming trajectory slices into training tensors, and
   throttles the sampler when more than `num_batches_to_accumulate` batches
   are queued.

4. **LearnerWorker** consumes a batch, runs `num_epochs × num_batches_per_epoch`
   SGD updates, bumps `policy_versions_tensor`, then signals the batcher to
   release the buffer.

The "APPO" part is that the InferenceWorker keeps serving actions during the
learner's gradient steps — no `optimizer.step()` blocks experience collection.
Off-policy slack is bounded by `max_policy_lag=1000` (default), the
`policy_versions` deltas the learner tolerates per trajectory.

**Throughput plumbing** — everything is a torch tensor with `.share_memory_()`
(`shared_buffers.py:35-57`), CPU affinity is pinned per worker
(`rollout_worker.py:55-65`), and `torch.set_num_threads(1)` stops worker
threads from fighting over BLAS pools.

---

## 2. Why APPO converges on LunarLander

The convergence story is essentially **PPO with three stabilizers baked into
the model itself**, plus a fourth one in the loss. They all live in
`algo/learning/learner.py` and `model/actor_critic.py`.

### 2.1 Observation normalization inside the model

`sample_factory/model/actor_critic.py:32`
```python
self.obs_normalizer: ObservationNormalizer = ObservationNormalizer(obs_space, cfg)
```

`actor_critic.py:98-99`
```python
def normalize_obs(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return self.obs_normalizer(obs)
```

LunarLander's 8 obs dims have wildly different scales — `Box([-2.5, -2.5, -10,
-10, -6.28, -10, 0, 0], [2.5, 2.5, 10, 10, 6.28, 10, 1, 1])`. Without
whitening, the angular-velocity dim (range ±10) dominates the first-layer
gradient and starves the leg-contact bits (range {0,1}) for capacity. SF
maintains a `RunningMeanStdInPlace` and applies it on every forward —
`sample_factory/algo/utils/running_mean_std.py:64-110`:
```python
x.sub_(μ).mul_(1 / σ).clamp_(-clip, clip)   # clip default 5.0
```
The normalizer's running stats live **inside the actor-critic Module**, so the
inference workers' polled `state_dict` carries them forward automatically.

### 2.2 Return normalization with denorm-for-GAE

`learner.py:969-1019` (excerpt):
```python
if self.cfg.normalize_returns:
    denormalized_values = buff["values"].clone()
    self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
else:
    denormalized_values = buff["values"]

buff["advantages"] = gae_advantages(buff["rewards"], buff["dones"],
                                    denormalized_values, ..., gamma, gae_lambda)
buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]
if self.cfg.normalize_returns:
    self.actor_critic.returns_normalizer(buff["returns"])   # in-place
```
The value head trains on returns-with-running-mean-std applied. GAE runs on
the **denormalized** values so deltas are in environment units. This decouples
the value head's loss magnitude from the reward scale, which on LunarLander
swings from −500 to +300.

### 2.3 Symmetric ratio clip + clipped value loss

`learner.py:430-459`:
```python
@staticmethod
def _policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids):
    clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
    loss = torch.min(ratio * adv, clipped_ratio * adv)
    return -masked_select(loss, valids, num_invalids).mean()

def _value_loss(self, new_values, old_values, target, clip_value, ...):
    value_clipped = old_values + torch.clamp(new_values - old_values,
                                             -clip_value, clip_value)
    value_loss = torch.max((new_values - target).pow(2),
                           (value_clipped - target).pow(2))
    value_loss = masked_select(value_loss, valids, num_invalids).mean()
    value_loss *= self.cfg.value_loss_coeff
```
And `learner.py:543-547`:
```python
clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio   # 1.2 for clip=0.2
clip_ratio_low  = 1.0 / clip_ratio_high           # 0.833, not 0.8
```
The ratio clamp is **symmetric in log-space** (`[1/(1+ε), 1+ε]`), and the
value head uses **PPO-style clipping** that bounds how far a single update can
move the critic — critical because the learner sees off-policy data from
older policy versions.

Belt-and-braces ratio guard against numerical blow-ups —
`learner.py:594`:
```python
ratio = torch.clamp(ratio, 0.05, 20.0)
```

### 2.4 KL-adaptive LR

`learner.py:46-65`:
```python
if mean_kl > 2.0 * self.lr_schedule_kl_threshold:
    lr = max(current_lr / 1.5, self.min_lr)
if mean_kl < (0.5 * self.lr_schedule_kl_threshold):
    lr = min(current_lr * 1.5, self.max_lr)
```
Optional but on by default in the standard configs. Keeps KL near a target
band rather than letting `target_kl` early-stop the entire iteration.

### 2.5 Orthogonal init with scaled gains

`model/actor_critic.py:73-88`:
```python
if self.cfg.policy_initialization == "orthogonal":
    if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
        nn.init.orthogonal_(layer.weight.data, gain=gain)
```
Hidden layers use `gain=sqrt(2)` (ReLU); the policy head uses a small gain so
initial logits are near-uniform — both standard PPO tricks.

### 2.6 GAE on transposed `[T, E]` data, then normalize

`algo/utils/rl_utils.py:77-94`:
```python
@torch.jit.script
def gae_advantages(rewards, dones, values, valids, γ, λ):
    rewards = rewards.transpose(0, 1)   # [E, T] -> [T, E]
    deltas = (rewards - values[:-1]) * valids[:-1] \
             + (1 - dones) * (γ * values[1:] * valids[1:])
    advantages = calculate_discounted_sum_torch(deltas, dones, valids[:-1], γ * λ)
```
And `learner.py:646-647`:
```python
adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)
```

---

## 3. Where RelayRL is now

RelayRL has the *form* of PPO right. The matched run in
`bench_beta4/patches/relayrl_algorithms/src/algorithms/PPO/kernel.rs:298-336`
implements the clipped surrogate and combined `pi + vf_coef * vf` loss:

```rust
let ratio = (logp.clone() - logp_old_tensor).exp();
let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);
let clip_obj = (ratio.clone() * adv_tensor.clone())
    .min_pair(clipped_ratio * adv_tensor)
    .mean();
let entropy_t = (log_probs_full.clone().exp() * log_probs_full)
    .neg().sum_dim(1).reshape([n]).mean();
let pi_loss_t = -(clip_obj + ent_coef * entropy_t.clone());
...
let vf_loss_t = (v_pred - ret_tensor).powf_scalar(2.0).mean();
let total_loss = pi_loss_t.clone() + vf_loss_t.clone() * vf_coef_t;
```

Grad clipping is on
(`bench_beta4/patches/relayrl_algorithms/src/algorithms/PPO/kernel.rs:243-245`):
```rust
let optimizer = AdamConfig::new()
    .init::<TB, ActorCriticMlp<TB>>()
    .with_grad_clipping(GradientClipping::Norm(4.0));
```

Orthogonal init with the right gains is there
(`kernel.rs:163-178`):
```rust
let gain = if i < pi_n - 1 { 2.0f64.sqrt() } else { 0.01 };
...
layer.weight = Initializer::Orthogonal { gain }
    .init_with([w[0], w[1]], Some(w[0]), Some(w[1]), device);
```

Advantage normalization runs at drain time
(`bench_beta4/patches/relayrl_algorithms/src/algorithms/PPO/replay_buffer.rs:331-333`):
```rust
let (adv_mean, adv_std) = scalar_stats(&fresh_adv);
let adv_norm = compute_normed_advantages(&fresh_adv, adv_mean, adv_std.max(1e-8));
```

Return normalization is also wired (`replay_buffer.rs:335-340`).

So what's missing? Five things.

### 3.1 No observation normalization — anywhere

```
$ grep -rn "RunningMean\|running_mean\|obs_subtract_mean" bench_beta4/patches/
(empty)
```

The 8-dim LunarLander observation is fed raw through `[128, 128]`. With
orthogonal init at gain √2, the angular-velocity dim (σ ≈ 3-5 in practice)
roughly **30× dominates the leg-contact bits in the first-layer
pre-activation**. For comparison, in
`bench_beta4/lunarlander-rl/src/lib.rs` the env returns raw Box-space floats
straight from the Python `gymnasium.make("LunarLander-v3")` call without any
wrapping. This is the single largest convergence delta.

### 3.2 Asymmetric ratio clamp, no clipped value loss

`kernel.rs:315-318`:
```rust
let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);
let clip_obj = (ratio.clone() * adv_tensor.clone())
    .min_pair(clipped_ratio * adv_tensor)
    .mean();
```
- Ratio clamp `[0.8, 1.2]` is fine for on-policy data but biased low-side
  for ratios > 1, because the lower clip kicks in earlier in log-space than
  the upper one.
- The value loss is plain MSE (`kernel.rs:332`) — no `value_clipped`
  branch. With a slow-updating ORT inference snapshot (see §3.4), the value
  head can fit a moving target unboundedly per epoch.

### 3.3 Stale TorchScript inference snapshot, no policy-lag accounting

The inference path uses an ONNX/TorchScript snapshot of the actor-critic that
is **refreshed only after a full epoch's `apply_epoch_result`**
(`bench_beta4/patches/relayrl_framework/src/network/client/runtime/coordination/state_manager.rs:1048-1063`):
```rust
trainer.apply_epoch_result(output);
...
let pi_module = trainer.acquire_model_module();
let vf_module = trainer.acquire_value_module();
if pi_module.is_some() || vf_module.is_some() {
    tokio::spawn(async move {
        if let Some(m) = pi_module {
            let _ = rt.perform_refresh_model(m, dev.clone()).await;
        }
        if let Some(v) = vf_module {
            let _ = rt.perform_refresh_value_model(v, dev).await;
        }
    });
}
```
The lag is bounded by the version filter in
`bench_beta4/patches/relayrl_algorithms/src/algorithms/PPO/independent/mod.rs:485-490`:
```rust
match slot.replay_buffer.finalize_and_drain_first_n_blocking(
    fresh_values, current_version, max_version_lag, n,
    self.hyperparams.normalize_returns)
{
    Some(batch) => jobs.push((kernel, batch)),
    None => { slot.kernel = Some(kernel); continue; }
}
```
…but `max_version_lag` defaults to `1`
(`PPO/independent/mod.rs:121-133`). That means episodes more than one version
behind get **dropped, not corrected**. SF's `max_policy_lag=1000` plus
ratio-clipping ε ensures stale data still contributes a corrected gradient,
which is much more sample-efficient than the drop strategy when the inference
snapshot is genuinely lagging.

### 3.4 KL is too tight, doesn't adapt

`target_kl=0.05` is set in the bench, and the inner `pi_iters` loop early-stops
on `1.5 × target_kl`. There's no LR adaptation
(`PPO/kernel.rs:256-263`):
```rust
fn effective_lr(&self) -> f64 {
    match self.lr_schedule_steps {
        Some(total) if total > 0 => {
            let frac = 1.0 - (self.grad_step_count as f64 / total as f64).min(1.0);
            self.lr * frac.max(0.0)
        }
        _ => self.lr,
    }
}
```
— just a linear decay if `lr_schedule_steps` is set. The `results_sf_matched.txt`
log shows `KL: 0.005` and `Entropy: 0.78`, well below the KL band SF sustains.
That means the policy is barely moving on each epoch even though
`StopIter=4.0` (the full `train_pi_iters` ran). Combined with the un-normalized
obs, this looks like a small, biased gradient signal being applied repeatedly.

### 3.5 Single agent-slot for "Independent" PPO

`PPO/independent/mod.rs:300` registers one slot per actor — but
`bench_lunar_ppo_64env.rs:67-72` configures `actor_count(1)`. So the 64 envs
share one replay buffer and one kernel. That's fine in itself, but
`traj_per_epoch=320` means the trainer waits for 320 complete episodes
(`bench_lunar_ppo_64env.rs:40`) before each gradient step. On a 64-env
vectorized rollout that's roughly 5 episodes per env, so one update lands
every ~5,760 env steps — which actually matches SF's `batch_size=5760`. Good.

The structural difference is **who computes value targets**. SF computes them
on the **same** GPU model that just trained; RelayRL recomputes `V(s_t)` from
the **current burn-tensor kernel** before GAE
(`PPO/independent/mod.rs:474-482`):
```rust
let kernel = slot.kernel.take()?;
let (obs_flat, obs_dim_peek) = slot.replay_buffer.get_obs_flat_for_first_n_episodes(n);
let fresh_values = if !obs_flat.is_empty() {
    kernel.value_forward_only_flat(&obs_flat, obs_dim_peek)
} else {
    Vec::new()
};
```
Good — but the `logp_old` in the buffer still comes from the **TorchScript
inference snapshot**, which is older than the current burn kernel. So `ratio`
is computed against a mismatched-model `logp_old`, biasing every PPO update.
The asymmetric clamp then chews on that biased ratio.

---

## 4. What RelayRL can adopt without giving up throughput

The pattern that makes APPO work on LunarLander isn't async-vs-sync, and it
isn't model size. It's **a small set of training-side stabilizers that cost
microseconds per update**. None of them serialize the env loop. Listed in
order of expected impact for LunarLander:

### 4.1 Observation normalization living in the kernel

Maintain `running_mean[obs_dim], running_var[obs_dim], count` inside
`ActorCriticMlp` (`PPO/kernel.rs:144-150`). Update them on every minibatch
inside `train_step_flat` (after computing the loss, before the optimizer
step), and apply `(x - μ) / sqrt(σ² + ε)` on both `pi_forward` and
`vf_forward`. **Critically**, the running stats must travel into the
TorchScript/ONNX snapshot that `acquire_model_module` exports
(`PPO/independent/mod.rs:570-583`) so the inference path normalizes the same
way the training path does. SF achieves this by parameterizing the normalizer
as `nn.Module` state — `model/actor_critic.py:32`. RelayRL would parameterize
it as a fixed `Linear` layer with frozen weights, baked into the ONNX export
in `algorithms/onnx_builder.rs` /  `algorithms/pt_builder.rs`.

Cost at runtime: 8 fused-mul-adds per forward. ~0 fps impact, given the env
step already dominates collection wall time.

### 4.2 Symmetric ratio clamp + ratio guard

Two-line change in
`bench_beta4/patches/relayrl_algorithms/src/algorithms/PPO/kernel.rs:314-315`:
```rust
let ratio = (logp.clone() - logp_old_tensor).exp().clamp(0.05, 20.0);
let high = 1.0 + clip_ratio;
let low  = 1.0 / high;
let clipped_ratio = ratio.clone().clamp(low, high);
```
Adopts SF's `_policy_loss` shape (`learner.py:431-439`, `learner.py:543-547`,
`learner.py:594`). No effect on on-policy data; protects against blow-up when
`logp_old` came from a stale snapshot.

### 4.3 Clipped value loss

Replace plain MSE
(`PPO/kernel.rs:328-332`) with the same `torch.max` pattern as
`learner.py:441-459`. Needs `old_values` flowing through the buffer — RelayRL
already stores them (`replay_buffer.rs:200, 494-510`, key `"value"`), they
just aren't routed into the loss. This is the single highest-leverage
change for stability under any policy-version lag.

### 4.4 Return normalization with denorm-for-GAE

The current path (`replay_buffer.rs:335-340`) normalizes returns *after* GAE
has already been computed on un-normalized values. SF instead normalizes
returns AS A TRAINING TARGET and **denormalizes values before GAE**
(`learner.py:969-1019`). This is the correct ordering — otherwise the value
head learns to predict normalized returns but GAE is fed inconsistent units.
Wire it up by storing a `RunningMeanStd` for returns next to the obs
normalizer, denormalizing `val_flat` at the top of `finalize_and_drain`,
running GAE, then normalizing the resulting `returns` slice.

### 4.5 Tighten the inference-snapshot loop, or accept the lag with V-trace

Two viable paths:

**Cheaper:** Refresh the TorchScript snapshot **every N minibatches** during
`run_ppo_sgd_flat` (`PPO/independent/mod.rs:603-690`), not only at epoch
boundaries. The bottleneck is `acquire_model_module` cost — currently builds
ONNX bytes from scratch per refresh. A direct shared-tensor handoff (a la
SF's `_shared_model_weights`, `model_sharing.py`) would avoid the
ONNX-serialize/deserialize round-trip entirely. On CPU, sharing weight
pointers between the burn autodiff kernel and the inference path is cheap.

**Principled:** Keep the lag, but compute V-trace ρ/c clipping in
`run_ppo_sgd_flat` so off-policy gradients are corrected rather than dropped.
SF does this when `with_vtrace=True` — `learner.py:601-640`. The drop-on-stale
behavior currently in `finalize_and_drain_first_n_blocking` wastes the data
that RelayRL's throughput is good at generating.

### 4.6 Bigger `target_kl`, or a KL-adaptive LR

`target_kl=0.05` with `train_pi_iters=10` and a small batch is rarely going
to be the binding constraint — `KL=0.005` in the logs means we're 10× below
the threshold and still moving slowly. Either:

- Raise `target_kl` to ~0.02 and increase `train_pi_iters` to ~20 so each
  epoch gets more learning, OR
- Implement the SF KL-adaptive LR (`learner.py:46-65`) — multiply lr by 1.5
  when mean KL is below `0.5×threshold`, divide by 1.5 when above
  `2×threshold`. This is purely a learner-side change.

---

## 5. Summary

Sample Factory converges on LunarLander not because of asynchronous sampling
— `bench_sf_lunarlander.py` runs with `--async_rl=False` — but because the
*learner-side training recipe* contains five pieces of convergence-engineering
that RelayRL is missing or has half-built:

| Mechanism | SF location | RelayRL state |
|---|---|---|
| Obs running mean/std normalization | `actor_critic.py:32`, `running_mean_std.py:64-110` | **Missing entirely** |
| Returns normalization, GAE on denorm values | `learner.py:969-1019` | Inverted ordering (`replay_buffer.rs:331-340`) |
| Clipped value loss | `learner.py:441-459` | Plain MSE (`kernel.rs:328-332`) |
| Symmetric ratio clip + ratio clamp guard | `learner.py:543-547, 594` | Asymmetric `[1-ε, 1+ε]` (`kernel.rs:315`) |
| Off-policy correction (V-trace or freq snapshot) | `learner.py:601-640`, `model_sharing.py:128-143` | Drop on lag (`PPO/independent/mod.rs:485-490`) |

None of these touch the env loop, the inference path's hot edges, or the
shared-memory plumbing that gives RelayRL its 9.5× throughput advantage on
this benchmark. They live entirely inside `kernel.rs`, `replay_buffer.rs`, and
the snapshot-refresh hop in `state_manager.rs`. The order to try them is the
order above: §4.1 (obs norm) alone should close most of the LunarLander gap;
§4.3 (clipped value) and §4.4 (returns-denorm-then-GAE) close the rest; §4.5
(tighter snapshot or V-trace) is what keeps the recipe working as RelayRL
scales actors and the snapshot lag grows.
