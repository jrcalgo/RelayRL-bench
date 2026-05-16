use crate::algorithms::{compute_normed_advantages, discounted_cumsum, scalar_stats};
use crate::templates::base_replay_buffer::{Batch, GenericReplayBuffer, ReplayBufferError};
use async_trait::async_trait;
use relayrl_types::prelude::action::RelayRLData;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use std::any::Any;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Online Welford mean/variance accumulator for return normalization.
/// Persists across epochs so the scale stays stable as the policy improves.
#[derive(Default)]
struct RunningMeanStd {
    mean:  f64,
    m2:    f64,
    count: u64,
}
impl RunningMeanStd {
    fn update_batch(&mut self, xs: &[f32]) {
        for &x in xs {
            self.count += 1;
            let delta = x as f64 - self.mean;
            self.mean += delta / self.count as f64;
            self.m2   += delta * (x as f64 - self.mean);
        }
    }
    fn std(&self) -> f32 {
        if self.count < 2 { return 1.0; }
        ((self.m2 / (self.count - 1) as f64) as f32).sqrt().max(1e-8)
    }
    fn mean_f32(&self) -> f32 { self.mean as f32 }
}

struct Buffers {
    obs_flat: Vec<f32>,
    obs_dim: usize,
    act_flat: Vec<i64>,
    logp_flat: Vec<f32>,
    rewards: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    values: Vec<f32>,
    // Episode boundaries for deferred GAE: (path_start, path_end, is_truncated)
    episode_boundaries: Vec<(usize, usize, bool)>,
}

struct BufferMetadata {
    gamma: f32,
    lam: f32,
    buffer_size: usize,
    buffer_pointer: AtomicUsize,
    buffer_path_start_idx: AtomicUsize,
}

pub struct PPOFlatBatch {
    pub obs_flat: Vec<f32>,
    pub obs_dim: usize,
    pub act_flat: Vec<i64>,
    pub logp_flat: Vec<f32>,
    pub adv_norm: Vec<f32>,
    pub ret_flat: Vec<f32>,
    pub val_flat: Vec<f32>,
}

pub struct PPOReplayBuffer {
    buffers: Arc<Mutex<Buffers>>,
    metadata: Arc<BufferMetadata>,
    return_running: Mutex<RunningMeanStd>,
}

impl Default for PPOReplayBuffer {
    fn default() -> Self {
        Self::new(1_000_000, 0.99, 0.97)
    }
}

impl PPOReplayBuffer {
    pub fn new(buffer_size: usize, gamma: f32, lam: f32) -> Self {
        let buffers = Buffers {
            obs_flat: Vec::with_capacity(buffer_size * 8),
            obs_dim: 0,
            act_flat: Vec::with_capacity(buffer_size),
            logp_flat: Vec::with_capacity(buffer_size),
            rewards: Vec::with_capacity(buffer_size),
            advantages: Vec::with_capacity(buffer_size),
            returns: Vec::with_capacity(buffer_size),
            values: Vec::with_capacity(buffer_size),
            episode_boundaries: Vec::new(),
        };
        Self {
            buffers: Arc::new(Mutex::new(buffers)),
            metadata: Arc::new(BufferMetadata {
                gamma,
                lam,
                buffer_size,
                buffer_pointer: AtomicUsize::new(0),
                buffer_path_start_idx: AtomicUsize::new(0),
            }),
            return_running: Mutex::new(RunningMeanStd::default()),
        }
    }

    /// Compute GAE for one episode [start, end) using values already in buffers.values.
    /// bootstrap: 0.0 for terminal episodes, V(s_T) for truncated.
    fn compute_gae_episode(
        buffers: &mut Buffers,
        gamma: f32,
        lam: f32,
        start: usize,
        end: usize,
        bootstrap: f32,
    ) {
        if start >= end {
            return;
        }
        let mut rews = buffers.rewards[start..end].to_vec();
        let mut vals = buffers.values[start..end].to_vec();
        rews.push(bootstrap);
        vals.push(bootstrap);

        let deltas: Vec<f32> = (0..rews.len() - 1)
            .map(|i| rews[i] + gamma * vals[i + 1] - vals[i])
            .collect();
        let advantages = discounted_cumsum(&deltas, gamma * lam);
        buffers.advantages[start..end].copy_from_slice(&advantages);

        let full_returns = discounted_cumsum(&rews, gamma);
        buffers.returns[start..end].copy_from_slice(&full_returns[..full_returns.len() - 1]);
    }

    /// Return flat f32 observations for all buffered steps, used for deferred GAE value inference.
    pub fn get_obs_flat_for_gae_blocking(&self) -> (Vec<f32>, usize) {
        let buffers = self.buffers.lock().unwrap();
        let ptr = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        let obs_dim = buffers.obs_dim;
        if obs_dim == 0 || ptr == 0 {
            return (Vec::new(), 1);
        }
        (buffers.obs_flat[..ptr * obs_dim].to_vec(), obs_dim)
    }

    /// Return obs for exactly the first `n` complete episodes.
    /// Returns empty if fewer than `n` complete episodes exist in the buffer.
    pub fn get_obs_flat_for_first_n_episodes(&self, n: usize) -> (Vec<f32>, usize) {
        let buffers = self.buffers.lock().unwrap();
        if buffers.episode_boundaries.len() < n {
            return (Vec::new(), 1);
        }
        let obs_dim = buffers.obs_dim;
        if obs_dim == 0 {
            return (Vec::new(), 1);
        }
        let cut_step = buffers.episode_boundaries[n - 1].1;
        (buffers.obs_flat[..cut_step * obs_dim].to_vec(), obs_dim)
    }

    /// Fill values buffer and compute GAE for all recorded episode boundaries.
    pub fn finalize_gae_blocking(&self, values: Vec<f32>) {
        let mut buffers = self.buffers.lock().unwrap();
        let gamma = self.metadata.gamma;
        let lam = self.metadata.lam;

        let fill_len = values.len().min(buffers.values.len());
        buffers.values[..fill_len].copy_from_slice(&values[..fill_len]);

        let boundaries: Vec<_> = buffers.episode_boundaries.clone();
        for (start, end, is_truncated) in boundaries {
            let bootstrap = if is_truncated {
                values.get(end.saturating_sub(1)).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            Self::compute_gae_episode(&mut buffers, gamma, lam, start, end, bootstrap);
        }
    }

    /// Single-lock combine: fill values + GAE + normalize adv + drain buffer.
    /// Replaces separate finalize_gae_blocking + sample_buffer_blocking calls.
    pub fn finalize_and_drain_blocking(&self, values: Vec<f32>) -> Option<PPOFlatBatch> {
        let mut buffers = self.buffers.lock().unwrap();
        let gamma = self.metadata.gamma;
        let lam = self.metadata.lam;
        let capacity = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        if capacity == 0 {
            return None;
        }
        let obs_dim = buffers.obs_dim;

        // Fill values only when external values provided; else use inline stored values
        if !values.is_empty() {
            let fill_len = values.len().min(buffers.values.len());
            buffers.values[..fill_len].copy_from_slice(&values[..fill_len]);
        }

        // Compute GAE for each completed episode
        let boundaries: Vec<_> = buffers.episode_boundaries.clone();
        for (start, end, is_truncated) in boundaries {
            let bootstrap = if is_truncated {
                buffers.values.get(end.saturating_sub(1)).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            Self::compute_gae_episode(&mut buffers, gamma, lam, start, end, bootstrap);
        }

        // Normalize advantages across the full epoch batch
        let adv_raw = &buffers.advantages[..capacity];
        let (adv_mean, adv_std) = scalar_stats(adv_raw);
        let adv_norm = compute_normed_advantages(adv_raw, adv_mean, adv_std.max(1e-8));

        // Extract flat arrays (clone out before clearing)
        let obs_end = (capacity * obs_dim).min(buffers.obs_flat.len());
        let obs_flat = buffers.obs_flat[..obs_end].to_vec();
        let act_flat = buffers.act_flat[..capacity.min(buffers.act_flat.len())].to_vec();
        let logp_flat = buffers.logp_flat[..capacity.min(buffers.logp_flat.len())].to_vec();
        let ret_raw = &buffers.returns[..capacity];
        let ret_flat = {
            let mut rrs = self.return_running.lock().unwrap();
            rrs.update_batch(ret_raw);
            compute_normed_advantages(ret_raw, rrs.mean_f32(), rrs.std())
        };
        let val_flat = buffers.values[..capacity].to_vec();

        // Clear for next epoch
        self.metadata.buffer_pointer.store(0, Ordering::Relaxed);
        self.metadata.buffer_path_start_idx.store(0, Ordering::Relaxed);
        buffers.obs_flat.clear();
        buffers.obs_dim = 0;
        buffers.act_flat.clear();
        buffers.logp_flat.clear();
        buffers.rewards.clear();
        buffers.advantages.clear();
        buffers.returns.clear();
        buffers.values.clear();
        buffers.episode_boundaries.clear();

        Some(PPOFlatBatch {
            obs_flat,
            obs_dim,
            act_flat,
            logp_flat,
            adv_norm,
            ret_flat,
            val_flat,
        })
    }

    /// Drain exactly the first `n` complete episodes: compute GAE, extract batch, shift
    /// remainder to front of buffer. Returns None if fewer than `n` episodes are complete.
    pub fn finalize_and_drain_first_n_blocking(&self, values: Vec<f32>, n: usize) -> Option<PPOFlatBatch> {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.episode_boundaries.len() < n {
            return None;
        }
        let gamma = self.metadata.gamma;
        let lam = self.metadata.lam;
        let obs_dim = buffers.obs_dim;
        let cut_step = buffers.episode_boundaries[n - 1].1;

        // Fill values only when external values provided; else use inline stored values
        if !values.is_empty() {
            let fill_len = values.len().min(cut_step).min(buffers.values.len());
            buffers.values[..fill_len].copy_from_slice(&values[..fill_len]);
        }

        // Compute GAE for the first n episodes only
        let boundaries_n: Vec<_> = buffers.episode_boundaries[..n].to_vec();
        for (start, end, is_truncated) in &boundaries_n {
            let bootstrap = if *is_truncated {
                buffers.values.get(end.saturating_sub(1)).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            Self::compute_gae_episode(&mut buffers, gamma, lam, *start, *end, bootstrap);
        }

        // Normalize advantages across these n episodes only
        let adv_raw = &buffers.advantages[..cut_step];
        let (adv_mean, adv_std) = scalar_stats(adv_raw);
        let adv_norm = compute_normed_advantages(adv_raw, adv_mean, adv_std.max(1e-8));

        // Extract batch for 0..cut_step
        let obs_end = cut_step * obs_dim;
        let obs_flat  = buffers.obs_flat[..obs_end].to_vec();
        let act_flat  = buffers.act_flat[..cut_step].to_vec();
        let logp_flat = buffers.logp_flat[..cut_step].to_vec();
        let ret_raw = &buffers.returns[..cut_step];
        let ret_flat = {
            let mut rrs = self.return_running.lock().unwrap();
            rrs.update_batch(ret_raw);
            compute_normed_advantages(ret_raw, rrs.mean_f32(), rrs.std())
        };
        let val_flat  = buffers.values[..cut_step].to_vec();

        // Shift remaining data to front of buffer
        let total_steps = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        let remaining = total_steps - cut_step;

        let obs_total = obs_end + remaining * obs_dim;
        buffers.obs_flat.copy_within(obs_end..obs_total, 0);
        buffers.obs_flat.truncate(remaining * obs_dim);

        buffers.act_flat.copy_within(cut_step..total_steps, 0);
        buffers.act_flat.truncate(remaining);

        buffers.logp_flat.copy_within(cut_step..total_steps, 0);
        buffers.logp_flat.truncate(remaining);

        buffers.rewards.copy_within(cut_step..total_steps, 0);
        buffers.rewards.truncate(remaining);

        buffers.advantages.copy_within(cut_step..total_steps, 0);
        buffers.advantages.truncate(remaining);

        buffers.returns.copy_within(cut_step..total_steps, 0);
        buffers.returns.truncate(remaining);

        buffers.values.copy_within(cut_step..total_steps, 0);
        buffers.values.truncate(remaining);

        // Remove first n episode boundaries and re-offset the rest
        let remaining_boundaries: Vec<_> = buffers.episode_boundaries[n..]
            .iter()
            .map(|&(s, e, trunc)| (s - cut_step, e - cut_step, trunc))
            .collect();
        buffers.episode_boundaries = remaining_boundaries;

        // Update atomics
        self.metadata.buffer_pointer.store(remaining, Ordering::Relaxed);
        let old_path_start = self.metadata.buffer_path_start_idx.load(Ordering::Relaxed);
        self.metadata.buffer_path_start_idx.store(
            old_path_start.saturating_sub(cut_step),
            Ordering::Relaxed,
        );

        Some(PPOFlatBatch {
            obs_flat,
            obs_dim,
            act_flat,
            logp_flat,
            adv_norm,
            ret_flat,
            val_flat,
        })
    }

    /// Number of complete episodes currently in the buffer.
    pub fn get_episode_count(&self) -> usize {
        self.buffers.lock().unwrap().episode_boundaries.len()
    }

    /// Total steps across all complete episodes (step index of last boundary end).
    pub fn get_complete_step_count(&self) -> usize {
        let buffers = self.buffers.lock().unwrap();
        buffers.episode_boundaries.last().map(|&(_, end, _)| end).unwrap_or(0)
    }

    /// Minimum number of complete episodes needed so their total steps >= min_steps.
    /// Returns 0 if no episodes exist or buffer hasn't accumulated min_steps yet.
    pub fn episodes_needed_for_steps(&self, min_steps: usize) -> usize {
        let buffers = self.buffers.lock().unwrap();
        for (i, &(_, end, _)) in buffers.episode_boundaries.iter().enumerate() {
            if end >= min_steps {
                return i + 1;
            }
        }
        0  // not enough complete steps yet
    }
}

#[async_trait]
impl GenericReplayBuffer for PPOReplayBuffer {
    async fn insert_trajectory(
        &self,
        trajectory: RelayRLTrajectory,
    ) -> Result<Box<dyn Any>, ReplayBufferError> {
        let mut buffers = self.buffers.lock().unwrap();
        let mut episode_return = 0.0f32;
        let mut episode_length = 0i32;

        for action in &trajectory.actions {
            episode_length += 1;
            let reward = action.get_rew();
            episode_return += reward;

            // Obs: bytemuck cast bytes → f32, extend flat buffer
            if let Some(obs_td) = action.get_obs() {
                let floats: &[f32] = bytemuck::cast_slice(&obs_td.data);
                buffers.obs_flat.extend_from_slice(floats);
                if buffers.obs_dim == 0 {
                    buffers.obs_dim = floats.len();
                }
            }

            // Act: extract as i64 (stored as i64 bytes or f32 bytes depending on backend)
            let act_i64 = if let Some(act_td) = action.get_act() {
                if act_td.data.len() >= 8 {
                    bytemuck::cast_slice::<u8, i64>(&act_td.data[..8])
                        .first()
                        .copied()
                        .unwrap_or(0)
                } else if act_td.data.len() >= 4 {
                    bytemuck::cast_slice::<u8, f32>(&act_td.data[..4])
                        .first()
                        .copied()
                        .unwrap_or(0.0) as i64
                } else {
                    0
                }
            } else {
                0
            };
            buffers.act_flat.push(act_i64);

            // LogP: extract f32 scalar from auxiliary data map
            let logp = if let Some(map) = action.get_data() {
                if let Some(RelayRLData::Tensor(logp_td)) = map.get("logp_a") {
                    bytemuck::cast_slice::<u8, f32>(&logp_td.data)
                        .first()
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            buffers.logp_flat.push(logp);

            // Value: extract f32 scalar pre-computed during rollout (0.0 if not present)
            let value = if let Some(map) = action.get_data() {
                if let Some(RelayRLData::Tensor(val_td)) = map.get("value") {
                    bytemuck::cast_slice::<u8, f32>(&val_td.data)
                        .first()
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            buffers.rewards.push(reward);
            buffers.advantages.push(0.0);
            buffers.returns.push(0.0);
            buffers.values.push(value);

            let next = self.metadata.buffer_pointer.load(Ordering::Relaxed) + 1;
            self.metadata.buffer_pointer.store(next, Ordering::Relaxed);

            if action.get_done() {
                let start = self.metadata.buffer_path_start_idx.load(Ordering::Relaxed);
                let end = self.metadata.buffer_pointer.load(Ordering::Relaxed);
                buffers.episode_boundaries.push((start, end, trajectory.is_truncated));
                self.metadata.buffer_path_start_idx.store(end, Ordering::Relaxed);
            }
        }

        Ok(Box::new((episode_return, episode_length)))
    }

    /// Not used in the optimized training path — callers use finalize_and_drain_blocking.
    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError> {
        Err(ReplayBufferError::BufferSamplingError(
            "PPOReplayBuffer: use finalize_and_drain_blocking instead of sample_buffer".to_string(),
        ))
    }
}
