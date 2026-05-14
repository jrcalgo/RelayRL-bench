use crate::algorithms::{compute_normed_advantages, discounted_cumsum, scalar_stats};
use crate::templates::base_replay_buffer::{
    Batch, BatchKey, BufferSample, BufferTensors, GenericReplayBuffer, ReplayBufferError,
    SampleScalars,
};
use async_trait::async_trait;
use relayrl_types::prelude::action::RelayRLData;
use relayrl_types::prelude::tensor::relayrl::TensorData;
use relayrl_types::prelude::trajectory::RelayRLTrajectory;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Mutex;

struct Buffers {
    observations: BufferTensors,
    actions: BufferTensors,
    masks: BufferTensors,
    rewards: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    logprobs: BufferTensors,
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

pub struct PPOReplayBuffer {
    buffers: Arc<Mutex<Buffers>>,
    metadata: Arc<BufferMetadata>,
}

impl Default for PPOReplayBuffer {
    fn default() -> Self {
        Self::new(1_000_000, 0.99, 0.97)
    }
}

impl PPOReplayBuffer {
    pub fn new(buffer_size: usize, gamma: f32, lam: f32) -> Self {
        let buffers = Buffers {
            observations: Vec::with_capacity(buffer_size),
            actions: Vec::with_capacity(buffer_size),
            masks: Vec::with_capacity(buffer_size),
            rewards: Vec::with_capacity(buffer_size),
            advantages: Vec::with_capacity(buffer_size),
            returns: Vec::with_capacity(buffer_size),
            logprobs: Vec::with_capacity(buffer_size),
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
        }
    }

    fn tensor_scalar_f32(data: &TensorData) -> f32 {
        let values: &[f32] = bytemuck::cast_slice(&data.data);
        values.first().copied().unwrap_or(0.0)
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
    /// Called from train_model (sync context) before sample_buffer.
    pub fn get_obs_flat_for_gae_blocking(&self) -> (Vec<f32>, usize) {
        let buffers_arc = Arc::clone(&self.buffers);
        let ptr = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        std::thread::scope(|s| {
            s.spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move {
                        let buffers = buffers_arc.lock().await;
                        let obs_dim = buffers
                            .observations
                            .iter()
                            .find_map(|o| o.as_ref())
                            .map(|td| td.shape[0])
                            .unwrap_or(1);
                        let flat: Vec<f32> = buffers.observations[..ptr]
                            .iter()
                            .filter_map(|opt| opt.as_ref())
                            .flat_map(|td| {
                                bytemuck::cast_slice::<u8, f32>(&td.data).iter().copied()
                            })
                            .collect();
                        (flat, obs_dim)
                    })
            })
            .join()
            .unwrap_or_else(|_| (Vec::new(), 1))
        })
    }

    /// Fill values buffer and compute GAE for all recorded episode boundaries.
    /// Must be called after get_obs_flat_for_gae_blocking + value inference, before sample_buffer.
    pub fn finalize_gae_blocking(&self, values: Vec<f32>) {
        let buffers_arc = Arc::clone(&self.buffers);
        let gamma = self.metadata.gamma;
        let lam = self.metadata.lam;
        std::thread::scope(|s| {
            s.spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move {
                        let mut buffers = buffers_arc.lock().await;

                        // Pre-fill values with batch-inferred results
                        let fill_len = values.len().min(buffers.values.len());
                        buffers.values[..fill_len].copy_from_slice(&values[..fill_len]);

                        // Compute GAE for each completed episode
                        let boundaries: Vec<_> = buffers.episode_boundaries.clone();
                        for (start, end, is_truncated) in boundaries {
                            let bootstrap = if is_truncated {
                                values.get(end.saturating_sub(1)).copied().unwrap_or(0.0)
                            } else {
                                0.0
                            };
                            Self::compute_gae_episode(&mut buffers, gamma, lam, start, end, bootstrap);
                        }
                    })
            })
            .join()
            .ok();
        });
    }
}

#[async_trait]
impl GenericReplayBuffer for PPOReplayBuffer {
    async fn insert_trajectory(
        &self,
        trajectory: RelayRLTrajectory,
    ) -> Result<Box<dyn Any>, ReplayBufferError> {
        let mut buffers = self.buffers.lock().await;
        let mut episode_return = 0.0f32;
        let mut episode_length = 0i32;

        for action in &trajectory.actions {
            episode_length += 1;
            let reward = action.get_rew();
            episode_return += reward;

            buffers.observations.push(action.get_obs().cloned());
            buffers.actions.push(action.get_act().cloned());
            buffers.masks.push(action.get_mask().cloned());
            buffers.logprobs.push(None);
            buffers.rewards.push(reward);
            buffers.advantages.push(0.0);
            buffers.returns.push(0.0);
            // Placeholder value — overwritten by finalize_gae_blocking at epoch end
            buffers.values.push(0.0);

            // Extract logp_a from auxiliary data (val removed — deferred to training time)
            if let Some(map) = action.get_data() {
                if let Some(RelayRLData::Tensor(logp)) = map.get("logp_a") {
                    if let Some(slot) = buffers.logprobs.last_mut() {
                        *slot = Some(logp.clone());
                    }
                }
            }

            let next = self.metadata.buffer_pointer.load(Ordering::Relaxed) + 1;
            self.metadata.buffer_pointer.store(next, Ordering::Relaxed);

            if action.get_done() {
                let start = self.metadata.buffer_path_start_idx.load(Ordering::Relaxed);
                let end = self.metadata.buffer_pointer.load(Ordering::Relaxed);
                // Record episode boundary for deferred GAE; is_truncated from trajectory flag
                buffers.episode_boundaries.push((start, end, trajectory.is_truncated));
                self.metadata.buffer_path_start_idx.store(end, Ordering::Relaxed);
            }
        }

        Ok(Box::new((episode_return, episode_length)))
    }

    /// Returns normalized advantages and the full batch for training.
    /// After sampling, buffer is cleared (epoch-level buffer, not replay).
    async fn sample_buffer(&self) -> Result<Batch, ReplayBufferError> {
        let mut buffers = self.buffers.lock().await;
        let capacity = self.metadata.buffer_pointer.load(Ordering::Relaxed);
        if capacity == 0 {
            return Err(ReplayBufferError::BufferSamplingError(
                "PPO replay buffer is empty".to_string(),
            ));
        }

        // Normalize advantages across the full epoch batch
        let adv_raw = &buffers.advantages[..capacity];
        let (adv_mean, adv_std) = scalar_stats(adv_raw);
        let adv_norm = compute_normed_advantages(adv_raw, adv_mean, adv_std.max(1e-8));

        let obs: Vec<TensorData> = buffers.observations[..capacity]
            .iter()
            .filter_map(|x| x.clone())
            .collect();
        let act: Vec<TensorData> = buffers.actions[..capacity]
            .iter()
            .filter_map(|x| x.clone())
            .collect();
        let mask: Vec<TensorData> = buffers.masks[..capacity]
            .iter()
            .filter_map(|x| x.clone())
            .collect();
        let logp: Vec<TensorData> = buffers.logprobs[..capacity]
            .iter()
            .filter_map(|x| x.clone())
            .collect();
        let ret: Vec<f32> = buffers.returns[..capacity].to_vec();
        let vals: Vec<f32> = buffers.values[..capacity].to_vec();

        // Reset buffer for next epoch
        self.metadata.buffer_pointer.store(0, Ordering::Relaxed);
        self.metadata
            .buffer_path_start_idx
            .store(0, Ordering::Relaxed);
        buffers.observations.clear();
        buffers.actions.clear();
        buffers.masks.clear();
        buffers.rewards.clear();
        buffers.advantages.clear();
        buffers.returns.clear();
        buffers.logprobs.clear();
        buffers.values.clear();
        buffers.episode_boundaries.clear();

        let mut batch: HashMap<BatchKey, BufferSample> = HashMap::new();
        batch.insert(BatchKey::Obs, BufferSample::Tensors(obs.into_boxed_slice()));
        batch.insert(BatchKey::Act, BufferSample::Tensors(act.into_boxed_slice()));
        batch.insert(
            BatchKey::Mask,
            BufferSample::Tensors(mask.into_boxed_slice()),
        );
        batch.insert(
            BatchKey::Custom("Adv".to_string()),
            BufferSample::Scalars(SampleScalars::F32(adv_norm.into_boxed_slice())),
        );
        batch.insert(
            BatchKey::Custom("Ret".to_string()),
            BufferSample::Scalars(SampleScalars::F32(ret.into_boxed_slice())),
        );
        batch.insert(
            BatchKey::Custom("LogP".to_string()),
            BufferSample::Tensors(logp.into_boxed_slice()),
        );
        batch.insert(
            BatchKey::Custom("VVals".to_string()),
            BufferSample::Scalars(SampleScalars::F32(vals.into_boxed_slice())),
        );

        Ok(batch)
    }
}
