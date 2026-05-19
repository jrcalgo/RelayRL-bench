//! SyncLunarVectorEnv — vectorised LunarLander using a flat observation buffer.
//!
//! Each sub-environment is an independent `LunarLanderEnv<B>` instance.  The
//! "SOA" aspect applied here is a single contiguous flat observation buffer:
//!
//!   observations : Vec<f32>   len = num_envs × OBS_DIM  (row-major)
//!
//! All N sub-environments own their complete physics state (the underlying
//! `PhysicsState` is private, so we cannot decompose it further), but
//! observations are extracted into the shared flat buffer after each step so
//! that `get_stacked_obs` is a single Vec clone — no per-env heap reads.
//!
//! `step_all` fans work out with rayon by zipping mutable slices of the env
//! Vec and the observation buffer.  Each thread gets exclusive `&mut` access to
//! one env and one 8-float obs slice — no Mutex acquisitions, no contention.
//!
//! `SyncLunarVectorEnvFramework` wraps `SyncLunarVectorEnv` with the
//! `VectorEnvironment` trait so it can be passed to `agent.set_env()`.

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

use rayon::prelude::*;

use burn_ndarray::NdArray;
use burn_tensor::backend::Backend;

use relayrl_env_trait::{
    DynVectorEnv, EnvDType, EnvNdArrayDType, EnvironmentError, EnvironmentHandle, EnvironmentKind,
    EnvironmentUuid, Uuid, VectorEnvReset, VectorEnvironment,
};

use super::LunarLanderEnv;

// ─────────────────────────────────────────────────────────────────────────────

/// LunarLander observation dimension (matches PhysicsState::compute_obs).
pub const OBS_DIM: usize = 8;
/// LunarLander discrete action count.
pub const ACT_DIM: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────

/// Vectorised LunarLander using rayon parallel stepping + a flat observation buffer.
///
/// All N sub-environments are independent instances stepped in parallel on every
/// `step_all` call.  Episodes end individually; each sub-env resets itself inline
/// within the same rayon task.
pub struct SyncLunarVectorEnv<B: Backend>
where
    B::Device: Clone + Send,
{
    /// N independent LunarLander environments.
    envs:          Vec<LunarLanderEnv<B>>,
    pub obs_dim:   usize,   // = OBS_DIM = 8
    pub num_envs:  usize,

    // ── Flat observation buffer ───────────────────────────────────────────────
    /// Row-major: env `i`'s observation is `observations[i*OBS_DIM..(i+1)*OBS_DIM]`.
    observations:  Vec<f32>,

    _phantom: PhantomData<B>,
}

// ─────────────────────────────────────────────────────────────────────────────

impl<B: Backend> SyncLunarVectorEnv<B>
where
    B::Device: Clone + Send,
{
    /// Construct `num_envs` parallel LunarLander environments.
    ///
    /// Each sub-env is initialised with the same `max_steps` limit but gets an
    /// independent random seed via `LunarLanderEnv::new` (seed 12345) followed
    /// by a `reset()` call to generate an environment-unique trajectory.
    pub fn new(
        num_envs:  usize,
        max_steps: usize,
        device:    B::Device,
    ) -> Result<Self, EnvironmentError> {
        let mut envs: Vec<LunarLanderEnv<B>> = (0..num_envs)
            .map(|_| LunarLanderEnv::new(max_steps, device.clone()))
            .collect();

        // Build initial observations: reset each env and record its first obs.
        let mut observations = vec![0.0f32; num_envs * OBS_DIM];
        for (i, env) in envs.iter_mut().enumerate() {
            // Each env already ran one settle step in PhysicsState::build.
            // Explicit reset generates a new random seed, diversifying episodes.
            env.reset();
            let obs = env.get_observation(0);
            observations[i * OBS_DIM..(i + 1) * OBS_DIM].copy_from_slice(&obs);
        }

        Ok(Self {
            envs,
            obs_dim:  OBS_DIM,
            num_envs,
            observations,
            _phantom: PhantomData,
        })
    }

    /// Reset all sub-environments in place (sequential).
    pub fn reset_all(&mut self) {
        for (i, env) in self.envs.iter_mut().enumerate() {
            env.reset();
            let obs = env.get_observation(0);
            self.observations[i * OBS_DIM..(i + 1) * OBS_DIM].copy_from_slice(&obs);
        }
    }

    /// Step all sub-environments in parallel using rayon.
    ///
    /// `actions` must have length `num_envs`.  Each sub-env that finishes its
    /// episode (physics done or max-steps) resets itself inline within the same
    /// rayon task.  The returned observation buffer reflects post-reset state.
    ///
    /// Returns `Vec<(reward, episode_done)>` of length `num_envs`.
    pub fn step_all(&mut self, actions: &[u8]) -> Vec<(f32, bool)> {
        assert_eq!(actions.len(), self.num_envs, "actions.len() must equal num_envs");

        // Disjoint field borrows: envs and observations are separate Vec fields.
        let envs = self.envs.as_mut_slice();
        let obs  = self.observations.as_mut_slice();

        envs.par_iter_mut()
            .zip(obs.par_chunks_mut(OBS_DIM))
            .zip(actions.par_iter())
            .map(|((env, obs_chunk), &act)| {
                // ── Step the environment ─────────────────────────────────────
                let (reward, done) = env.step(0, act).unwrap_or((0.0, true));

                // ── Update observation chunk with post-step state ─────────────
                let new_obs = env.get_observation(0);
                obs_chunk.copy_from_slice(&new_obs);

                // ── Inline reset if episode finished ──────────────────────────
                if done {
                    env.reset();
                    let reset_obs = env.get_observation(0);
                    obs_chunk.copy_from_slice(&reset_obs);
                }

                (reward, done)
            })
            .collect()
    }

    /// Return the flat `[num_envs × OBS_DIM]` observation array (cloned).
    pub fn get_stacked_obs(&self) -> Vec<f32> {
        self.observations.clone()
    }

    /// Borrow the flat observation buffer without copying.
    pub fn get_stacked_obs_ref(&self) -> &[f32] {
        &self.observations
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SyncLunarVectorEnvFramework — VectorEnvironment adapter for set_env / run_env
// ─────────────────────────────────────────────────────────────────────────────

/// Inner mutable state, protected by a single Mutex.
struct VecFrameworkInner {
    env: SyncLunarVectorEnv<NdArray>,
    /// UUID → env index in the inner Vec.
    uuid_to_idx: HashMap<EnvironmentUuid, usize>,
    /// env index → UUID (for building step results in order).
    idx_to_uuid: Vec<EnvironmentUuid>,
}

/// Framework-compatible wrapper around `SyncLunarVectorEnv`.
///
/// Implements [`VectorEnvironment`] so it can be passed to `agent.set_env()`.
/// `step_all` uses rayon to step all sub-envs in parallel; auto-reset is
/// handled inline within each rayon task.  The framework's double-reset path
/// is suppressed by always returning `terminated = false`.
pub struct SyncLunarVectorEnvFramework {
    inner: Mutex<VecFrameworkInner>,
    num_envs: usize,
    max_steps: usize,
}

impl SyncLunarVectorEnvFramework {
    pub fn new(num_envs: usize, max_steps: usize) -> Result<Self, EnvironmentError> {
        let env = SyncLunarVectorEnv::<NdArray>::new(num_envs, max_steps, Default::default())
            .map_err(|e| EnvironmentError::EnvironmentError(e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(VecFrameworkInner {
                env,
                uuid_to_idx: HashMap::new(),
                idx_to_uuid: Vec::new(),
            }),
            num_envs,
            max_steps,
        })
    }
}

impl relayrl_env_trait::Environment for SyncLunarVectorEnvFramework {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        Ok(())
    }
    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(Vec::<f32>::new()))
    }
    fn observation_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }
    fn action_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }
    fn observation_dim(&self) -> usize { OBS_DIM }
    fn action_dim(&self) -> usize { ACT_DIM }
    fn flat_observation_bytes(&self) -> Vec<u8> {
        let obs = self.inner.lock().unwrap().env.get_stacked_obs();
        bytemuck::cast_slice::<f32, u8>(&obs).to_vec()
    }
    fn action_is_discrete(&self) -> bool { true }
    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Vector
    }
    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Vector(self as Box<DynVectorEnv>)
    }
}

impl VectorEnvironment for SyncLunarVectorEnvFramework {
    /// Called once by the framework with `count = ENV_COUNT`.
    /// Generates stable UUIDs for each sub-env and stores the index mapping.
    fn init_num_envs(
        &self,
        num_envs: usize,
    ) -> Result<Vec<EnvironmentUuid>, EnvironmentError> {
        let uuids: Vec<EnvironmentUuid> = (0..num_envs).map(|_| Uuid::new_v4()).collect();
        let mut inner = self.inner.lock().unwrap();
        inner.idx_to_uuid = uuids.clone();
        inner.uuid_to_idx = uuids.iter().copied().enumerate().map(|(i, u)| (u, i)).collect();
        Ok(uuids)
    }

    /// Returns current observations for the requested env IDs (used by the
    /// framework's reset_all at startup and reset_where for done envs).
    /// Since step_all handles all resets inline, this just echoes the current
    /// observation buffer without actually re-initialising any physics.
    fn reset(
        &self,
        env_ids: &[EnvironmentUuid],
    ) -> Result<Vec<VectorEnvReset>, EnvironmentError> {
        let inner = self.inner.lock().unwrap();
        let stacked_obs = inner.env.get_stacked_obs();
        let results = env_ids
            .iter()
            .filter_map(|uuid| {
                let &idx = inner.uuid_to_idx.get(uuid)?;
                let start = idx * OBS_DIM;
                let obs_bytes = bytemuck::cast_slice::<f32, u8>(
                    &stacked_obs[start..start + OBS_DIM]
                ).to_vec();
                Some(VectorEnvReset {
                    env_id: *uuid,
                    observation: obs_bytes,
                    info: None,
                })
            })
            .collect();
        Ok(results)
    }

    fn n_envs(&self) -> usize { self.num_envs }

    fn step_bytes(&self, actions: &[u8]) -> Option<(Vec<u8>, Vec<f32>, Vec<bool>)> {
        let mut inner = self.inner.lock().unwrap();
        let step_results = inner.env.step_all(actions);
        let new_obs_f32 = inner.env.get_stacked_obs();
        let mut rewards = Vec::with_capacity(step_results.len());
        let mut dones   = Vec::with_capacity(step_results.len());
        for (r, d) in step_results {
            rewards.push(r);
            dones.push(d);
        }
        Some((bytemuck::cast_slice::<f32, u8>(&new_obs_f32).to_vec(), rewards, dones))
    }
}
