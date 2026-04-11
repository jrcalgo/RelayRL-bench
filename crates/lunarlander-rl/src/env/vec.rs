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

use std::marker::PhantomData;

use rayon::prelude::*;

use relayrl_env_trait::environment_traits::EnvironmentError;
use relayrl_types::prelude::tensor::burn::backend::Backend;

use super::LunarLanderEnv;

// ─────────────────────────────────────────────────────────────────────────────

/// LunarLander observation dimension (matches PhysicsState::compute_obs).
pub const OBS_DIM: usize = 8;

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

    /// Return the flat `[num_envs × OBS_DIM]` observation array.
    ///
    /// Because observations are maintained in-place in a single contiguous
    /// buffer, this is a single Vec clone — no per-env locking or scatter-gather.
    pub fn get_stacked_obs(&self) -> Vec<f32> {
        self.observations.clone()
    }
}
