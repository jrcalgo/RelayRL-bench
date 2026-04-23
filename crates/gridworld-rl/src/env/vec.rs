//! SyncVectorEnv — Structure-of-Arrays (SOA) vectorised GridWorld.
//!
//! Instead of N independent `GridWorldEnv` instances (each with its own heap
//! allocations), all per-env state lives in a small set of contiguous flat
//! arrays:
//!
//!   positions    : Vec<(isize, isize)>  len = num_envs
//!   init_pos     : Vec<(isize, isize)>  len = num_envs
//!   done_flags   : Vec<bool>            len = num_envs
//!   step_counts  : Vec<usize>           len = num_envs
//!   last_rewards : Vec<f32>             len = num_envs
//!   observations : Vec<f32>             len = num_envs × obs_dim  (row-major)
//!
//! A single `static_obs_template` pre-bakes walls (1.0) and the goal cell (2.0)
//! once; each step only writes the actor's new position (4.0) on top of a copy
//! of the template, so `get_stacked_obs` is a single contiguous Vec clone.
//!
//! `step_all` fans work out with rayon by zipping mutable slices of each SOA
//! column.  There are no Mutex acquisitions during stepping; shared read-only
//! data (walls, template, config) is referenced across rayon threads safely
//! because it is never mutated once constructed.

use std::marker::PhantomData;

use rayon::prelude::*;

use relayrl_env_trait::EnvironmentError;
use relayrl_types::prelude::tensor::burn::backend::Backend;

use super::RewardConfig;

// ─────────────────────────────────────────────────────────────────────────────

/// Vectorised GridWorld using a flat SOA layout + rayon parallel stepping.
///
/// All N sub-environments share the same grid configuration (size, walls, goal)
/// and are stepped in parallel on every `step_all` call.  Episodes end
/// individually; each sub-env resets itself inline within the same rayon task.
pub struct SyncVectorEnv<B: Backend>
where
    B::Device: Clone,
{
    // ── Shared, immutable config ──────────────────────────────────────────────
    walls:          Vec<(isize, isize)>,
    end:            (isize, isize),
    pub grid_size:  usize,
    pub obs_dim:    usize,   // grid_size × grid_size
    pub num_envs:   usize,
    max_steps:      usize,
    reward_cfg:     RewardConfig,

    // Pre-baked grid overlay: walls = 1.0, goal = 2.0, actor slot = 0.0.
    // Copied into each env's obs slice on every step (overwrite then set actor).
    static_template: Vec<f32>,   // len = obs_dim

    // ── SOA mutable state (each Vec has length num_envs) ─────────────────────
    positions:   Vec<(isize, isize)>,
    init_pos:    Vec<(isize, isize)>,
    done_flags:  Vec<bool>,
    step_counts: Vec<usize>,
    last_rewards: Vec<f32>,

    // ── Flat observation buffer ───────────────────────────────────────────────
    /// Row-major: env `i`'s observation is `observations[i*obs_dim..(i+1)*obs_dim]`.
    observations: Vec<f32>,

    _phantom: PhantomData<B>,
}

// ─────────────────────────────────────────────────────────────────────────────

fn action_delta(act: u8) -> (isize, isize) {
    match act {
        0 => (-1,  0), // Up
        1 => ( 1,  0), // Down
        2 => ( 0, -1), // Left
        3 => ( 0,  1), // Right
        _ => ( 0,  0), // no-op
    }
}

fn build_template(
    walls: &[(isize, isize)],
    end:   (isize, isize),
    gs:    usize,
) -> Vec<f32> {
    let mut t = vec![0.0f32; gs * gs];
    for &(wr, wc) in walls {
        t[wr as usize * gs + wc as usize] = 1.0;
    }
    let (er, ec) = end;
    t[er as usize * gs + ec as usize] = 2.0;
    t
}

// ─────────────────────────────────────────────────────────────────────────────

impl<B: Backend> SyncVectorEnv<B>
where
    B::Device: Clone,
{
    /// Construct `num_envs` parallel single-actor GridWorld environments.
    ///
    /// All sub-envs use the same `grid_size × grid_size` layout, default walls
    /// (10×10 only), goal at `(gs-1, gs-1)`, and the single actor at `(0, 0)`.
    pub fn new(
        num_envs:  usize,
        grid_size: usize,
        _device:   B::Device,   // kept for API compatibility; not used in SOA
    ) -> Result<Self, EnvironmentError> {
        let gs  = grid_size;
        let end = (gs as isize - 1, gs as isize - 1);

        let walls: Vec<(isize, isize)> = if gs == 10 {
            vec![
                (2,1),(2,2),(2,3),(2,4),
                (3,4),(4,4),(5,4),(6,4),(7,4),
                (2,6),(2,7),(2,8),
            ]
        } else {
            vec![]
        };

        let reward_cfg     = RewardConfig::default();
        let static_template = build_template(&walls, end, gs);
        let obs_dim         = gs * gs;

        let init = (0isize, 0isize);

        // Build initial observations: template + actor at (0,0) = 4.0
        let mut observations = static_template.repeat(num_envs);
        for i in 0..num_envs {
            let off = i * obs_dim;
            observations[off + init.0 as usize * gs + init.1 as usize] = 4.0;
        }

        Ok(Self {
            walls,
            end,
            grid_size: gs,
            obs_dim,
            num_envs,
            max_steps: 200,
            reward_cfg,
            static_template,
            positions:    vec![init; num_envs],
            init_pos:     vec![init; num_envs],
            done_flags:   vec![false; num_envs],
            step_counts:  vec![0usize; num_envs],
            last_rewards: vec![0.0f32; num_envs],
            observations,
            _phantom: PhantomData,
        })
    }

    /// Reset all sub-environments in place.
    pub fn reset_all(&mut self) {
        let gs       = self.grid_size;
        let obs_dim  = self.obs_dim;
        let template = &self.static_template;

        for i in 0..self.num_envs {
            let init = self.init_pos[i];
            self.positions[i]    = init;
            self.done_flags[i]   = false;
            self.step_counts[i]  = 0;
            self.last_rewards[i] = 0.0;

            let obs = &mut self.observations[i * obs_dim..(i + 1) * obs_dim];
            obs.copy_from_slice(template);
            obs[init.0 as usize * gs + init.1 as usize] = 4.0;
        }
    }

    /// Step all sub-environments in parallel using rayon.
    ///
    /// `actions` must have length `num_envs`.  Each sub-env that finishes its
    /// episode (done or max-steps) resets itself inline.
    ///
    /// Returns `Vec<(reward, episode_done)>` of length `num_envs`.
    pub fn step_all(&mut self, actions: &[u8]) -> Vec<(f32, bool)> {
        assert_eq!(actions.len(), self.num_envs, "actions.len() must equal num_envs");

        let gs        = self.grid_size;
        let obs_dim   = self.obs_dim;
        let max_steps = self.max_steps;
        let end       = self.end;

        // Extract shared read-only refs before the mutable zips.
        // Rust allows disjoint field borrows: these share-borrow fields that are
        // NOT included in the mutable zip below.
        let walls     = self.walls.as_slice();
        let reward_cfg = &self.reward_cfg;
        let template  = self.static_template.as_slice();
        let init_pos  = self.init_pos.as_slice();

        self.positions.par_iter_mut()
            .zip(self.done_flags.par_iter_mut())
            .zip(self.step_counts.par_iter_mut())
            .zip(self.last_rewards.par_iter_mut())
            .zip(self.observations.par_chunks_mut(obs_dim))
            .zip(init_pos.par_iter())
            .zip(actions.par_iter())
            .map(|((((((pos, done), steps), last_r), obs_chunk), init), &act)| {
                // ── Move resolution ──────────────────────────────────────────
                let (cr, cc) = *pos;
                let (dr, dc) = action_delta(act);
                let np       = (cr + dr, cc + dc);

                let in_bounds = np.0 >= 0 && np.0 < gs as isize
                             && np.1 >= 0 && np.1 < gs as isize;
                let is_wall   = walls.contains(&np);
                let is_end    = np == end;

                let (new_pos, reward, end_reached) = if !in_bounds || is_wall {
                    (*pos, reward_cfg.collision_reward, false)
                } else if is_end {
                    (np,   reward_cfg.end_state_reward, true)
                } else {
                    (np,   reward_cfg.step_reward,      false)
                };

                *pos    = new_pos;
                *last_r = reward;
                *steps += 1;
                if end_reached { *done = true; }

                let episode_done = *done || *steps >= max_steps;

                // ── Update observation for current position ───────────────────
                obs_chunk.copy_from_slice(template);
                obs_chunk[pos.0 as usize * gs + pos.1 as usize] = 4.0;

                // ── Inline reset if episode finished ──────────────────────────
                if episode_done {
                    *pos   = *init;
                    *done  = false;
                    *steps = 0;
                    obs_chunk.copy_from_slice(template);
                    obs_chunk[pos.0 as usize * gs + pos.1 as usize] = 4.0;
                }

                (reward, episode_done)
            })
            .collect()
    }

    /// Return the flat `[num_envs × obs_dim]` observation array.
    ///
    /// Because observations are maintained in-place in a single contiguous
    /// buffer, this is a single Vec clone — no locking, no scatter-gather.
    pub fn get_stacked_obs(&self) -> Vec<f32> {
        self.observations.clone()
    }
}
