//! SyncVectorEnv — Structure-of-Arrays vectorised GridWorld (inlined for beta.2 comparison).
//!
//! Identical logic to crates/gridworld-rl/src/env/vec.rs; only the import path
//! for `Backend` is changed to use `burn_tensor` directly (no relayrl_types shim).

use std::marker::PhantomData;

use rayon::prelude::*;

use burn_tensor::backend::Backend;

// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RewardConfig {
    pub collision_reward: f32,
    pub end_state_reward: f32,
    pub step_reward:      f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            collision_reward: -1.0,
            end_state_reward: 10.0,
            step_reward:      -0.01,
        }
    }
}

pub struct SyncVectorEnv<B: Backend>
where
    B::Device: Clone,
{
    walls:           Vec<(isize, isize)>,
    end:             (isize, isize),
    pub grid_size:   usize,
    pub obs_dim:     usize,
    pub num_envs:    usize,
    max_steps:       usize,
    reward_cfg:      RewardConfig,
    static_template: Vec<f32>,
    positions:       Vec<(isize, isize)>,
    init_pos:        Vec<(isize, isize)>,
    done_flags:      Vec<bool>,
    step_counts:     Vec<usize>,
    last_rewards:    Vec<f32>,
    observations:    Vec<f32>,
    _phantom:        PhantomData<B>,
}

fn action_delta(act: u8) -> (isize, isize) {
    match act {
        0 => (-1,  0),
        1 => ( 1,  0),
        2 => ( 0, -1),
        3 => ( 0,  1),
        _ => ( 0,  0),
    }
}

fn build_template(walls: &[(isize, isize)], end: (isize, isize), gs: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; gs * gs];
    for &(wr, wc) in walls {
        t[wr as usize * gs + wc as usize] = 1.0;
    }
    let (er, ec) = end;
    t[er as usize * gs + ec as usize] = 2.0;
    t
}

impl<B: Backend> SyncVectorEnv<B>
where
    B::Device: Clone,
{
    pub fn new(
        num_envs:  usize,
        grid_size: usize,
        _device:   B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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

        let reward_cfg      = RewardConfig::default();
        let static_template = build_template(&walls, end, gs);
        let obs_dim         = gs * gs;
        let init            = (0isize, 0isize);

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

    pub fn reset_all(&mut self) {
        let gs      = self.grid_size;
        let obs_dim = self.obs_dim;
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

    pub fn step_all(&mut self, actions: &[u8]) -> Vec<(f32, bool)> {
        assert_eq!(actions.len(), self.num_envs);

        let gs        = self.grid_size;
        let obs_dim   = self.obs_dim;
        let max_steps = self.max_steps;
        let end       = self.end;
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

                obs_chunk.copy_from_slice(template);
                obs_chunk[pos.0 as usize * gs + pos.1 as usize] = 4.0;

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

    pub fn get_stacked_obs(&self) -> Vec<f32> {
        self.observations.clone()
    }
}
