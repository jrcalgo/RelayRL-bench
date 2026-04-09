use burn_tensor::backend::Backend;
use gridworld_rl::env::{GridWorldEnv, RewardConfig};
use lunarlander_rl::env::LunarLanderEnv;
use relayrl_env_trait::environment_traits::EnvironmentError;

/// Minimal environment interface used by the training loop.
/// Both GridWorldEnv and LunarLanderEnv satisfy this via blanket delegates.
pub trait SimpleEnv {
    fn reset(&self);
    fn step(&self, actor_idx: usize, action: u8) -> Result<(f32, bool), EnvironmentError>;
    fn get_observation(&self, actor_idx: usize) -> Vec<f32>;
    fn get_last_reward(&self, actor_idx: usize) -> f32;
    fn all_done(&self) -> bool;
    fn is_max_steps_reached(&self) -> bool;
    fn actor_count(&self) -> usize;
}

impl<B: Backend> SimpleEnv for GridWorldEnv<B>
where
    B::Device: Clone,
{
    fn reset(&self) { GridWorldEnv::reset(self); }
    fn step(&self, actor_idx: usize, action: u8) -> Result<(f32, bool), EnvironmentError> {
        GridWorldEnv::step(self, actor_idx, action)
    }
    fn get_observation(&self, actor_idx: usize) -> Vec<f32> {
        GridWorldEnv::get_observation(self, actor_idx)
    }
    fn get_last_reward(&self, actor_idx: usize) -> f32 {
        GridWorldEnv::get_last_reward(self, actor_idx)
    }
    fn all_done(&self) -> bool { GridWorldEnv::all_done(self) }
    fn is_max_steps_reached(&self) -> bool { GridWorldEnv::is_max_steps_reached(self) }
    fn actor_count(&self) -> usize { GridWorldEnv::actor_count(self) }
}

impl<B: Backend> SimpleEnv for LunarLanderEnv<B>
where
    B::Device: Clone,
{
    fn reset(&self) { LunarLanderEnv::reset(self); }
    fn step(&self, actor_idx: usize, action: u8) -> Result<(f32, bool), EnvironmentError> {
        LunarLanderEnv::step(self, actor_idx, action)
    }
    fn get_observation(&self, actor_idx: usize) -> Vec<f32> {
        LunarLanderEnv::get_observation(self, actor_idx)
    }
    fn get_last_reward(&self, actor_idx: usize) -> f32 {
        LunarLanderEnv::get_last_reward(self, actor_idx)
    }
    fn all_done(&self) -> bool { LunarLanderEnv::all_done(self) }
    fn is_max_steps_reached(&self) -> bool { LunarLanderEnv::is_max_steps_reached(self) }
    fn actor_count(&self) -> usize { LunarLanderEnv::actor_count(self) }
}

/// Build a `GridWorldEnv` for `actor_count` actors on a `grid_size × grid_size` grid.
///
/// Actors are placed starting at (0,0) and spread left-to-right, top-to-bottom.
/// The goal is always at `(grid_size-1, grid_size-1)`.
/// Default walls are used for the 10×10 grid; no walls are placed for other sizes.
pub fn build_gridworld_env<B>(
    actor_count: usize,
    grid_size: usize,
    max_steps: usize,
    device: B::Device,
) -> Result<GridWorldEnv<B>, EnvironmentError>
where
    B: Backend,
    B::Device: Clone,
{
    let end_position = ((grid_size - 1) as isize, (grid_size - 1) as isize);

    // Spread actors from the top-left corner, wrapping to the next row as needed.
    let actor_positions: Vec<(isize, isize)> = (0..actor_count)
        .map(|i| ((i / grid_size) as isize, (i % grid_size) as isize))
        .collect();

    // Use the standard wall layout only for the default 10×10 grid, filtering
    // out any wall cells that an actor is already occupying.
    let actor_set: std::collections::HashSet<(isize, isize)> =
        actor_positions.iter().copied().collect();
    let wall_positions: Vec<(isize, isize)> = if grid_size == 10 {
        vec![
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (2, 6),
            (2, 7),
            (2, 8),
        ]
        .into_iter()
        .filter(|w| !actor_set.contains(w))
        .collect()
    } else {
        vec![]
    };

    GridWorldEnv::new(
        true,
        grid_size,
        grid_size,
        wall_positions,
        end_position,
        actor_positions,
        Some(RewardConfig::default()),
        Some(max_steps),
        device,
    )
}

/// Build a single-actor `LunarLanderEnv`.
pub fn build_lunarlander_env<B>(
    max_steps: usize,
    device: B::Device,
) -> Result<LunarLanderEnv<B>, EnvironmentError>
where
    B: Backend,
    B::Device: Clone,
{
    Ok(LunarLanderEnv::new(max_steps, device))
}
