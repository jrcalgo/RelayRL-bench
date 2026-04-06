use burn_tensor::backend::Backend;
use gridworld_rl::env::{GridWorldEnv, RewardConfig};
use relayrl_env_trait::environment_traits::EnvironmentError;

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

    // Use the standard wall layout only for the default 10×10 grid.
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
