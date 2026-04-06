use relayrl_types::prelude::tensor::burn::backend::Backend;
use relayrl_types::prelude::tensor::burn::{Float, Tensor, TensorData};
use relayrl_env_trait::environment_traits::{EnvironmentTrait, TrainingPerformanceReturnFn, EnvironmentError};
use std::any::Any;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub struct RewardConfig {
    pub collision_reward: f32,
    pub end_state_reward: f32,
    pub step_reward: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            collision_reward: -1.0,
            end_state_reward: 10.0,
            step_reward: -0.01,
        }
    }
}

enum Move {
    Up,
    Down,
    Left,
    Right,
}

impl Move {
    fn from_action(action: u8) -> Option<Self> {
        match action {
            0 => Some(Move::Up),
            1 => Some(Move::Down),
            2 => Some(Move::Left),
            3 => Some(Move::Right),
            _ => None,
        }
    }

    fn delta(&self) -> (isize, isize) {
        match self {
            Move::Up    => (-1,  0),
            Move::Down  => ( 1,  0),
            Move::Left  => ( 0, -1),
            Move::Right => ( 0,  1),
        }
    }
}

enum MoveResult {
    Valid,
    AtEnd,
    WallCollision,
    AgentCollision,
    OutOfBounds,
}

#[derive(Clone, Debug)]
pub struct Actor {
    pub id: usize,
    pub initial_position: (isize, isize),
    pub current_position: (isize, isize),
    pub done: bool,
    pub last_reward: f32,
    pub cumulative_reward: f32,
}

impl Actor {
    fn reset(&mut self) {
        self.current_position = self.initial_position;
        self.done = false;
        self.last_reward = 0.0;
        self.cumulative_reward = 0.0;
    }
}

pub struct GridWorldEnv<B: Backend>
where
    B::Device: Clone,
{
    pub training: bool,
    pub length: usize,
    pub width: usize,
    pub wall_positions: Vec<(isize, isize)>,
    pub end_position: (isize, isize),
    pub reward_config: RewardConfig,
    pub max_steps: usize,
    pub device: B::Device,
    actors: RefCell<Vec<Actor>>,
    step_count: RefCell<usize>,
    last_observations: RefCell<Vec<Vec<f32>>>,
    episode_returns: RefCell<Vec<f32>>,
    running: AtomicBool,
}

impl<B: Backend> Default for GridWorldEnv<B>
where
    B::Device: Clone + Default,
{
    fn default() -> Self {
        let wall_positions: Vec<(isize, isize)> = vec![
            (2, 1), (2, 2), (2, 3), (2, 4),
            (3, 4), (4, 4), (5, 4), (6, 4), (7, 4),
            (2, 6), (2, 7), (2, 8),
        ];
        let num_actors = 1;
        let cells = 10 * 10;
        Self {
            training: true,
            length: 10,
            width: 10,
            wall_positions,
            end_position: (9, 9),
            reward_config: RewardConfig::default(),
            max_steps: 200,
            device: B::Device::default(),
            actors: RefCell::new(vec![Actor {
                id: 0,
                initial_position: (0, 0),
                current_position: (0, 0),
                done: false,
                last_reward: 0.0,
                cumulative_reward: 0.0,
            }]),
            step_count: RefCell::new(0),
            last_observations: RefCell::new(vec![vec![0.0f32; cells]; num_actors]),
            episode_returns: RefCell::new(vec![0.0f32; num_actors]),
            running: AtomicBool::new(false),
        }
    }
}

impl<B: Backend> GridWorldEnv<B>
where
    B::Device: Clone,
{
    pub fn new(
        training: bool,
        length: usize,
        width: usize,
        wall_positions: Vec<(isize, isize)>,
        end_position: (isize, isize),
        initial_actor_positions: Vec<(isize, isize)>,
        reward_config: Option<RewardConfig>,
        max_steps: Option<usize>,
        device: B::Device,
    ) -> Result<Self, EnvironmentError> {
        let (er, ec) = end_position;
        if er < 0 || er >= length as isize || ec < 0 || ec >= width as isize {
            return Err(EnvironmentError::EnvironmentError(format!(
                "End position ({},{}) is out of bounds for {}x{} grid",
                er, ec, length, width
            )));
        }

        for &(wr, wc) in &wall_positions {
            if wr < 0 || wr >= length as isize || wc < 0 || wc >= width as isize {
                return Err(EnvironmentError::EnvironmentError(format!(
                    "Wall at ({},{}) is out of bounds for {}x{} grid",
                    wr, wc, length, width
                )));
            }
            if (wr, wc) == end_position {
                return Err(EnvironmentError::EnvironmentError(format!(
                    "Wall at ({},{}) conflicts with end position ({},{})",
                    wr, wc, er, ec
                )));
            }
        }

        for (i, &(ar, ac)) in initial_actor_positions.iter().enumerate() {
            if ar < 0 || ar >= length as isize || ac < 0 || ac >= width as isize {
                return Err(EnvironmentError::EnvironmentError(format!(
                    "Actor {} at ({},{}) is out of bounds for {}x{} grid",
                    i, ar, ac, length, width
                )));
            }
            if (ar, ac) == end_position {
                return Err(EnvironmentError::EnvironmentError(format!(
                    "Actor {} at ({},{}) conflicts with end position ({},{})",
                    i, ar, ac, er, ec
                )));
            }
            for &(wr, wc) in &wall_positions {
                if (ar, ac) == (wr, wc) {
                    return Err(EnvironmentError::EnvironmentError(format!(
                        "Actor {} at ({},{}) conflicts with wall at ({},{})",
                        i, ar, ac, wr, wc
                    )));
                }
            }
            for (j, &(br, bc)) in initial_actor_positions.iter().enumerate() {
                if i != j && (ar, ac) == (br, bc) {
                    return Err(EnvironmentError::EnvironmentError(format!(
                        "Actors {} and {} both start at ({},{})",
                        i, j, ar, ac
                    )));
                }
            }
        }

        let num_actors = initial_actor_positions.len();
        let cells = length * width;
        let actors: Vec<Actor> = initial_actor_positions
            .into_iter()
            .enumerate()
            .map(|(id, pos)| Actor {
                id,
                initial_position: pos,
                current_position: pos,
                done: false,
                last_reward: 0.0,
                cumulative_reward: 0.0,
            })
            .collect();

        Ok(Self {
            training,
            length,
            width,
            wall_positions,
            end_position,
            reward_config: reward_config.unwrap_or_default(),
            max_steps: max_steps.unwrap_or(200),
            device,
            actors: RefCell::new(actors),
            step_count: RefCell::new(0),
            last_observations: RefCell::new(vec![vec![0.0f32; cells]; num_actors]),
            episode_returns: RefCell::new(vec![0.0f32; num_actors]),
            running: AtomicBool::new(false),
        })
    }

    pub fn reset(&self) {
        let num_actors = {
            let mut actors = self.actors.borrow_mut();
            for actor in actors.iter_mut() {
                actor.reset();
            }
            actors.len()
        };
        *self.step_count.borrow_mut() = 0;
        let cells = self.length * self.width;
        *self.last_observations.borrow_mut() = vec![vec![0.0f32; cells]; num_actors];
        *self.episode_returns.borrow_mut() = vec![0.0f32; num_actors];
        self.update_observations();
    }

    pub fn step(&self, actor_idx: usize, action: u8) -> Result<(f32, bool), EnvironmentError> {
        let mv = Move::from_action(action).ok_or_else(|| {
            EnvironmentError::EnvironmentError(format!("Invalid action: {}", action))
        })?;

        let (is_done, last_r) = {
            let actors = self.actors.borrow();
            if actor_idx >= actors.len() {
                return Err(EnvironmentError::EnvironmentError(format!(
                    "Actor index {} out of range ({})",
                    actor_idx,
                    actors.len()
                )));
            }
            (actors[actor_idx].done, actors[actor_idx].last_reward)
        };

        if is_done {
            *self.step_count.borrow_mut() += 1;
            let episode_done = self.all_done() || self.is_max_steps_reached();
            return Ok((last_r, episode_done));
        }

        let (cr, cc) = self.actors.borrow()[actor_idx].current_position;
        let (dr, dc) = mv.delta();
        let new_pos = (cr + dr, cc + dc);
        let move_result = self.validate_move_result(actor_idx, new_pos);

        let reward = {
            let mut actors = self.actors.borrow_mut();
            let r = match &move_result {
                MoveResult::Valid => {
                    actors[actor_idx].current_position = new_pos;
                    self.reward_config.step_reward
                }
                MoveResult::AtEnd => {
                    actors[actor_idx].current_position = new_pos;
                    actors[actor_idx].done = true;
                    self.reward_config.end_state_reward
                }
                MoveResult::WallCollision
                | MoveResult::AgentCollision
                | MoveResult::OutOfBounds => self.reward_config.collision_reward,
            };
            actors[actor_idx].last_reward = r;
            actors[actor_idx].cumulative_reward += r;
            r
        };

        *self.step_count.borrow_mut() += 1;
        self.update_observations();

        let episode_done = self.all_done() || self.is_max_steps_reached();
        Ok((reward, episode_done))
    }

    pub fn get_observation(&self, actor_idx: usize) -> Vec<f32> {
        self.last_observations.borrow()[actor_idx].clone()
    }

    pub fn get_last_reward(&self, actor_idx: usize) -> f32 {
        self.actors.borrow()[actor_idx].last_reward
    }

    pub fn get_episode_return(&self, actor_idx: usize) -> f32 {
        self.actors.borrow()[actor_idx].cumulative_reward
    }

    pub fn all_done(&self) -> bool {
        self.actors.borrow().iter().all(|a| a.done)
    }

    pub fn is_max_steps_reached(&self) -> bool {
        *self.step_count.borrow() >= self.max_steps
    }

    pub fn actor_count(&self) -> usize {
        self.actors.borrow().len()
    }

    /// Returns the current grid position of every actor, in actor-index order.
    #[cfg(feature = "render")]
    pub fn get_actor_positions(&self) -> Vec<(isize, isize)> {
        self.actors.borrow().iter().map(|a| a.current_position).collect()
    }

    fn is_in_bounds(&self, pos: (isize, isize)) -> bool {
        pos.0 >= 0
            && pos.0 < self.length as isize
            && pos.1 >= 0
            && pos.1 < self.width as isize
    }

    fn is_wall(&self, pos: (isize, isize)) -> bool {
        self.wall_positions.contains(&pos)
    }

    fn is_actor_at(&self, pos: (isize, isize), exclude_idx: usize) -> bool {
        self.actors
            .borrow()
            .iter()
            .enumerate()
            .any(|(i, a)| i != exclude_idx && !a.done && a.current_position == pos)
    }

    fn validate_move_result(&self, actor_idx: usize, new_pos: (isize, isize)) -> MoveResult {
        if !self.is_in_bounds(new_pos) {
            MoveResult::OutOfBounds
        } else if self.is_wall(new_pos) {
            MoveResult::WallCollision
        } else if new_pos == self.end_position {
            MoveResult::AtEnd
        } else if self.is_actor_at(new_pos, actor_idx) {
            MoveResult::AgentCollision
        } else {
            MoveResult::Valid
        }
    }

    fn build_actor_observation(&self, actor_idx: usize) -> Vec<f32> {
        let cells = self.length * self.width;
        let mut obs = vec![0.0f32; cells];

        for &(wr, wc) in &self.wall_positions {
            obs[wr as usize * self.width + wc as usize] = 1.0;
        }

        let (er, ec) = self.end_position;
        obs[er as usize * self.width + ec as usize] = 2.0;

        let actors = self.actors.borrow();
        for (i, actor) in actors.iter().enumerate() {
            if i != actor_idx && !actor.done {
                let (r, c) = actor.current_position;
                obs[r as usize * self.width + c as usize] = 3.0;
            }
        }
        let (ar, ac) = actors[actor_idx].current_position;
        obs[ar as usize * self.width + ac as usize] = 4.0;

        obs
    }

    fn update_observations(&self) {
        let num_actors = self.actors.borrow().len();
        let mut obs = self.last_observations.borrow_mut();
        for i in 0..num_actors {
            obs[i] = self.build_actor_observation(i);
        }
    }
}

impl<B: Backend> EnvironmentTrait for GridWorldEnv<B>
where
    B::Device: Clone,
{
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        self.running.store(true, Ordering::SeqCst);
        self.reset();
        Ok(())
    }

    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        self.update_observations();
        let obs = self.last_observations.borrow();
        let n = obs.len();
        let cells = self.length * self.width;
        let flat: Vec<f32> = obs.iter().flat_map(|row| row.iter().copied()).collect();
        let tensor = Tensor::<B, 2, Float>::from_data(
            TensorData::new(flat, [n, cells]),
            &self.device,
        );
        Ok(Box::new(tensor))
    }
}

impl<B: Backend> TrainingPerformanceReturnFn for GridWorldEnv<B> {
	fn calculate_performance_return(&self) -> Result<Box<dyn Any>, EnvironmentError> {
		{
			let actors = self.actors.borrow();
			let mut returns = self.episode_returns.borrow_mut();
			for (i, actor) in actors.iter().enumerate() {
			returns[i] = actor.cumulative_reward;
			}
		}	
		let flat = self.episode_returns.borrow().clone();
		let n = flat.len();
		let tensor = Tensor::<B, 1, Float>::from_data(
			TensorData::new(flat, [n]),
			&self.device,
		);
		Ok(Box::new(tensor))
}

}
