//! Simple 5×5 gridworld for PPO convergence benchmarking.
//!
//! Agent starts at (0,0) and must reach (4,4).
//! Observation: 25-dim one-hot encoding of agent position.
//! Actions: 0=Up, 1=Down, 2=Left, 3=Right (boundary-clamped).
//! Reward: +1.0 on goal, -0.01 per step otherwise.
//! Episode ends on goal or after MAX_STEPS steps.

use std::sync::Mutex;
use std::any::Any;

use relayrl_env_trait::{
    DynScalarEnvironment, EnvDType, EnvNdArrayDType, EnvironmentHandle, EnvironmentKind,
    ScalarEnvReset, EnvironmentError,
};

pub const GRID_SIZE: usize = 5;
pub const OBS_DIM: usize = GRID_SIZE * GRID_SIZE;
pub const ACT_DIM: usize = 4;
pub const MAX_STEPS: usize = 100;

pub struct GridWorldBenchEnv {
    state: Mutex<GwState>,
}

struct GwState {
    row: usize,
    col: usize,
    steps: usize,
}

impl GwState {
    fn new() -> Self {
        Self { row: 0, col: 0, steps: 0 }
    }

    fn reset(&mut self) {
        self.row = 0;
        self.col = 0;
        self.steps = 0;
    }

    fn obs_bytes(&self) -> Vec<u8> {
        let mut obs = [0.0f32; OBS_DIM];
        obs[self.row * GRID_SIZE + self.col] = 1.0;
        bytemuck::cast_slice::<f32, u8>(&obs).to_vec()
    }

    fn manhattan_dist(&self) -> i32 {
        ((GRID_SIZE - 1 - self.row) + (GRID_SIZE - 1 - self.col)) as i32
    }

    fn step(&mut self, action: usize) -> (f32, bool) {
        self.steps += 1;
        let old_dist = self.manhattan_dist();
        let (dr, dc): (i32, i32) = match action {
            0 => (-1, 0),
            1 => (1, 0),
            2 => (0, -1),
            3 => (0, 1),
            _ => (0, 0),
        };
        self.row = (self.row as i32 + dr).clamp(0, GRID_SIZE as i32 - 1) as usize;
        self.col = (self.col as i32 + dc).clamp(0, GRID_SIZE as i32 - 1) as usize;
        let new_dist = self.manhattan_dist();

        let at_goal = self.row == GRID_SIZE - 1 && self.col == GRID_SIZE - 1;
        let timeout = self.steps >= MAX_STEPS;
        let done = at_goal || timeout;
        // Dense shaping: +0.05 per step closer, -0.05 per step farther, -0.01 step cost, +1 goal
        let shaping = 0.05 * (old_dist - new_dist) as f32;
        let reward = if at_goal { 1.0 } else { -0.01 + shaping };
        (reward, done)
    }
}

impl GridWorldBenchEnv {
    pub fn new() -> Self {
        Self { state: Mutex::new(GwState::new()) }
    }
}

impl Default for GridWorldBenchEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GridWorldBenchEnv {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl relayrl_env_trait::Environment for GridWorldBenchEnv {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        Ok(())
    }

    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        let s = self.state.lock().unwrap();
        let mut obs = vec![0.0f32; OBS_DIM];
        obs[s.row * GRID_SIZE + s.col] = 1.0;
        Ok(Box::new(obs))
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
        self.state.lock().unwrap().obs_bytes()
    }

    fn action_is_discrete(&self) -> bool { true }

    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Scalar
    }

    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Scalar(self as Box<dyn DynScalarEnvironment>)
    }
}

impl relayrl_env_trait::ScalarEnvironment for GridWorldBenchEnv {
    fn step_bytes(&self, action: &[u8]) -> Option<(Vec<u8>, f32, bool)> {
        let act = *action.first()? as usize;
        let (reward, done) = {
            let mut s = self.state.lock().unwrap();
            s.step(act)
        };
        if done {
            let mut s = self.state.lock().unwrap();
            s.reset();
        }
        let obs = self.state.lock().unwrap().obs_bytes();
        Some((obs, reward, done))
    }

    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        let mut s = self.state.lock().unwrap();
        s.reset();
        let obs = s.obs_bytes();
        Ok(ScalarEnvReset { observation: obs, info: None })
    }
}
