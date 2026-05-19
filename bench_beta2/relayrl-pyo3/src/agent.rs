//! [`PyRelayRLAgent`] — PyO3-exposed RelayRL agent.
//!
//! Wraps `RelayRLAgent<NdArray, 2, 2, Float, Float>` behind a Tokio runtime
//! and a `Mutex`, so every async framework call is driven via `block_on` from
//! the Python thread.

use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use burn_ndarray::NdArray;
use burn_tensor::Float;

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;
use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode, RelayRLActorEnv,
    RelayRLAgentActors,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};
use relayrl_types::data::tensor::{DType, NdArrayDType};
use relayrl_types::model::{ModelFileType, ModelMetadata};

use crate::py_env::{PyScalarEnv, PyVectorEnv};

type B = NdArray;
type AgentT = relayrl_framework::prelude::network::RelayRLAgent<B, 2, 2, Float, Float>;

// ─────────────────────────── model helpers ───────────────────────────────────

/// Build (or load) an ONNX model suitable for `ModelModule<NdArray>`.
///
/// If `onnx_bytes` is `None`, a 3-layer MLP bootstrap model is generated:
/// `obs_dim → hidden_dim → hidden_dim → act_dim` with constant 0.01 weights.
pub(crate) fn make_model(
    obs_dim: usize,
    act_dim: usize,
    hidden_dim: usize,
    onnx_bytes: Option<Vec<u8>>,
) -> Result<ModelModule<B>, Box<dyn std::error::Error>>
where
    B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>,
{
    let bytes = onnx_bytes.unwrap_or_else(|| {
        let specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = vec![
            (obs_dim,    hidden_dim, vec![0.01f32; hidden_dim * obs_dim],    vec![0.0f32; hidden_dim]),
            (hidden_dim, hidden_dim, vec![0.01f32; hidden_dim * hidden_dim], vec![0.0f32; hidden_dim]),
            (hidden_dim, act_dim,    vec![0.01f32; act_dim    * hidden_dim], vec![0.0f32; act_dim]),
        ];
        build_onnx_mlp_bytes(&specs)
    });
    let metadata = ModelMetadata {
        model_file:     "model.onnx".to_string(),
        model_type:     ModelFileType::Onnx,
        input_dtype:    DType::NdArray(NdArrayDType::F32),
        output_dtype:   DType::NdArray(NdArrayDType::F32),
        input_shape:    vec![1, obs_dim],
        output_shape:   vec![1, act_dim],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(bytes, metadata)?)
}

// ─────────────────────────── PyRelayRLAgent ──────────────────────────────────

/// Python-accessible RelayRL agent.
///
/// ## Quick start
///
/// ```python
/// import relayrl_pyo3 as rl
/// import gymnasium as gym
///
/// agent = rl.RelayRLAgent(obs_dim=8, act_dim=4)
/// ids   = agent.get_actor_ids()
///
/// # Register a scalar env (cloned N times by the framework)
/// agent.set_scalar_env(ids[0], factory=lambda: gym.make("LunarLander-v3"),
///                      obs_dim=8, act_dim=4, discrete=True, count=32)
/// agent.run_env(ids[0], steps=50_000)
/// agent.shutdown()
/// ```
///
/// ## Batched vec-env (SB3-compatible)
///
/// ```python
/// from stable_baselines3.common.vec_env import DummyVecEnv
/// vec_env = DummyVecEnv([lambda: gym.make("LunarLander-v3")] * 32)
///
/// agent.set_vector_env(ids[0], env=vec_env, n_envs=32,
///                      obs_dim=8, act_dim=4, discrete=True)
/// agent.run_env(ids[0], steps=50_000)
/// ```
#[pyclass(name = "RelayRLAgent")]
pub struct PyRelayRLAgent {
    runtime: tokio::runtime::Runtime,
    agent:   Mutex<AgentT>,
}

#[pymethods]
impl PyRelayRLAgent {
    /// Create a new agent.
    ///
    /// Args:
    ///   obs_dim:      Observation space dimensionality.
    ///   act_dim:      Action space dimensionality.
    ///   actor_count:  Number of parallel actors to initialise (default 1).
    ///   router_scale: Internal router scale factor (default 1).
    ///   hidden_dim:   Hidden layer size for the bootstrap MLP (default 64).
    ///   onnx_bytes:   Pre-trained ONNX model as Python `bytes`.  If omitted,
    ///                 a random bootstrap MLP is used.
    #[new]
    #[pyo3(signature = (obs_dim, act_dim, actor_count=1, router_scale=1, hidden_dim=64, onnx_bytes=None))]
    fn new(
        obs_dim:      usize,
        act_dim:      usize,
        actor_count:  u32,
        router_scale: u32,
        hidden_dim:   usize,
        onnx_bytes:   Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let model = make_model(obs_dim, act_dim, hidden_dim, onnx_bytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let config_path = std::path::PathBuf::from("./config.json");
        let mut builder = AgentBuilder::<B, 2, 2, Float, Float>::builder()
            .actor_count(actor_count)
            .default_device(DeviceType::Cpu)
            .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
            .actor_training_data_mode(ActorTrainingDataMode::Disabled)
            .default_model(model)
            .router_scale(router_scale);
        if config_path.exists() {
            builder = builder.config_path(config_path);
        }

        let (mut agent, params) = runtime
            .block_on(builder.build())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        runtime
            .block_on(agent.start(params))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self { runtime, agent: Mutex::new(agent) })
    }

    /// Return all actor IDs as UUID strings.
    fn get_actor_ids(&self) -> PyResult<Vec<String>> {
        self.agent
            .lock()
            .unwrap()
            .get_actor_ids()
            .map(|ids| ids.iter().map(|u| u.to_string()).collect())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Register a Python scalar env with an actor.
    ///
    /// `factory` is called `count` times by the framework to populate each
    /// parallel env slot — it must accept no arguments and return a fresh env.
    ///
    /// Env protocol: gymnasium (`reset → (obs, info)`, `step → 5-tuple`) or
    /// SB3-style (`reset → obs`, `step → 4-tuple`).
    #[pyo3(signature = (actor_id, factory, obs_dim, act_dim, discrete=true, count=1))]
    fn set_scalar_env(
        &self,
        actor_id: String,
        factory:  PyObject,
        obs_dim:  usize,
        act_dim:  usize,
        discrete: bool,
        count:    u32,
    ) -> PyResult<()> {
        let id = actor_id
            .parse::<uuid::Uuid>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Create the first env instance eagerly so factory errors surface here.
        let env_obj = Python::with_gil(|py| {
            factory
                .call0(py)
                .map_err(|e| PyRuntimeError::new_err(format!("factory() failed: {e}")))
        })?;

        let py_env = PyScalarEnv::new(env_obj, factory, obs_dim, act_dim, discrete);
        let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(py_env);

        self.runtime
            .block_on(self.agent.lock().unwrap().set_env(id, boxed, count))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Register a Python batched vec-env with an actor.
    ///
    /// `env` must implement the batched-env protocol (see [`PyVectorEnv`] docs).
    /// Both SB3 `DummyVecEnv` / `SubprocVecEnv` and gymnasium `VectorEnv` are
    /// supported.
    #[pyo3(signature = (actor_id, env, n_envs, obs_dim, act_dim, discrete=true))]
    fn set_vector_env(
        &self,
        actor_id: String,
        env:      PyObject,
        n_envs:   usize,
        obs_dim:  usize,
        act_dim:  usize,
        discrete: bool,
    ) -> PyResult<()> {
        let id = actor_id
            .parse::<uuid::Uuid>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let py_env = PyVectorEnv::new(env, n_envs, obs_dim, act_dim, discrete);
        let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(py_env);

        self.runtime
            .block_on(self.agent.lock().unwrap().set_env(id, boxed, n_envs as u32))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Step all registered envs for `steps` loop iterations.
    ///
    /// Each iteration steps every sub-env once (via the framework's internal
    /// `ScalarVecEnv` or `BatchVecEnv`), runs model inference, and dispatches
    /// the resulting actions.
    fn run_env(&self, actor_id: String, steps: usize) -> PyResult<()> {
        let id = actor_id
            .parse::<uuid::Uuid>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.runtime
            .block_on(self.agent.lock().unwrap().run_env(id, steps))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove the env registered to an actor.
    fn remove_env(&self, actor_id: String) -> PyResult<()> {
        let id = actor_id
            .parse::<uuid::Uuid>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.runtime
            .block_on(self.agent.lock().unwrap().remove_env(id))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the number of envs registered to an actor.
    fn get_env_count(&self, actor_id: String) -> PyResult<u32> {
        let id = actor_id
            .parse::<uuid::Uuid>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.runtime
            .block_on(self.agent.lock().unwrap().get_env_count(id))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Shut down all actors and release framework resources.
    fn shutdown(&self) -> PyResult<()> {
        self.runtime
            .block_on(self.agent.lock().unwrap().shutdown())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
