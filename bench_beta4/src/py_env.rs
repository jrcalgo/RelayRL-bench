//! PyVectorEnv — thin PyO3 bridge from a gymnasium SyncVectorEnv to
//! relayrl_env_trait::VectorEnvironment.
//!
//! Efficiency notes vs the beta2 implementation:
//!   - `numpy` module is cached at construction; no `py.import` on every step.
//!   - `ascontiguousarray(obs, 'float32')` replaces `astype().flatten()` (fewer
//!     Python method calls; no-op when obs is already C-contiguous float32).
//!   - Actions use `numpy.frombuffer(PyBytes, 'int64')` — zero-copy numpy view of
//!     a Rust-owned byte slice rather than building a Python list.
//!   - Single `Python::with_gil` acquisition per `step_bytes` call covers obs,
//!     rewards, and dones together.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use uuid::Uuid;

use relayrl_env_trait::{
    EnvDType, EnvNdArrayDType, EnvironmentError, EnvironmentHandle,
    EnvironmentKind, EnvironmentUuid, VectorEnvReset, VectorEnvironment,
};

// ─────────────────────────── PyVectorEnv ─────────────────────────────────────

/// Wraps a gymnasium `SyncVectorEnv` (or any compatible batched vec-env) as a
/// [`VectorEnvironment`].  All 64 environments are stepped with a single Python
/// call; the GIL is held only for that call and released immediately after.
pub struct PyVectorEnv {
    env_obj: PyObject,
    np: PyObject,
    n_envs: usize,
    obs_dim: usize,
    act_dim: usize,
    act_discrete: bool,
    uuids: Mutex<Vec<EnvironmentUuid>>,
    uuid_to_idx: Mutex<HashMap<EnvironmentUuid, usize>>,
    /// Cached flat `[n_envs × obs_dim × 4]` bytes; updated on every step/reset.
    obs_cache: Mutex<Vec<u8>>,
}

// Py<PyAny> is Send + Sync in PyO3 0.14+. Mutex<...> is Send + Sync.
// Both fields are safe to move across threads as long as GIL is acquired before use.
unsafe impl Send for PyVectorEnv {}
unsafe impl Sync for PyVectorEnv {}

impl PyVectorEnv {
    /// Create a new wrapper.  `env_obj` must already be reset (or reset lazily
    /// on the first `VectorEnvironment::reset()` call from the framework).
    pub fn new(
        env_obj: PyObject,
        n_envs: usize,
        obs_dim: usize,
        act_dim: usize,
        act_discrete: bool,
    ) -> Self {
        let np: PyObject = Python::with_gil(|py| py.import("numpy").unwrap().into());
        Self {
            env_obj,
            np,
            n_envs,
            obs_dim,
            act_dim,
            act_discrete,
            uuids: Mutex::new(Vec::new()),
            uuid_to_idx: Mutex::new(HashMap::new()),
            obs_cache: Mutex::new(vec![0u8; n_envs * obs_dim * 4]),
        }
    }

    /// Extract a flat `Vec<u8>` of f32 bytes from a `[n_envs, obs_dim]` numpy array.
    /// Uses `ascontiguousarray` which is a no-op when the array is already C-contiguous
    /// float32 (the common case for gymnasium LunarLander).
    fn extract_flat_obs(&self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
        let np = self.np.bind(py);
        let contiguous = np.call_method1("ascontiguousarray", (obj, "float32")).ok()?;
        let raw: Vec<u8> = contiguous.call_method0("tobytes").ok()?.extract().ok()?;
        if raw.len() == self.n_envs * self.obs_dim * 4 {
            Some(raw)
        } else {
            None
        }
    }
}

/// Convert 1-byte-per-env discrete action slice to `int64[n_envs]` numpy array.
/// Uses `frombuffer` on a `PyBytes` wrapping the cast slice — avoids building a
/// Python list and performs a single memcpy via numpy's buffer protocol.
fn batch_actions_to_py<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    actions: &[u8],
    n_envs: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let i64s: Vec<i64> = actions[..n_envs.min(actions.len())]
        .iter()
        .map(|&b| b as i64)
        .collect();
    let i64_bytes: &[u8] = bytemuck::cast_slice(&i64s);
    let bytes_obj = PyBytes::new(py, i64_bytes);
    np.call_method1("frombuffer", (bytes_obj, "int64"))
}

// ─────────────────────────── Environment trait ───────────────────────────────

impl relayrl_env_trait::Environment for PyVectorEnv {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        Ok(())
    }

    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Err(EnvironmentError::EnvironmentError(
            "PyVectorEnv: call VectorEnvironment::reset instead of build_observation".into(),
        ))
    }

    fn observation_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }

    fn action_dtype(&self) -> EnvDType {
        EnvDType::NdArray(EnvNdArrayDType::F32)
    }

    fn observation_dim(&self) -> usize {
        self.obs_dim
    }

    fn action_dim(&self) -> usize {
        self.act_dim
    }

    fn flat_observation_bytes(&self) -> Vec<u8> {
        self.obs_cache.lock().unwrap().clone()
    }

    fn action_is_discrete(&self) -> bool {
        self.act_discrete
    }

    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Vector
    }

    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Vector(self as Box<dyn VectorEnvironment>)
    }
}

// ─────────────────────────── VectorEnvironment trait ─────────────────────────

impl VectorEnvironment for PyVectorEnv {
    fn init_num_envs(
        &self,
        num_envs: usize,
    ) -> Result<Vec<EnvironmentUuid>, EnvironmentError> {
        let ids: Vec<Uuid> = (0..num_envs).map(|_| Uuid::new_v4()).collect();
        let map: HashMap<_, _> = ids
            .iter()
            .copied()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect();
        *self.uuids.lock().unwrap() = ids.clone();
        *self.uuid_to_idx.lock().unwrap() = map;
        Ok(ids)
    }

    fn reset(
        &self,
        env_ids: &[EnvironmentUuid],
    ) -> Result<Vec<VectorEnvReset>, EnvironmentError> {
        Python::with_gil(|py| {
            let result = self
                .env_obj
                .bind(py)
                .call_method0("reset")
                .map_err(|e| EnvironmentError::EnvironmentError(e.to_string()))?;

            // gymnasium 1.x returns (obs, info); accept bare array too
            let obs_obj = if let Ok(tup) = result.downcast::<PyTuple>() {
                tup.get_item(0)
                    .map_err(|e| EnvironmentError::ObservationBuildingError(e.to_string()))?
            } else {
                result.clone()
            };

            let flat = self.extract_flat_obs(py, &obs_obj).ok_or_else(|| {
                EnvironmentError::ObservationBuildingError(format!(
                    "reset: expected [{} × {}] float32 obs",
                    self.n_envs, self.obs_dim
                ))
            })?;
            *self.obs_cache.lock().unwrap() = flat.clone();

            let stride = self.obs_dim * 4;
            let idx_map = self.uuid_to_idx.lock().unwrap();
            let resets = env_ids
                .iter()
                .filter_map(|uuid| {
                    let &idx = idx_map.get(uuid)?;
                    Some(VectorEnvReset {
                        env_id: *uuid,
                        observation: flat[idx * stride..(idx + 1) * stride].to_vec(),
                        info: None,
                    })
                })
                .collect();
            Ok(resets)
        })
    }

    fn n_envs(&self) -> usize {
        self.n_envs
    }

    fn step_bytes(&self, actions: &[u8]) -> Option<(Vec<u8>, Vec<f32>, Vec<bool>)> {
        Python::with_gil(|py| {
            let np = self.np.bind(py);
            let py_actions = batch_actions_to_py(py, &np, actions, self.n_envs).ok()?;

            let result = self
                .env_obj
                .bind(py)
                .call_method1("step", (py_actions,))
                .ok()?;
            let tup = result.downcast::<PyTuple>().ok()?;

            // obs: [n_envs, obs_dim] — auto-reset obs for done envs (gymnasium semantics)
            let flat_obs = self.extract_flat_obs(py, &tup.get_item(0).ok()?)?;
            *self.obs_cache.lock().unwrap() = flat_obs.clone();

            // rewards: gymnasium returns float64 by default; cast to float32
            let rew_item = tup.get_item(1).ok()?;
            let rew_bytes: Vec<u8> = rew_item
                .call_method1("astype", ("float32",))
                .ok()?
                .call_method0("tobytes")
                .ok()?
                .extract()
                .ok()?;
            let rewards: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rew_bytes).to_vec();

            // dones: terminated | truncated (gymnasium 5-tuple)
            let term = tup.get_item(2).ok()?;
            let trunc = tup.get_item(3).ok()?;
            let t_raw: Vec<u8> = term
                .call_method1("astype", ("bool",))
                .ok()?
                .call_method0("tobytes")
                .ok()?
                .extract()
                .ok()?;
            let u_raw: Vec<u8> = trunc
                .call_method1("astype", ("bool",))
                .ok()?
                .call_method0("tobytes")
                .ok()?
                .extract()
                .ok()?;
            let dones: Vec<bool> = t_raw
                .iter()
                .zip(u_raw.iter())
                .map(|(&t, &u)| t != 0 || u != 0)
                .collect();

            Some((flat_obs, rewards, dones))
        })
    }
}

// ─────────────────────────── gymnasium factory helpers ───────────────────────

/// Create a `gymnasium.SyncVectorEnv` wrapping `num_envs` LunarLander-v3 instances.
/// Returns the Python object (already reset) and a `PyVectorEnv` wrapping it.
pub fn make_lunar_lander_vec(
    num_envs: usize,
    obs_dim: usize,
    act_dim: usize,
) -> PyResult<PyVectorEnv> {
    Python::with_gil(|py| {
        let gym = py.import("gymnasium")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("num_envs", num_envs as i64)?;
        kwargs.set_item("max_episode_steps", 500i64)?;
        let vec_env = gym.call_method("make_vec", ("LunarLander-v3",), Some(&kwargs))?;
        // Initial reset so the env is in a clean state before set_env hands it to the framework
        vec_env.call_method0("reset")?;
        Ok(PyVectorEnv::new(
            vec_env.unbind(),
            num_envs,
            obs_dim,
            act_dim,
            true, // discrete
        ))
    })
}
