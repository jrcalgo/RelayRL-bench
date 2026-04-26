//! Python → Rust environment trait adapters.
//!
//! [`PyScalarEnv`] wraps a single gymnasium-compatible Python env and presents
//! it as a [`ScalarEnvironment`].  The framework clones it N times (via the
//! blanket `DynScalarEnvironment` impl) by calling the user-supplied `factory`
//! callable, so each parallel slot gets its own independent Python env object.
//!
//! [`PyVectorEnv`] wraps a batched Python vec-env (SB3 DummyVecEnv /
//! SubprocVecEnv or any compatible object) and presents it as a
//! [`VectorEnvironment`].  All sub-envs are stepped with a single Python call.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use relayrl_env_trait::{
    DynScalarEnvironment, DynVectorEnv, EnvDType, EnvNdArrayDType, EnvironmentError,
    EnvironmentHandle, EnvironmentKind, EnvironmentUuid, ScalarEnvReset, ScalarEnvironment,
    VectorEnvReset, VectorEnvironment,
};
use uuid::Uuid;

// ─────────────────────────── conversion helpers ──────────────────────────────

/// Convert a Python obs object (numpy array or sequence) to flat f32 bytes.
///
/// Tries `.astype('float32').tobytes()` first (numpy fast path), then falls
/// back to iterating as a Python sequence.
fn obs_to_bytes(_py: Python<'_>, obj: &Bound<'_, PyAny>, obs_dim: usize) -> PyResult<Vec<u8>> {
    // Fast path: numpy array → cast to f32 → tobytes
    if let Ok(as_f32) = obj.call_method1("astype", ("float32",)) {
        if let Ok(bytes_obj) = as_f32.call_method0("tobytes") {
            if let Ok(raw) = bytes_obj.extract::<Vec<u8>>() {
                if raw.len() == obs_dim * 4 {
                    return Ok(raw);
                }
            }
        }
    }
    // Slow fallback: iterate as Python sequence
    let f32s: Vec<f32> = (0..obs_dim)
        .map(|i| obj.get_item(i).and_then(|v| v.extract::<f32>()).unwrap_or(0.0))
        .collect();
    if f32s.len() == obs_dim {
        Ok(bytemuck::cast_slice::<f32, u8>(&f32s).to_vec())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "obs conversion: expected {obs_dim} elements, got {}",
            f32s.len()
        )))
    }
}

/// Convert framework action bytes to a Python scalar (discrete) or numpy
/// array (continuous) suitable for passing to a single-env `step()`.
fn scalar_action_to_py<'py>(
    py: Python<'py>,
    action: &[u8],
    act_dim: usize,
    discrete: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if discrete {
        let idx = *action
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty action bytes"))? as i64;
        Ok(idx.into_pyobject(py)?.into_any())
    } else {
        let numpy = py.import("numpy")?;
        let len = act_dim.min(action.len() / 4) * 4;
        let bytes_obj = PyBytes::new(py, &action[..len]);
        numpy.call_method1("frombuffer", (bytes_obj, "float32"))
    }
}

/// Convert flat framework action bytes to a Python numpy array for a
/// batched vec-env's `step()`.
///
/// Discrete: produces `int64[n_envs]`.
/// Continuous: produces `float32[n_envs, act_dim]`.
fn batch_actions_to_py<'py>(
    py: Python<'py>,
    actions: &[u8],
    n_envs: usize,
    act_dim: usize,
    discrete: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    if discrete {
        let ints: Vec<i64> = actions[..n_envs.min(actions.len())]
            .iter()
            .map(|&b| b as i64)
            .collect();
        numpy.call_method1("array", (ints,))
    } else {
        let bytes_obj = PyBytes::new(py, actions);
        let flat = numpy.call_method1("frombuffer", (bytes_obj, "float32"))?;
        flat.call_method1("reshape", ((n_envs as i64, act_dim as i64),))
    }
}

// ─────────────────────────── PyScalarEnv ─────────────────────────────────────

/// Wraps a single Python gymnasium-compatible env as a Rust
/// [`ScalarEnvironment`].
///
/// ## Python protocol
///
/// | Call | Signature | Notes |
/// |------|-----------|-------|
/// | `factory()` | `() → env` | Returns a fresh env instance; called once per parallel slot |
/// | `env.reset()` | `() → (obs, info)` or `obs` | Gymnasium or bare array |
/// | `env.step(action)` | `(int) → (obs, r, terminated, truncated, info)` | Gymnasium 5-tuple |
/// |  | `(int) → (obs, r, done, info)` | SB3 4-tuple also accepted |
///
/// Observations must be float32-castable numpy arrays of length `obs_dim`.
pub struct PyScalarEnv {
    env_obj:      PyObject,
    factory:      PyObject,
    obs_dim:      usize,
    act_dim:      usize,
    act_discrete: bool,
    /// Cached last observation bytes (for `flat_observation_bytes`).
    last_obs:     Mutex<Vec<u8>>,
}

// Py<PyAny> is Send + Sync since PyO3 0.14 — the GIL guards actual access.
unsafe impl Send for PyScalarEnv {}
unsafe impl Sync for PyScalarEnv {}

impl PyScalarEnv {
    pub fn new(
        env_obj: PyObject,
        factory: PyObject,
        obs_dim: usize,
        act_dim: usize,
        act_discrete: bool,
    ) -> Self {
        Self {
            env_obj,
            factory,
            obs_dim,
            act_dim,
            act_discrete,
            last_obs: Mutex::new(vec![0u8; obs_dim * 4]),
        }
    }
}

/// `Clone` calls the factory to create a fresh Python env, then resets it.
/// The `DynScalarEnvironment` blanket impl uses this to populate each parallel
/// slot when `ScalarVecEnv::init_boxed` is called by the framework.
impl Clone for PyScalarEnv {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let new_env = self
                .factory
                .call0(py)
                .expect("PyScalarEnv: factory() must be a zero-arg callable that returns an env");
            let _ = new_env.call_method0(py, "reset");
            PyScalarEnv::new(
                new_env,
                self.factory.clone_ref(py),
                self.obs_dim,
                self.act_dim,
                self.act_discrete,
            )
        })
    }
}

impl relayrl_env_trait::Environment for PyScalarEnv {
    fn run_environment(&self) -> Result<(), EnvironmentError> {
        Ok(())
    }

    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(self.last_obs.lock().unwrap().clone()))
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

    fn flat_observation_bytes(&self) -> Option<Vec<u8>> {
        Some(self.last_obs.lock().unwrap().clone())
    }

    fn action_is_discrete(&self) -> bool {
        self.act_discrete
    }

    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Scalar
    }

    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Scalar(self as Box<dyn DynScalarEnvironment>)
    }
}

impl ScalarEnvironment for PyScalarEnv {
    fn reset(&self) -> Result<ScalarEnvReset, EnvironmentError> {
        Python::with_gil(|py| {
            let result = self
                .env_obj
                .bind(py)
                .call_method0("reset")
                .map_err(|e| EnvironmentError::EnvironmentError(e.to_string()))?;

            // Accept both (obs, info) tuple and bare obs array.
            let obs_obj = if let Ok(tup) = result.downcast::<PyTuple>() {
                tup.get_item(0)
                    .map_err(|e| EnvironmentError::ObservationBuildingError(e.to_string()))?
            } else {
                result.clone()
            };

            let obs_bytes = obs_to_bytes(py, &obs_obj, self.obs_dim)
                .map_err(|e| EnvironmentError::ObservationBuildingError(e.to_string()))?;
            *self.last_obs.lock().unwrap() = obs_bytes.clone();
            Ok(ScalarEnvReset { observation: obs_bytes, info: None })
        })
    }

    fn step_bytes(&self, action: &[u8]) -> Option<(Vec<u8>, f32, bool)> {
        Python::with_gil(|py| {
            let py_action =
                scalar_action_to_py(py, action, self.act_dim, self.act_discrete).ok()?;
            let result = self
                .env_obj
                .bind(py)
                .call_method1("step", (py_action,))
                .ok()?;
            let tup = result.downcast::<PyTuple>().ok()?;

            let obs_obj  = tup.get_item(0).ok()?;
            let reward: f32 = tup.get_item(1).ok()?.extract().ok()?;

            // 5-tuple: (obs, reward, terminated, truncated, info)  — gymnasium
            // 4-tuple: (obs, reward, done, info)                   — SB3 / older gym
            let done = if tup.len() >= 5 {
                let t: bool = tup.get_item(2).ok()?.extract().ok()?;
                let u: bool = tup.get_item(3).ok()?.extract().ok()?;
                t || u
            } else {
                tup.get_item(2).ok()?.extract().ok()?
            };

            let obs_bytes = obs_to_bytes(py, &obs_obj, self.obs_dim).ok()?;
            *self.last_obs.lock().unwrap() = obs_bytes.clone();
            Some((obs_bytes, reward, done))
        })
    }
}

// The blanket impl in relayrl_env_trait provides DynScalarEnvironment for any
// T: ScalarEnvironment + Clone + Send + Sync + 'static, so PyScalarEnv gets it
// automatically and `clone_box()` delegates to our Clone impl above.

// ─────────────────────────── PyVectorEnv ─────────────────────────────────────

/// Wraps a batched Python vec-env as a Rust [`VectorEnvironment`].
///
/// ## Python protocol
///
/// | Call | Signature | Notes |
/// |------|-----------|-------|
/// | `env.reset()` | `() → obs[n,d]` or `(obs, info)` | float32 array |
/// | `env.step(acts)` | `(acts) → (obs, rew, term, trunc, info)` | gymnasium 5-tuple |
/// |  | `(acts) → (obs, rew, done, info)` | SB3 4-tuple |
///
/// Actions: `int64[n_envs]` (discrete) or `float32[n_envs, act_dim]` (continuous).
/// Both SB3 `DummyVecEnv`/`SubprocVecEnv` and raw gymnasium vec-envs are compatible.
pub struct PyVectorEnv {
    env_obj:      PyObject,
    n_envs:       usize,
    obs_dim:      usize,
    act_dim:      usize,
    act_discrete: bool,
    uuids:        Mutex<Vec<EnvironmentUuid>>,
    uuid_to_idx:  Mutex<HashMap<EnvironmentUuid, usize>>,
}

unsafe impl Send for PyVectorEnv {}
unsafe impl Sync for PyVectorEnv {}

impl PyVectorEnv {
    pub fn new(
        env_obj: PyObject,
        n_envs: usize,
        obs_dim: usize,
        act_dim: usize,
        act_discrete: bool,
    ) -> Self {
        Self {
            env_obj,
            n_envs,
            obs_dim,
            act_dim,
            act_discrete,
            uuids: Mutex::new(Vec::new()),
            uuid_to_idx: Mutex::new(HashMap::new()),
        }
    }

    /// Extract flat f32 bytes from a `[n_envs, obs_dim]` numpy array.
    fn extract_flat_obs(&self, _py: Python<'_>, obj: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
        let expected = self.n_envs * self.obs_dim * 4;
        let as_f32 = obj.call_method1("astype", ("float32",)).ok()?;
        let flat = as_f32.call_method0("flatten").ok()?;
        let raw: Vec<u8> = flat.call_method0("tobytes").ok()?.extract().ok()?;
        if raw.len() == expected { Some(raw) } else { None }
    }
}

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

    fn flat_observation_bytes(&self) -> Option<Vec<u8>> {
        None
    }

    fn action_is_discrete(&self) -> bool {
        self.act_discrete
    }

    fn kind(&self) -> EnvironmentKind {
        EnvironmentKind::Vector
    }

    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Vector(self as Box<DynVectorEnv>)
    }
}

impl VectorEnvironment for PyVectorEnv {
    fn init_num_envs(
        &self,
        num_envs: usize,
    ) -> Result<Vec<EnvironmentUuid>, EnvironmentError> {
        let ids: Vec<Uuid> = (0..num_envs).map(|_| Uuid::new_v4()).collect();
        let map: HashMap<_, _> = ids.iter().copied().enumerate().map(|(i, u)| (u, i)).collect();
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
            let py_actions = batch_actions_to_py(
                py, actions, self.n_envs, self.act_dim, self.act_discrete,
            )
            .ok()?;

            let result = self
                .env_obj
                .bind(py)
                .call_method1("step", (py_actions,))
                .ok()?;
            let tup = result.downcast::<PyTuple>().ok()?;
            let tup_len = tup.len();

            let obs_obj  = tup.get_item(0).ok()?;
            let flat_obs = self.extract_flat_obs(py, &obs_obj)?;

            // Rewards: [n_envs] float32
            let rew_obj = tup.get_item(1).ok()?;
            let rew_f32 = rew_obj
                .call_method1("astype", ("float32",))
                .unwrap_or_else(|_| rew_obj.clone());
            let rew_bytes: Vec<u8> = rew_f32.call_method0("tobytes").ok()?.extract().ok()?;
            let rewards: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rew_bytes).to_vec();

            // Dones: handle both gymnasium 5-tuple and SB3 4-tuple
            let dones: Vec<bool> = if tup_len >= 5 {
                let term = tup.get_item(2).ok()?;
                let trunc = tup.get_item(3).ok()?;
                let t_b = term.call_method1("astype", ("bool",)).ok()?;
                let u_b = trunc.call_method1("astype", ("bool",)).ok()?;
                let t_raw: Vec<u8> = t_b.call_method0("tobytes").ok()?.extract().ok()?;
                let u_raw: Vec<u8> = u_b.call_method0("tobytes").ok()?.extract().ok()?;
                t_raw
                    .iter()
                    .zip(u_raw.iter())
                    .map(|(&t, &u)| t != 0 || u != 0)
                    .collect()
            } else {
                let done_obj = tup.get_item(2).ok()?;
                let done_b = done_obj.call_method1("astype", ("bool",)).ok()?;
                let d_raw: Vec<u8> = done_b.call_method0("tobytes").ok()?.extract().ok()?;
                d_raw.iter().map(|&b| b != 0).collect()
            };

            Some((flat_obs, rewards, dones))
        })
    }
}
