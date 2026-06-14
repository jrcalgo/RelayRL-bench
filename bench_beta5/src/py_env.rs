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

/// Welford online mean/variance for observation normalization.
struct ObsRunningStats {
    mean: Vec<f64>,
    var: Vec<f64>,
    count: f64,
}

impl ObsRunningStats {
    fn new(dim: usize) -> Self {
        Self { mean: vec![0.0; dim], var: vec![1.0; dim], count: 0.0 }
    }

    fn update_batch(&mut self, flat: &[f32], n_envs: usize) {
        let dim = self.mean.len();
        for i in 0..n_envs {
            for d in 0..dim {
                let x = flat[i * dim + d] as f64;
                self.count += 1.0;
                let delta = x - self.mean[d];
                self.mean[d] += delta / self.count;
                let delta2 = x - self.mean[d];
                self.var[d] += delta * delta2;
            }
        }
    }

    /// Normalize `flat` and write the result directly as little-endian f32 bytes,
    /// avoiding the intermediate `Vec<f32>` allocation/copy of the old
    /// `normalize()` + `bytemuck::cast_slice().to_vec()` chain.
    fn normalize_to_bytes(&self, flat: &[f32]) -> Vec<u8> {
        let dim = self.mean.len();
        let denom = (self.count - 1.0).max(1.0);
        let mut out = Vec::with_capacity(flat.len() * 4);
        for (i, &v) in flat.iter().enumerate() {
            let d = i % dim;
            let std = (self.var[d] / denom).sqrt().max(1e-8);
            let normed = ((v as f64 - self.mean[d]) / std).clamp(-10.0, 10.0) as f32;
            out.extend_from_slice(&normed.to_le_bytes());
        }
        out
    }
}

use pyo3::buffer::PyBuffer;
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
    /// Required by the eval loop (`flat_observation_bytes()` is polled per step there).
    obs_cache: Mutex<Vec<u8>>,
    obs_stats: Mutex<ObsRunningStats>,
    /// Reused scratch buffers to avoid per-step heap allocations.
    scratch_obs_f32: Mutex<Vec<f32>>,
    scratch_rewards_f64: Mutex<Vec<f64>>,
    scratch_actions_i64: Mutex<Vec<i64>>,
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
            obs_stats: Mutex::new(ObsRunningStats::new(obs_dim)),
            scratch_obs_f32: Mutex::new(vec![0f32; n_envs * obs_dim]),
            scratch_rewards_f64: Mutex::new(vec![0f64; n_envs]),
            scratch_actions_i64: Mutex::new(vec![0i64; n_envs]),
        }
    }

    /// Update running obs stats and return normalized f32 bytes.
    fn normalize_obs(&self, floats: &[f32]) -> Vec<u8> {
        let mut stats = self.obs_stats.lock().unwrap();
        stats.update_batch(floats, self.n_envs);
        stats.normalize_to_bytes(floats)
    }

    /// Fill `out` (length `n_envs * obs_dim`) with the f32 observations from a
    /// `[n_envs, obs_dim]` numpy array.
    ///
    /// Fast path: `PyBuffer::<f32>::get` gives a zero-copy view of the numpy
    /// array's underlying memory, copied directly into `out` — no Python
    /// `bytes` object or intermediate `Vec<u8>` is allocated.
    ///
    /// Falls back to the old `ascontiguousarray` -> `tobytes` -> `extract`
    /// chain if the buffer isn't a matching, C-contiguous f32 array (e.g. a
    /// non-numpy or non-contiguous wrapper), so correctness never depends on
    /// the fast path succeeding.
    fn extract_flat_obs_f32(&self, py: Python<'_>, obj: &Bound<'_, PyAny>, out: &mut [f32]) -> bool {
        if let Ok(buf) = PyBuffer::<f32>::get(obj) {
            if buf.item_count() == out.len()
                && buf.is_c_contiguous()
                && buf.copy_to_slice(py, out).is_ok()
            {
                return true;
            }
        }

        let np = self.np.bind(py);
        if let Ok(contiguous) = np.call_method1("ascontiguousarray", (obj, "float32")) {
            if let Ok(raw) = contiguous
                .call_method0("tobytes")
                .and_then(|b| b.extract::<Vec<u8>>())
            {
                if raw.len() == out.len() * 4 {
                    out.copy_from_slice(bytemuck::cast_slice(&raw));
                    return true;
                }
            }
        }
        false
    }
}

/// Convert a numpy `float64[n_envs]` rewards array to `Vec<f32>`.
///
/// Fast path: `PyBuffer::<f64>::get` copies directly into `scratch` (no Python
/// `bytes` allocation), then casts to f32 — same rounding as numpy's
/// `astype("float32")`. Falls back to `astype` -> `tobytes` -> `extract` for
/// non-matching buffers.
fn extract_rewards_f32(py: Python<'_>, rew_item: &Bound<'_, PyAny>, scratch: &mut [f64]) -> Option<Vec<f32>> {
    if let Ok(buf) = PyBuffer::<f64>::get(rew_item) {
        if buf.item_count() == scratch.len() && buf.copy_to_slice(py, scratch).is_ok() {
            return Some(scratch.iter().map(|&v| v as f32).collect());
        }
    }
    let rew_bytes: Vec<u8> = rew_item
        .call_method1("astype", ("float32",))
        .ok()?
        .call_method0("tobytes")
        .ok()?
        .extract()
        .ok()?;
    Some(bytemuck::cast_slice::<u8, f32>(&rew_bytes).to_vec())
}

/// Convert 1-byte-per-env discrete action slice to `int64[n_envs]` numpy array.
/// Uses `frombuffer` on a `PyBytes` wrapping the cast slice — avoids building a
/// Python list and performs a single memcpy via numpy's buffer protocol.
/// `scratch` is reused across calls to avoid a per-step `Vec<i64>` allocation.
fn batch_actions_to_py<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    actions: &[u8],
    n_envs: usize,
    scratch: &mut Vec<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    scratch.clear();
    scratch.extend(actions[..n_envs.min(actions.len())].iter().map(|&b| b as i64));
    let i64_bytes: &[u8] = bytemuck::cast_slice(scratch.as_slice());
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

    fn flat_mask_bytes(&self) -> Option<Vec<u8>> {
        None
    }

    fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(Option::<Vec<u8>>::None))
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

            let flat = {
                let mut scratch = self.scratch_obs_f32.lock().unwrap();
                if !self.extract_flat_obs_f32(py, &obs_obj, &mut scratch) {
                    return Err(EnvironmentError::ObservationBuildingError(format!(
                        "reset: expected [{} × {}] float32 obs",
                        self.n_envs, self.obs_dim
                    )));
                }
                self.normalize_obs(&scratch)
            };
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

    fn step_bytes(&self, actions: &[u8]) -> Option<(Vec<u8>, Option<Vec<u8>>, Vec<f32>, Vec<bool>, Vec<bool>)> {
        Python::with_gil(|py| {
            let np = self.np.bind(py);
            let py_actions = {
                let mut scratch = self.scratch_actions_i64.lock().unwrap();
                batch_actions_to_py(py, &np, actions, self.n_envs, &mut scratch).ok()?
            };

            let result = self
                .env_obj
                .bind(py)
                .call_method1("step", (py_actions,))
                .ok()?;
            let tup = result.downcast::<PyTuple>().ok()?;

            // obs: [n_envs, obs_dim] — auto-reset obs for done envs (gymnasium semantics)
            let flat_obs = {
                let mut scratch = self.scratch_obs_f32.lock().unwrap();
                if !self.extract_flat_obs_f32(py, &tup.get_item(0).ok()?, &mut scratch) {
                    return None;
                }
                self.normalize_obs(&scratch)
            };
            *self.obs_cache.lock().unwrap() = flat_obs.clone();

            // rewards: gymnasium returns float64 by default; cast to float32
            let rew_item = tup.get_item(1).ok()?;
            let rewards = {
                let mut scratch = self.scratch_rewards_f64.lock().unwrap();
                extract_rewards_f32(py, &rew_item, &mut scratch)?
            };

            // terminated and truncated (gymnasium 5-tuple) — already bool dtype,
            // no astype("bool") needed.
            let term = tup.get_item(2).ok()?;
            let trunc = tup.get_item(3).ok()?;
            let t_raw: Vec<u8> = term.call_method0("tobytes").ok()?.extract().ok()?;
            let u_raw: Vec<u8> = trunc.call_method0("tobytes").ok()?.extract().ok()?;
            let terminated: Vec<bool> = t_raw.iter().map(|&t| t != 0).collect();
            let truncated: Vec<bool> = u_raw.iter().map(|&u| u != 0).collect();

            Some((flat_obs, None, rewards, terminated, truncated))
        })
    }
}

// ─────────────────────────── EnvPoolVecEnv ───────────────────────────────────

/// Wraps an EnvPool gymnasium-mode env as a [`VectorEnvironment`].
///
/// EnvPool differences vs gymnasium SyncVectorEnv:
///   - Releases the GIL during `env.step()` and steps all envs on a C++ thread pool.
///   - Returns obs/rewards as float32 natively — no `astype` conversions needed.
///   - Returns term/trunc as bool natively — no `astype` conversions needed.
///   - Expects actions as int32, not int64.
pub struct EnvPoolVecEnv {
    env_obj: PyObject,
    np: PyObject,
    n_envs: usize,
    obs_dim: usize,
    act_dim: usize,
    act_discrete: bool,
    uuids: Mutex<Vec<EnvironmentUuid>>,
    uuid_to_idx: Mutex<HashMap<EnvironmentUuid, usize>>,
    obs_cache: Mutex<Vec<u8>>,
    obs_stats: Mutex<ObsRunningStats>,
}

unsafe impl Send for EnvPoolVecEnv {}
unsafe impl Sync for EnvPoolVecEnv {}

impl EnvPoolVecEnv {
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
            obs_stats: Mutex::new(ObsRunningStats::new(obs_dim)),
        }
    }

    fn normalize_obs(&self, raw: Vec<u8>) -> Vec<u8> {
        let floats: &[f32] = bytemuck::cast_slice(&raw);
        let mut stats = self.obs_stats.lock().unwrap();
        stats.update_batch(floats, self.n_envs);
        stats.normalize_to_bytes(floats)
    }

    /// Extract flat f32 bytes from a numpy array already guaranteed float32 C-contiguous.
    /// Skips the `ascontiguousarray` cast used by PyVectorEnv since EnvPool guarantees this.
    fn extract_flat_obs(&self, _py: Python<'_>, obj: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
        let raw: Vec<u8> = obj.call_method0("tobytes").ok()?.extract().ok()?;
        if raw.len() == self.n_envs * self.obs_dim * 4 {
            Some(raw)
        } else {
            None
        }
    }
}

/// Convert 1-byte-per-env discrete actions to `int32[n_envs]` numpy array for EnvPool.
fn batch_actions_to_py_i32<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyAny>,
    actions: &[u8],
    n_envs: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let i32s: Vec<i32> = actions[..n_envs.min(actions.len())]
        .iter()
        .map(|&b| b as i32)
        .collect();
    let i32_bytes: &[u8] = bytemuck::cast_slice(&i32s);
    let bytes_obj = PyBytes::new(py, i32_bytes);
    np.call_method1("frombuffer", (bytes_obj, "int32"))
}

impl relayrl_env_trait::Environment for EnvPoolVecEnv {
    fn run_environment(&self) -> Result<(), EnvironmentError> { Ok(()) }

    fn build_observation(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Err(EnvironmentError::EnvironmentError(
            "EnvPoolVecEnv: call VectorEnvironment::reset instead of build_observation".into(),
        ))
    }

    fn observation_dtype(&self) -> EnvDType { EnvDType::NdArray(EnvNdArrayDType::F32) }
    fn action_dtype(&self) -> EnvDType { EnvDType::NdArray(EnvNdArrayDType::F32) }
    fn observation_dim(&self) -> usize { self.obs_dim }
    fn action_dim(&self) -> usize { self.act_dim }
    fn flat_observation_bytes(&self) -> Vec<u8> { self.obs_cache.lock().unwrap().clone() }

    fn flat_mask_bytes(&self) -> Option<Vec<u8>> { None }

    fn build_mask(&self) -> Result<Box<dyn Any>, EnvironmentError> {
        Ok(Box::new(Option::<Vec<u8>>::None))
    }

    fn action_is_discrete(&self) -> bool { self.act_discrete }
    fn kind(&self) -> EnvironmentKind { EnvironmentKind::Vector }

    fn into_handle(self: Box<Self>) -> EnvironmentHandle {
        EnvironmentHandle::Vector(self as Box<dyn VectorEnvironment>)
    }
}

impl VectorEnvironment for EnvPoolVecEnv {
    fn init_num_envs(&self, num_envs: usize) -> Result<Vec<EnvironmentUuid>, EnvironmentError> {
        let ids: Vec<Uuid> = (0..num_envs).map(|_| Uuid::new_v4()).collect();
        let map: HashMap<_, _> = ids.iter().copied().enumerate().map(|(i, u)| (u, i)).collect();
        *self.uuids.lock().unwrap() = ids.clone();
        *self.uuid_to_idx.lock().unwrap() = map;
        Ok(ids)
    }

    fn reset(&self, env_ids: &[EnvironmentUuid]) -> Result<Vec<VectorEnvReset>, EnvironmentError> {
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

            let raw = self.extract_flat_obs(py, &obs_obj).ok_or_else(|| {
                EnvironmentError::ObservationBuildingError(format!(
                    "reset: expected [{} × {}] float32 obs",
                    self.n_envs, self.obs_dim
                ))
            })?;
            let flat = self.normalize_obs(raw);
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

    fn n_envs(&self) -> usize { self.n_envs }

    fn step_bytes(&self, actions: &[u8]) -> Option<(Vec<u8>, Option<Vec<u8>>, Vec<f32>, Vec<bool>, Vec<bool>)> {
        Python::with_gil(|py| {
            let np = self.np.bind(py);
            // EnvPool expects int32 actions
            let py_actions = batch_actions_to_py_i32(py, &np, actions, self.n_envs).ok()?;

            let result = self
                .env_obj
                .bind(py)
                .call_method1("step", (py_actions,))
                .ok()?;
            let tup = result.downcast::<PyTuple>().ok()?;

            // obs: move into cache (no clone) — framework eval loop reads via flat_observation_bytes().
            let raw_obs = self.extract_flat_obs(py, &tup.get_item(0).ok()?)?;
            let flat_obs = self.normalize_obs(raw_obs);
            *self.obs_cache.lock().unwrap() = flat_obs;

            // rewards: already float32 from EnvPool — tobytes directly
            let rew_bytes: Vec<u8> = tup
                .get_item(1).ok()?
                .call_method0("tobytes").ok()?
                .extract().ok()?;
            let rewards: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rew_bytes).to_vec();

            // terminated: needed for episode tracking
            let t_raw: Vec<u8> = tup.get_item(2).ok()?.call_method0("tobytes").ok()?.extract().ok()?;
            let terminated: Vec<bool> = t_raw.iter().map(|&t| t != 0).collect();

            // Obs return and truncated omitted: eval loop discards the full step result.
            // flat_observation_bytes() on the next iter reads from obs_cache above.
            Some((Vec::new(), None, rewards, terminated, Vec::new()))
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
        // max_episode_steps is a TimeLimit wrapper kwarg passed via wrappers in gymnasium 1.x
        // LunarLander-v3 has a built-in 500-step limit; no extra wrapper needed
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

/// Builds `gymnasium.make_vec("LunarLander-v2", num_envs=N, vectorization_mode="sync",
/// wrappers=[functools.partial(TimeLimit, max_episode_steps=500)])` — matches
/// scripts/sf_lunar_bench.py's `gym.make("LunarLander-v2", max_episode_steps=500)` exactly.
///
/// Uses `vectorization_mode="sync"` (single-process, sequential) rather than the default
/// "async" (multiprocessing fork): forking a process that already has tokio + rayon + LibTorch
/// threads running is unsafe, and sync mode measured faster anyway (23.7k vs 15.2k
/// env-frames/sec in isolated benchmarking with 64 envs).
pub fn make_sf_matched_lunar_lander_vec(
    num_envs: usize,
    obs_dim: usize,
    act_dim: usize,
) -> PyResult<PyVectorEnv> {
    Python::with_gil(|py| {
        let gym = py.import("gymnasium")?;
        let functools = py.import("functools")?;
        let time_limit_cls = py.import("gymnasium.wrappers")?.getattr("TimeLimit")?;
        let partial_kwargs = PyDict::new(py);
        partial_kwargs.set_item("max_episode_steps", 500)?;
        let time_limit_partial =
            functools.call_method("partial", (time_limit_cls,), Some(&partial_kwargs))?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("num_envs", num_envs as i64)?;
        kwargs.set_item("vectorization_mode", "sync")?;
        kwargs.set_item("wrappers", vec![time_limit_partial])?;
        let vec_env = gym.call_method("make_vec", ("LunarLander-v2",), Some(&kwargs))?;
        vec_env.call_method0("reset")?;
        Ok(PyVectorEnv::new(vec_env.unbind(), num_envs, obs_dim, act_dim, true))
    })
}

/// Create an EnvPool `LunarLander-v3` vec-env with `num_envs` parallel envs.
/// EnvPool releases the GIL during step and uses a C++ thread pool internally.
pub fn make_envpool_lunar_lander_vec(
    num_envs: usize,
    obs_dim: usize,
    act_dim: usize,
) -> PyResult<EnvPoolVecEnv> {
    Python::with_gil(|py| {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let ep = py.import("envpool")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("env_type", "gymnasium")?;
        kwargs.set_item("num_envs", num_envs as i64)?;
        kwargs.set_item("seed", 0i64)?;
        kwargs.set_item("num_threads", num_threads as i64)?;
        let env = ep.call_method("make", ("LunarLander-v3",), Some(&kwargs))?;
        env.call_method0("reset")?;
        Ok(EnvPoolVecEnv::new(
            env.unbind(),
            num_envs,
            obs_dim,
            act_dim,
            true, // discrete
        ))
    })
}
