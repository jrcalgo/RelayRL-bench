//! `relayrl_pyo3` — Python bindings for the RelayRL beta.2 framework.
//!
//! ## Build (maturin)
//!
//! ```bash
//! cd bench_beta2/relayrl-pyo3
//! maturin develop --release          # installs into the active venv
//! # or
//! maturin build --release            # produces a wheel in target/wheels/
//! ```
//!
//! ## Usage
//!
//! ```python
//! import gymnasium as gym
//! import relayrl_pyo3 as rl
//!
//! agent = rl.RelayRLAgent(obs_dim=8, act_dim=4, actor_count=1)
//! ids   = agent.get_actor_ids()
//!
//! # Scalar path — framework clones the env `count` times via the factory
//! agent.set_scalar_env(
//!     ids[0],
//!     factory  = lambda: gym.make("LunarLander-v3"),
//!     obs_dim  = 8,
//!     act_dim  = 4,
//!     discrete = True,
//!     count    = 32,
//! )
//! agent.run_env(ids[0], steps=50_000)
//! agent.shutdown()
//! ```

pub mod agent;
pub mod py_env;

use pyo3::prelude::*;
use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;

/// Build a bootstrap ONNX MLP and return its raw bytes.
///
/// The model is a 3-layer fully-connected network:
/// `obs_dim → hidden_dim → hidden_dim → act_dim`
/// with all weights initialised to 0.01 and biases to 0.
///
/// Useful for initialisation before a real trained model is available, or
/// for verifying that the pipeline is wired correctly.
///
/// Args:
///   obs_dim:    Input (observation) dimensionality.
///   act_dim:    Output (action) dimensionality.
///   hidden_dim: Hidden layer width (default 64).
///
/// Returns:
///   `bytes` — raw ONNX protobuf that can be passed as `onnx_bytes` to
///   `RelayRLAgent(...)`.
#[pyfunction]
#[pyo3(signature = (obs_dim, act_dim, hidden_dim=64))]
fn build_bootstrap_model_bytes(obs_dim: usize, act_dim: usize, hidden_dim: usize) -> Vec<u8> {
    let specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = vec![
        (obs_dim,    hidden_dim, vec![0.01f32; hidden_dim * obs_dim],    vec![0.0f32; hidden_dim]),
        (hidden_dim, hidden_dim, vec![0.01f32; hidden_dim * hidden_dim], vec![0.0f32; hidden_dim]),
        (hidden_dim, act_dim,    vec![0.01f32; act_dim    * hidden_dim], vec![0.0f32; act_dim]),
    ];
    build_onnx_mlp_bytes(&specs)
}

#[pymodule]
fn relayrl_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<agent::PyRelayRLAgent>()?;
    m.add_function(wrap_pyfunction!(build_bootstrap_model_bytes, m)?)?;
    Ok(())
}
