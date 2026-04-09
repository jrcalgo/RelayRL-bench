use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, LocalTrajectoryFileParams,
    LocalTrajectoryFileType, ModelMode, RelayRLAgentActors,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;

use relayrl_algorithms::algorithms::PPO::{IPPOParams, PPOPolicyWithBaseline};
use relayrl_algorithms::algorithms::REINFORCE::ActivationKind;
use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;
use relayrl_algorithms::{AlgorithmTrait, PpoTrainer, RelayRLTrainer, TrainerArgs};

use relayrl_types::data::tensor::{DType, NdArrayDType};
use relayrl_types::model::{ModelFileType, ModelMetadata};
use relayrl_types::prelude::records::ArrowTrajectory;

use std::path::PathBuf;
use std::sync::Mutex;

// ─── Type aliases ────────────────────────────────────────────────────────────

type B = NdArray;
type AgentT = relayrl_framework::prelude::network::RelayRLAgent<B, 2, 2, Float, Float>;
type TrainerT = PpoTrainer<B, Float, Float, PPOPolicyWithBaseline<B, Float, Float>>;
type ActorUuid = uuid::Uuid;

// ─── Bootstrap model helper ──────────────────────────────────────────────────

fn make_bootstrap_model(
    obs_dim: usize,
    act_dim: usize,
) -> Result<ModelModule<B>, Box<dyn std::error::Error>> {
    let layer_specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = vec![
        (obs_dim, 64, vec![0.01f32; 64 * obs_dim], vec![0.0f32; 64]),
        (64, 64, vec![0.01f32; 64 * 64], vec![0.0f32; 64]),
        (64, act_dim, vec![0.01f32; act_dim * 64], vec![0.0f32; act_dim]),
    ];
    let onnx_bytes = build_onnx_mlp_bytes(&layer_specs);
    let metadata = ModelMetadata {
        model_file: "bootstrap.onnx".to_string(),
        model_type: ModelFileType::Onnx,
        input_dtype: DType::NdArray(NdArrayDType::F32),
        output_dtype: DType::NdArray(NdArrayDType::F32),
        input_shape: vec![1, obs_dim],
        output_shape: vec![1, act_dim],
        default_device: Some(relayrl_types::data::tensor::DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(onnx_bytes, metadata)?)
}

// ─── Inner state ─────────────────────────────────────────────────────────────

struct Inner {
    agent: AgentT,
    trainer: TrainerT,
    actor_ids: Vec<ActorUuid>,
    traj_dir: PathBuf,
    save_prefix: String,
    traj_per_epoch: usize,
    episodes_this_epoch: usize,
    obs_dim: usize,
    device: <B as burn_tensor::backend::Backend>::Device,
}

// ─── PyO3 class ──────────────────────────────────────────────────────────────

/// Python-accessible RelayRL IPPO agent for use with Python gym environments.
///
/// Usage::
///
///     agent = RelayRLPPOAgent(
///         obs_dim=8, act_dim=4,
///         model_path="./model_lunar",
///         traj_dir="./trajectories_pyo3",
///         save_model_dir="./trained_model_pyo3",
///         traj_per_epoch=8,
///     )
///     action = agent.get_action(obs.tolist())
///     trained = agent.end_episode(terminal_reward)
///     agent.shutdown()
#[pyclass]
pub struct RelayRLPPOAgent {
    /// Tokio runtime stored separately to allow split borrows with `inner`.
    runtime: tokio::runtime::Runtime,
    inner: Mutex<Inner>,
}

#[pymethods]
impl RelayRLPPOAgent {
    #[new]
    #[pyo3(signature = (obs_dim, act_dim, model_path, traj_dir, save_model_dir, traj_per_epoch=8))]
    fn new(
        obs_dim: usize,
        act_dim: usize,
        model_path: &str,
        traj_dir: &str,
        save_model_dir: &str,
        traj_per_epoch: usize,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;

        let model_path = PathBuf::from(model_path);
        let traj_dir_path = PathBuf::from(traj_dir);
        let save_dir_path = PathBuf::from(save_model_dir);
        let save_prefix = save_dir_path.to_string_lossy().into_owned();

        std::fs::create_dir_all(&traj_dir_path)
            .map_err(|e| PyRuntimeError::new_err(format!("create traj_dir: {e}")))?;
        std::fs::create_dir_all(&save_dir_path)
            .map_err(|e| PyRuntimeError::new_err(format!("create save_model_dir: {e}")))?;

        let device: <B as burn_tensor::backend::Backend>::Device = Default::default();

        // Load model — fall back to bootstrap weights if obs_dim doesn't match.
        let initial_model = {
            match ModelModule::<B>::load_from_path(&model_path) {
                Ok(m) => {
                    let saved_obs = m.metadata.input_shape.get(1).copied().unwrap_or(obs_dim);
                    if saved_obs != obs_dim {
                        make_bootstrap_model(obs_dim, act_dim)
                            .map_err(|e| PyRuntimeError::new_err(format!("bootstrap model: {e}")))?
                    } else {
                        m
                    }
                }
                Err(_) => make_bootstrap_model(obs_dim, act_dim)
                    .map_err(|e| PyRuntimeError::new_err(format!("bootstrap model: {e}")))?,
            }
        };

        let traj_params =
            LocalTrajectoryFileParams::new(traj_dir_path.clone(), LocalTrajectoryFileType::Arrow)
                .map_err(|e| PyRuntimeError::new_err(format!("traj params: {e}")))?;

        // Build and start the framework agent.
        let (agent, actor_ids) = runtime.block_on(async {
            let mut builder = AgentBuilder::<B, 2, 2, Float, Float>::builder()
                .actor_count(1)
                .default_device(DeviceType::Cpu)
                .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
                .actor_training_data_mode(ActorTrainingDataMode::Offline(Some(traj_params)))
                .default_model(initial_model);

            // Use config.json to cap max_traj_length and avoid the 100M-element default.
            let config_path = std::path::PathBuf::from("./config.json");
            if config_path.exists() {
                builder = builder.config_path(config_path);
            }

            let (mut agent, params) = builder
                .build()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("agent build: {e}")))?;
            agent
                .start(params)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("agent start: {e}")))?;
            let actor_ids = agent
                .get_actor_ids()
                .map_err(|e| PyRuntimeError::new_err(format!("get_actor_ids: {e}")))?;
            Ok::<_, PyErr>((agent, actor_ids))
        })?;

        // Build the IPPO trainer.
        // Set traj_per_epoch = u64::MAX to disable the auto-trigger inside
        // receive_trajectory; the training schedule is managed externally.
        let hyperparams = IPPOParams {
            traj_per_epoch: u64::MAX,
            ..Default::default()
        };
        let trainer_args = TrainerArgs {
            env_dir: traj_dir_path.clone(),
            save_model_path: save_dir_path,
            obs_dim,
            act_dim,
            buffer_size: traj_per_epoch * 1_000,
        };
        let kernel = PPOPolicyWithBaseline::<B, Float, Float>::new(
            obs_dim,
            act_dim,
            true,
            &[64, 64],
            ActivationKind::ReLU,
            3e-4,
            1e-3,
            &device,
        );
        let trainer = RelayRLTrainer::ppo::<B, Float, Float, _>(
            trainer_args,
            Some(hyperparams),
            kernel,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("trainer: {e}")))?;

        Ok(Self {
            runtime,
            inner: Mutex::new(Inner {
                agent,
                trainer,
                actor_ids,
                traj_dir: traj_dir_path,
                save_prefix,
                traj_per_epoch,
                episodes_this_epoch: 0,
                obs_dim,
                device,
            }),
        })
    }

    /// Request an action for the given observation vector.
    /// Returns the discrete action index (0–3 for LunarLander-v3).
    fn get_action(&self, obs: Vec<f32>) -> PyResult<i64> {
        let mut inner = self.inner.lock().unwrap();
        let obs_tensor = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs, [1, inner.obs_dim]),
            &inner.device,
        );
        let ids = inner.actor_ids.clone();
        let actions = self
            .runtime
            .block_on(inner.agent.request_action(ids, obs_tensor, None, 0.0))
            .map_err(|e| PyRuntimeError::new_err(format!("request_action: {e}")))?;

        let action = actions
            .first()
            .and_then(|(_, relay_action)| relay_action.get_act())
            .map(|act_data| {
                act_data
                    .data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            })
            .unwrap_or(0);

        Ok(action)
    }

    /// Signal the end of an episode with the terminal reward.
    /// Returns `True` when a training step was triggered (every `traj_per_epoch` episodes).
    fn end_episode(&self, terminal_reward: f32) -> PyResult<bool> {
        {
            let mut inner = self.inner.lock().unwrap();
            let ids = inner.actor_ids.clone();
            self.runtime
                .block_on(inner.agent.flag_last_action(ids, Some(terminal_reward)))
                .map_err(|e| PyRuntimeError::new_err(format!("flag_last_action: {e}")))?;

            inner.episodes_this_epoch += 1;
            if inner.episodes_this_epoch < inner.traj_per_epoch {
                return Ok(false);
            }
            // Reset counter before dropping the lock so train_step can re-lock.
            inner.episodes_this_epoch = 0;
        }

        self.train_step()?;
        Ok(true)
    }

    /// Shut down the underlying framework agent cleanly.
    fn shutdown(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        self.runtime
            .block_on(inner.agent.shutdown())
            .map_err(|e| PyRuntimeError::new_err(format!("shutdown: {e}")))?;
        Ok(())
    }
}

impl RelayRLPPOAgent {
    /// Internal: load collected Arrow files, train, push updated model, clean up.
    fn train_step(&self) -> PyResult<()> {
        // ── 1. Collect Arrow files ───────────────────────────────────────────
        let traj_files: Vec<PathBuf> = {
            let inner = self.inner.lock().unwrap();
            std::fs::read_dir(&inner.traj_dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("arrow"))
                        .collect()
                })
                .unwrap_or_default()
        };

        if traj_files.is_empty() {
            return Ok(());
        }

        // ── 2. Feed trajectories into the replay buffer ──────────────────────
        {
            let mut inner = self.inner.lock().unwrap();
            for file in &traj_files {
                match ArrowTrajectory::new(None).from_arrow(file, None, None) {
                    Ok(traj) => {
                        let _ = self.runtime.block_on(inner.trainer.receive_trajectory(traj));
                    }
                    Err(e) => eprintln!("relayrl_pyo3: skipping {}: {e}", file.display()),
                }
            }
        }

        // ── 3. Train ─────────────────────────────────────────────────────────
        // sample_buffer_blocking uses std::thread::scope so calling train_model
        // from inside a block_on future is safe — no nested block_on panic.
        {
            let mut inner = self.inner.lock().unwrap();
            let save_prefix = inner.save_prefix.clone();
            self.runtime.block_on(async {
                <TrainerT as AlgorithmTrait<ArrowTrajectory>>::train_model(&mut inner.trainer);
            });
            <TrainerT as AlgorithmTrait<ArrowTrajectory>>::log_epoch(&mut inner.trainer);
            <TrainerT as AlgorithmTrait<ArrowTrajectory>>::save(&inner.trainer, &save_prefix);
            // Reset trajectory counts so the auto-trigger guard stays quiet.
            inner.trainer.reset_epoch();
        }

        // ── 4. Push updated model weights to the inference agent ─────────────
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(model) = inner.trainer.acquire_model_module() {
                let ids = inner.actor_ids.clone();
                let _ = self
                    .runtime
                    .block_on(inner.agent.update_model(model, Some(ids)));
            }
        }

        // ── 5. Delete trajectory files to avoid re-processing ────────────────
        for file in &traj_files {
            let _ = std::fs::remove_file(file);
        }

        Ok(())
    }
}

// ─── Module registration ─────────────────────────────────────────────────────

#[pymodule]
fn relayrl_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RelayRLPPOAgent>()?;
    Ok(())
}
