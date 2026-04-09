pub mod environments;
use environments::build_gridworld_env;

use std::path::PathBuf;

use clap::Parser;
use futures::future::join_all;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, LocalTrajectoryFileParams,
    LocalTrajectoryFileType, ModelMode, RelayRLAgentActors, ToAnyBurnTensor,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::PPO::PPOPolicyWithBaseline;
use relayrl_algorithms::algorithms::REINFORCE::ActivationKind;
use relayrl_algorithms::{AlgorithmTrait, MultiagentTrainer, RelayRLTrainer, TrainerArgs};

use relayrl_types::prelude::records::{ArrowTrajectory, CsvTrajectory};

// ─────────────────────────────── CLI ────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "relayrl-e2e",
    about = "RelayRL offline training loop with GridWorld"
)]
struct Args {
    /// Number of actors (agents) in the environment
    #[arg(long, default_value_t = 1)]
    actor_count: usize,

    /// Model mode: "independent" or "shared"
    #[arg(long, default_value = "independent")]
    model_mode: String,

    /// Algorithm: "ppo" (default) or "reinforce"
    #[arg(long, default_value = "ppo")]
    algorithm: String,

    /// Path to the initial model directory (must contain metadata.json)
    #[arg(long, default_value = "./model")]
    model_path: PathBuf,

    /// Directory for trajectory output files
    #[arg(long, default_value = "./trajectories")]
    traj_dir: PathBuf,

    /// Directory to save trained model checkpoints
    #[arg(long, default_value = "./trained_model")]
    save_model_path: PathBuf,

    /// Grid size (length = width)
    #[arg(long, default_value_t = 10)]
    grid_size: usize,

    /// Maximum steps per episode
    #[arg(long, default_value_t = 200)]
    max_steps: usize,

    /// Number of training epochs
    #[arg(long, default_value_t = 100)]
    num_epochs: usize,

    /// Number of episodes collected per training epoch
    #[arg(long, default_value_t = 8)]
    traj_per_epoch: usize,

    /// Trajectory file format: "arrow" (default) or "csv"
    #[arg(long, default_value = "arrow")]
    traj_file_type: String,

    /// Compute backend: "ndarray" (CPU), "tch-cpu", "tch-cuda", or "tch-mps"
    #[arg(long, default_value = "ndarray")]
    backend: String,

    /// CUDA device index (only used when backend = "tch-cuda")
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,

    /// Number of router tasks (parallelism for action dispatch)
    #[arg(long, default_value_t = 1)]
    router_scale: u32,

    /// Disable trajectory collection (no file writes, inference-only)
    #[arg(long, default_value_t = false)]
    disable_traj: bool,
}

// ─────────────────────────────── Entry point ─────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.backend.as_str() {
        "ndarray" => run_training_loop::<NdArray>(args, DeviceType::Cpu).await?,
        #[cfg(feature = "tch")]
        "tch-cpu" => {
            run_training_loop::<burn_tch::LibTorch<f32>>(args, DeviceType::Cpu).await?
        }
        #[cfg(feature = "tch")]
        "tch-cuda" => {
            let dev = DeviceType::Cuda(args.cuda_device);
            run_training_loop::<burn_tch::LibTorch<f32>>(args, dev).await?
        }
        #[cfg(feature = "tch")]
        "tch-mps" => {
            run_training_loop::<burn_tch::LibTorch<f32>>(args, DeviceType::Mps).await?
        }
        other => {
            eprintln!(
                "Unknown backend '{}'. Supported: ndarray, tch-cpu, tch-cuda, tch-mps",
                other
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

// ─────────────────── Helper: train on Arrow trajectory files ─────────────────

async fn train_on_arrow<Tr: AlgorithmTrait<ArrowTrajectory>>(
    trainer: &mut Tr,
    traj_files: &[PathBuf],
    save_prefix: &str,
) {
    for file in traj_files {
        match ArrowTrajectory::new(None).from_arrow(file, None, None) {
            Ok(traj) => {
                let _ = trainer.receive_trajectory(traj).await;
            }
            Err(e) => eprintln!("Warning: could not read Arrow {}: {}", file.display(), e),
        }
    }
    // train_model is sync but calls Handle::block_on internally; use
    // block_in_place so we don't panic inside the Tokio runtime.
    tokio::task::block_in_place(|| trainer.train_model());
    trainer.log_epoch();
    trainer.save(save_prefix);
}

// ─────────────────── Helper: train on CSV trajectory files ───────────────────

async fn train_on_csv<Tr: AlgorithmTrait<CsvTrajectory>>(
    trainer: &mut Tr,
    traj_files: &[PathBuf],
    save_prefix: &str,
) {
    for file in traj_files {
        match CsvTrajectory::new(None).from_csv(file, 65_536, None, None) {
            Ok(traj) => {
                let _ = trainer.receive_trajectory(traj).await;
            }
            Err(e) => eprintln!("Warning: could not read CSV {}: {}", file.display(), e),
        }
    }
    tokio::task::block_in_place(|| trainer.train_model());
    trainer.log_epoch();
    trainer.save(save_prefix);
}

// ─────────────────────────────── Training loop ───────────────────────────────

async fn run_training_loop<B>(
    args: Args,
    device_type: DeviceType,
) -> Result<(), Box<dyn std::error::Error>>
where
    B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>,
    B::Device: Clone + Default,
    Tensor<B, 2, Float>: ToAnyBurnTensor<B, 2>,
{
    let obs_dim = args.grid_size * args.grid_size;
    let act_dim: usize = 4;

    // ── Burn device ─────────────────────────────────────────────────────────
    let device: B::Device = B::get_device(&device_type).unwrap_or_default();

    // ── GridWorld environment ────────────────────────────────────────────────
    let env = build_gridworld_env::<B>(
        args.actor_count,
        args.grid_size,
        args.max_steps,
        device.clone(),
    )?;

    // ── Trajectory sink configuration ───────────────────────────────────────
    std::fs::create_dir_all(&args.traj_dir)?;
    std::fs::create_dir_all(&args.save_model_path)?;

    let traj_file_type = match args.traj_file_type.as_str() {
        "csv" => LocalTrajectoryFileType::Csv,
        _ => LocalTrajectoryFileType::Arrow,
    };
    let traj_params =
        LocalTrajectoryFileParams::new(args.traj_dir.clone(), traj_file_type.clone())?;

    // ── Agent construction ───────────────────────────────────────────────────
    let model_mode = if args.model_mode == "shared" {
        ModelMode::Shared
    } else {
        ModelMode::Independent
    };

    let initial_model = ModelModule::<B>::load_from_path(&args.model_path)?;

    // Use a local config.json to cap max_traj_length and avoid the 100M-element
    // default allocation.  Fall back gracefully if the file doesn't exist.
    let config_path = PathBuf::from("./config.json");

    let training_data_mode = if args.disable_traj {
        ActorTrainingDataMode::Disabled
    } else {
        ActorTrainingDataMode::Offline(Some(traj_params))
    };

    let builder = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(args.actor_count as u32)
        .default_device(device_type.clone())
        .actor_inference_mode(ActorInferenceMode::Local(model_mode))
        .actor_training_data_mode(training_data_mode)
        .default_model(initial_model)
        .router_scale(args.router_scale);

    let builder = if config_path.exists() {
        builder.config_path(config_path)
    } else {
        builder
    };

    let (mut agent, params) = builder.build().await?;

    agent.start(params).await?;
    let actor_ids = agent.get_actor_ids()?;

    // ── Algorithm / trainer setup ────────────────────────────────────────────
    let trainer_args = TrainerArgs {
        env_dir: args.traj_dir.clone(),
        save_model_path: args.save_model_path.clone(),
        obs_dim,
        act_dim,
        buffer_size: args.traj_per_epoch * args.max_steps,
    };
    let save_prefix = args
        .save_model_path
        .to_str()
        .unwrap_or("./trained_model")
        .to_owned();
    let use_csv = args.traj_file_type == "csv";
    let expected_ext = if use_csv { "csv" } else { "arrow" };

    // ── Inner epoch loop body shared across trainer variants ─────────────────
    //
    // The macro avoids code duplication: data collection and model-update are
    // identical regardless of which algorithm variant is active.  The training
    // phase dispatches to the type-explicit helpers `train_on_arrow` /
    // `train_on_csv` so the compiler can resolve which `AlgorithmTrait<T>`
    // implementation to call.
    macro_rules! training_loop {
        ($trainer:ident) => {{
            for epoch in 0..args.num_epochs {
                // ── Phase 1 : Clear stale trajectory files from previous epochs
                // to prevent stale UUIDs from creating default-initialized agent
                // slots (obs_dim=1) in the algorithm's registry.
                if let Ok(entries) = std::fs::read_dir(&args.traj_dir) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.extension().and_then(|e| e.to_str()) == Some(expected_ext) {
                            let _ = std::fs::remove_file(&p);
                        }
                    }
                }

                // ── Phase 2 : Data collection ────────────────────────────────
                let mut episodes_done = 0usize;
                while episodes_done < args.traj_per_epoch {
                    env.reset();

                    'episode: loop {
                        // Collect all observations upfront (synchronous env access).
                        let obs_vecs: Vec<Vec<f32>> = (0..env.actor_count())
                            .map(|i| env.get_observation(i))
                            .collect();

                        // Fire all action requests concurrently.
                        let action_futures = actor_ids.iter().zip(obs_vecs.iter()).map(|(id, obs_vec)| {
                            let obs_tensor = Tensor::<B, 2, Float>::from_data(
                                TensorData::new(obs_vec.clone(), [1, obs_dim]),
                                &device,
                            );
                            agent.request_action(vec![*id], obs_tensor, None, 0.0)
                        });
                        let all_results = join_all(action_futures).await;

                        // Process results into step_actions.
                        let mut step_actions: Vec<u8> = vec![0u8; env.actor_count()];
                        for (i, result) in all_results.into_iter().enumerate() {
                            if let Ok(actions) = result {
                                if let Some((_, relay_action)) = actions.first() {
                                    // The ONNX model outputs raw logits [1, act_dim].
                                    // Take argmax to get discrete action index 0-3.
                                    let action_u8 = relay_action
                                        .get_act()
                                        .map(|act_data| {
                                            act_data
                                                .data
                                                .chunks_exact(4)
                                                .map(|b| {
                                                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                                                })
                                                .enumerate()
                                                .max_by(|(_, a), (_, b)| {
                                                    a.partial_cmp(b)
                                                        .unwrap_or(std::cmp::Ordering::Equal)
                                                })
                                                .map(|(idx, _)| idx as u8)
                                                .unwrap_or(0)
                                        })
                                        .unwrap_or(0);
                                    step_actions[i] = action_u8;
                                }
                            }
                        }

                        // Step the environment for every actor.
                        for (i, &action) in step_actions.iter().enumerate() {
                            let _ = env.step(i, action);
                        }

                        // Check for episode termination.
                        if env.all_done() || env.is_max_steps_reached() {
                            let actor_n = env.actor_count() as f32;
                            let avg_terminal_reward: f32 =
                                (0..env.actor_count())
                                    .map(|i| env.get_last_reward(i))
                                    .sum::<f32>()
                                    / actor_n;

                            agent
                                .flag_last_action(actor_ids.clone(), Some(avg_terminal_reward))
                                .await?;

                            episodes_done += 1;
                            break 'episode;
                        }
                    }
                }

                // ── Phase 2 : Train on collected trajectories ────────────────
                let traj_files: Vec<PathBuf> = std::fs::read_dir(&args.traj_dir)
                    .map(|rd| {
                        rd.filter_map(|e| e.ok().map(|e| e.path()))
                            .filter(|p| {
                                p.extension()
                                    .and_then(|x| x.to_str())
                                    .map_or(false, |x| x == expected_ext)
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if use_csv {
                    train_on_csv(&mut $trainer, &traj_files, &save_prefix).await;
                } else {
                    train_on_arrow(&mut $trainer, &traj_files, &save_prefix).await;
                }

                // Reset per-agent trajectory counters after our manual train step
                // so that receive_trajectory doesn't auto-call train_model (which
                // uses Handle::block_on and panics inside the Tokio runtime).
                $trainer.reset_epoch();

                // ── Phase 3 : Model update ───────────────────────────────────
                // Prefer in-memory weight export so no disk I/O is required.
                // Fall back to the original model path if the trainer has no
                // weights yet (empty replay buffer on epoch 0).
                let updated_model = $trainer
                    .acquire_model_module()
                    .or_else(|| ModelModule::<B>::load_from_path(&args.model_path).ok());
                match updated_model {
                    Some(new_model) => {
                        let _ = agent
                            .update_model(new_model, Some(actor_ids.clone()))
                            .await;
                    }
                    None => eprintln!("Warning: could not acquire updated model weights"),
                }

                // Clean up trajectory files so they are not replayed next epoch.
                for file in &traj_files {
                    let _ = std::fs::remove_file(file);
                }

                println!("Epoch {}/{} complete", epoch + 1, args.num_epochs);
            }
        }};
    }

    // ── Dispatch on model-mode × algorithm ──────────────────────────────────
    match (args.model_mode.as_str(), args.algorithm.as_str()) {
        ("shared", "reinforce") => {
            let mut trainer =
                MultiagentTrainer::<B, Float, Float>::mareinforce(trainer_args, None)?;
            training_loop!(trainer);
        }
        ("shared", _) => {
            let mut trainer =
                MultiagentTrainer::<B, Float, Float>::mappo(trainer_args, None)?;
            training_loop!(trainer);
        }
        (_, _) => {
            // Independent PPO (default; "reinforce" flag also falls here since
            // PolicyWithBaseline does not implement Default in 0.1.0).
            let kernel = PPOPolicyWithBaseline::<B, Float, Float>::new(
                obs_dim,
                act_dim,
                true, // discrete actions
                &[64, 64],
                ActivationKind::ReLU,
                3e-4, // policy learning rate
                1e-3, // value-function learning rate
                &device,
            );
            let mut trainer =
                RelayRLTrainer::ppo::<B, Float, Float, _>(trainer_args, None, kernel)?;
            training_loop!(trainer);
        }
    }

    agent.shutdown().await?;
    Ok(())
}
