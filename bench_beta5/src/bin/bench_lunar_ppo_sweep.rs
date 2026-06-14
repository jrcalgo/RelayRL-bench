//! bench_lunar_ppo_sweep — configurable PPO hyperparameter-sweep harness for
//! LunarLander-discrete, derived from bench_lunar_ppo_scalar1.
//!
//! All hyperparameters and run isolation paths are controlled via env vars so the
//! same binary can be re-invoked many times across a sweep without code edits and
//! without state from a previous run leaking into the next (separate env_dir /
//! save_model_path per run, derived from SWEEP_RUN_ID).
//!
//! Env vars (all optional, defaults match bench_lunar_ppo_scalar1):
//!   SWEEP_RUN_ID          (default "default")  — isolates ./envs/<id> and ./models/<id>
//!   SWEEP_SEED            (default 12345)       — env RNG seed
//!   SWEEP_TOTAL_STEPS     (default 23438)
//!   SWEEP_GAMMA           (default 0.999)
//!   SWEEP_LAM             (default 0.98)
//!   SWEEP_CLIP_RATIO      (default 0.2)
//!   SWEEP_PI_LR           (default 2.5e-4)
//!   SWEEP_VF_LR           (default 2.5e-4)
//!   SWEEP_TRAIN_PI_ITERS  (default 10)
//!   SWEEP_TRAIN_VF_ITERS  (default 10)
//!   SWEEP_TARGET_KL       (default 0.05)
//!   SWEEP_TRAJ_PER_EPOCH  (default 128)
//!   SWEEP_ENT_COEF        (default 0.05)
//!   SWEEP_MINI_BATCH      (default 64)
//!   SWEEP_NORMALIZE_RETURNS (default 0/false)
//!
//! Build & run:
//!   cargo build --release -p bench-beta5 --bin bench_lunar_ppo_sweep
//!   SWEEP_RUN_ID=trial1 SWEEP_TOTAL_STEPS=300000 ./target/release/bench_lunar_ppo_sweep

use std::path::PathBuf;
use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::Float;

use relayrl_algorithms::TrainerArgs;
use relayrl_algorithms::algorithms::PPO::kernel::{DiscretePPOPolicyHead, PPOPolicyHead};
use relayrl_algorithms::algorithms::PPO::{IPPOParams, PPONetworkArgs, PPOTrainerSpec};
use relayrl_algorithms::algorithms::{ActivationKind, GenericMlp, WeightProvider, acquire_model_module};
use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode, RelayRLActorEnv,
    RelayRLAgentActors, ReplayBufferSize,
};
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;
use relayrl_types::data::tensor::{DType, NdArrayDType};

use lunarlander_rl::env::LunarLanderEnv;

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const MAX_STEPS: usize = 500;
const ENV_COUNT: u32 = 64;
const BUFFER_SIZE: ReplayBufferSize = 100_000;

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|s| matches!(s.as_str(), "1" | "true" | "True" | "TRUE"))
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.26.0",
    );

    type B = NdArray;

    let run_id = std::env::var("SWEEP_RUN_ID").unwrap_or_else(|_| "default".to_string());
    let seed = env_u64("SWEEP_SEED", 12345);
    let total_steps = env_usize("SWEEP_TOTAL_STEPS", 23_438);
    let gamma = env_f32("SWEEP_GAMMA", 0.999);
    let lam = env_f32("SWEEP_LAM", 0.98);
    let clip_ratio = env_f32("SWEEP_CLIP_RATIO", 0.2);
    let pi_lr = env_f64("SWEEP_PI_LR", 2.5e-4);
    let vf_lr = env_f64("SWEEP_VF_LR", 2.5e-4);
    let train_pi_iters = env_u64("SWEEP_TRAIN_PI_ITERS", 10);
    let train_vf_iters = env_u64("SWEEP_TRAIN_VF_ITERS", 10);
    let target_kl = env_f32("SWEEP_TARGET_KL", 0.05);
    let traj_per_epoch = env_u64("SWEEP_TRAJ_PER_EPOCH", 128);
    let ent_coef = env_f32("SWEEP_ENT_COEF", 0.05);
    let mini_batch = env_usize("SWEEP_MINI_BATCH", 64);
    let normalize_returns = env_bool("SWEEP_NORMALIZE_RETURNS", false);

    // Fresh, isolated state per run — avoid loading stale checkpoints/persisted
    // training-session state from a previous sweep trial.
    let run_root = PathBuf::from("./sweep_runs").join(&run_id);
    let _ = std::fs::remove_dir_all(&run_root);

    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let total_env_frames = total_steps * ENV_COUNT as usize;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.5 — PPO sweep — LunarLander discrete — run_id={run_id}");
    println!("  seed={seed}  loop_steps={total_steps}  env-frames={total_env_frames}");
    println!(
        "  gamma={gamma}  lam={lam}  clip={clip_ratio}  pi_lr={pi_lr}  vf_lr={vf_lr}  ent_coef={ent_coef}"
    );
    println!(
        "  pi_iters={train_pi_iters}  vf_iters={train_vf_iters}  target_kl={target_kl}  traj/epoch={traj_per_epoch}  mb={mini_batch}  normalize_returns={normalize_returns}"
    );
    println!("  {num_cores} logical cores");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let burn_device = <B as burn_tensor::backend::Backend>::Device::default();
    let obs_dtype = DType::NdArray(NdArrayDType::F32);
    let act_dtype = DType::NdArray(NdArrayDType::F32);

    let pi_mlp = GenericMlp::<B, Float, Float>::new(
        OBS_DIM,
        obs_dtype.clone(),
        &[128, 128],
        ACT_DIM,
        act_dtype.clone(),
        ActivationKind::ReLU(burn_nn::activation::Relu::new()),
        &burn_device,
    );
    let initial_model = acquire_model_module::<B>(
        "ppo_pi",
        pi_mlp.get_layer_specs(),
        obs_dtype.clone(),
        act_dtype.clone(),
        vec![1, OBS_DIM],
        vec![1, ACT_DIM],
        Some(DeviceType::Cpu),
    );
    let pi_head = PPOPolicyHead::Discrete(DiscretePPOPolicyHead::new(pi_mlp)?);
    let vf_mlp = GenericMlp::<B, Float, Float>::new(
        OBS_DIM,
        obs_dtype.clone(),
        &[128, 128],
        1,
        DType::NdArray(NdArrayDType::F32),
        ActivationKind::ReLU(burn_nn::activation::Relu::new()),
        &burn_device,
    );

    let trainer_args = TrainerArgs {
        env_dir: run_root.join("envs"),
        save_model_path: run_root.join("models"),
        obs_dim: OBS_DIM,
        obs_dtype,
        act_dim: ACT_DIM,
        act_dtype,
        buffer_size: BUFFER_SIZE,
        device: DeviceType::Cpu,
    };
    let hyperparams = IPPOParams {
        discrete: true,
        gamma,
        lam,
        clip_ratio,
        pi_lr: pi_lr as f32,
        vf_lr: vf_lr as f32,
        train_pi_iters,
        train_vf_iters,
        target_kl,
        traj_per_epoch,
        ent_coef,
        max_episode_steps: Some(MAX_STEPS),
        minibatch: Some(mini_batch),
        normalize_returns,
        ..Default::default()
    };
    let trainer_spec = PPOTrainerSpec::ppo(
        trainer_args,
        Some(hyperparams),
        PPONetworkArgs { pi_head, vf_mlp },
    );

    let builder = AgentBuilder::<B>::builder()
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .router_scale(1);
    // Sweep runs intentionally avoid ./config.json so each trial uses identical
    // transport defaults regardless of the working directory's config.

    let (mut agent, params) = builder.build().await?;
    agent.start(params).await?;
    agent
        .new_actor::<OBS_DIM, ACT_DIM>(DeviceType::Cpu, MAX_STEPS, initial_model)
        .await?;
    let actor_ids = agent.get_actor_ids()?;
    let actor_id = actor_ids[0];

    let env = LunarLanderEnv::<B>::new_with_seed(MAX_STEPS, Default::default(), seed);
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!("set_env OK — registered {} LunarLander env with actor {}\n", ENV_COUNT, actor_id);

    println!("Starting PPO training ({total_steps} steps)...\n");

    let t0 = Instant::now();
    agent
        .run_env_with_ppo::<Float, Float, _>(actor_id, total_steps, MAX_STEPS, trainer_spec)
        .await?;
    let wall = t0.elapsed().as_secs_f64();

    let loop_steps_per_sec = total_steps as f64 / wall;
    let env_frames_per_sec = loop_steps_per_sec * ENV_COUNT as f64;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  PPO sweep run complete — run_id={run_id}");
    println!("  wall time         : {:.2}s", wall);
    println!("  loop steps/sec    : {:.0}  (each step = {} env transitions)", loop_steps_per_sec, ENV_COUNT);
    println!("  env-frames/sec    : {:.0}", env_frames_per_sec);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
