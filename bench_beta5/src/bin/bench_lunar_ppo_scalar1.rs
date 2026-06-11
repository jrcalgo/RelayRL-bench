//! bench_lunar_ppo_scalar1 — PPO convergence benchmark on LunarLander-discrete.
//!
//! Creates a [128, 128] MLP PPO kernel and trains on 64 parallel scalar LunarLander
//! environments via set_env + run_env_with_ppo. Prints per-epoch stats.
//!
//! Root-cause fix applied: ONNX input node was named "obs" but ORT inference sent
//! inputs as "input", causing silent fallback to all-zero logits (uniform policy).
//! After rename to "input" in onnx_builder.rs, ORT correctly uses trained weights.
//!
//! Hyperparameters: full-batch GAE (mini_batch_size=None), target_kl=0.05,
//! gamma=0.999, lam=0.98, ent_coef=0.01 — matches gridworld convergence config.
//!
//! Build & run:
//!   cargo build --release -p bench-beta5 --bin bench_lunar_ppo_scalar1
//!   ./target/release/bench_lunar_ppo_scalar1

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

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const MAX_STEPS: usize = 500;
const ENV_COUNT: u32 = 64;
// Hyperparameters: SB3 Zoo LunarLander-v2 base, adjusted for full-batch training.
const GAMMA: f32 = 0.999;
const LAM: f32 = 0.98;
const CLIP_RATIO: f32 = 0.2;
const PI_LR: f64 = 2.5e-4;
const VF_LR: f64 = 2.5e-4;
const TRAIN_PI_ITERS: u64 = 10;
const TRAIN_VF_ITERS: u64 = 10;
// 0.05: higher target_kl allows larger policy updates to escape local optima.
// SB3 Zoo exact value is 0.015, but that causes StopIter=1 (essentially no training)
// when the policy is near a local optimum. Relaxed to 0.05 to allow exploration.
const TARGET_KL: f32 = 0.05;
const TRAJ_PER_EPOCH: u64 = 128;
// 1,500,000 env-frames / 64 envs ≈ 23,438 loop steps — SB3 Zoo trains for ~1M frames
// to reach mean_ret ≥ 200; 1.5M provides clear convergence signal.
const TOTAL_STEPS: usize = 23_438;
// SB3 Zoo mini-batch size — matches the 64-sample batches from SB3 LunarLander-v2 config.
// With 128 traj × ~100 steps/ep = ~12,800 transitions/epoch and 10 pi_iters:
//   12,800 / 64 = 200 mini-batches per pi_iter → up to 2,000 grad steps/epoch.
//   target_kl=0.015 early stopping keeps it to ~3-5 pi_iters (~600-1,000 grad steps/epoch).
const MINI_BATCH_SIZE: usize = 64;
const BUFFER_SIZE: ReplayBufferSize = 100_000;

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make ORT dylib available for policy ONNX inference during rollouts.
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.26.0",
    );

    type B = NdArray;

    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let total_env_frames = TOTAL_STEPS * ENV_COUNT as usize;
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.5 — PPO — LunarLander discrete — {ENV_COUNT} envs");
    println!("  obs={OBS_DIM}  act={ACT_DIM}  MLP=[128,128]  max_steps={MAX_STEPS}");
    println!("  loop_steps={TOTAL_STEPS}  env-frames={total_env_frames}");
    println!("  gamma={GAMMA}  lam={LAM}  clip={CLIP_RATIO}  pi_lr={PI_LR}  vf_lr={VF_LR}");
    println!("  pi_iters={TRAIN_PI_ITERS}  vf_iters={TRAIN_VF_ITERS}  target_kl={TARGET_KL}  ent_coef=0.05  traj/epoch={TRAJ_PER_EPOCH}  mb={MINI_BATCH_SIZE}");
    println!("  {num_cores} logical cores");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── PPO networks: [128, 128] MLPs for obs=8, act=4 ──────────────────────
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
    // Seed the actor with an ONNX export of the freshly-initialized policy so
    // `new_actor` doesn't try (and fail) to load a model from local_model_path.
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
        env_dir: PathBuf::from("./envs/lunar_ppo"),
        save_model_path: PathBuf::from("./models/lunar_ppo"),
        obs_dim: OBS_DIM,
        obs_dtype,
        act_dim: ACT_DIM,
        act_dtype,
        buffer_size: BUFFER_SIZE,
        device: DeviceType::Cpu,
    };
    let hyperparams = IPPOParams {
        discrete: true,
        gamma: GAMMA,
        lam: LAM,
        clip_ratio: CLIP_RATIO,
        pi_lr: PI_LR as f32,
        vf_lr: VF_LR as f32,
        train_pi_iters: TRAIN_PI_ITERS,
        train_vf_iters: TRAIN_VF_ITERS,
        target_kl: TARGET_KL,
        traj_per_epoch: TRAJ_PER_EPOCH,
        ent_coef: 0.05, // 5x SB3 default to prevent entropy collapse before landing is discovered
        max_episode_steps: Some(MAX_STEPS),
        minibatch: Some(MINI_BATCH_SIZE), // SB3: 64-sample batches, KL early stop
        ..Default::default()
    };
    let trainer_spec = PPOTrainerSpec::ppo(
        trainer_args,
        Some(hyperparams),
        PPONetworkArgs { pi_head, vf_mlp },
    );

    // ── Agent setup ─────────────────────────────────────────────────────────
    let config_path = PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B>::builder()
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .router_scale(1);
    if config_path.exists() {
        builder = builder.config_path(config_path);
    }

    let (mut agent, params) = builder.build().await?;
    agent.start(params).await?;
    agent
        .new_actor::<OBS_DIM, ACT_DIM>(DeviceType::Cpu, MAX_STEPS, initial_model)
        .await?;
    let actor_ids = agent.get_actor_ids()?;
    let actor_id = actor_ids[0];

    // ── Environment ──────────────────────────────────────────────────────────
    let env = LunarLanderEnv::<B>::new(MAX_STEPS, Default::default());
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!(
        "set_env OK — registered {} LunarLander env with actor {}\n",
        ENV_COUNT, actor_id
    );

    // ── PPO training ─────────────────────────────────────────────────────────
    println!("Starting PPO training ({TOTAL_STEPS} steps)...\n");
    println!(
        "{:>12}  {:>8}  {:>14}  {:>10}",
        "epoch", "episodes", "mean_ret(100)", "last_ep"
    );
    println!("{}", "─".repeat(52));

    let t0 = Instant::now();
    agent
        .run_env_with_ppo::<Float, Float, _>(actor_id, TOTAL_STEPS, MAX_STEPS, trainer_spec)
        .await?;
    let wall = t0.elapsed().as_secs_f64();

    // ── Final stats ──────────────────────────────────────────────────────────
    let loop_steps_per_sec = TOTAL_STEPS as f64 / wall;
    let env_frames_per_sec = loop_steps_per_sec * ENV_COUNT as f64;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  PPO training complete");
    println!("  wall time         : {:.2}s", wall);
    println!("  loop steps/sec    : {:.0}  (each step = {} env transitions)", loop_steps_per_sec, ENV_COUNT);
    println!("  env-frames/sec    : {:.0}", env_frames_per_sec);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
