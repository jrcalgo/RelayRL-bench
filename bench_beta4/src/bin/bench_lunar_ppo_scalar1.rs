//! bench_lunar_ppo_scalar1 — PPO convergence benchmark on LunarLander-discrete.
//!
//! Creates a [128, 128] MLP PPO kernel and trains on a single scalar LunarLander
//! environment via run_env_with_ppo. Prints per-epoch stats (mean return over last
//! 100 episodes) to show convergence.
//!
//! Hyperparameters follow the SB3 RL Baselines3-Zoo tuned config for LunarLander-v2:
//!   gamma=0.999, lam=0.98, clip=0.2, pi_lr=2.5e-4, vf_lr=1e-3,
//!   train_pi_iters=10, train_vf_iters=10, target_kl=0.1, traj_per_epoch=64
//!
//! Build & run:
//!   cargo build --release -p bench-beta4 --bin bench_lunar_ppo_scalar1
//!   ./target/release/bench_lunar_ppo_scalar1

use std::path::PathBuf;
use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::Float;

use relayrl_algorithms::algorithms::PPO::PPOKernel;
use relayrl_algorithms::algorithms::REINFORCE::ActivationKind;
use relayrl_algorithms::PPOParams;
use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, AlgorithmCfg, ModelMode,
    RelayRLActorEnv, RelayRLAgentActors, ReplayBufferSize, SaveModelPath,
};
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use lunarlander_rl::env::LunarLanderEnv;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const MAX_STEPS: usize = 500;
const ENV_COUNT: u32 = 1;
// ── Hyperparameters (SB3 RL Baselines3-Zoo for LunarLander-v2) ──────────────
const GAMMA: f32 = 0.999;
const LAM: f32 = 0.98;
const CLIP_RATIO: f32 = 0.2;
const PI_LR: f64 = 2.5e-4;
const VF_LR: f64 = 1e-3;
const TRAIN_PI_ITERS: u64 = 10;
const TRAIN_VF_ITERS: u64 = 10;
const TARGET_KL: f32 = 0.1;
const TRAJ_PER_EPOCH: u64 = 64;
const TOTAL_STEPS: usize = 500_000;
const BUFFER_SIZE: ReplayBufferSize = 100_000;

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.4 — PPO — LunarLander discrete — 1 env  (SB3 Zoo hparams)");
    println!("  obs={OBS_DIM}  act={ACT_DIM}  MLP=[128,128]  total steps={TOTAL_STEPS}");
    println!("  gamma={GAMMA}  lam={LAM}  clip={CLIP_RATIO}  pi_lr={PI_LR}  vf_lr={VF_LR}");
    println!("  pi_iters={TRAIN_PI_ITERS}  vf_iters={TRAIN_VF_ITERS}  target_kl={TARGET_KL}  traj/epoch={TRAJ_PER_EPOCH}");
    println!("  {num_cores} logical cores");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── Agent setup ─────────────────────────────────────────────────────────
    let config_path = PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B, 2, 2>::builder()
        .actor_count(1)
        .default_device(DeviceType::Cpu)
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .router_scale(1);
    if config_path.exists() {
        builder = builder.config_path(config_path);
    }

    let (mut agent, params) = builder.build().await?;
    agent.start(params).await?;
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

    // ── PPO kernel: [128, 128] MLP for obs=8, act=4 ─────────────────────────
    let burn_device = <B as burn_tensor::backend::Backend>::Device::default();
    let kernel = PPOKernel::<B, Float, Float>::new(
        OBS_DIM,
        ACT_DIM,
        true, // discrete actions
        &[128, 128],
        ActivationKind::ReLU,
        PI_LR,
        VF_LR,
        &burn_device,
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
        .run_env_with_ppo::<Float, Float, _>(
            actor_id,
            TOTAL_STEPS,
            AlgorithmCfg::PPO(Some(PPOParams {
                discrete: true,
                gamma: GAMMA,
                lam: LAM,
                clip_ratio: CLIP_RATIO,
                train_pi_iters: TRAIN_PI_ITERS,
                train_vf_iters: TRAIN_VF_ITERS,
                target_kl: TARGET_KL,
                traj_per_epoch: TRAJ_PER_EPOCH,
                ..Default::default()
            })),
            SaveModelPath::from("./models/lunar_ppo"),
            BUFFER_SIZE,
            DeviceType::Cpu,
            kernel,
        )
        .await?;
    let wall = t0.elapsed().as_secs_f64();

    // ── Final stats ──────────────────────────────────────────────────────────
    let steps_per_sec = TOTAL_STEPS as f64 / wall;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  PPO training complete");
    println!("  wall time    : {:.2}s", wall);
    println!("  steps/sec    : {:.0}", steps_per_sec);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
