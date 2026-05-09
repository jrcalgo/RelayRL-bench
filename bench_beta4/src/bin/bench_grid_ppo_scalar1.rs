//! bench_grid_ppo_scalar1 — PPO convergence benchmark on a 5×5 discrete GridWorld.
//!
//! Simple environment to verify that the PPO implementation converges.
//! Agent starts at (0,0) and must reach goal (4,4) in a 5×5 grid.
//! Observation: 25-dim one-hot position. Actions: Up/Down/Left/Right.
//! Reward: +1.0 at goal, -0.01/step otherwise. Max 100 steps/episode.
//!
//! Run-2 structural fix — full-batch training (matches RL4Sys/spinning-up):
//!   mini_batch_size: None → 1 gradient step per pi_iter (all data at once).
//!   KL early stopping across pi_iters controls effective count (typically 3-7 steps).
//!   This prevents entropy collapse: fewer gradient steps, larger per-step KL signal.
//!   Compared to run-1: 216 gradient steps/epoch → 3-7 steps/epoch.
//!
//! Build & run:
//!   cargo build --release -p bench-beta4 --bin bench_grid_ppo_scalar1
//!   ./target/release/bench_grid_ppo_scalar1

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
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;

use gridworld_bench::GridWorldBenchEnv;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = gridworld_bench::OBS_DIM;   // 25 (5×5 one-hot)
const ACT_DIM: usize = gridworld_bench::ACT_DIM;   // 4
const MAX_STEPS: usize = gridworld_bench::MAX_STEPS; // 100
const ENV_COUNT: u32 = 64;

// Hyperparameters — tuned to prevent entropy collapse seen in LunarLander run-16.
const GAMMA: f32 = 0.99;
const LAM: f32 = 0.95;
const CLIP_RATIO: f32 = 0.2;
const PI_LR: f64 = 3e-4;
const VF_LR: f64 = 3e-4;
// 10 pi_iters; KL early stopping across iters controls effective count (3-7 full-batch steps).
const TRAIN_PI_ITERS: u64 = 10;
const TRAIN_VF_ITERS: u64 = 10;
// target_kl=0.05: each full-batch step produces larger KL than mini-batch; KL stops at 3-7 iters.
const TARGET_KL: f32 = 0.05;
const TRAJ_PER_EPOCH: u64 = 64;
// Enough steps for well over 100 epochs (each epoch ~20 loop steps at avg EpLen≈20).
const TOTAL_STEPS: usize = 50_000;
const BUFFER_SIZE: ReplayBufferSize = 50_000;

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.25.0",
    );

    type B = NdArray;

    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let total_env_frames = TOTAL_STEPS * ENV_COUNT as usize;
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.4 — PPO — 5×5 GridWorld — {ENV_COUNT} envs");
    println!("  obs={OBS_DIM} (one-hot)  act={ACT_DIM}  MLP=[64,64]  max_steps={MAX_STEPS}");
    println!("  loop_steps={TOTAL_STEPS}  env-frames={total_env_frames}");
    println!("  gamma={GAMMA}  lam={LAM}  clip={CLIP_RATIO}  pi_lr={PI_LR}  vf_lr={VF_LR}");
    println!("  pi_iters={TRAIN_PI_ITERS}  vf_iters={TRAIN_VF_ITERS}  target_kl={TARGET_KL}  ent_coef=0.01  traj/epoch={TRAJ_PER_EPOCH}  mb=FULL-BATCH");
    println!("  Run-2 fix: full-batch (mini_batch_size=None) = 1 grad step/pi_iter; KL stops at 3-7 iters");
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
    let env = GridWorldBenchEnv::new();
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!(
        "set_env OK — registered {} GridWorld envs with actor {}\n",
        ENV_COUNT, actor_id
    );

    // ── PPO kernel: [64, 64] MLP for obs=25, act=4 ──────────────────────────
    let burn_device = <B as burn_tensor::backend::Backend>::Device::default();
    let kernel = PPOKernel::<B, Float, Float>::new(
        OBS_DIM,
        ACT_DIM,
        true, // discrete actions
        &[64, 64],
        ActivationKind::ReLU,
        PI_LR,
        VF_LR,
        &burn_device,
    );

    // ── PPO training ─────────────────────────────────────────────────────────
    println!("Starting PPO training ({TOTAL_STEPS} loop steps)...\n");
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
                ent_coef: 0.01,
                max_episode_steps: Some(MAX_STEPS),
                mini_batch_size: None, // full-batch: 1 grad step per pi_iter
                ..Default::default()
            })),
            SaveModelPath::from("./models/grid_ppo"),
            BUFFER_SIZE,
            DeviceType::Cpu,
            kernel,
        )
        .await?;
    let wall = t0.elapsed().as_secs_f64();

    // ── Final stats ──────────────────────────────────────────────────────────
    let loop_steps_per_sec = TOTAL_STEPS as f64 / wall;
    let env_frames_per_sec = loop_steps_per_sec * ENV_COUNT as f64;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  PPO GridWorld training complete");
    println!("  wall time         : {:.2}s", wall);
    println!("  loop steps/sec    : {:.0}  (each step = {} env transitions)", loop_steps_per_sec, ENV_COUNT);
    println!("  env-frames/sec    : {:.0}", env_frames_per_sec);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
