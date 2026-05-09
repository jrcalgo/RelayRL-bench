//! bench_lunar_ppo_scalar1 — PPO convergence benchmark on LunarLander-discrete.
//!
//! Creates a [128, 128] MLP PPO kernel and trains on a single scalar LunarLander
//! environment via run_env_with_ppo. Prints per-epoch stats (mean return over last
//! 100 episodes) to show convergence.
//!
//! Hyperparameters follow the RL4Sys+SB3 config for LunarLander-v2:
//!   gamma=0.999, lam=0.98, clip=0.2, pi_lr=2.5e-4, vf_lr=2.5e-4,
//!   train_pi_iters=10, train_vf_iters=4, target_kl=0.015, traj_per_epoch=128, mb=64
//!   run-16: Per-mb KL stop + fresh logp_old from Burn (fixes async ORT staleness / StopIter=1 bug).
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
const ENV_COUNT: u32 = 64;
// ── Hyperparameters (SB3 RL Baselines3-Zoo for LunarLander-v2) ──────────────
const GAMMA: f32 = 0.999;
const LAM: f32 = 0.98;
const CLIP_RATIO: f32 = 0.2;
// run-12: 2.5e-4 — exact SB3 Zoo value.
const PI_LR: f64 = 2.5e-4;
// run-13: 2.5e-4 — match pi_lr; run-12 VF_LR=1e-3 caused VF overfitting (880 gradient
// steps at 4x lr vs SB3's balanced equal-lr joint update → VF reset high every epoch).
const VF_LR: f64 = 2.5e-4;
// run-15: 10 — allow many pi iters; per-mini-batch KL check controls effective count.
// With target_kl=0.015, early stop triggers ~halfway through pi training (dynamic).
const TRAIN_PI_ITERS: u64 = 10;
// run-15: 4 — keep VF at 4 iters; no KL constraint for VF.
const TRAIN_VF_ITERS: u64 = 4;
// run-15: 0.015 — matches RL4Sys (vs 0.1 in run-14). With per-mini-batch KL checking
// this prevents excess policy drift (ClipFrac was 0.55 in run-14, now expect ~0.25).
const TARGET_KL: f32 = 0.015;
// run-7: 128 — 2x larger batches (~11,520 transitions/epoch vs 5,760).
const TRAJ_PER_EPOCH: u64 = 128;
// 12_288_000_000 env-frames / 64 envs = 192_000_000 loop iterations (doubled from run 14).
const TOTAL_STEPS: usize = 192_000_000;
const BUFFER_SIZE: ReplayBufferSize = 100_000;

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Make ORT dylib available for policy ONNX inference during rollouts.
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
    println!("  RelayRL beta.4 — PPO — LunarLander discrete — {ENV_COUNT} envs  (SB3 Zoo hparams)");
    println!("  inference: ORT policy (categorical) + ORT value-head (GAE) + OpenBLAS training");
    println!("  obs={OBS_DIM}  act={ACT_DIM}  MLP=[128,128]  loop steps={TOTAL_STEPS}  env-frames={total_env_frames}  (run-16: fresh logp_old+per-mb KL, target_kl=0.015, pi_iters=10)");
    println!("  gamma={GAMMA}  lam={LAM}  clip={CLIP_RATIO}  pi_lr={PI_LR}  vf_lr={VF_LR}  grad_clip_norm=0.5 (pi only)");
    println!("  pi_iters={TRAIN_PI_ITERS}  vf_iters={TRAIN_VF_ITERS}  target_kl={TARGET_KL}  ent_coef=0.01  traj/epoch={TRAJ_PER_EPOCH}  mb=64  (fresh Burn logp_old, per-mb KL, RL4Sys target_kl)");
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
                ent_coef: 0.01,
                max_episode_steps: Some(MAX_STEPS),
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
