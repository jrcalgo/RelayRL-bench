//! bench_lunar_ppo_tch — RelayRL PPO on LunarLander, 64 envs, 100k steps, LibTorch backend.
//!
//! Same hyperparameters as bench_lunar_ppo_64env but uses burn-tch (LibTorch/PyTorch) for training.

use std::path::PathBuf;
use std::time::Instant;

use burn_tch::LibTorch;
use burn_tensor::Float;

use relayrl_algorithms::algorithms::PPO::PPOKernel;
use relayrl_algorithms::algorithms::REINFORCE::ActivationKind;
use relayrl_algorithms::PPOParams;
use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, AlgorithmCfg, ModelMode,
    RelayRLActorEnv, RelayRLAgentActors, ReplayBufferSize, SaveModelPath,
};
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;

use lunarlander_rl::env::LunarLanderEnv;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const MAX_STEPS: usize = 500;
const ENV_COUNT: u32 = 64;

const GAMMA: f32 = 0.999;
const LAM: f32 = 0.98;
const CLIP_RATIO: f32 = 0.2;
const PI_LR: f64 = 2.5e-4;
const VF_LR: f64 = 2.5e-4;
const TRAIN_PI_ITERS: u64 = 10;
const TRAIN_VF_ITERS: u64 = 10;
const TARGET_KL: f32 = 0.05;
const MINI_BATCH_SIZE: usize = 64;
const ENT_COEF: f32 = 0.05;

const TRAJ_PER_EPOCH: u64 = 320;
const TOTAL_STEPS: usize = 100;
const BUFFER_SIZE: ReplayBufferSize = 200_000;

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.25.0",
    );

    type B = LibTorch;

    println!("============================================================");
    println!("  RelayRL PPO — LunarLander-v3 — 64 envs — LibTorch backend");
    println!("  lr={PI_LR}  n_steps≈{}  batch={MINI_BATCH_SIZE}  epochs={TRAIN_PI_ITERS}", TRAJ_PER_EPOCH as usize * 200);
    println!("  gamma={GAMMA}  lam={LAM}  clip={CLIP_RATIO}  ent={ENT_COEF}  target_kl={TARGET_KL}");
    println!("  net=[128,128]  seed=42  max_ep_steps={MAX_STEPS}  env_count={ENV_COUNT}");
    println!("============================================================");

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

    let env = LunarLanderEnv::<B>::new_with_seed(MAX_STEPS, Default::default(), 42);
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;

    let burn_device = <B as burn_tensor::backend::Backend>::Device::default();
    let kernel = PPOKernel::<B, Float, Float>::new_with_schedule(
        OBS_DIM,
        ACT_DIM,
        true,
        &[128, 128],
        ActivationKind::ReLU,
        PI_LR,
        VF_LR,
        None,
        &burn_device,
    );

    let total_frames = TOTAL_STEPS * ENV_COUNT as usize;
    println!("Starting PPO training ({TOTAL_STEPS} loop iters × {ENV_COUNT} envs = {total_frames} env frames)...\n");

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
                ent_coef: ENT_COEF,
                max_episode_steps: Some(MAX_STEPS),
                mini_batch_size: Some(MINI_BATCH_SIZE),
                ..Default::default()
            })),
            SaveModelPath::from("./models/lunar_ppo_tch"),
            BUFFER_SIZE,
            DeviceType::Cpu,
            kernel,
        )
        .await?;
    let wall = t0.elapsed().as_secs_f64();

    let total_env_frames = TOTAL_STEPS * ENV_COUNT as usize;
    let env_frames_per_sec = total_env_frames as f64 / wall;

    println!();
    println!("============================================================");
    println!("  RelayRL RESULTS (LibTorch backend)");
    println!("============================================================");
    println!("  n_envs            : {}", ENV_COUNT);
    println!("  loop iterations   : {}", TOTAL_STEPS);
    println!("  total env frames  : {}", total_env_frames);
    println!("  wall time         : {:.1}s", wall);
    println!("  steps/sec         : {:.0}", env_frames_per_sec);
    println!("============================================================");

    agent.shutdown().await?;
    Ok(())
}
