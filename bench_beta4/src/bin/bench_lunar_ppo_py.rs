//! bench_lunar_ppo_py — RelayRL PPO on LunarLander-v3, 64 envs, Python/gymnasium backend.
//!
//! Identical hyperparameters to bench_lunar_ppo_tch but the environment is
//! gymnasium's LunarLander-v3 accessed via a thin PyO3 binding layer.

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

use bench_beta4::py_env::make_lunar_lander_vec;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const MAX_STEPS: usize = 500;
const ENV_COUNT: u32 = 64;

const GAMMA: f32 = 0.99;
const LAM: f32 = 0.98;
const CLIP_RATIO: f32 = 0.2;
#[allow(dead_code)]
const PI_LR: f64 = 2.5e-4;
const VF_COEF: f32 = 1.0;
const TRAIN_PI_ITERS: u64 = 4;
const TRAIN_VF_ITERS: u64 = 4;
const TARGET_KL: f32 = 1.0;
const MINI_BATCH_SIZE: usize = 5760;
const ENT_COEF: f32 = 0.01;
const NORMALIZE_RETURNS: bool = true;

const TRAJ_PER_EPOCH: u64 = 64;
const MIN_STEPS_PER_EPOCH: u64 = MINI_BATCH_SIZE as u64;
const MAX_BUFFERED_EPISODES: u64 = 128;
const TOTAL_STEPS: usize = 600_000;
const BUFFER_SIZE: ReplayBufferSize = 500_000;

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.25.0",
    );

    type B = LibTorch;

    println!("============================================================");
    println!("  RelayRL PPO — LunarLander-v3 — 64 envs — Python/gymnasium backend");
    println!("  lr={PI_LR}  batch={MINI_BATCH_SIZE}  epochs={TRAIN_PI_ITERS}  normalize_returns={NORMALIZE_RETURNS}");
    println!("  gamma={GAMMA}  lam={LAM}  clip={CLIP_RATIO}  ent={ENT_COEF}  vf_coef={VF_COEF}");
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

    // Build the gymnasium LunarLander-v3 vec-env and wrap it for RelayRL.
    // GIL is acquired and released inside make_lunar_lander_vec before any await.
    let py_env = make_lunar_lander_vec(ENV_COUNT as usize, OBS_DIM, ACT_DIM)
        .map_err(|e| format!("gymnasium env creation failed: {e}"))?;
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(py_env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;

    let burn_device = <B as burn_tensor::backend::Backend>::Device::default();
    let kernel = PPOKernel::<B, Float, Float>::new_with_schedule(
        OBS_DIM,
        ACT_DIM,
        true,
        &[128, 128],
        ActivationKind::ReLU,
        PI_LR,
        VF_COEF,
        None,
        &burn_device,
    );

    let total_frames = TOTAL_STEPS * ENV_COUNT as usize;
    println!("Starting PPO convergence run ({TOTAL_STEPS} loop iters × {ENV_COUNT} envs = {total_frames} env frames)...\n");

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
                vf_coef: VF_COEF,
                min_steps_per_epoch: Some(MIN_STEPS_PER_EPOCH),
                max_buffered_episodes: Some(MAX_BUFFERED_EPISODES),
                max_version_lag: 1,
                normalize_returns: NORMALIZE_RETURNS,
                ..Default::default()
            })),
            SaveModelPath::from("./models/lunar_ppo_py"),
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
    println!("  RelayRL RESULTS (Python/gymnasium backend)");
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
