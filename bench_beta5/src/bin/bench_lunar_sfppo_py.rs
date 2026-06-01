//! bench_lunar_sfppo_py — RelayRL SFPPO on LunarLander-v3, 64 envs, Python/gymnasium.
//!
//! Uses Sample Factory APPO–aligned hyperparameters:
//!   clip=0.1, epochs=1, rollout=32, lr=1e-4, normalize_returns=true
//! The kernel and loss formula are identical to PPO/IPPO; only the defaults differ.
//!
//! Build:
//!   cargo build --release -p bench-beta5 --bin bench_lunar_sfppo_py
//!
//! Run:
//!   LD_LIBRARY_PATH=... LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
//!     ./target/release/bench_lunar_sfppo_py

use std::path::PathBuf;
use std::time::Instant;

use burn_tch::LibTorch;
use burn_tensor::Float;

use relayrl_algorithms::algorithms::PPO::PPOKernel;
use relayrl_algorithms::algorithms::REINFORCE::ActivationKind;
use relayrl_algorithms::SFPPOParams;
use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, AlgorithmCfg, ModelMode,
    RelayRLActorEnv, RelayRLAgentActors, ReplayBufferSize, SaveModelPath,
};
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;

use bench_beta5::py_env::make_lunar_lander_vec;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const ENV_COUNT: u32 = 64;

// SF APPO defaults for LunarLander (rollout_len=32 → mini_batch=64*32=2048)
const ROLLOUT_LEN: usize = 32;
const MINI_BATCH_SIZE: usize = ENV_COUNT as usize * ROLLOUT_LEN;
const TOTAL_STEPS: usize = 600_000;
const BUFFER_SIZE: ReplayBufferSize = 500_000;

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.26.0",
    );

    type B = LibTorch;

    let params = SFPPOParams {
        rollout_len: Some(ROLLOUT_LEN),
        mini_batch_size: Some(MINI_BATCH_SIZE),
        min_steps_per_epoch: Some(MINI_BATCH_SIZE as u64),
        ..Default::default()
    };

    println!("════════════════════════════════════════════════════════════════");
    println!("  RelayRL SFPPO — LunarLander-v3 — {} envs — Python/gymnasium", ENV_COUNT);
    println!("  lr={:.0e}  rollout={}  batch={}  epochs={}  norm_returns={}",
             params.pi_lr, ROLLOUT_LEN, MINI_BATCH_SIZE, params.train_pi_iters, params.normalize_returns);
    println!("  gamma={}  lam={}  clip={}  ent={}  vf_coef={}",
             params.gamma, params.lam, params.clip_ratio, params.ent_coef, params.vf_coef);
    println!("  net=[128,128]  seed=1  max_ep_steps={}  total_steps={}",
             params.max_episode_steps.unwrap_or(500), TOTAL_STEPS);
    println!("════════════════════════════════════════════════════════════════");

    let config_path = PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B>::builder()
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .router_scale(1);
    if config_path.exists() {
        builder = builder.config_path(config_path);
    }

    let (mut agent, start_params) = builder.build().await?;
    agent.start(start_params).await?;
    let actor_ids = agent.get_actor_ids()?;
    let actor_id = actor_ids[0];

    let py_env = make_lunar_lander_vec(ENV_COUNT as usize, OBS_DIM, ACT_DIM)
        .map_err(|e| format!("gymnasium env creation failed: {e}"))?;
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(py_env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;

    let burn_device = <B as burn_tensor::backend::Backend>::Device::default();
    <B as burn_tensor::backend::Backend>::seed(&burn_device, 1);
    let kernel = PPOKernel::<B, Float, Float>::new_with_schedule(
        OBS_DIM,
        ACT_DIM,
        true,
        &[128, 128],
        ActivationKind::ReLU,
        params.pi_lr as f64,
        params.vf_coef,
        None,
        &burn_device,
    );

    let total_frames = TOTAL_STEPS * ENV_COUNT as usize;
    println!("\nStarting SFPPO run ({TOTAL_STEPS} loop iters × {ENV_COUNT} envs = {total_frames} frames)...\n");

    let t0 = Instant::now();
    agent
        .run_env_with_ppo::<Float, Float, _>(
            actor_id,
            TOTAL_STEPS,
            AlgorithmCfg::SFPPO(Some(params)),
            SaveModelPath::from("./models/lunar_sfppo_py"),
            BUFFER_SIZE,
            DeviceType::Cpu,
            kernel,
        )
        .await?;
    let wall = t0.elapsed().as_secs_f64();

    let env_frames_per_sec = total_frames as f64 / wall;

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("  SFPPO RESULTS — LunarLander-v3 — Python/gymnasium — {} envs", ENV_COUNT);
    println!("════════════════════════════════════════════════════════════════");
    println!("  loop iterations   : {}", TOTAL_STEPS);
    println!("  total env frames  : {}", total_frames);
    println!("  wall time         : {:.1} s", wall);
    println!("  env frames/sec    : {:.0}", env_frames_per_sec);
    println!("════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
