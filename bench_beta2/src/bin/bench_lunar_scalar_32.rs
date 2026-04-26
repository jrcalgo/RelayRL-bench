//! bench_lunar_scalar_32 — RelayRL scalar LunarLander, 32 envs, 50k steps.
//!
//! Uses the ScalarVecEnv path (LunarLanderEnv stepped sequentially by the
//! framework's env runner) — the direct apples-to-apples comparison against
//! SB3 DummyVecEnv and SubprocVecEnv at the same env count.
//!
//! Build & run:
//!   cargo build --release -p bench-beta2
//!   ORT_DYLIB_PATH=... ./bench_beta2/target/release/bench_lunar_scalar_32

use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::Float;

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode,
    RelayRLActorEnv, RelayRLAgentActors,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;

use lunarlander_rl::env::LunarLanderEnv;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM:      usize = 8;
const ACT_DIM:      usize = 4;
const MAX_STEPS:    usize = 500;
const TARGET_STEPS: usize = 50_000;
const ENV_COUNT:    u32   = 32;

// ─────────────────────────── Bootstrap model ────────────────────────────────

fn make_bootstrap_model<B>() -> Result<ModelModule<B>, Box<dyn std::error::Error>>
where
    B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>,
{
    use relayrl_types::data::tensor::{DType, NdArrayDType};
    use relayrl_types::model::{ModelFileType, ModelMetadata};

    let layer_specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = vec![
        (OBS_DIM, 64, vec![0.01f32; 64 * OBS_DIM], vec![0.0f32; 64]),
        (64,      64, vec![0.01f32; 64 * 64],       vec![0.0f32; 64]),
        (64, ACT_DIM, vec![0.01f32; ACT_DIM * 64],  vec![0.0f32; ACT_DIM]),
    ];
    let onnx_bytes = build_onnx_mlp_bytes(&layer_specs);
    let metadata = ModelMetadata {
        model_file:     "bootstrap.onnx".to_string(),
        model_type:     ModelFileType::Onnx,
        input_dtype:    DType::NdArray(NdArrayDType::F32),
        output_dtype:   DType::NdArray(NdArrayDType::F32),
        input_shape:    vec![1, OBS_DIM],
        output_shape:   vec![1, ACT_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(onnx_bytes, metadata)?)
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.2 — scalar LunarLander — {} envs — {} steps",
        ENV_COUNT, TARGET_STEPS);
    println!("  env backend: ScalarVecEnv (LunarLanderEnv × {})", ENV_COUNT);
    println!("  {} total env transitions · ndarray · {} logical cores",
        TARGET_STEPS * ENV_COUNT as usize, num_cores);
    println!("═══════════════════════════════════════════════════════════════════\n");

    let initial_model = make_bootstrap_model::<B>()?;
    let config_path = std::path::PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(1)
        .default_device(DeviceType::Cpu)
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .default_model(initial_model)
        .router_scale(1);
    if config_path.exists() {
        builder = builder.config_path(config_path);
    }

    let (mut agent, params) = builder.build().await?;
    agent.start(params).await?;
    let actor_ids = agent.get_actor_ids()?;
    let actor_id  = actor_ids[0];

    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let env = LunarLanderEnv::<B>::new(MAX_STEPS, device);
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!("set_env OK — registered {} scalar LunarLander envs with actor {}",
        ENV_COUNT, actor_id);

    println!("Warming up (500 loop iters = {} env transitions)…", 500 * ENV_COUNT);
    agent.run_env(actor_id, 500).await?;
    println!("Warm-up done. Starting timed benchmark ({} loop iters)…\n", TARGET_STEPS);

    let t0 = Instant::now();
    agent.run_env(actor_id, TARGET_STEPS).await?;
    let wall = t0.elapsed().as_secs_f64();

    let loop_iters_per_sec  = TARGET_STEPS as f64 / wall;
    let env_transitions_sec = loop_iters_per_sec * ENV_COUNT as f64;
    let total_transitions   = TARGET_STEPS * ENV_COUNT as usize;
    let us_per_loop_iter    = 1_000_000.0 / loop_iters_per_sec;
    let us_per_transition   = 1_000_000.0 / env_transitions_sec;

    let mut rss_kb: u64 = 0;
    let mut vol_ctx: u64 = 0;
    let mut nonvol_ctx: u64 = 0;
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            let mut it = line.splitn(2, ':');
            let k = it.next().unwrap_or("").trim();
            let v = it.next().unwrap_or("").trim();
            match k {
                "VmRSS" => rss_kb = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"    => vol_ctx    = v.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => nonvol_ctx = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    let total_ctx    = vol_ctx + nonvol_ctx;
    let ctx_per_iter = total_ctx as f64 / TARGET_STEPS as f64;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.2 — scalar — FINAL RESULTS  ({} envs)", ENV_COUNT);
    println!("═══════════════════════════════════════════════════════════════════\n");
    println!("─── Throughput ──────────────────────────────────────────────────────");
    println!("  env count                : {:>10}", ENV_COUNT);
    println!("  loop iterations          : {:>10}", TARGET_STEPS);
    println!("  total env transitions    : {:>10}", total_transitions);
    println!("  wall time                : {:>10.2} s", wall);
    println!("  loop iters/sec           : {:>10.0}", loop_iters_per_sec);
    println!("  env transitions/sec      : {:>10.0}", env_transitions_sec);
    println!("  µs / loop iter           : {:>10.3}", us_per_loop_iter);
    println!("  µs / env transition      : {:>10.3}", us_per_transition);
    println!();
    println!("─── OS ──────────────────────────────────────────────────────────────");
    println!("  RSS                      : {:>7.1} MB", rss_kb as f64 / 1024.0);
    println!("  context switches total   : {:>10}", total_ctx);
    println!("  context switches/iter    : {:>10.4}", ctx_per_iter);
    println!("  logical cores            : {:>10}", num_cores);
    println!();
    println!("─── vs SB3 baselines (32 envs, 50k iters) ───────────────────────────");
    const SB3_DUMMY_SPS:   f64 = 23_222.0;
    const SB3_SUBPROC_SPS: f64 = 13_130.0;
    println!("  SB3 DummyVecEnv          : {:>10.0}  transitions/sec", SB3_DUMMY_SPS);
    println!("  SB3 SubprocVecEnv        : {:>10.0}  transitions/sec", SB3_SUBPROC_SPS);
    println!("  RelayRL scalar           : {:>10.0}  transitions/sec", env_transitions_sec);
    println!("  speedup vs DummyVecEnv   : {:>10.2}×", env_transitions_sec / SB3_DUMMY_SPS);
    println!("  speedup vs SubprocVecEnv : {:>10.2}×", env_transitions_sec / SB3_SUBPROC_SPS);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
