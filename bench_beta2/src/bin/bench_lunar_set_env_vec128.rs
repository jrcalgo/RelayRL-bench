//! bench_lunar_set_env_vec128 — beta.2 throughput using SyncLunarVectorEnvFramework.
//!
//! Compares rayon-parallel vector env vs sequential scalar env, both through
//! the same set_env / run_env API with 128 parallel environments.
//!
//! Key difference from bench_lunar_set_env_128:
//!   Scalar path: ScalarVecEnv steps 128 LunarLander instances sequentially
//!   Vector path: SyncLunarVectorEnvFramework steps all 128 in parallel via rayon
//!
//! Build & run:
//!   cargo build --release -p bench-beta2
//!   ORT_DYLIB_PATH=... ./bench_beta2/target/release/bench_lunar_set_env_vec128

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

use lunarlander_rl::env::vec::SyncLunarVectorEnvFramework;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM:      usize = 8;
const ACT_DIM:      usize = 4;
const MAX_STEPS:    usize = 500;
const TARGET_STEPS: usize = 500_000;
const ENV_COUNT:    u32   = 1024;

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
    println!("  RelayRL beta.2 — set_env / run_env — VectorLunarLander — {} envs", ENV_COUNT);
    println!("  {} loop iterations · {} total env transitions · ndarray · {} logical cores",
        TARGET_STEPS, TARGET_STEPS * ENV_COUNT as usize, num_cores);
    println!("  env backend: SyncLunarVectorEnvFramework (rayon parallel stepping)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── Agent: 1 actor, local independent, trajectories disabled ─────────────
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

    // ── Create vectorised env (128 rayon-parallel sub-envs) ──────────────────
    let vec_env = SyncLunarVectorEnvFramework::new(ENV_COUNT as usize, MAX_STEPS)
        .map_err(|e| format!("Failed to create vector env: {e}"))?;
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(vec_env);

    // set_env with count=ENV_COUNT triggers BatchVecEnv path → init_num_envs(128)
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!("set_env OK — registered {} rayon-parallel LunarLander sub-envs with actor {}",
        ENV_COUNT, actor_id);

    // ── Confirm actor is alive with a manual request_action ──────────────────
    {
        use burn_tensor::TensorData;
        let obs = vec![0.0f32; OBS_DIM];
        let obs_t = burn_tensor::Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs, [1, OBS_DIM]), &Default::default());
        let result = agent.request_action(vec![actor_id], obs_t, None, 0.0).await;
        println!("manual request_action smoke test: {:?}", result.is_ok());
    }

    // ── Warm-up: 500 loop iterations ─────────────────────────────────────────
    println!("Warming up (500 loop iters = {} env transitions)…", 500 * ENV_COUNT);
    agent.run_env(actor_id, 500).await?;
    println!("Warm-up done. Starting timed benchmark ({} loop iters)…\n", TARGET_STEPS);

    // ── Timed run ─────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    agent.run_env(actor_id, TARGET_STEPS).await?;
    let wall = t0.elapsed().as_secs_f64();

    let loop_iters_per_sec  = TARGET_STEPS as f64 / wall;
    let env_transitions_sec = loop_iters_per_sec * ENV_COUNT as f64;
    let total_transitions   = TARGET_STEPS * ENV_COUNT as usize;
    let us_per_loop_iter    = 1_000_000.0 / loop_iters_per_sec;
    let us_per_transition   = 1_000_000.0 / env_transitions_sec;

    // ── /proc stats ───────────────────────────────────────────────────────────
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
    println!("  RelayRL beta.2 — VectorLunarLander — FINAL RESULTS  ({} envs)", ENV_COUNT);
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

    // Compare against scalar-128 and 1-env baselines
    const SCALAR_128_TRANSITIONS_SEC: f64 = 205_656.0;
    const SINGLE_ENV_SPS: f64 = 147_240.0;
    println!("─── vs baselines ────────────────────────────────────────────────────");
    println!("  scalar-128 transitions/s : {:>10.0}  (sequential stepping)", SCALAR_128_TRANSITIONS_SEC);
    println!("  vector-128 transitions/s : {:>10.0}", env_transitions_sec);
    println!("  speedup vs scalar-128    : {:>10.2}×", env_transitions_sec / SCALAR_128_TRANSITIONS_SEC);
    println!("  1-env baseline           : {:>10.0}  loop iters/sec", SINGLE_ENV_SPS);
    println!("  loop iter speedup        : {:>10.2}×  vs 1-env", loop_iters_per_sec / SINGLE_ENV_SPS);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
