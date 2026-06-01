//! bench_lunar_set_env_vec1024 — beta.3 throughput using set_env / run_env with 32 vectorised envs.
//!
//! The framework owns the step loop: after set_env the coordinator calls
//! VectorEnvironment::step / reset through the internal run_env path.
//! One actor, 32 rayon-parallel LunarLander sub-envs via SyncLunarVectorEnvFramework.
//!
//! Build & run:
//!   cargo build --release -p bench-beta3 --bin bench_lunar_set_env_vec1024
//!   ORT_DYLIB_PATH=... perf stat ./bench_beta3/target/release/bench_lunar_set_env_vec1024

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
const ENV_COUNT:    u32   = 1024;
const TARGET_STEPS: usize = 50_000;
const WARMUP_ITERS: usize = 500;

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
    println!("  RelayRL beta.3 — set_env / run_env — VectorLunarLander — {} envs", ENV_COUNT);
    println!("  {} loop iters · {} total transitions · ndarray · {} logical cores",
             TARGET_STEPS, TARGET_STEPS * ENV_COUNT as usize, num_cores);
    println!("  env backend: SyncLunarVectorEnvFramework (rayon parallel)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── Agent: 1 actor, local independent, trajectories disabled ─────────────
    let initial_model = make_bootstrap_model::<B>()?;
    let config_path   = std::path::PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B, 2, 2, Float, Float>::builder()
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

    // ── Create vectorised env (32 rayon-parallel sub-envs) ───────────────────
    let vec_env = SyncLunarVectorEnvFramework::new(ENV_COUNT as usize, MAX_STEPS)
        .map_err(|e| format!("Failed to create vector env: {e}"))?;
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(vec_env);

    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!("set_env OK — registered {} rayon-parallel LunarLander sub-envs with actor {}",
             ENV_COUNT, actor_id);

    // ── Smoke test ────────────────────────────────────────────────────────────
    {
        use burn_tensor::TensorData;
        let obs_t = burn_tensor::Tensor::<B, 2, Float>::from_data(
            TensorData::new(vec![0.0f32; OBS_DIM], [1, OBS_DIM]), &Default::default());
        let result = agent.request_action(vec![actor_id], obs_t, None, 0.0).await;
        println!("smoke test request_action: {}", if result.is_ok() { "OK" } else { "FAILED" });
    }

    // ── Warm-up ───────────────────────────────────────────────────────────────
    println!("Warming up ({} iters = {} transitions)…",
             WARMUP_ITERS, WARMUP_ITERS * ENV_COUNT as usize);
    agent.run_env_eval(actor_id, WARMUP_ITERS).await?;
    println!("Warm-up done. Starting timed run ({} loop iters)…\n", TARGET_STEPS);

    // ── Timed run ─────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    agent.run_env_eval(actor_id, TARGET_STEPS).await?;
    let wall = t0.elapsed().as_secs_f64();

    let loop_iters_per_sec  = TARGET_STEPS as f64 / wall;
    let env_transitions_sec = loop_iters_per_sec * ENV_COUNT as f64;
    let total_transitions   = TARGET_STEPS * ENV_COUNT as usize;
    let us_per_loop_iter    = 1_000_000.0 / loop_iters_per_sec;
    let us_per_transition   = 1_000_000.0 / env_transitions_sec;

    // ── /proc stats ───────────────────────────────────────────────────────────
    let mut rss_kb:    u64 = 0;
    let mut vol_ctx:   u64 = 0;
    let mut nvol_ctx:  u64 = 0;
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            let mut it = line.splitn(2, ':');
            let k = it.next().unwrap_or("").trim();
            let v = it.next().unwrap_or("").trim();
            match k {
                "VmRSS"                    => rss_kb   = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"  => vol_ctx  = v.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => nvol_ctx = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    let total_ctx    = vol_ctx + nvol_ctx;
    let ctx_per_iter = total_ctx as f64 / TARGET_STEPS as f64;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.3 — VectorLunarLander — FINAL RESULTS  ({} envs)", ENV_COUNT);
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
    println!("  context switches (vol)   : {:>10}", vol_ctx);
    println!("  context switches (nonvol): {:>10}", nvol_ctx);
    println!("  context switches (total) : {:>10}", total_ctx);
    println!("  context switches/iter    : {:>10.4}", ctx_per_iter);
    println!("  logical cores            : {:>10}", num_cores);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
