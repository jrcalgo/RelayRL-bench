//! bench_lunar_dual_actor — two actors running concurrently via join_all.
//!
//! Two independent actors are each given 32 rayon-parallel LunarLander
//! sub-environments.  Both run_env calls are launched simultaneously and
//! awaited together with futures::future::try_join_all, so the tokio
//! runtime interleaves inference + env-step work from both actors on the
//! same thread pool.
//!
//! Build & run:
//!   cargo build --release -p bench-beta2
//!   ORT_DYLIB_PATH=... ./bench_beta2/target/release/bench_lunar_dual_actor

use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::Float;
use futures::future::try_join_all;

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
const TARGET_STEPS: usize = 50_000;
const ENV_COUNT:    u32   = 32;
const ACTOR_COUNT:  u32   = 2;

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
    let total_transitions = TARGET_STEPS * ENV_COUNT as usize * ACTOR_COUNT as usize;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.2 — dual-actor concurrent run_env — LunarLander");
    println!("  {} actors · {} envs/actor · {} steps · {} total transitions",
        ACTOR_COUNT, ENV_COUNT, TARGET_STEPS, total_transitions);
    println!("  env backend: SyncLunarVectorEnvFramework (rayon parallel stepping)");
    println!("  {} logical cores", num_cores);
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── Agent: 2 actors, local independent, trajectories disabled ────────────
    let initial_model = make_bootstrap_model::<B>()?;
    let config_path = std::path::PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(ACTOR_COUNT)
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
    assert_eq!(actor_ids.len(), 2, "expected 2 actor IDs from actor_count(2)");
    let (actor_a, actor_b) = (actor_ids[0], actor_ids[1]);
    println!("Actors: A={} B={}", actor_a, actor_b);

    // ── Register 32 envs per actor ───────────────────────────────────────────
    for &actor_id in &[actor_a, actor_b] {
        let vec_env = SyncLunarVectorEnvFramework::new(ENV_COUNT as usize, MAX_STEPS)
            .map_err(|e| format!("failed to create vector env: {e}"))?;
        let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(vec_env);
        agent.set_env(actor_id, boxed, ENV_COUNT).await?;
        println!("set_env OK — actor {} registered {} envs", actor_id, ENV_COUNT);
    }

    // ── Warm-up: 500 iters on both actors concurrently ───────────────────────
    println!("\nWarming up (500 iters × {} envs × {} actors)…",
        ENV_COUNT, ACTOR_COUNT);
    try_join_all(vec![
        Box::pin(agent.run_env(actor_a, 500)),
        Box::pin(agent.run_env(actor_b, 500)),
    ]).await?;
    println!("Warm-up done. Starting timed benchmark ({} iters)…\n", TARGET_STEPS);

    // ── Timed run: both actors concurrently via join_all ─────────────────────
    let t0 = Instant::now();
    try_join_all(vec![
        Box::pin(agent.run_env(actor_a, TARGET_STEPS)),
        Box::pin(agent.run_env(actor_b, TARGET_STEPS)),
    ]).await?;
    let wall = t0.elapsed().as_secs_f64();

    let env_transitions_sec = total_transitions as f64 / wall;
    let loop_iters_per_sec  = TARGET_STEPS as f64 / wall;
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
    println!("  RelayRL beta.2 — dual-actor — FINAL RESULTS");
    println!("═══════════════════════════════════════════════════════════════════\n");
    println!("─── Configuration ───────────────────────────────────────────────────");
    println!("  actors                   : {:>10}", ACTOR_COUNT);
    println!("  envs / actor             : {:>10}", ENV_COUNT);
    println!("  loop iterations / actor  : {:>10}", TARGET_STEPS);
    println!();
    println!("─── Throughput ──────────────────────────────────────────────────────");
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
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
