//! bench_request_action_latency — measures coordinator::request_action hot path.
//!
//! Calls agent.request_action() in a tight loop for TARGET_ITERS iterations,
//! timing only the dispatch + response round-trip through the coordinator.
//! This is the path the hot path patch optimises: RwLock bypass via cached
//! HotPathChannels → direct filter channel send.
//!
//! Build & run:
//!   cargo build --release -p bench-beta3 --bin bench_request_action_latency
//!   ORT_DYLIB_PATH=... ./target/release/bench_request_action_latency

use std::sync::Arc;
use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode,
    RelayRLAgentActors,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM:       usize = 8;
const ACT_DIM:       usize = 4;
const TARGET_ITERS:  usize = 100_000;
const WARMUP_ITERS:  usize = 10_000;

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
    println!("  RelayRL beta.3 — request_action latency benchmark");
    println!("  1 actor · {} warmup · {} timed iters · {} logical cores",
             WARMUP_ITERS, TARGET_ITERS, num_cores);
    println!("  measures: coordinator dispatch + inference + response round-trip");
    println!("═══════════════════════════════════════════════════════════════════\n");

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

    // Fixed observation tensor (reused every call — measures dispatch overhead, not alloc)
    let obs = Tensor::<B, 2, Float>::from_data(
        TensorData::new(vec![0.1f32; OBS_DIM], [1, OBS_DIM]),
        &Default::default(),
    );

    // ── Warm-up ───────────────────────────────────────────────────────────────
    println!("Warming up ({} iters)…", WARMUP_ITERS);
    for _ in 0..WARMUP_ITERS {
        agent.request_action(vec![actor_id], obs.clone().detach(), None, 0.0).await?;
    }
    println!("Warm-up done. Starting timed run ({} iters)…\n", TARGET_ITERS);

    // ── Timed run ─────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    for _ in 0..TARGET_ITERS {
        agent.request_action(vec![actor_id], obs.clone().detach(), None, 0.0).await?;
    }
    let wall = t0.elapsed().as_secs_f64();

    let calls_per_sec   = TARGET_ITERS as f64 / wall;
    let us_per_call     = 1_000_000.0 / calls_per_sec;
    let ns_per_call     = 1_000_000_000.0 / calls_per_sec;

    // ── /proc stats ───────────────────────────────────────────────────────────
    let mut rss_kb:   u64 = 0;
    let mut vol_ctx:  u64 = 0;
    let mut nvol_ctx: u64 = 0;
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            let mut it = line.splitn(2, ':');
            let k = it.next().unwrap_or("").trim();
            let v = it.next().unwrap_or("").trim();
            match k {
                "VmRSS"                      => rss_kb   = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"    => vol_ctx  = v.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => nvol_ctx = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    let total_ctx     = vol_ctx + nvol_ctx;
    let ctx_per_call  = total_ctx as f64 / TARGET_ITERS as f64;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  request_action latency — FINAL RESULTS");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────");
    println!("  iterations               : {:>10}", TARGET_ITERS);
    println!("  wall time                : {:>10.3} s", wall);
    println!("  calls/sec                : {:>10.0}", calls_per_sec);
    println!("  µs / call                : {:>10.3}", us_per_call);
    println!("  ns / call                : {:>10.1}", ns_per_call);
    println!();

    println!("─── OS ──────────────────────────────────────────────────────────────");
    println!("  RSS                      : {:>7.1} MB", rss_kb as f64 / 1024.0);
    println!("  context switches (vol)   : {:>10}", vol_ctx);
    println!("  context switches (nonvol): {:>10}", nvol_ctx);
    println!("  context switches (total) : {:>10}", total_ctx);
    println!("  context switches/call    : {:>10.4}", ctx_per_call);
    println!("  logical cores            : {:>10}", num_cores);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
