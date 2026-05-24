//! bench_start_latency — measures how long agent.start() takes to return.
//!
//! Runs agent.build() + agent.start() then immediately shuts down.
//! Reports wall-clock time for each phase so the caller can see the startup cost.

use std::path::PathBuf;
use std::time::Instant;

use burn_tch::LibTorch;

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode,
};
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.25.0",
    );

    type B = LibTorch;

    println!("=== RelayRL start() latency benchmark ===");

    let config_path = PathBuf::from("./config.json");

    // ── Phase 1: builder.build() ─────────────────────────────────────────────
    let t_build = Instant::now();
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
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
    println!("builder.build()  : {build_ms:.2} ms");

    // ── Phase 2: agent.start() ───────────────────────────────────────────────
    let t_start = Instant::now();
    agent.start(params).await?;
    let start_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    println!("agent.start()    : {start_ms:.2} ms");

    // ── Phase 3: shutdown ────────────────────────────────────────────────────
    let t_shut = Instant::now();
    agent.shutdown().await?;
    let shut_ms = t_shut.elapsed().as_secs_f64() * 1000.0;
    println!("agent.shutdown() : {shut_ms:.2} ms");

    let total_ms = t_build.elapsed().as_secs_f64() * 1000.0;
    println!("-----------------------------------------");
    println!("total (build+start+shutdown): {total_ms:.2} ms");

    Ok(())
}
