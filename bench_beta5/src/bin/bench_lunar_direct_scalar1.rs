//! bench_lunar_direct_scalar1 — live integration path, 1 scalar env, 1 actor. (beta.5)
//!
//! Each loop iteration:
//!   1. Collect real obs from LunarLanderEnv  → Tensor[1, 8]
//!   2. request_action                        → RelayRLAction (act = [1, 4] logits)
//!   3. Argmax → discrete action u8
//!   4. env.step()  (inline reset on done)
//!   5. flag_last_action with real reward
//!
//! This is the canonical single-actor integration path — measures full
//! coordinator dispatch round-trip including real obs / action / reward flow.
//!
//! Build & run:
//!   cargo build --release -p bench-beta3 --bin bench_lunar_direct_scalar1
//!   ORT_DYLIB_PATH=... perf stat ./target/release/bench_lunar_direct_scalar1

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

use lunarlander_rl::env::LunarLanderEnv;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM:      usize = 8;
const ACT_DIM:      usize = 4;
const MAX_STEPS:    usize = 500;
const TARGET_ITERS: usize = 100_000;
const WARMUP_ITERS: usize = 10_000;

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

// ─────────────────────────── Helper ─────────────────────────────────────────

/// Argmax over ACT_DIM f32 values packed in little-endian bytes.
fn argmax_action(data: &[u8]) -> u8 {
    let mut best_idx = 0u8;
    let mut best_val = f32::NEG_INFINITY;
    for j in 0..ACT_DIM {
        let off = j * 4;
        let val = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        if val > best_val {
            best_val = val;
            best_idx = j as u8;
        }
    }
    best_idx
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.5 — direct integration path — 1 scalar env, 1 actor");
    println!("  request_action + flag_last_action per step · {} logical cores", num_cores);
    println!("  {} warmup · {} timed iters", WARMUP_ITERS, TARGET_ITERS);
    println!("═══════════════════════════════════════════════════════════════════\n");

    let initial_model = make_bootstrap_model::<B>()?;
    let config_path   = std::path::PathBuf::from("./config.json");
    // beta.5: AgentBuilder no longer has KindIn/KindOut type params
    let mut builder = AgentBuilder::<B>::builder()
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

    let env = LunarLanderEnv::<B>::new(MAX_STEPS, Default::default());
    env.reset();

    let mut prev_reward = 0.0f32;

    // ── Warm-up ───────────────────────────────────────────────────────────────
    println!("Warming up ({WARMUP_ITERS} iters)…");
    for _ in 0..WARMUP_ITERS {
        let obs_vec = env.get_observation(0);
        let obs_tensor = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_vec, [1, OBS_DIM]),
            &Default::default(),
        );

        let results = agent.request_action(vec![actor_id], obs_tensor, None::<Tensor<B, 2, Float>>, prev_reward).await?;

        let action = results.first()
            .and_then(|(_, a)| a.get_act())
            .map(|td| argmax_action(&td.data))
            .unwrap_or(0);

        let (reward, done) = env.step(0, action).unwrap_or((0.0, true));
        if done { env.reset(); }
        prev_reward = reward;

        agent.flag_last_action(vec![actor_id], Some(reward)).await?;
    }
    println!("Warm-up done. Starting timed run ({TARGET_ITERS} iters)…\n");

    // ── Timed run ─────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    for _ in 0..TARGET_ITERS {
        let obs_vec = env.get_observation(0);
        let obs_tensor = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_vec, [1, OBS_DIM]),
            &Default::default(),
        );

        let results = agent.request_action(vec![actor_id], obs_tensor, None::<Tensor<B, 2, Float>>, prev_reward).await?;

        let action = results.first()
            .and_then(|(_, a)| a.get_act())
            .map(|td| argmax_action(&td.data))
            .unwrap_or(0);

        let (reward, done) = env.step(0, action).unwrap_or((0.0, true));
        if done { env.reset(); }
        prev_reward = reward;

        agent.flag_last_action(vec![actor_id], Some(reward)).await?;
    }
    let wall = t0.elapsed().as_secs_f64();

    let iters_per_sec = TARGET_ITERS as f64 / wall;
    let us_per_iter   = 1_000_000.0 / iters_per_sec;
    let ns_per_iter   = 1_000_000_000.0 / iters_per_sec;

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
    let total_ctx    = vol_ctx + nvol_ctx;
    let ctx_per_iter = total_ctx as f64 / TARGET_ITERS as f64;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.5 — direct integration — FINAL RESULTS  (1 scalar env, 1 actor)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────");
    println!("  iterations               : {:>10}", TARGET_ITERS);
    println!("  wall time                : {:>10.3} s", wall);
    println!("  iters/sec                : {:>10.0}", iters_per_sec);
    println!("  µs / iter                : {:>10.3}", us_per_iter);
    println!("  ns / iter                : {:>10.1}", ns_per_iter);
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
