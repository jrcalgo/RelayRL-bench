//! bench_lunar_set_env — beta.2 throughput test using set_env / run_env API.
//!
//! The framework owns the step loop: after set_env the coordinator calls
//! ScalarEnvironment::step / reset through the internal run_env path.
//! This measures the overhead of that API path vs the manual request_action loop.
//!
//! Build & run:
//!   cargo build --release -p bench-beta2
//!   ORT_DYLIB_PATH=... ./bench_beta2/target/release/bench_lunar_set_env

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

const OBS_DIM:    usize = 8;
const ACT_DIM:    usize = 4;
const MAX_STEPS:  usize = 500;
const TARGET_STEPS: usize = 1_000_000;

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
    println!("  RelayRL beta.2 — set_env / run_env — LunarLander — 1 env");
    println!("  {} steps · ndarray · {} logical cores", TARGET_STEPS, num_cores);
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

    // ── Create env and register it with the actor ─────────────────────────────
    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let env = LunarLanderEnv::<B>::new(MAX_STEPS, device);
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(env);

    agent.set_env(actor_id, boxed, 1).await?;
    println!("set_env OK — registered 1 LunarLander env with actor {}", actor_id);

    // ── Confirm actor is alive with a manual request_action ──────────────────
    {
        use burn_tensor::TensorData;
        let obs = vec![0.0f32; OBS_DIM];
        let obs_t = burn_tensor::Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs, [1, OBS_DIM]), &Default::default());
        let result = agent.request_action(vec![actor_id], obs_t, None, 0.0).await;
        println!("manual request_action smoke test: {:?}", result.is_ok());
    }

    // ── Warm-up: 5 000 steps via run_env ─────────────────────────────────────
    println!("Warming up (5 000 steps)…");
    agent.run_env(actor_id, 5_000).await?;
    println!("Warm-up done. Starting timed benchmark ({} steps)…\n", TARGET_STEPS);

    // ── Timed run ─────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    agent.run_env(actor_id, TARGET_STEPS).await?;
    let wall = t0.elapsed().as_secs_f64();

    let sps     = TARGET_STEPS as f64 / wall;
    let us_step = 1_000_000.0 / sps;

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
    let ctx_per_step = total_ctx as f64 / TARGET_STEPS as f64;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.2 — set_env/run_env — FINAL RESULTS");
    println!("═══════════════════════════════════════════════════════════════════\n");
    println!("─── Throughput ──────────────────────────────────────────────────────");
    println!("  total steps              : {:>10}", TARGET_STEPS);
    println!("  wall time                : {:>10.2} s", wall);
    println!("  steps/sec                : {:>10.0}", sps);
    println!("  µs/step                  : {:>10.3}", us_step);
    println!();
    println!("─── OS ──────────────────────────────────────────────────────────────");
    println!("  RSS                      : {:>7.1} MB", rss_kb as f64 / 1024.0);
    println!("  context switches total   : {:>10}", total_ctx);
    println!("  context switches/step    : {:>10.4}", ctx_per_step);
    println!("  logical cores            : {:>10}", num_cores);
    println!();

    // Compare against the manual request_action baseline (beta.2)
    const BETA2_MANUAL_SPS: f64 = 19_138.0;
    const PATCHED_SPS: f64 = 213_390.0;
    println!("─── vs baselines ────────────────────────────────────────────────────");
    println!("  vs beta.2 manual loop    : {:>10.2}×", sps / BETA2_MANUAL_SPS);
    println!("  vs patched (inline infer): {:>10.2}×", sps / PATCHED_SPS);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
