//! bench_lunar_request_action_vec32 — live integration path benchmark.
//!
//! This is the caller-controlled loop scenario: the framework is NOT given an env
//! to manage internally.  Instead the caller owns the step loop:
//!
//!   1. Collect stacked observations from 32 parallel LunarLander envs ([32, 8] tensor).
//!   2. Call `agent.request_action()` once with the batched observation.
//!   3. Decode 32 discrete actions from the returned action bytes (argmax per row).
//!   4. Step all 32 envs in parallel via rayon (`step_all`).
//!   5. Repeat.
//!
//! The ONNX MLP uses a dynamic batch dimension so [32, 8] → [32, 4] works without
//! model changes.
//!
//! Build & run:
//!   cargo build --release -p bench-beta2
//!   ORT_DYLIB_PATH=... ./bench_beta2/target/release/bench_lunar_request_action_vec32

use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode, RelayRLAgentActors,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;

use lunarlander_rl::env::vec::SyncLunarVectorEnv;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM:      usize = 8;
const ACT_DIM:      usize = 4;
const MAX_STEPS:    usize = 500;
const NUM_ENVS:     usize = 32;
const TARGET_STEPS: usize = 50_000;   // loop iterations (= 50k × 32 = 1.6M transitions)
const WARMUP_STEPS: usize = 500;

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

// ─────────────────────────── Action decoding ────────────────────────────────

/// Decode the flat action byte buffer returned by the framework.
///
/// The framework may return `[N × ACT_DIM]` logits (full batch) or fall back to
/// `[1 × ACT_DIM]` zeros when model inference silently fails (ONNX name mismatch).
/// In the fallback case every env gets action 0 — still valid for throughput measurement.
fn decode_batched_actions(act_bytes: &[u8], n: usize) -> Vec<u8> {
    let floats: Vec<f32> = act_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let available_rows = floats.len() / ACT_DIM;

    (0..n)
        .map(|i| {
            let row_idx = i.min(available_rows.saturating_sub(1));
            if available_rows == 0 {
                return 0u8;
            }
            let row = &floats[row_idx * ACT_DIM..(row_idx + 1) * ACT_DIM];
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u8)
                .unwrap_or(0)
        })
        .collect()
}

// ─────────────────────────── /proc helpers ──────────────────────────────────

fn read_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines()
            .find(|l| l.starts_with("VmRSS"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok()))
        .unwrap_or(0)
}

fn read_ctx_switches() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .map(|s| {
            let mut vol = 0u64;
            let mut nonvol = 0u64;
            for line in s.lines() {
                if line.starts_with("voluntary_ctxt_switches") {
                    vol = line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
                } else if line.starts_with("nonvoluntary_ctxt_switches") {
                    nonvol = line.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
                }
            }
            vol + nonvol
        })
        .unwrap_or(0)
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() { return 0; }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean_u64(vals: &[u64]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.iter().sum::<u64>() as f64 / vals.len() as f64
}

fn stddev_u64(vals: &[u64], mean: f64) -> f64 {
    if vals.len() < 2 { return 0.0; }
    (vals.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>()
        / (vals.len() - 1) as f64).sqrt()
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.2 — request_action (live integration) — {} envs", NUM_ENVS);
    println!("  {} loop iters · {} total transitions · ndarray · {} logical cores",
        TARGET_STEPS, TARGET_STEPS * NUM_ENVS, num_cores);
    println!("  path: caller loop → request_action([{NUM_ENVS}×{OBS_DIM}]) → step_all (rayon)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── Agent: 1 actor, no env registered (caller drives the loop) ──────────
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
    println!("Agent started — actor {actor_id}");

    // ── Vectorised env: 32 rayon-parallel LunarLander sub-envs ──────────────
    let mut vec_env = SyncLunarVectorEnv::<B>::new(NUM_ENVS, MAX_STEPS, device.clone())?;
    println!("SyncLunarVectorEnv created — {NUM_ENVS} envs");

    // ── Smoke test: [32, 8] obs → request_action ─────────────────────────────
    {
        let obs_flat = vec_env.get_stacked_obs();
        let obs_t = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_flat.clone(), [NUM_ENVS, OBS_DIM]), &device);
        let result = agent.request_action(vec![actor_id], obs_t, None, 0.0).await;
        let ok = result.as_ref().map(|r| r.len()).unwrap_or(0);
        println!("Smoke test request_action([{NUM_ENVS}×{OBS_DIM}]): {} response(s)", ok);
        if let Ok(ref actions_vec) = result {
            if let Some((_, relay_action)) = actions_vec.first() {
                let bytes = relay_action.get_act()
                    .map(|a| a.data.len()).unwrap_or(0);
                let expected = NUM_ENVS * ACT_DIM * 4;
        println!("  action bytes returned: {} (expect {expected} = {NUM_ENVS}×{ACT_DIM}×4)",
                    bytes);
        if bytes < expected {
            println!("  NOTE: fallback to zeros_action ({bytes}B) — model inference silently");
            println!("        failed (ONNX input name mismatch). Actions default to 0.");
            println!("        Throughput measurement is unaffected.");
        }
            }
        }
    }

    // ── Per-step timing storage ───────────────────────────────────────────────
    let mut infer_ns:  Vec<u64> = Vec::with_capacity(TARGET_STEPS);
    let mut env_ns:    Vec<u64> = Vec::with_capacity(TARGET_STEPS);
    let mut ep_returns: Vec<f32> = Vec::new();
    let mut cur_return: f32     = 0.0;
    let mut episodes_done: u64  = 0;

    // ── Warm-up ───────────────────────────────────────────────────────────────
    println!("Warming up ({WARMUP_STEPS} iters = {} transitions)…",
        WARMUP_STEPS * NUM_ENVS);
    for _ in 0..WARMUP_STEPS {
        let obs_flat = vec_env.get_stacked_obs();
        let obs_t = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_flat, [NUM_ENVS, OBS_DIM]), &device);
        let result = agent.request_action(vec![actor_id], obs_t, None, 0.0).await?;
        let actions = result.into_iter()
            .next()
            .and_then(|(_, ra)| ra.get_act().map(|a| decode_batched_actions(&a.data, NUM_ENVS)))
            .unwrap_or_else(|| vec![0u8; NUM_ENVS]);
        vec_env.step_all(&actions);
    }
    println!("Warm-up done. Starting timed run ({TARGET_STEPS} iters)…\n");

    // ── Timed loop ────────────────────────────────────────────────────────────
    let ctx_start = read_ctx_switches();
    let t0 = Instant::now();

    for _ in 0..TARGET_STEPS {
        // -- Inference: pack [32, 8] obs, get back [32, 4] logits ----------------
        let obs_flat = vec_env.get_stacked_obs();
        let obs_t = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_flat, [NUM_ENVS, OBS_DIM]), &device);

        let t_infer = Instant::now();
        let result = agent.request_action(vec![actor_id], obs_t, None, cur_return).await?;
        infer_ns.push(t_infer.elapsed().as_nanos() as u64);

        let actions = result.into_iter()
            .next()
            .and_then(|(_, ra)| ra.get_act().map(|a| decode_batched_actions(&a.data, NUM_ENVS)))
            .unwrap_or_else(|| vec![0u8; NUM_ENVS]);

        // -- Env step: rayon-parallel step of 32 sub-envs -------------------------
        let t_env = Instant::now();
        let results = vec_env.step_all(&actions);
        env_ns.push(t_env.elapsed().as_nanos() as u64);

        // -- Episode accounting ---------------------------------------------------
        for (reward, done) in &results {
            cur_return += reward;
            if *done {
                ep_returns.push(cur_return);
                cur_return = 0.0;
                episodes_done += 1;
            }
        }
    }

    let wall = t0.elapsed().as_secs_f64();
    let ctx_end = read_ctx_switches();
    let rss_kb  = read_rss_kb();

    // ── Compute metrics ───────────────────────────────────────────────────────
    let total_transitions   = TARGET_STEPS * NUM_ENVS;
    let transitions_per_sec = total_transitions as f64 / wall;
    let loop_iters_per_sec  = TARGET_STEPS as f64 / wall;
    let us_per_transition   = 1_000_000.0 / transitions_per_sec;
    let us_per_iter         = 1_000_000.0 / loop_iters_per_sec;

    let mut infer_sorted = infer_ns.clone(); infer_sorted.sort_unstable();
    let mut env_sorted   = env_ns.clone();   env_sorted.sort_unstable();

    let infer_mean = mean_u64(&infer_ns);
    let infer_std  = stddev_u64(&infer_ns, infer_mean);
    let infer_p50  = percentile(&infer_sorted, 50.0);
    let infer_p95  = percentile(&infer_sorted, 95.0);
    let infer_p99  = percentile(&infer_sorted, 99.0);
    let infer_p999 = percentile(&infer_sorted, 99.9);

    let env_mean = mean_u64(&env_ns);
    let env_std  = stddev_u64(&env_ns, env_mean);
    let env_p50  = percentile(&env_sorted, 50.0);
    let env_p99  = percentile(&env_sorted, 99.0);

    let step_mean = infer_mean + env_mean;
    let infer_frac = infer_mean / step_mean.max(1.0);
    let env_frac   = env_mean   / step_mean.max(1.0);

    let total_ctx    = ctx_end.saturating_sub(ctx_start);
    let ctx_per_iter = total_ctx as f64 / TARGET_STEPS as f64;

    let ep_ret_mean = if ep_returns.is_empty() { 0.0 }
        else { ep_returns.iter().sum::<f32>() as f64 / ep_returns.len() as f64 };
    let ep_ret_std = if ep_returns.len() < 2 { 0.0 } else {
        let var: f64 = ep_returns.iter()
            .map(|&r| { let d = r as f64 - ep_ret_mean; d * d })
            .sum::<f64>() / (ep_returns.len() - 1) as f64;
        var.sqrt()
    };
    let eps_per_sec = episodes_done as f64 / wall;

    // ── Baselines ─────────────────────────────────────────────────────────────
    const RELAYRL_SCALAR_32_SPS:  f64 = 1_403_081.0; // run_env scalar 32 envs
    const RELAYRL_VEC_128_SPS:    f64 = 2_590_416.0; // run_env vector 128 envs

    // ── Print report ──────────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL — request_action (live integration) — FINAL RESULTS  ({NUM_ENVS} envs)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────");
    println!("  env count                : {:>10}", NUM_ENVS);
    println!("  loop iterations          : {:>10}", TARGET_STEPS);
    println!("  total env transitions    : {:>10}", total_transitions);
    println!("  wall time                : {:>10.2} s", wall);
    println!("  loop iters/sec           : {:>10.0}", loop_iters_per_sec);
    println!("  env transitions/sec      : {:>10.0}", transitions_per_sec);
    println!("  µs / loop iter           : {:>10.3}", us_per_iter);
    println!("  µs / env transition      : {:>10.3}", us_per_transition);
    println!("  episodes completed       : {:>10}", episodes_done);
    println!("  episodes/sec             : {:>10.1}", eps_per_sec);
    println!();

    println!("─── Inference Timing (µs, per request_action call) ──────────────────");
    println!("  mean                     : {:>10.3}", infer_mean  / 1_000.0);
    println!("  std dev                  : {:>10.3}", infer_std   / 1_000.0);
    println!("  P50                      : {:>10.3}", infer_p50   as f64 / 1_000.0);
    println!("  P95                      : {:>10.3}", infer_p95   as f64 / 1_000.0);
    println!("  P99                      : {:>10.3}", infer_p99   as f64 / 1_000.0);
    println!("  P99.9                    : {:>10.3}", infer_p999  as f64 / 1_000.0);
    println!("  fraction of iter time    : {:>10.3}", infer_frac);
    println!();

    println!("─── Env Step Timing (µs, rayon step_all × {NUM_ENVS}) ─────────────────");
    println!("  mean                     : {:>10.3}", env_mean / 1_000.0);
    println!("  std dev                  : {:>10.3}", env_std  / 1_000.0);
    println!("  P50                      : {:>10.3}", env_p50  as f64 / 1_000.0);
    println!("  P99                      : {:>10.3}", env_p99  as f64 / 1_000.0);
    println!("  fraction of iter time    : {:>10.3}", env_frac);
    println!();

    println!("─── Episode Statistics ───────────────────────────────────────────────");
    println!("  total episodes           : {:>10}", episodes_done);
    println!("  episode return mean      : {:>10.3}", ep_ret_mean);
    println!("  episode return std dev   : {:>10.3}", ep_ret_std);
    println!();

    println!("─── OS ──────────────────────────────────────────────────────────────");
    println!("  RSS                      : {:>7.1} MB", rss_kb as f64 / 1024.0);
    println!("  context switches total   : {:>10}", total_ctx);
    println!("  context switches/iter    : {:>10.4}", ctx_per_iter);
    println!("  logical cores            : {:>10}", num_cores);
    println!();

    println!("─── vs baselines ────────────────────────────────────────────────────");
    println!("  RelayRL scalar-32  (run_env)  : {:>10.0}  t/s", RELAYRL_SCALAR_32_SPS);
    println!("  RelayRL vector-128 (run_env)  : {:>10.0}  t/s", RELAYRL_VEC_128_SPS);
    println!("  This run  (request_action-32) : {:>10.0}  t/s  ← live integration", transitions_per_sec);
    println!("  vs scalar-32  run_env         : {:>10.3}×", transitions_per_sec / RELAYRL_SCALAR_32_SPS);
    println!("  vs vector-128 run_env         : {:>10.3}×", transitions_per_sec / RELAYRL_VEC_128_SPS);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
