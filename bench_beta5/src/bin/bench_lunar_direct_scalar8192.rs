//! bench_lunar_direct_scalar8192 — live integration path, 8192 scalar envs.
//!
//! 8192 independent LunarLanderEnv instances stepped in parallel via rayon.
//! Each loop iteration:
//!   1. Collect obs in parallel → Tensor[8192, 8]
//!   2. request_action  → RelayRLAction (act = Tensor[8192, 4] logits)
//!   3. Argmax → Vec<u8> discrete actions
//!   4. step_all in parallel → Vec<(reward, done)>
//!   5. flag_last_action with mean reward
//!
//! This is the path a practitioner drives directly — no set_env / run_env.
//!
//! Build & run:
//!   cargo build --release -p bench-beta3 --bin bench_lunar_direct_scalar8192
//!   ORT_DYLIB_PATH=... perf stat ./target/release/bench_lunar_direct_scalar8192

use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use rayon::prelude::*;

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
const ENV_COUNT:    usize = 8192;
const TARGET_STEPS: usize = 5_000;
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

// ─────────────────────────── Helpers ────────────────────────────────────────

/// Collect flat [ENV_COUNT * OBS_DIM] observations from all envs in parallel.
fn collect_obs(envs: &[LunarLanderEnv<NdArray>]) -> Vec<f32> {
    let mut obs = vec![0.0f32; envs.len() * OBS_DIM];
    obs.par_chunks_mut(OBS_DIM)
        .zip(envs.par_iter())
        .for_each(|(chunk, env): (&mut [f32], &LunarLanderEnv<NdArray>)| {
            chunk.copy_from_slice(&env.get_observation(0));
        });
    obs
}

/// Step all envs in parallel; returns (reward, done) per env.
fn step_all(envs: &[LunarLanderEnv<NdArray>], actions: &[u8]) -> Vec<(f32, bool)> {
    envs.par_iter()
        .zip(actions.par_iter())
        .map(|(env, &act): (&LunarLanderEnv<NdArray>, &u8)| {
            let (reward, done) = env.step(0, act).unwrap_or((0.0, true));
            if done {
                env.reset();
            }
            (reward, done)
        })
        .collect()
}

/// Argmax each row of a row-major [N, ACT_DIM] F32 buffer (little-endian bytes).
fn logits_to_discrete_actions(data: &[u8], num_envs: usize) -> Vec<u8> {
    let mut actions = Vec::with_capacity(num_envs);
    for i in 0..num_envs {
        let mut best_idx = 0u8;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..ACT_DIM {
            let off = (i * ACT_DIM + j) * 4;
            let val = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            if val > best_val {
                best_val = val;
                best_idx = j as u8;
            }
        }
        actions.push(best_idx);
    }
    actions
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.3 — direct integration path — {} scalar envs", ENV_COUNT);
    println!("  request_action + flag_last_action per step · {} logical cores", num_cores);
    println!("  obs: [{ENV_COUNT}×{OBS_DIM}]  inference: batched ONNX ndarray");
    println!("  env stepping: rayon parallel ({ENV_COUNT} scalar LunarLanderEnv)");
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

    // ── Create ENV_COUNT independent scalar envs ──────────────────────────────
    let envs: Vec<LunarLanderEnv<B>> = (0..ENV_COUNT)
        .map(|_| {
            let e = LunarLanderEnv::<B>::new(MAX_STEPS, Default::default());
            e.reset();
            e
        })
        .collect();
    println!("Created {ENV_COUNT} scalar LunarLanderEnv instances");

    let mut prev_reward = 0.0f32;

    // ── Warm-up ───────────────────────────────────────────────────────────────
    println!("Warming up ({WARMUP_ITERS} iters = {} transitions)…",
             WARMUP_ITERS * ENV_COUNT);
    for _ in 0..WARMUP_ITERS {
        let obs_flat = collect_obs(&envs);
        let obs_tensor = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_flat, [ENV_COUNT, OBS_DIM]),
            &Default::default(),
        );

        let results = agent.request_action(vec![actor_id], obs_tensor, None, prev_reward).await?;

        let actions = if let Some((_, action)) = results.first() {
            if let Some(act_td) = action.get_act() {
                logits_to_discrete_actions(&act_td.data, ENV_COUNT)
            } else {
                vec![0u8; ENV_COUNT]
            }
        } else {
            vec![0u8; ENV_COUNT]
        };

        let step_results = step_all(&envs, &actions);
        prev_reward = step_results.iter().map(|(r, _)| r).sum::<f32>() / ENV_COUNT as f32;

        agent.flag_last_action(vec![actor_id], Some(prev_reward)).await?;
    }
    println!("Warm-up done. Starting timed run ({TARGET_STEPS} loop iters)…\n");

    // ── Timed run ─────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    for _ in 0..TARGET_STEPS {
        let obs_flat = collect_obs(&envs);
        let obs_tensor = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_flat, [ENV_COUNT, OBS_DIM]),
            &Default::default(),
        );

        let results = agent.request_action(vec![actor_id], obs_tensor, None, prev_reward).await?;

        let actions = if let Some((_, action)) = results.first() {
            if let Some(act_td) = action.get_act() {
                logits_to_discrete_actions(&act_td.data, ENV_COUNT)
            } else {
                vec![0u8; ENV_COUNT]
            }
        } else {
            vec![0u8; ENV_COUNT]
        };

        let step_results = step_all(&envs, &actions);
        prev_reward = step_results.iter().map(|(r, _)| r).sum::<f32>() / ENV_COUNT as f32;

        agent.flag_last_action(vec![actor_id], Some(prev_reward)).await?;
    }
    let wall = t0.elapsed().as_secs_f64();

    let loop_iters_per_sec  = TARGET_STEPS as f64 / wall;
    let env_transitions_sec = loop_iters_per_sec * ENV_COUNT as f64;
    let total_transitions   = TARGET_STEPS * ENV_COUNT;
    let us_per_loop_iter    = 1_000_000.0 / loop_iters_per_sec;
    let us_per_transition   = 1_000_000.0 / env_transitions_sec;

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
    let ctx_per_iter = total_ctx as f64 / TARGET_STEPS as f64;

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.3 — direct integration — FINAL RESULTS  ({ENV_COUNT} scalar envs)");
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
