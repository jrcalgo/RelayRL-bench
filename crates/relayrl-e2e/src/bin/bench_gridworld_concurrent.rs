//! bench_gridworld_concurrent — N actors, each owning one GridWorld env,
//! all running in independent tokio tasks concurrently.
//!
//! Tests whether per-actor throughput remains stable as actor count grows.
//! Each tokio task issues request_action for its own actor_id in a tight loop,
//! independent of every other task.  No block_on or sequential joins in the
//! hot path; all tasks run until they finish --steps-per-actor, then
//! futures::future::join_all awaits them asynchronously.
//!
//! Build:
//!   cargo build --release -p relayrl-e2e
//!
//! Run:
//!   ORT_DYLIB_PATH=... ./target/release/bench_gridworld_concurrent
//!   ORT_DYLIB_PATH=... ./target/release/bench_gridworld_concurrent --actor-count 128

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use clap::Parser;

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode, RelayRLAgentActors,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;

use gridworld_rl::env::vec::SyncVectorEnv;

// ─────────────────────────── CLI ────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "bench_gridworld_concurrent",
          about = "RelayRL GridWorld — N concurrent actors, 1 env each")]
struct Args {
    /// Number of concurrent actors (each gets its own tokio task + env)
    #[arg(long, default_value_t = 32)]
    actor_count: usize,

    /// Steps per actor (total env steps = actor_count × steps_per_actor)
    #[arg(long, default_value_t = 100_000u64)]
    steps_per_actor: u64,
}

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM:  usize = 100;  // 10×10 grid
const ACT_DIM:  usize = 4;
const GRID_SIZE: usize = 10;

// ─────────────────────────── Bootstrap model ────────────────────────────────

fn bootstrap_model<B>() -> Result<ModelModule<B>, Box<dyn std::error::Error>>
where
    B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>,
{
    use relayrl_types::data::tensor::{DType, NdArrayDType};
    use relayrl_types::model::{ModelFileType, ModelMetadata};
    let specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = vec![
        (OBS_DIM, 64, vec![0.01f32; 64 * OBS_DIM], vec![0.0f32; 64]),
        (64,      64, vec![0.01f32; 64 * 64],       vec![0.0f32; 64]),
        (64, ACT_DIM, vec![0.01f32; ACT_DIM * 64],  vec![0.0f32; ACT_DIM]),
    ];
    let bytes = build_onnx_mlp_bytes(&specs);
    let meta = ModelMetadata {
        model_file:     "bootstrap.onnx".into(),
        model_type:     ModelFileType::Onnx,
        input_dtype:    DType::NdArray(NdArrayDType::F32),
        output_dtype:   DType::NdArray(NdArrayDType::F32),
        input_shape:    vec![1, OBS_DIM],
        output_shape:   vec![1, ACT_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(bytes, meta)?)
}

// ─────────────────────────── Action decoding ────────────────────────────────

fn decode_action(relay: &relayrl_types::data::action::RelayRLAction) -> u8 {
    relay.get_act()
        .map(|d| d.data.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u8)
            .unwrap_or(0))
        .unwrap_or(0)
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let args        = Args::parse();
    let actor_count = args.actor_count;
    let steps_per   = args.steps_per_actor;
    let total_steps = actor_count as u64 * steps_per;
    let num_cores   = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL GridWorld — concurrent {}-actor benchmark", actor_count);
    println!("  {} actors × {} steps/actor = {} total env steps",
             actor_count, steps_per, total_steps);
    println!("  {} independent tokio tasks, 1 SyncVectorEnv(1) per task",
             actor_count);
    println!("  {} logical cores", num_cores);
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── Agent ────────────────────────────────────────────────────────────────
    let model   = bootstrap_model::<B>()?;
    let cfgpath = std::path::PathBuf::from("./config.json");
    let mut bld = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(actor_count as u32)
        .default_device(DeviceType::Cpu)
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .default_model(model)
        .router_scale(actor_count as u32);
    if cfgpath.exists() { bld = bld.config_path(cfgpath); }

    let (mut agent, params) = bld.build().await?;
    agent.start(params).await?;
    let actor_ids = agent.get_actor_ids()?;
    assert_eq!(actor_ids.len(), actor_count, "expected {} actor IDs", actor_count);

    let agent = Arc::new(agent);

    // ── Warm-up: 200 steps per actor, sequential ─────────────────────────────
    println!("Warming up (200 steps × {} actors)…", actor_count);
    {
        let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
        for &actor_id in &actor_ids {
            let mut env = SyncVectorEnv::<B>::new(1, GRID_SIZE, device.clone())?;
            for _ in 0..200u32 {
                let obs   = env.get_stacked_obs();
                let obs_t = Tensor::<B, 2, Float>::from_data(
                    TensorData::new(obs, [1, OBS_DIM]), &device);
                let _ = agent.request_action(vec![actor_id], obs_t, None, 0.0).await;
                env.step_all(&[0u8]);
            }
        }
    }
    println!("Warm-up done. Spawning {} concurrent tasks…\n", actor_count);

    // ── Shared progress counter ───────────────────────────────────────────────
    let total_done = Arc::new(AtomicU64::new(0));

    // ── Spawn N independent tokio tasks ──────────────────────────────────────
    let t_start = Instant::now();

    let mut handles = Vec::with_capacity(actor_count);
    for i in 0..actor_count {
        let agent_arc = agent.clone();
        let actor_id  = actor_ids[i];
        let counter   = total_done.clone();

        handles.push(tokio::spawn(async move {
            let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
            let mut env = SyncVectorEnv::<B>::new(1, GRID_SIZE, device.clone())
                .expect("SyncVectorEnv init");

            let mut steps_done:    u64 = 0;
            let mut call_total_ns: u64 = 0;
            let mut cur_reward:    f32 = 0.0;

            while steps_done < steps_per {
                let obs   = env.get_stacked_obs();
                let obs_t = Tensor::<B, 2, Float>::from_data(
                    TensorData::new(obs, [1, OBS_DIM]), &device);

                let call_t0 = Instant::now();
                let result  = agent_arc
                    .request_action(vec![actor_id], obs_t, None, cur_reward)
                    .await;
                call_total_ns += call_t0.elapsed().as_nanos() as u64;

                let act = result.as_ref().ok()
                    .and_then(|v| v.first())
                    .map(|(_, a)| decode_action(a))
                    .unwrap_or(0);

                let outcomes = env.step_all(&[act]);
                if let Some(&(r, done)) = outcomes.first() {
                    cur_reward = r;
                    if done {
                        let _ = agent_arc
                            .flag_last_action(vec![actor_id], Some(cur_reward))
                            .await;
                        cur_reward = 0.0;
                    }
                }

                steps_done                       += 1;
                counter.fetch_add(1, Ordering::Relaxed);
            }

            (steps_done, call_total_ns)
        }));
    }

    // ── Progress printer (async, does not block worker tasks) ────────────────
    let progress_counter = total_done.clone();
    let printer = tokio::spawn(async move {
        let mut prev = 0u64;
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            let done = progress_counter.load(Ordering::Relaxed);
            let rate = done.saturating_sub(prev);
            prev = done;
            println!("  [{:>10} / {} steps] {:>8.0} steps/sec",
                     done, total_steps, rate as f64);
            if done >= total_steps { break; }
        }
    });

    // ── Await all worker tasks (async join, not block_on) ────────────────────
    let results = futures::future::join_all(handles).await;
    printer.abort();

    let elapsed = t_start.elapsed().as_secs_f64();

    // ── Aggregate stats ───────────────────────────────────────────────────────
    let mut all_steps:     u64 = 0;
    let mut total_call_ns: u64 = 0;
    let mut task_errors:   u32 = 0;
    for r in &results {
        match r {
            Ok((steps, call_ns)) => { all_steps += steps; total_call_ns += call_ns; }
            Err(_) => task_errors += 1,
        }
    }

    let env_sps       = all_steps as f64 / elapsed;
    let per_actor_sps = env_sps / actor_count as f64;
    let avg_call_us   = total_call_ns as f64 / 1_000.0 / all_steps.max(1) as f64;

    // /proc snapshot
    let mut rss_kb:    u64 = 0;
    let mut ctx_total: u64 = 0;
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            let mut it = line.splitn(2, ':');
            let k = it.next().unwrap_or("").trim();
            let v = it.next().unwrap_or("").trim();
            match k {
                "VmRSS" => rss_kb = v.split_whitespace().next()
                                      .and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches" | "nonvoluntary_ctxt_switches" =>
                    ctx_total += v.parse::<u64>().unwrap_or(0),
                _ => {}
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL GridWorld — {} concurrent actors — FINAL RESULTS", actor_count);
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────");
    println!("  actor count                  : {:>10}", actor_count);
    println!("  steps per actor              : {:>10}", steps_per);
    println!("  total env steps              : {:>10}", all_steps);
    println!("  wall time                    : {:>10.2}s", elapsed);
    println!("  env steps/sec (total)        : {:>10.0}", env_sps);
    println!("  env steps/sec per actor      : {:>10.0}", per_actor_sps);
    println!("  steps/sec / logical core     : {:>10.0}", env_sps / num_cores as f64);
    println!("  task errors                  : {:>10}", task_errors);
    println!();

    println!("─── Call Timing ──────────────────────────────────────────────────────");
    println!("  avg call latency (mean/actor): {:>10.3} µs", avg_call_us);
    println!();

    println!("─── Memory / OS ─────────────────────────────────────────────────────");
    println!("  RSS (final)                  : {:>7.1} MB", rss_kb as f64 / 1024.0);
    println!("  context switches (total)     : {:>10}", ctx_total);
    println!("  context switches/step        : {:>10.6}",
             ctx_total as f64 / all_steps.max(1) as f64);
    println!("  logical cores                : {:>10}", num_cores);
    println!();

    println!("─── vs baselines ────────────────────────────────────────────────────");
    const BASELINE_1A_SPS:  f64 = 305_166.0;  // bench_gridworld --actor-count 1
    const BASELINE_32A_SPS: f64 = 170_371.0;  // bench_gridworld_concurrent --actor-count 32
    println!("  1-actor sequential           : {:>10.0}  steps/sec", BASELINE_1A_SPS);
    println!("  32-actor concurrent (prior)  : {:>10.0}  steps/sec  (total)",
             BASELINE_32A_SPS);
    println!("  this run (per-actor)         : {:>10.0}  steps/sec", per_actor_sps);
    println!("  this run (total)             : {:>10.0}  steps/sec", env_sps);
    println!("  per-actor throughput retained: {:>10.2}%",
             per_actor_sps / BASELINE_1A_SPS * 100.0);
    println!("  aggregate speedup vs 1-actor : {:>10.2}×",
             env_sps / BASELINE_1A_SPS);
    println!("  ideal linear speedup         : {:>10.2}×  ({}×)",
             actor_count as f64, actor_count);
    println!("  scaling efficiency           : {:>10.2}%",
             env_sps / BASELINE_1A_SPS / actor_count as f64 * 100.0);
    println!("═══════════════════════════════════════════════════════════════════");

    drop(results);
    let mut agent_owned = Arc::try_unwrap(agent)
        .expect("Arc<agent> still borrowed after all tasks completed");
    agent_owned.shutdown().await?;
    Ok(())
}
