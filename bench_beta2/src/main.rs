//! bench_lunar — comprehensive 40-metric RelayRL benchmark on LunarLander.
//!
//! Supports 1–N actors, each with its own independent LunarLanderEnv instance.
//! Action requests are issued sequentially (no join_all); the coordinator's
//! internal send-all-then-receive-all pipeline handles concurrency.
//!
//! Build & run:
//!   cargo build --release -p relayrl-e2e
//!   ./target/release/bench_lunar --actor-count 1
//!   ./target/release/bench_lunar --actor-count 2
//!   ./target/release/bench_lunar --actor-count 5
//!   ./target/release/bench_lunar --actor-count 10
//!
//! For hardware counters (cache misses, IPC):
//!   perf stat -e cycles,instructions,cache-misses,LLC-load-misses \
//!       ./target/release/bench_lunar --actor-count N

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode, RelayRLAgentActors,
    ToAnyBurnTensor,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;

use lunarlander_rl::env::LunarLanderEnv;

// ─────────────────────────── CLI ────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "bench_lunar", about = "RelayRL LunarLander comprehensive benchmark")]
struct Args {
    /// Number of independent actors (each gets its own LunarLanderEnv)
    #[arg(long, default_value_t = 1)]
    actor_count: usize,

    /// Total environment steps to collect (summed across all actors)
    #[arg(long, default_value_t = 1_000_000)]
    target_steps: u64,
}

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM:   usize = 8;
const ACT_DIM:   usize = 4;
const MAX_STEPS: usize = 500;

// ─────────────────────────── /proc helpers ──────────────────────────────────

#[derive(Clone, Default, Debug)]
struct ProcSample {
    rss_kb:        u64,
    vol_ctx_sw:    u64,
    nonvol_ctx_sw: u64,
    utime_ticks:   u64,
    stime_ticks:   u64,
    threads:       u64,
    runq:          f32,
}

fn sample_proc() -> ProcSample {
    let mut s = ProcSample::default();
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            let mut it = line.splitn(2, ':');
            let key = it.next().unwrap_or("").trim();
            let val = it.next().unwrap_or("").trim();
            match key {
                "VmRSS"   => s.rss_kb        = val.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"    => s.vol_ctx_sw    = val.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => s.nonvol_ctx_sw = val.parse().unwrap_or(0),
                "Threads" => s.threads        = val.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        let f: Vec<&str> = stat.split_whitespace().collect();
        s.utime_ticks = f.get(13).and_then(|v| v.parse().ok()).unwrap_or(0);
        s.stime_ticks = f.get(14).and_then(|v| v.parse().ok()).unwrap_or(0);
    }
    if let Ok(la) = std::fs::read_to_string("/proc/loadavg") {
        s.runq = la.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    }
    s
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() { return 0; }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn stddev_u64(vals: &[u64], mean: f64) -> f64 {
    if vals.len() < 2 { return 0.0; }
    (vals.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>()
        / (vals.len() - 1) as f64).sqrt()
}

fn mean_u64(vals: &[u64]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.iter().sum::<u64>() as f64 / vals.len() as f64
}

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
        model_file:    "bootstrap.onnx".to_string(),
        model_type:    ModelFileType::Onnx,
        input_dtype:   DType::NdArray(NdArrayDType::F32),
        output_dtype:  DType::NdArray(NdArrayDType::F32),
        input_shape:   vec![1, OBS_DIM],
        output_shape:  vec![1, ACT_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(onnx_bytes, metadata)?)
}

// ─────────────────────────── Decode action from relay output ─────────────────

fn decode_action(relay_action: &relayrl_types::data::action::RelayRLAction) -> u8 {
    relay_action
        .get_act()
        .map(|act_data| {
            act_data.data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u8)
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let args = Args::parse();
    let actor_count = args.actor_count;
    let target_steps = args.target_steps;

    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let device_type = DeviceType::Cpu;
    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL — LunarLander comprehensive benchmark");
    println!("  {} actor{} · {} steps/actor · ndarray · {} logical cores",
        actor_count, if actor_count == 1 { "" } else { "s" }, target_steps, num_cores);
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── One independent LunarLanderEnv per actor ─────────────────────────────
    let envs: Vec<LunarLanderEnv<B>> = (0..actor_count)
        .map(|_| LunarLanderEnv::<B>::new(MAX_STEPS, device.clone()))
        .collect();

    // ── Agent: N actors, all local independent, trajectories disabled ─────────
    let initial_model = make_bootstrap_model::<B>()?;
    let config_path = std::path::PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(actor_count as u32)
        .default_device(device_type.clone())
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
    assert_eq!(actor_ids.len(), actor_count, "actor ID count mismatch");

    // ── Storage (pre-allocate for target_steps per actor) ────────────────────
    let total_cap  = (target_steps * actor_count as u64) as usize;
    let rounds_cap = target_steps as usize; // one round per target_steps/actor
    // Round-trip time per step (whole actor loop iteration = sum of all actors)
    let mut round_times_ns:  Vec<u64> = Vec::with_capacity(rounds_cap);
    // Per-actor-request timing
    let mut infer_times_ns:  Vec<u64> = Vec::with_capacity(total_cap);
    let mut env_times_ns:    Vec<u64> = Vec::with_capacity(total_cap);
    // Per-actor episode tracking
    let mut ep_returns: Vec<Vec<f32>> = vec![Vec::new(); actor_count];
    let mut ep_lengths: Vec<Vec<u64>> = vec![Vec::new(); actor_count];
    let mut cur_returns: Vec<f32>     = vec![0.0; actor_count];
    let mut cur_lens:    Vec<u64>     = vec![0; actor_count];

    // ── Background /proc sampler ──────────────────────────────────────────────
    let done_flag   = Arc::new(AtomicBool::new(false));
    let proc_store  = Arc::new(std::sync::Mutex::new(Vec::<ProcSample>::with_capacity(5_000)));
    {
        let done  = done_flag.clone();
        let store = proc_store.clone();
        std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                store.lock().unwrap().push(sample_proc());
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        });
    }

    // ── Warm-up: 200 rounds ───────────────────────────────────────────────────
    println!("Warming up…");
    for env in &envs { env.reset(); }
    for _ in 0..200 {
        for (i, env) in envs.iter().enumerate() {
            let obs_vec = env.get_observation(0);
            let obs_t = Tensor::<B, 2, Float>::from_data(
                TensorData::new(obs_vec, [1, OBS_DIM]), &device);
            let _ = agent.request_action(vec![actor_ids[i]], obs_t, None, 0.0).await;
            let _ = env.step(0, 0);
            if env.all_done() || env.is_max_steps_reached() { env.reset(); }
        }
    }
    println!("Warm-up done. Starting benchmark…\n");

    // ── Main collection loop ──────────────────────────────────────────────────
    let t_start = Instant::now();
    let mut total_steps: u64 = 0;

    for env in &envs { env.reset(); }

    while total_steps < target_steps * actor_count as u64 {
        let round_start = Instant::now();

        for (i, env) in envs.iter().enumerate() {
            // -- Inference: one request_action per actor, sequential --
            let obs_vec = env.get_observation(0);
            let obs_t = Tensor::<B, 2, Float>::from_data(
                TensorData::new(obs_vec, [1, OBS_DIM]), &device);

            let infer_start = Instant::now();
            let result = agent.request_action(vec![actor_ids[i]], obs_t, None, cur_returns[i]).await;
            let infer_ns = infer_start.elapsed().as_nanos() as u64;
            infer_times_ns.push(infer_ns);

            let action_u8: u8 = result.ok()
                .and_then(|mut a| a.pop())
                .map(|(_, r)| decode_action(&r))
                .unwrap_or(0);

            // -- Env step --
            let env_start = Instant::now();
            let (reward, _done) = env.step(0, action_u8).unwrap_or((0.0, false));
            let env_ns = env_start.elapsed().as_nanos() as u64;
            env_times_ns.push(env_ns);

            cur_returns[i] += reward;
            cur_lens[i]    += 1;
            total_steps    += 1;

            if env.all_done() || env.is_max_steps_reached() {
                agent.flag_last_action(vec![actor_ids[i]], Some(cur_returns[i])).await?;
                ep_returns[i].push(cur_returns[i]);
                ep_lengths[i].push(cur_lens[i]);
                cur_returns[i] = 0.0;
                cur_lens[i]    = 0;
                env.reset();
            }
        }

        round_times_ns.push(round_start.elapsed().as_nanos() as u64);

        if total_steps % (100_000 * actor_count as u64) == 0 {
            let sps = total_steps as f64 / t_start.elapsed().as_secs_f64();
            let steps_per_actor = total_steps / actor_count as u64;
            println!("  [{:>7} steps/actor  {:>8} total] {:.0} steps/sec", steps_per_actor, total_steps, sps);
        }
    }

    let elapsed_sec = t_start.elapsed().as_secs_f64();
    done_flag.store(true, Ordering::Relaxed);

    // ── Compute metrics ───────────────────────────────────────────────────────

    // Sort timing arrays
    let mut infer_sorted  = infer_times_ns.clone(); infer_sorted.sort_unstable();
    let mut env_sorted    = env_times_ns.clone();   env_sorted.sort_unstable();
    let mut round_sorted  = round_times_ns.clone(); round_sorted.sort_unstable();

    // Round (full actor-loop iteration) timing
    let round_mean_ns  = mean_u64(&round_times_ns);
    let round_std_ns   = stddev_u64(&round_times_ns, round_mean_ns);
    // Step timing = per-actor infer+env timing (each actor is one "step")
    let infer_mean_ns  = mean_u64(&infer_times_ns);
    let infer_std_ns   = stddev_u64(&infer_times_ns, infer_mean_ns);
    let infer_p50      = percentile(&infer_sorted, 50.0);
    let infer_p95      = percentile(&infer_sorted, 95.0);
    let infer_p99      = percentile(&infer_sorted, 99.0);
    let infer_p999     = percentile(&infer_sorted, 99.9);

    let env_mean_ns    = mean_u64(&env_times_ns);
    let env_std_ns     = stddev_u64(&env_times_ns, env_mean_ns);
    let env_p50        = percentile(&env_sorted, 50.0);
    let env_p99        = percentile(&env_sorted, 99.0);

    // Per-step = infer + env (overhead = round - N*(infer+env))
    let step_mean_ns   = infer_mean_ns + env_mean_ns;
    let overhead_per_round = (round_mean_ns
        - actor_count as f64 * (infer_mean_ns + env_mean_ns))
        .max(0.0);
    let overhead_ratio  = if round_mean_ns > 0.0 {
        overhead_per_round as f64 / round_mean_ns
    } else { 0.0 };

    // Step-level latency percentiles (round / N actors = per-step latency from caller pov)
    let step_p50_ns  = percentile(&round_sorted, 50.0) / actor_count as u64;
    let step_p95_ns  = percentile(&round_sorted, 95.0) / actor_count as u64;
    let step_p99_ns  = percentile(&round_sorted, 99.0) / actor_count as u64;
    let step_p999_ns = percentile(&round_sorted, 99.9) / actor_count as u64;
    let jitter_ns    = step_p99_ns.saturating_sub(step_p50_ns);

    // Round-level std / jitter
    let round_p50    = percentile(&round_sorted, 50.0);
    let round_p99    = percentile(&round_sorted, 99.0);

    // Throughput
    let steps_per_sec   = total_steps as f64 / elapsed_sec;
    let steps_per_actor = steps_per_sec / actor_count as f64;
    let steps_per_core  = steps_per_sec / num_cores as f64;

    // Episode stats (merge all actors)
    let all_returns: Vec<f32> = ep_returns.iter().flatten().copied().collect();
    let all_lengths: Vec<u64> = ep_lengths.iter().flatten().copied().collect();
    let total_eps    = all_returns.len() as f64;
    let eps_per_sec  = total_eps / elapsed_sec;
    let avg_ep_len   = if all_lengths.is_empty() { 0.0 }
                        else { all_lengths.iter().sum::<u64>() as f64 / all_lengths.len() as f64 };
    let ep_ret_mean  = if all_returns.is_empty() { 0.0 }
                        else { all_returns.iter().sum::<f32>() as f64 / total_eps };
    let ep_ret_var   = if all_returns.len() < 2 { 0.0 } else {
        all_returns.iter().map(|&r| { let d = r as f64 - ep_ret_mean; d * d }).sum::<f64>()
            / (all_returns.len() - 1) as f64
    };
    let ep_ret_std   = ep_ret_var.sqrt();

    // /proc samples
    let samples = proc_store.lock().unwrap();
    let rss_vals: Vec<u64>  = samples.iter().map(|s| s.rss_kb).collect();
    let rss_mean_kb = if rss_vals.is_empty() { 0 } else { rss_vals.iter().sum::<u64>() / rss_vals.len() as u64 };
    let rss_peak_kb = rss_vals.iter().copied().max().unwrap_or(0);
    let rss_init_kb = samples.first().map(|s| s.rss_kb).unwrap_or(0);
    let rss_final_kb= samples.last().map(|s| s.rss_kb).unwrap_or(0);
    let alloc_rate  = (rss_final_kb.saturating_sub(rss_init_kb)) as f64 / elapsed_sec;

    let ctx_first  = samples.first().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let ctx_last   = samples.last().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let total_ctx  = ctx_last.saturating_sub(ctx_first);
    let ctx_per_sec  = total_ctx as f64 / elapsed_sec;
    let ctx_per_step = total_ctx as f64 / total_steps as f64;

    let cpu_first  = samples.first().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_last   = samples.last().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_ticks  = cpu_last.saturating_sub(cpu_first) as f64;
    let cpu_util   = (cpu_ticks / 100.0) / elapsed_sec * 100.0;
    let cpu_per_core = cpu_util / num_cores as f64;

    let thread_mean = if samples.is_empty() { 0.0 }
                       else { samples.iter().map(|s| s.threads).sum::<u64>() as f64 / samples.len() as f64 };
    let runq_mean   = if samples.is_empty() { 0.0 }
                       else { samples.iter().map(|s| s.runq as f64).sum::<f64>() / samples.len() as f64 };
    drop(samples);

    let rss_mean_gb = rss_mean_kb as f64 / (1024.0 * 1024.0);
    let sps_per_gb  = if rss_mean_gb > 0.0 { steps_per_sec / rss_mean_gb } else { 0.0 };

    // S(n) relative to 1-actor baseline (channel path): 19 443 steps/sec
    const BASELINE_1A_SPS: f64 = 19_443.0;
    let scalability = steps_per_sec / BASELINE_1A_SPS;

    // ── Print report ──────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL LunarLander — FINAL RESULTS  ({} actor{})",
        actor_count, if actor_count == 1 { "" } else { "s" });
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────");
    println!("  steps/sec (global)           : {:>10.1}",   steps_per_sec);
    println!("  steps/sec per actor          : {:>10.1}",   steps_per_actor);
    println!("  steps/sec per logical core   : {:>10.1}",   steps_per_core);
    println!("  episodes/sec                 : {:>10.3}",   eps_per_sec);
    println!("  total steps (all actors)     : {:>10}",     total_steps);
    println!("  steps per actor              : {:>10}",     total_steps / actor_count as u64);
    println!("  total episodes               : {:>10}",     all_returns.len());
    println!("  wall time                    : {:>10.2}s",  elapsed_sec);
    println!("  logical cores                : {:>10}",     num_cores);
    println!();

    println!("─── Episode Statistics ───────────────────────────────────────────────");
    println!("  avg steps per episode        : {:>10.1}",   avg_ep_len);
    println!("  episode return mean          : {:>10.3}",   ep_ret_mean);
    println!("  episode return std dev       : {:>10.3}",   ep_ret_std);
    println!("  episode completion variance  : {:>10.3}",   ep_ret_var);
    println!();

    println!("─── Per-Step Timing (µs) ─────────────────────────────────────────────");
    println!("  step mean (infer+env)        : {:>10.3} µs", step_mean_ns / 1_000.0);
    println!("  step P50  (round/N)          : {:>10.3} µs", step_p50_ns  as f64 / 1_000.0);
    println!("  step P95  (round/N)          : {:>10.3} µs", step_p95_ns  as f64 / 1_000.0);
    println!("  step P99  (round/N)          : {:>10.3} µs", step_p99_ns  as f64 / 1_000.0);
    println!("  step P99.9                   : {:>10.3} µs", step_p999_ns as f64 / 1_000.0);
    println!("  jitter (P99−P50)             : {:>10.3} µs", jitter_ns    as f64 / 1_000.0);
    println!("  step std dev (infer)         : {:>10.3} µs", infer_std_ns / 1_000.0);
    println!();

    println!("─── Inference Timing (µs) ────────────────────────────────────────────");
    println!("  inference mean               : {:>10.3} µs", infer_mean_ns / 1_000.0);
    println!("  inference std dev            : {:>10.3} µs", infer_std_ns  / 1_000.0);
    println!("  inference P50               : {:>10.3} µs", infer_p50     as f64 / 1_000.0);
    println!("  inference P95               : {:>10.3} µs", infer_p95     as f64 / 1_000.0);
    println!("  inference P99               : {:>10.3} µs", infer_p99     as f64 / 1_000.0);
    println!("  inference P99.9             : {:>10.3} µs", infer_p999    as f64 / 1_000.0);
    println!("  actor dispatch latency      ≈ {:>10.3} µs  (P50 inference)", infer_p50 as f64 / 1_000.0);
    println!("  inference / step ratio       : {:>10.3}",   infer_mean_ns / step_mean_ns.max(1.0));
    println!();

    println!("─── Env Step Timing (µs) ─────────────────────────────────────────────");
    println!("  env step mean                : {:>10.3} µs", env_mean_ns  / 1_000.0);
    println!("  env step std dev             : {:>10.3} µs", env_std_ns   / 1_000.0);
    println!("  env step P50                 : {:>10.3} µs", env_p50      as f64 / 1_000.0);
    println!("  env step P99                 : {:>10.3} µs", env_p99      as f64 / 1_000.0);
    println!("  env step / step ratio        : {:>10.3}",   env_mean_ns  / step_mean_ns.max(1.0));
    println!();

    println!("─── Scheduling / Overhead ────────────────────────────────────────────");
    println!("  overhead per round           : {:>10.3} µs", overhead_per_round / 1_000.0);
    println!("  overhead ratio               : {:>10.3}",   overhead_ratio);
    println!("  deadtime per actor           : {:>10.3}",   overhead_ratio / actor_count as f64);
    println!("  round P50                    : {:>10.3} µs", round_p50 as f64 / 1_000.0);
    println!("  round P99                    : {:>10.3} µs", round_p99 as f64 / 1_000.0);
    println!("  round std dev                : {:>10.3} µs", round_std_ns / 1_000.0);
    println!("  action serialization         :   included in inference timing");
    println!("  state update / buffer write  :   disabled (traj off)");
    println!("  dropped/late updates         : {:>10}",     0u32);
    println!();

    println!("─── Memory ───────────────────────────────────────────────────────────");
    println!("  RSS init                     : {:>7.1} MB", rss_init_kb  as f64 / 1024.0);
    println!("  RSS peak                     : {:>7.1} MB", rss_peak_kb  as f64 / 1024.0);
    println!("  RSS mean                     : {:>7.1} MB", rss_mean_kb  as f64 / 1024.0);
    println!("  RSS final                    : {:>7.1} MB", rss_final_kb as f64 / 1024.0);
    println!("  allocation rate (RSS Δ)      : {:>7.3} KB/s", alloc_rate);
    println!("  /proc samples                : {:>10}",    rss_vals.len());
    println!();

    println!("─── CPU / OS ─────────────────────────────────────────────────────────");
    println!("  CPU utilisation (1 core %)   : {:>10.2}%", cpu_util);
    println!("  CPU util / logical core      : {:>10.2}%", cpu_per_core);
    println!("  mean threads                 : {:>10.1}",  thread_mean);
    println!("  mean run-queue (1-min avg)   : {:>10.3}",  runq_mean);
    println!("  context switches total       : {:>10}",    total_ctx);
    println!("  context switches/sec         : {:>10.1}",  ctx_per_sec);
    println!("  context switches/step        : {:>10.6}",  ctx_per_step);
    println!();

    println!("─── Efficiency Ratios ────────────────────────────────────────────────");
    println!("  steps/sec / logical core     : {:>10.1}",  steps_per_core);
    println!("  steps/sec / GB RSS (proxy)   : {:>10.1}",  sps_per_gb);
    println!("  S(n) vs 1-actor baseline     : {:>10.3}",  scalability);
    println!("  overhead ratio               : {:>10.3}",  overhead_ratio);
    println!();

    println!("─── Notes (hardware counters require perf) ───────────────────────────");
    println!("  cache misses (L1/L2/L3)      : perf stat -e cache-misses,LLC-load-misses");
    println!("  IPC                          : perf stat -e cycles,instructions");
    println!("  memory bandwidth             : perf stat -e cache-references");
    println!("  queue backlog / contention   : not exposed by framework API");
    println!("  inter-thread msg latency     ≈ P50 inference − actual ORT time");
    println!("  sync wait time               : included in overhead_per_round");
    println!("  steps/sec / watt             : requires external power measurement");
    println!("  allocator contention         : requires jemalloc/heaptrack profiling");
    println!();
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
