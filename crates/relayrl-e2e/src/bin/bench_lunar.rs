//! bench_lunar — comprehensive 40-metric RelayRL benchmark on LunarLander.
//!
//! 1 actor · 1 000 000 steps · disable-traj (pure inference throughput).
//!
//! Build & run:
//!   cargo build --release -p relayrl-e2e
//!   ./target/release/bench_lunar
//!
//! For hardware counters (cache misses, IPC):
//!   perf stat -e cycles,instructions,cache-misses,LLC-load-misses \
//!       ./target/release/bench_lunar

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures::future::join_all;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, LocalTrajectoryFileParams,
    LocalTrajectoryFileType, ModelMode, RelayRLAgentActors, ToAnyBurnTensor,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;
use relayrl_algorithms::algorithms::PPO::PPOPolicyWithBaseline;
use relayrl_algorithms::algorithms::REINFORCE::ActivationKind;
use relayrl_algorithms::{AlgorithmTrait, RelayRLTrainer, TrainerArgs};

use lunarlander_rl::env::LunarLanderEnv;

// ─────────────────────────── Constants ──────────────────────────────────────

const TARGET_STEPS: u64 = 1_000_000;
const OBS_DIM:      usize = 8;
const ACT_DIM:      usize = 4;
const MAX_STEPS:    usize = 500;

// ─────────────────────────── /proc helpers ──────────────────────────────────

/// Process status fields sampled in background.
#[derive(Clone, Default, Debug)]
struct ProcSample {
    rss_kb:          u64, // VmRSS
    vol_ctx_sw:      u64, // voluntary_ctxt_switches
    nonvol_ctx_sw:   u64, // nonvoluntary_ctxt_switches
    utime_ticks:     u64, // utime from /proc/self/stat
    stime_ticks:     u64, // stime from /proc/self/stat
    threads:         u64, // Threads from /proc/self/status
    runq:            f32, // /proc/loadavg first field (1-min load avg proxy)
    ts_ns:           u64, // Instant::now() as nanos since epoch (monotonic delta)
}

fn sample_proc(t0: Instant) -> ProcSample {
    let mut s = ProcSample {
        ts_ns: t0.elapsed().as_nanos() as u64,
        ..Default::default()
    };

    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            let mut it = line.splitn(2, ':');
            let key = it.next().unwrap_or("").trim();
            let val = it.next().unwrap_or("").trim();
            match key {
                "VmRSS"   => s.rss_kb        = val.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"    => s.vol_ctx_sw    = val.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => s.nonvol_ctx_sw = val.parse().unwrap_or(0),
                "Threads" => s.threads       = val.parse().unwrap_or(0),
                _ => {}
            }
        }
    }

    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        let fields: Vec<&str> = stat.split_whitespace().collect();
        s.utime_ticks = fields.get(13).and_then(|v| v.parse().ok()).unwrap_or(0);
        s.stime_ticks = fields.get(14).and_then(|v| v.parse().ok()).unwrap_or(0);
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

fn stddev(vals: &[u64], mean: f64) -> f64 {
    if vals.len() < 2 { return 0.0; }
    let var = vals.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / (vals.len() - 1) as f64;
    var.sqrt()
}

fn mean_f64(vals: &[u64]) -> f64 {
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

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let device_type = DeviceType::Cpu;

    // ── Number of logical cores ──────────────────────────────────────────────
    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL — LunarLander comprehensive benchmark");
    println!("  1 actor · {} steps · ndarray backend · {} logical cores", TARGET_STEPS, num_cores);
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ── Environment ──────────────────────────────────────────────────────────
    let env = LunarLanderEnv::<B>::new(MAX_STEPS, device.clone());

    // ── Agent ────────────────────────────────────────────────────────────────
    let initial_model = make_bootstrap_model::<B>()?;

    let config_path = std::path::PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(1)
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

    // ── Timing / stat storage ─────────────────────────────────────────────────
    // Pre-allocate for TARGET_STEPS to avoid realloc jitter.
    let cap = TARGET_STEPS as usize;
    let mut step_times_ns:  Vec<u64> = Vec::with_capacity(cap);
    let mut infer_times_ns: Vec<u64> = Vec::with_capacity(cap);
    let mut env_times_ns:   Vec<u64> = Vec::with_capacity(cap);
    let mut ep_returns:     Vec<f32> = Vec::new();
    let mut ep_lengths:     Vec<u64> = Vec::new();

    // ── Background /proc sampler ──────────────────────────────────────────────
    let done_flag = Arc::new(AtomicBool::new(false));
    let proc_samples: Arc<std::sync::Mutex<Vec<ProcSample>>> =
        Arc::new(std::sync::Mutex::new(Vec::with_capacity(5_000)));

    let done_flag_bg  = done_flag.clone();
    let proc_samples_bg = proc_samples.clone();
    let t0_bg = Instant::now();
    let sampler_thread = std::thread::spawn(move || {
        while !done_flag_bg.load(Ordering::Relaxed) {
            let s = sample_proc(t0_bg);
            proc_samples_bg.lock().unwrap().push(s);
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    });

    // ── Warm-up: 500 steps to prime caches / JIT ─────────────────────────────
    println!("Warming up (500 steps)…");
    env.reset();
    for _ in 0..500 {
        let obs_vec = env.get_observation(0);
        let obs_tensor = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_vec, [1, OBS_DIM]),
            &device,
        );
        let _ = join_all(
            actor_ids.iter().map(|id| agent.request_action(vec![*id], obs_tensor.clone(), None, 0.0))
        ).await;
        let _ = env.step(0, 0);
        if env.all_done() || env.is_max_steps_reached() { env.reset(); }
    }
    println!("Warm-up done. Starting benchmark…\n");

    // ── Main collection loop ──────────────────────────────────────────────────
    let t_start  = Instant::now();
    let mut total_steps: u64 = 0;
    let mut ep_return  = 0.0f32;
    let mut ep_len: u64 = 0;

    env.reset();

    while total_steps < TARGET_STEPS {
        let step_start = Instant::now();

        // -- Inference --
        let obs_vec = env.get_observation(0);
        let obs_tensor = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_vec, [1, OBS_DIM]),
            &device,
        );
        let action_futures = actor_ids.iter().map(|id| {
            agent.request_action(vec![*id], obs_tensor.clone(), None, ep_return)
        });
        let infer_start = Instant::now();
        let all_results = join_all(action_futures).await;
        let infer_ns = infer_start.elapsed().as_nanos() as u64;

        // Decode action (argmax over logits)
        let action_u8: u8 = all_results.into_iter().next()
            .and_then(|r| r.ok())
            .and_then(|a| a.into_iter().next())
            .and_then(|(_, relay_action)| relay_action.get_act().map(|act_data| {
                act_data.data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u8)
                    .unwrap_or(0)
            }))
            .unwrap_or(0);

        // -- Env step --
        let env_start = Instant::now();
        let step_result = env.step(0, action_u8);
        let env_ns = env_start.elapsed().as_nanos() as u64;

        let reward = step_result.ok().map(|(r, _)| r).unwrap_or(0.0);
        ep_return += reward;
        ep_len    += 1;
        total_steps += 1;

        let step_ns = step_start.elapsed().as_nanos() as u64;
        step_times_ns.push(step_ns);
        infer_times_ns.push(infer_ns);
        env_times_ns.push(env_ns);

        // Episode boundary
        if env.all_done() || env.is_max_steps_reached() {
            agent.flag_last_action(actor_ids.clone(), Some(ep_return)).await?;
            ep_returns.push(ep_return);
            ep_lengths.push(ep_len);
            ep_return = 0.0;
            ep_len    = 0;
            env.reset();
        }

        if total_steps % 100_000 == 0 {
            let elapsed = t_start.elapsed().as_secs_f64();
            let sps = total_steps as f64 / elapsed;
            println!("  [{:>7} steps] {:.0} steps/sec", total_steps, sps);
        }
    }

    let elapsed_sec = t_start.elapsed().as_secs_f64();

    // Signal background sampler to stop
    done_flag.store(true, Ordering::Relaxed);
    let _ = sampler_thread.join();

    // ── Compute all metrics ───────────────────────────────────────────────────

    // Sort timing arrays for percentiles
    let mut step_sorted   = step_times_ns.clone();
    let mut infer_sorted  = infer_times_ns.clone();
    let mut env_sorted    = env_times_ns.clone();
    step_sorted.sort_unstable();
    infer_sorted.sort_unstable();
    env_sorted.sort_unstable();

    let n = step_sorted.len() as f64;

    // Step timing (µs)
    let step_mean_ns  = mean_f64(&step_times_ns);
    let step_std_ns   = stddev(&step_times_ns, step_mean_ns);
    let step_p50_ns   = percentile(&step_sorted, 50.0);
    let step_p95_ns   = percentile(&step_sorted, 95.0);
    let step_p99_ns   = percentile(&step_sorted, 99.0);
    let step_p999_ns  = percentile(&step_sorted, 99.9);
    let jitter_ns     = step_p99_ns.saturating_sub(step_p50_ns);

    // Inference timing
    let infer_mean_ns = mean_f64(&infer_times_ns);
    let infer_std_ns  = stddev(&infer_times_ns, infer_mean_ns);
    let infer_p50_ns  = percentile(&infer_sorted, 50.0);
    let infer_p95_ns  = percentile(&infer_sorted, 95.0);
    let infer_p99_ns  = percentile(&infer_sorted, 99.0);

    // Env step timing
    let env_mean_ns   = mean_f64(&env_times_ns);
    let env_std_ns    = stddev(&env_times_ns, env_mean_ns);
    let env_p50_ns    = percentile(&env_sorted, 50.0);
    let env_p99_ns    = percentile(&env_sorted, 99.0);

    // Overhead (everything that's not inference or env step)
    let overhead_mean_ns = step_mean_ns - infer_mean_ns - env_mean_ns;
    let overhead_ratio   = if step_mean_ns > 0.0 { overhead_mean_ns / step_mean_ns } else { 0.0 };

    // Throughput
    let steps_per_sec  = total_steps as f64 / elapsed_sec;
    let steps_per_core = steps_per_sec / num_cores as f64;

    // Episode stats
    let num_episodes = ep_returns.len() as f64;
    let episodes_per_sec = num_episodes / elapsed_sec;
    let avg_ep_len = if ep_lengths.is_empty() { 0.0 }
                     else { ep_lengths.iter().sum::<u64>() as f64 / ep_lengths.len() as f64 };
    let ep_return_mean = if ep_returns.is_empty() { 0.0 }
                          else { ep_returns.iter().sum::<f32>() as f64 / num_episodes };
    let ep_return_var = if ep_returns.len() < 2 { 0.0 } else {
        ep_returns.iter().map(|&r| { let d = r as f64 - ep_return_mean; d * d }).sum::<f64>()
            / (ep_returns.len() - 1) as f64
    };
    let ep_return_std = ep_return_var.sqrt();

    // Context switches & CPU from /proc samples
    let samples = proc_samples.lock().unwrap();
    let rss_vals:   Vec<u64> = samples.iter().map(|s| s.rss_kb).collect();
    let rss_mean_kb = if rss_vals.is_empty() { 0 }
                       else { rss_vals.iter().sum::<u64>() / rss_vals.len() as u64 };
    let rss_peak_kb = rss_vals.iter().copied().max().unwrap_or(0);
    let rss_init_kb = samples.first().map(|s| s.rss_kb).unwrap_or(0);
    let rss_final_kb= samples.last().map(|s| s.rss_kb).unwrap_or(0);

    // Allocation rate (bytes/sec): delta RSS / elapsed
    let alloc_rate_kb_s = if elapsed_sec > 0.0 {
        (rss_final_kb.saturating_sub(rss_init_kb)) as f64 / elapsed_sec
    } else { 0.0 };

    // Context switches: total vol + nonvol across entire run (delta first→last)
    let ctx_sw_first = samples.first().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let ctx_sw_last  = samples.last().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let total_ctx_sw = ctx_sw_last.saturating_sub(ctx_sw_first);
    let ctx_sw_per_sec = total_ctx_sw as f64 / elapsed_sec;
    let ctx_sw_per_step = total_ctx_sw as f64 / total_steps as f64;

    // CPU utilisation: (utime + stime) ticks delta / wall ticks
    let hz: f64 = 100.0; // typical Linux CONFIG_HZ
    let cpu_first = samples.first().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_last  = samples.last().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_ticks_delta = cpu_last.saturating_sub(cpu_first) as f64;
    let cpu_util = (cpu_ticks_delta / hz) / elapsed_sec * 100.0; // percent of 1 core
    let cpu_util_per_core = cpu_util / num_cores as f64;

    // Thread count (mean)
    let thread_mean = if samples.is_empty() { 0.0 }
                       else { samples.iter().map(|s| s.threads).sum::<u64>() as f64 / samples.len() as f64 };

    // Run queue load average (mean)
    let runq_mean = if samples.is_empty() { 0.0 }
                     else { samples.iter().map(|s| s.runq as f64).sum::<f64>() / samples.len() as f64 };
    drop(samples);

    // Deadtime per actor: fraction of step time not doing infer or env step
    let deadtime_frac = overhead_ratio;

    // Steps/sec/GB memory bandwidth: approximate using RSS mean as working set
    // (Not true bandwidth, but a normalised efficiency proxy)
    let rss_mean_gb = rss_mean_kb as f64 / (1024.0 * 1024.0);
    let sps_per_gb  = if rss_mean_gb > 0.0 { steps_per_sec / rss_mean_gb } else { 0.0 };

    // S(1) = 1 by definition since this is 1-actor baseline
    let scalability_s1 = 1.0f64;

    // ── Print report ──────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL LunarLander Comprehensive Benchmark — FINAL RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    println!("─── Throughput ─────────────────────────────────────────────────────");
    println!("  Steps/sec (global)         : {:>10.1}",    steps_per_sec);
    println!("  Steps/sec per actor        : {:>10.1}",    steps_per_sec); // 1 actor
    println!("  Steps/sec per logical core : {:>10.1}",    steps_per_core);
    println!("  Episodes/sec               : {:>10.3}",    episodes_per_sec);
    println!("  Total steps                : {:>10}",      total_steps);
    println!("  Total episodes             : {:>10}",      ep_returns.len());
    println!("  Wall time                  : {:>10.2}s",   elapsed_sec);
    println!("  Logical cores              : {:>10}",      num_cores);
    println!();

    println!("─── Episode Statistics ──────────────────────────────────────────────");
    println!("  Avg steps per episode      : {:>10.1}",    avg_ep_len);
    println!("  Episode return mean        : {:>10.3}",    ep_return_mean);
    println!("  Episode return std dev     : {:>10.3}",    ep_return_std);
    println!("  Episode return variance    : {:>10.3}",    ep_return_var);
    println!("  Episode completion var     : {:>10.3}",    ep_return_std / ep_return_mean.abs().max(1.0));
    println!();

    println!("─── Per-Step Timing (µs) ────────────────────────────────────────────");
    println!("  Step time mean             : {:>10.3} µs", step_mean_ns  / 1_000.0);
    println!("  Step time std dev          : {:>10.3} µs", step_std_ns   / 1_000.0);
    println!("  Step P50                   : {:>10.3} µs", step_p50_ns   as f64 / 1_000.0);
    println!("  Step P95                   : {:>10.3} µs", step_p95_ns   as f64 / 1_000.0);
    println!("  Step P99                   : {:>10.3} µs", step_p99_ns   as f64 / 1_000.0);
    println!("  Step P99.9                 : {:>10.3} µs", step_p999_ns  as f64 / 1_000.0);
    println!("  Jitter (P99 - P50)         : {:>10.3} µs", jitter_ns     as f64 / 1_000.0);
    println!();

    println!("─── Inference Timing (µs) ───────────────────────────────────────────");
    println!("  Inference mean             : {:>10.3} µs", infer_mean_ns / 1_000.0);
    println!("  Inference std dev          : {:>10.3} µs", infer_std_ns  / 1_000.0);
    println!("  Inference P50              : {:>10.3} µs", infer_p50_ns  as f64 / 1_000.0);
    println!("  Inference P95              : {:>10.3} µs", infer_p95_ns  as f64 / 1_000.0);
    println!("  Inference P99              : {:>10.3} µs", infer_p99_ns  as f64 / 1_000.0);
    println!("  Inference / step ratio     : {:>10.3}",    infer_mean_ns / step_mean_ns.max(1.0));
    println!();

    println!("─── Env Step Timing (µs) ────────────────────────────────────────────");
    println!("  Env step mean              : {:>10.3} µs", env_mean_ns   / 1_000.0);
    println!("  Env step std dev           : {:>10.3} µs", env_std_ns    / 1_000.0);
    println!("  Env step P50               : {:>10.3} µs", env_p50_ns    as f64 / 1_000.0);
    println!("  Env step P99               : {:>10.3} µs", env_p99_ns    as f64 / 1_000.0);
    println!("  Env step / step ratio      : {:>10.3}",    env_mean_ns   / step_mean_ns.max(1.0));
    println!();

    println!("─── Scheduling / Overhead ───────────────────────────────────────────");
    println!("  Overhead mean (step−infer−env) : {:>7.3} µs", overhead_mean_ns / 1_000.0);
    println!("  Overhead ratio             : {:>10.3}",    overhead_ratio);
    println!("  Deadtime per actor         : {:>10.3}",    deadtime_frac);
    println!("  Dropped/late updates       : {:>10}",      0u32); // no update loss in offline mode
    println!();

    println!("─── Memory ──────────────────────────────────────────────────────────");
    println!("  RSS init                   : {:>7.1} MB", rss_init_kb  as f64 / 1024.0);
    println!("  RSS peak                   : {:>7.1} MB", rss_peak_kb  as f64 / 1024.0);
    println!("  RSS mean                   : {:>7.1} MB", rss_mean_kb  as f64 / 1024.0);
    println!("  RSS final                  : {:>7.1} MB", rss_final_kb as f64 / 1024.0);
    println!("  Allocation rate (RSS Δ)    : {:>7.3} KB/s", alloc_rate_kb_s);
    println!("  /proc samples              : {:>10}",     rss_vals.len());
    println!();

    println!("─── CPU / OS ────────────────────────────────────────────────────────");
    println!("  CPU utilisation (1 core)   : {:>10.2}%",  cpu_util);
    println!("  CPU util / logical core    : {:>10.2}%",  cpu_util_per_core);
    println!("  Mean threads               : {:>10.1}",   thread_mean);
    println!("  Mean run-queue (load avg)  : {:>10.3}",   runq_mean);
    println!("  Context switches total     : {:>10}",     total_ctx_sw);
    println!("  Context switches/sec       : {:>10.1}",   ctx_sw_per_sec);
    println!("  Context switches/step      : {:>10.6}",   ctx_sw_per_step);
    println!();

    println!("─── Efficiency Ratios ────────────────────────────────────────────────");
    println!("  Steps/sec / logical core   : {:>10.1}",   steps_per_core);
    println!("  Steps/sec / GB RSS (proxy) : {:>10.1}",   sps_per_gb);
    println!("  Scalability S(1)           : {:>10.3}",   scalability_s1);
    println!();

    println!("─── Notes (hardware counters require perf) ───────────────────────────");
    println!("  Cache misses (L1/L2/L3)    : run with `perf stat -e cache-misses,LLC-load-misses`");
    println!("  IPC                        : run with `perf stat -e cycles,instructions`");
    println!("  Memory bandwidth           : run with `perf stat -e cache-references`");
    println!("  Inter-thread msg latency   : measured via P50 inference timing above");
    println!("  Queue backlog / contention : not exposed by framework API (no hooks)");
    println!("  Sync wait time             : included in overhead_mean above");
    println!();

    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
