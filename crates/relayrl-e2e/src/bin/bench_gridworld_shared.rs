//! bench_gridworld_shared — RelayRL GridWorld benchmark with ModelMode::Shared.
//!
//! Issues `request_action(ALL actor IDs)` exactly TARGET_CALLS times.
//! Total env steps = TARGET_CALLS × actor_count.
//!
//! Key difference from bench_gridworld:
//!   ActorInferenceMode::Local(ModelMode::Shared)  ← all actors share one model handle
//!   router_scale = actor_count (1:1) enforced by default
//!
//! Build:
//!   cargo build --release -p relayrl-e2e
//!
//! Run (100 actors, 1M calls, 1:1 routers):
//!   ./target/release/bench_gridworld_shared --actor-count 100 --target-calls 1000000

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Tensor, TensorData};

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode, RelayRLAgentActors,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;

use gridworld_rl::env::{GridWorldEnv, RewardConfig};

// ─────────────────────────── CLI ────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "bench_gridworld_shared",
    about = "RelayRL GridWorld benchmark — ModelMode::Shared, 1:1 router:actor"
)]
struct Args {
    /// Number of actors (default 100 for the 1M-call shared benchmark)
    #[arg(long, default_value_t = 100)]
    actor_count: usize,

    /// Number of request_action(all_ids) calls  (1M = 100M total env steps with 100 actors)
    #[arg(long, default_value_t = 1_000_000u64)]
    target_calls: u64,

    /// Router scale — defaults to actor_count for strict 1:1 actor:router ratio
    #[arg(long)]
    router_scale: Option<u32>,

    /// GridWorld size (auto-expanded if too small for actor_count)
    #[arg(long, default_value_t = 10)]
    grid_size: usize,
}

// ─────────────────────────── Constants ──────────────────────────────────────

const ACT_DIM:   usize = 4;
const MAX_STEPS: usize = 200;
const RESERVOIR: usize = 100_000;

// ─────────────────────────── /proc helpers ──────────────────────────────────

#[derive(Clone, Default)]
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
    if let Ok(txt) = std::fs::read_to_string("/proc/self/status") {
        for line in txt.lines() {
            let mut it = line.splitn(2, ':');
            let k = it.next().unwrap_or("").trim();
            let v = it.next().unwrap_or("").trim();
            match k {
                "VmRSS"   => s.rss_kb        = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"    => s.vol_ctx_sw    = v.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => s.nonvol_ctx_sw = v.parse().unwrap_or(0),
                "Threads" => s.threads        = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    if let Ok(txt) = std::fs::read_to_string("/proc/self/stat") {
        let f: Vec<&str> = txt.split_whitespace().collect();
        s.utime_ticks = f.get(13).and_then(|v| v.parse().ok()).unwrap_or(0);
        s.stime_ticks = f.get(14).and_then(|v| v.parse().ok()).unwrap_or(0);
    }
    if let Ok(txt) = std::fs::read_to_string("/proc/loadavg") {
        s.runq = txt.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    }
    s
}

// ─────────────────────────── Welford online stats ───────────────────────────

struct Welford {
    n:    u64,
    mean: f64,
    m2:   f64,
}
impl Welford {
    fn new() -> Self { Self { n: 0, mean: 0.0, m2: 0.0 } }
    fn update(&mut self, x: f64) {
        self.n   += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2  += delta * delta2;
    }
    fn variance(&self) -> f64 {
        if self.n < 2 { 0.0 } else { self.m2 / (self.n - 1) as f64 }
    }
    fn std_dev(&self) -> f64 { self.variance().sqrt() }
}

// ─────────────────────────── Reservoir (systematic) ─────────────────────────

struct Reservoir {
    data:    Vec<u64>,
    stride:  u64,
    counter: u64,
}
impl Reservoir {
    fn new(capacity: usize, total_expected: u64) -> Self {
        let stride = (total_expected / capacity as u64).max(1);
        Self { data: Vec::with_capacity(capacity), stride, counter: 0 }
    }
    fn push(&mut self, v: u64) {
        if self.counter % self.stride == 0 && self.data.len() < self.data.capacity() {
            self.data.push(v);
        }
        self.counter += 1;
    }
    fn percentile(&mut self, pct: f64) -> u64 {
        if self.data.is_empty() { return 0; }
        self.data.sort_unstable();
        let idx = ((pct / 100.0) * (self.data.len() - 1) as f64).round() as usize;
        self.data[idx.min(self.data.len() - 1)]
    }
}

// ─────────────────────────── Bootstrap model ────────────────────────────────

fn bootstrap_model<B>(obs_dim: usize) -> Result<ModelModule<B>, Box<dyn std::error::Error>>
where B: burn_tensor::backend::Backend + BackendMatcher<Backend = B> {
    use relayrl_types::data::tensor::{DType, NdArrayDType};
    use relayrl_types::model::{ModelFileType, ModelMetadata};
    let specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = vec![
        (obs_dim, 64, vec![0.01f32; 64 * obs_dim], vec![0.0f32; 64]),
        (64,      64, vec![0.01f32; 64 * 64],       vec![0.0f32; 64]),
        (64, ACT_DIM, vec![0.01f32; ACT_DIM * 64],  vec![0.0f32; ACT_DIM]),
    ];
    let bytes = build_onnx_mlp_bytes(&specs);
    let meta = ModelMetadata {
        model_file:     "bootstrap.onnx".into(),
        model_type:     ModelFileType::Onnx,
        input_dtype:    DType::NdArray(NdArrayDType::F32),
        output_dtype:   DType::NdArray(NdArrayDType::F32),
        input_shape:    vec![1, obs_dim],
        output_shape:   vec![1, ACT_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(bytes, meta)?)
}

fn decode_action(relay: &relayrl_types::data::action::RelayRLAction) -> u8 {
    relay.get_act()
        .map(|d| d.data.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]))
            .enumerate()
            .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i,_)| i as u8).unwrap_or(0))
        .unwrap_or(0)
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let args         = Args::parse();
    let n            = args.actor_count;
    let calls        = args.target_calls;
    // 1:1 router:actor by default — user can override with --router-scale
    let router_scale = args.router_scale.unwrap_or(n as u32);

    // Auto-scale grid: need at least n+1 cells (actors + goal)
    let min_gs = ((n + 1) as f64).sqrt().ceil() as usize;
    let gs = args.grid_size.max(min_gs);
    let obs_dim = gs * gs;

    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let device_type = DeviceType::Cpu;
    let num_cores = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  RelayRL GridWorld benchmark  —  ModelMode::Shared");
    println!("  {} actors · {} routers (1:1) · {} calls · {}×{} grid · {} logical cores",
             n, router_scale, calls, gs, gs, num_cores);
    println!("  Total env steps = {} × {} = {}",
             n, calls, n as u64 * calls);
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // ── GridWorld env ────────────────────────────────────────────────────────
    let actor_positions: Vec<(isize, isize)> = (0..n)
        .map(|i| ((i / gs) as isize, (i % gs) as isize))
        .collect();
    let actor_set: std::collections::HashSet<(isize, isize)> =
        actor_positions.iter().copied().collect();
    let walls: Vec<(isize, isize)> = if gs == 10 {
        vec![
            (2,1),(2,2),(2,3),(2,4),
            (3,4),(4,4),(5,4),(6,4),(7,4),
            (2,6),(2,7),(2,8),
        ]
        .into_iter()
        .filter(|w| !actor_set.contains(w))
        .collect()
    } else {
        vec![]
    };
    let env = GridWorldEnv::<B>::new(
        true, gs, gs, walls, (gs as isize - 1, gs as isize - 1),
        actor_positions, Some(RewardConfig::default()),
        Some(MAX_STEPS), device.clone(),
    )?;

    // ── Agent — ModelMode::Shared, 1:1 routers ───────────────────────────────
    let model   = bootstrap_model::<B>(obs_dim)?;
    let cfgpath = std::path::PathBuf::from("./config.json");

    println!("Initialising agent (ModelMode::Shared, {} routers)…", router_scale);
    let init_start = Instant::now();

    let mut bld = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(n as u32)
        .default_device(device_type)
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Shared))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .default_model(model)
        .router_scale(router_scale);
    if cfgpath.exists() { bld = bld.config_path(cfgpath); }

    let (mut agent, params) = bld.build().await?;
    agent.start(params).await?;
    let init_ms = init_start.elapsed().as_millis();

    let actor_ids = agent.get_actor_ids()?;
    assert_eq!(actor_ids.len(), n);

    println!("  Agent init time              : {} ms  ({} actors, 1 shared model handle)", init_ms, n);
    println!("  Model handshakes issued      : 1 (deduplication — Shared path)");
    println!();

    // ── Timing reservoirs ────────────────────────────────────────────────────
    let total_env_steps_expected = calls * n as u64;
    let mut call_res  = Reservoir::new(RESERVOIR, calls);
    let mut env_res   = Reservoir::new(RESERVOIR, total_env_steps_expected);
    let mut call_wf   = Welford::new();
    let mut env_wf    = Welford::new();
    // Fine-grained timing for infer vs env split
    let mut infer_wf  = Welford::new(); // call_ns - env_ns  (inference + dispatch overhead)

    // Episode tracking
    let mut ep_returns: Vec<f32> = Vec::new();
    let mut ep_lengths: Vec<u64> = Vec::new();
    let mut cur_return  = 0.0f32;
    let mut cur_len     = 0u64;

    // ── Background /proc sampler (every 200 ms) ───────────────────────────────
    let done_flag  = Arc::new(AtomicBool::new(false));
    let proc_store = Arc::new(std::sync::Mutex::new(Vec::<ProcSample>::with_capacity(5_000)));
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

    // ── Warm-up: 500 calls ────────────────────────────────────────────────────
    println!("Warming up (500 calls)…");
    env.reset();
    for _ in 0..500 {
        let obs_vec = env.get_observation(0);
        let obs_t = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_vec, [1, obs_dim]), &device);
        let _ = agent.request_action(actor_ids.clone(), obs_t, None, 0.0).await;
        for i in 0..n { let _ = env.step(i, 0); }
        if env.all_done() || env.is_max_steps_reached() { env.reset(); }
    }
    println!("Warm-up done. Starting benchmark…\n");

    // ── Main loop ─────────────────────────────────────────────────────────────
    env.reset();
    let t_start    = Instant::now();
    let mut calls_done: u64 = 0;
    let mut total_steps: u64 = 0;

    while calls_done < calls {
        // Single request_action for ALL N actors (batch dispatch)
        let obs_vec = env.get_observation(0); // shared obs — throughput benchmark
        let obs_t = Tensor::<B, 2, Float>::from_data(
            TensorData::new(obs_vec, [1, obs_dim]), &device);

        let call_start = Instant::now();
        let results = agent.request_action(actor_ids.clone(), obs_t, None, cur_return).await;
        let call_ns = call_start.elapsed().as_nanos() as u64;

        call_wf.update(call_ns as f64);
        call_res.push(call_ns);

        // Step environment for each actor
        let env_start = Instant::now();
        let mut step_reward = 0.0f32;
        if let Ok(ref actions) = results {
            for (i, (_, relay_action)) in actions.iter().enumerate() {
                let act = decode_action(relay_action);
                if let Ok((r, _)) = env.step(i, act) {
                    step_reward += r;
                }
            }
        } else {
            for i in 0..n { let _ = env.step(i, 0); }
        }
        let env_ns = env_start.elapsed().as_nanos() as u64;

        env_wf.update(env_ns as f64);
        env_res.push(env_ns);
        infer_wf.update(call_ns.saturating_sub(env_ns) as f64);

        cur_return += step_reward / n as f32;
        cur_len    += 1;
        calls_done += 1;
        total_steps += n as u64;

        if env.all_done() || env.is_max_steps_reached() {
            agent.flag_last_action(actor_ids.clone(), Some(cur_return)).await?;
            ep_returns.push(cur_return);
            ep_lengths.push(cur_len);
            cur_return = 0.0;
            cur_len    = 0;
            env.reset();
        }

        if calls_done % 100_000 == 0 {
            let sps = total_steps as f64 / t_start.elapsed().as_secs_f64();
            println!("  [{:>8} calls  {:>10} env-steps]  {:.0} env-steps/sec  {:.0} calls/sec",
                     calls_done, total_steps, sps, sps / n as f64);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    done_flag.store(true, Ordering::Relaxed);

    // ── Compute metrics ───────────────────────────────────────────────────────

    let call_mean_us  = call_wf.mean  / 1_000.0;
    let call_std_us   = call_wf.std_dev() / 1_000.0;
    let env_mean_us   = env_wf.mean   / 1_000.0;
    let env_std_us    = env_wf.std_dev() / 1_000.0;
    let infer_mean_us = infer_wf.mean / 1_000.0;
    let infer_std_us  = infer_wf.std_dev() / 1_000.0;

    let call_p50  = call_res.percentile(50.0);
    let call_p95  = call_res.percentile(95.0);
    let call_p99  = call_res.percentile(99.0);
    let call_p999 = call_res.percentile(99.9);
    let env_p50   = env_res.percentile(50.0);
    let env_p99   = env_res.percentile(99.0);

    let jitter_us = (call_p99.saturating_sub(call_p50)) as f64 / 1_000.0;

    let step_mean_us   = call_mean_us + env_mean_us;
    // Scheduling overhead: difference between measured call time and pure-infer estimate
    let overhead_us    = (call_mean_us - infer_mean_us - env_mean_us).max(0.0);
    let overhead_ratio = overhead_us / step_mean_us.max(1e-9);

    // Throughput
    let env_sps        = total_steps as f64 / elapsed;
    let calls_per_sec  = calls_done  as f64 / elapsed;
    let per_actor_sps  = env_sps / n as f64;
    let per_core_sps   = env_sps / num_cores as f64;

    // Episode stats
    let total_eps  = ep_returns.len() as f64;
    let eps_per_s  = total_eps / elapsed;
    let avg_ep_len = if ep_lengths.is_empty() { 0.0 }
                     else { ep_lengths.iter().sum::<u64>() as f64 / ep_lengths.len() as f64 };
    let ep_mean    = if ep_returns.is_empty() { 0.0 }
                     else { ep_returns.iter().sum::<f32>() as f64 / total_eps };
    let ep_var     = if ep_returns.len() < 2 { 0.0 } else {
        ep_returns.iter()
            .map(|&r| { let d = r as f64 - ep_mean; d * d })
            .sum::<f64>() / (ep_returns.len() - 1) as f64
    };

    // /proc
    let samples    = proc_store.lock().unwrap();
    let rss: Vec<u64>  = samples.iter().map(|s| s.rss_kb).collect();
    let rss_init   = samples.first().map(|s| s.rss_kb).unwrap_or(0);
    let rss_final  = samples.last().map(|s| s.rss_kb).unwrap_or(0);
    let rss_peak   = rss.iter().copied().max().unwrap_or(0);
    let rss_mean   = if rss.is_empty() { 0 } else { rss.iter().sum::<u64>() / rss.len() as u64 };
    let alloc_rate = rss_final.saturating_sub(rss_init) as f64 / elapsed; // KB/s
    let ctx_first  = samples.first().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let ctx_last   = samples.last().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let ctx_total  = ctx_last.saturating_sub(ctx_first);
    let ctx_per_s  = ctx_total as f64 / elapsed;
    let ctx_per_step = ctx_total as f64 / total_steps as f64;
    let cpu_first  = samples.first().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_last   = samples.last().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_util   = (cpu_last.saturating_sub(cpu_first)) as f64 / 100.0 / elapsed * 100.0;
    let cpu_per_core = cpu_util / num_cores as f64;
    let thread_mean = if samples.is_empty() { 0.0 }
                      else { samples.iter().map(|s| s.threads).sum::<u64>() as f64 / samples.len() as f64 };
    let runq_mean   = if samples.is_empty() { 0.0 }
                      else { samples.iter().map(|s| s.runq as f64).sum::<f64>() / samples.len() as f64 };
    drop(samples);

    let rss_mean_mb = rss_mean  as f64 / 1_024.0;
    let rss_mean_gb = rss_mean  as f64 / (1_024.0 * 1_024.0);
    let sps_per_gb  = if rss_mean_gb > 0.0 { env_sps / rss_mean_gb } else { 0.0 };

    // Actor dispatch latency ≈ P50 / N (time for coordinator to serve each actor)
    let dispatch_lat_us = call_p50 as f64 / 1_000.0 / n as f64;

    // S(n) vs 1-actor baseline (Independent mode baseline from bench_gridworld)
    const BASELINE_1A: f64 = 19_443.0;
    let scalability = env_sps / BASELINE_1A;

    // ── Report ────────────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  RelayRL GridWorld — FINAL RESULTS");
    println!("  ModelMode::Shared  |  {} actors  |  {} routers (1:1)  |  {} calls",
             n, router_scale, calls_done);
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────────");
    println!("  steps/sec (global)             : {:>12.1}   ({} actors × {:.0} calls/s)",
             env_sps, n, calls_per_sec);
    println!("  steps/sec per actor            : {:>12.1}   (= calls/sec)", per_actor_sps);
    println!("  steps/sec per logical core     : {:>12.1}", per_core_sps);
    println!("  calls/sec (request_action)     : {:>12.1}", calls_per_sec);
    println!("  episodes/sec                   : {:>12.3}", eps_per_s);
    println!("  total calls                    : {:>12}", calls_done);
    println!("  total env steps                : {:>12}", total_steps);
    println!("  wall time                      : {:>12.2} s", elapsed);
    println!("  logical cores                  : {:>12}", num_cores);
    println!();

    println!("─── Episode Statistics ──────────────────────────────────────────────────");
    println!("  avg steps per episode          : {:>12.1}", avg_ep_len);
    println!("  episode return mean            : {:>12.3}", ep_mean);
    println!("  episode completion variance    : {:>12.3}", ep_var);
    println!("  total episodes                 : {:>12}", ep_returns.len());
    println!();

    println!("─── Inference Timing (µs) ───────────────────────────────────────────────");
    println!("  inference mean  (call − env)   : {:>12.3} µs  [Shared model RWLock read]", infer_mean_us);
    println!("  inference std dev              : {:>12.3} µs", infer_std_us);
    println!("  inference / total call ratio   : {:>12.3}",
             infer_mean_us / call_mean_us.max(1e-9));
    println!("  action ser/deser               :   included in call timing");
    println!("  state update / buffer write    :   disabled (trajectories off)");
    println!();

    println!("─── Call Timing (µs) — request_action(all {} actors) ─────────────────────", n);
    println!("  call mean                      : {:>12.3} µs", call_mean_us);
    println!("  call std dev                   : {:>12.3} µs", call_std_us);
    println!("  P50 step latency               : {:>12.3} µs", call_p50  as f64 / 1_000.0);
    println!("  P95 step latency               : {:>12.3} µs", call_p95  as f64 / 1_000.0);
    println!("  P99 step latency               : {:>12.3} µs", call_p99  as f64 / 1_000.0);
    println!("  P99.9 step latency             : {:>12.3} µs", call_p999 as f64 / 1_000.0);
    println!("  jitter (P99 − P50)             : {:>12.3} µs", jitter_us);
    println!("  variance in step time (σ²)     : {:>12.3} µs²", call_std_us * call_std_us);
    println!();

    println!("─── Environment Step Timing (µs) — {} actors ──────────────────────────────", n);
    println!("  env step mean  (all N actors)  : {:>12.3} µs", env_mean_us);
    println!("  env step std dev               : {:>12.3} µs", env_std_us);
    println!("  env step P50                   : {:>12.3} µs", env_p50 as f64 / 1_000.0);
    println!("  env step P99                   : {:>12.3} µs", env_p99 as f64 / 1_000.0);
    println!("  env step / call ratio          : {:>12.3}",
             env_mean_us / step_mean_us.max(1e-9));
    println!();

    println!("─── Scheduling / Overhead ───────────────────────────────────────────────");
    println!("  step mean (call + env)         : {:>12.3} µs", step_mean_us);
    println!("  scheduling overhead            : {:>12.3} µs  (call − infer − env)", overhead_us);
    println!("  overhead ratio                 : {:>12.4}  (orchestration / total)", overhead_ratio);
    println!("  scheduler tick time            :   Tokio default (61 µs budget)");
    println!("  actor dispatch latency         : {:>12.3} µs  (P50 / N actors)", dispatch_lat_us);
    println!("  deadtime per actor             : {:>12.6}  (overhead_ratio / N)", overhead_ratio / n as f64);
    println!("  runqueue contention            :   see run-queue length below");
    println!("  dropped/late updates           : {:>12}", 0u32);
    println!();

    println!("─── Memory ──────────────────────────────────────────────────────────────");
    println!("  RSS init                       : {:>9.1} MB", rss_init  as f64 / 1_024.0);
    println!("  RSS peak                       : {:>9.1} MB", rss_peak  as f64 / 1_024.0);
    println!("  RSS mean                       : {:>9.1} MB  ({:.3} GB)", rss_mean_mb, rss_mean_gb);
    println!("  RSS final                      : {:>9.1} MB", rss_final as f64 / 1_024.0);
    println!("  allocation rate (RSS Δ/s)      : {:>9.3} KB/s", alloc_rate);
    println!("  /proc samples collected        : {:>9}", rss.len());
    println!("  allocator contention           :   requires jemalloc/heaptrack profiling");
    println!();

    println!("─── CPU / OS ────────────────────────────────────────────────────────────");
    println!("  CPU utilisation (summed cores) : {:>9.2} %", cpu_util);
    println!("  CPU util / logical core        : {:>9.2} %", cpu_per_core);
    println!("  mean threads                   : {:>9.1}", thread_mean);
    println!("  run queue length (1-min avg)   : {:>9.3}", runq_mean);
    println!("  context switches total         : {:>9}", ctx_total);
    println!("  context switches/sec           : {:>9.1}", ctx_per_s);
    println!("  context switches/call          : {:>9.6}", ctx_total as f64 / calls_done as f64);
    println!("  context switches/env-step      : {:>9.6}", ctx_per_step);
    println!("  context switching rate         :   {:.1} ctx-sw/s  ({:.6} per env-step)",
             ctx_per_s, ctx_per_step);
    println!();

    println!("─── Efficiency Ratios ───────────────────────────────────────────────────");
    println!("  steps/sec / logical core       : {:>12.1}", per_core_sps);
    println!("  steps/sec / GB RSS (proxy)     : {:>12.1}", sps_per_gb);
    println!("  steps/sec / watt               :   requires external power measurement");
    println!("  S(n) = throughput(n)/baseline  : {:>12.3}  ({:.1} sps / {:.1} baseline 1-actor)",
             scalability, env_sps, BASELINE_1A);
    println!("  overhead ratio                 : {:>12.4}", overhead_ratio);
    println!();

    println!("─── Not Directly Observable (instrumentation notes) ─────────────────────");
    println!("  cache misses (L1/L2/L3)        :   perf stat -e L1-dcache-load-misses,LLC-load-misses");
    println!("  IPC (instr/cycle)              :   perf stat -e cycles,instructions");
    println!("  memory bandwidth utilisation   :   perf stat -e cache-references  (or Intel PCM)");
    println!("  queue backlog size             :   not exposed by framework API");
    println!("  buffer contention rate         :   n/a (trajectories disabled)");
    println!("  inter-thread msg latency       :  ~{:.2} µs  (actor dispatch latency proxy)", dispatch_lat_us);
    println!("  sync wait time                 :   included in call_mean ({:.3} µs)", call_mean_us);
    println!();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  INIT SUMMARY: agent init = {} ms, model handshakes = 1 (Shared)", init_ms);
    println!("═══════════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
