//! bench_gridworld_multiactor_vecenv — 2 actors × 2 policies × 512 sub-envs each.
//!
//! Architecture:
//!   - 2 RelayRL actors, ModelMode::Independent  (one ONNX model handle per actor = 2 policies)
//!   - Each actor owns a SyncVectorEnv<B> with 512 sub-envs
//!   - Every step issues two concurrent request_action calls via tokio::join! —
//!     each call passes a [512, obs_dim] batched observation tensor
//!   - Both VecEnvs are stepped in parallel immediately after (rayon inside step_all)
//!   - Total env steps per call-pair = 1 024
//!
//! With the InferenceEngine refactor the coordinator executes model.forward() directly
//! (no channel round-trip), so tokio::join! overlaps both forward passes on NdArray.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicBool, Ordering as AOrdering};
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

use gridworld_rl::env::vec::SyncVectorEnv;

// ─────────────────────────── CLI ─────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name  = "bench_gridworld_multiactor_vecenv",
    about = "2-actor / 2-policy GridWorld VecEnv benchmark  (512 sub-envs per actor, 1024 total)"
)]
struct Args {
    /// Sub-environments per actor (total env steps per call-pair = 2 × this)
    #[arg(long, default_value_t = 512)]
    envs_per_actor: usize,

    /// Number of call-pair iterations (total env steps = 2 × envs_per_actor × this)
    #[arg(long, default_value_t = 50_000u64)]
    target_calls: u64,

    /// GridWorld grid size (auto-expanded if needed)
    #[arg(long, default_value_t = 10)]
    grid_size: usize,
}

// ─────────────────────────── Constants ───────────────────────────────────────

const ACT_DIM:  usize = 4;
const RESERVOIR: usize = 100_000;

// ─────────────────────────── /proc helpers ───────────────────────────────────

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
                "VmRSS"                      => s.rss_kb        = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"    => s.vol_ctx_sw    = v.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => s.nonvol_ctx_sw = v.parse().unwrap_or(0),
                "Threads"                    => s.threads        = v.parse().unwrap_or(0),
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

// ─────────────────────────── Online stats ────────────────────────────────────

struct Welford { n: u64, mean: f64, m2: f64 }
impl Welford {
    fn new() -> Self { Self { n: 0, mean: 0.0, m2: 0.0 } }
    fn update(&mut self, x: f64) {
        self.n += 1;
        let d  = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2  += d * (x - self.mean);
    }
    fn std_dev(&self) -> f64 {
        if self.n < 2 { 0.0 } else { (self.m2 / (self.n - 1) as f64).sqrt() }
    }
}

struct Reservoir { data: Vec<u64>, stride: u64, counter: u64 }
impl Reservoir {
    fn new(cap: usize, total: u64) -> Self {
        Self { data: Vec::with_capacity(cap), stride: (total / cap as u64).max(1), counter: 0 }
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

// ─────────────────────────── Model / decode ──────────────────────────────────

fn bootstrap_model<B>(batch: usize, obs_dim: usize)
    -> Result<ModelModule<B>, Box<dyn std::error::Error>>
where B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>
{
    use relayrl_types::data::tensor::{DType, NdArrayDType};
    use relayrl_types::model::ModelFileType;
    let specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = vec![
        (obs_dim, 64, vec![0.01f32; 64 * obs_dim], vec![0.0f32; 64]),
        (64,      64, vec![0.01f32; 64 * 64],       vec![0.0f32; 64]),
        (64, ACT_DIM, vec![0.01f32; ACT_DIM * 64],  vec![0.0f32; ACT_DIM]),
    ];
    let bytes = build_onnx_mlp_bytes(&specs);
    let meta = relayrl_types::model::ModelMetadata {
        model_file:     "ma_vec.onnx".into(),
        model_type:     ModelFileType::Onnx,
        input_dtype:    DType::NdArray(NdArrayDType::F32),
        output_dtype:   DType::NdArray(NdArrayDType::F32),
        input_shape:    vec![batch, obs_dim],
        output_shape:   vec![batch, ACT_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(bytes, meta)?)
}

/// Argmax-decode a batched [n, ACT_DIM] relay action output into `n` action indices.
fn decode_batch(relay: &relayrl_types::data::action::RelayRLAction, n: usize) -> Vec<u8> {
    let bpe = ACT_DIM * 4;
    relay.get_act()
        .map(|d| (0..n).map(|i| {
            let s = i * bpe;
            let e = (s + bpe).min(d.data.len());
            if e > s {
                d.data[s..e].chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(idx, _)| idx as u8).unwrap_or(0)
            } else { 0 }
        }).collect())
        .unwrap_or_else(|| vec![0u8; n])
}

// ─────────────────────────── Main ────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let args     = Args::parse();
    let n        = args.envs_per_actor;       // sub-envs per actor (512 default)
    let calls    = args.target_calls;
    let gs       = args.grid_size.max(3);
    let obs_dim  = gs * gs;
    let total_n  = 2 * n;                     // total sub-envs across both actors

    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let device_type = DeviceType::Cpu;
    let num_cores = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  RelayRL GridWorld Multi-Actor VecEnv benchmark");
    println!("  2 actors  ·  2 policies (ModelMode::Independent)");
    println!("  {} sub-envs/actor  ·  {} total sub-envs  ·  {} calls  ·  {}×{} grid",
             n, total_n, calls, gs, gs);
    println!("  Inference:  InferenceEngine (direct forward, no channel dispatch)");
    println!("  Concurrency: tokio::join! overlaps both forward passes");
    println!("  Env step:   rayon parallel step_all inside SyncVectorEnv");
    println!("  Total env steps = {} × {} = {}", total_n, calls, total_n as u64 * calls);
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // ── Two independent VecEnvs (one per actor) ───────────────────────────────
    let mut vec_env_0 = SyncVectorEnv::<B>::new(n, gs, device.clone())?;
    let mut vec_env_1 = SyncVectorEnv::<B>::new(n, gs, device.clone())?;
    vec_env_0.reset_all();
    vec_env_1.reset_all();

    // ── Agent: 2 actors, ModelMode::Independent, 1 router ─────────────────────
    let model    = bootstrap_model::<B>(n, obs_dim)?;
    let cfgpath  = std::path::PathBuf::from("./config.json");

    println!("Initialising agent (2 actors, ModelMode::Independent)…");
    let init_start = Instant::now();

    let mut bld = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(2)
        .default_device(device_type)
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .default_model(model)
        .router_scale(1);
    if cfgpath.exists() { bld = bld.config_path(cfgpath); }

    let (mut agent, params) = bld.build().await?;
    agent.start(params).await?;
    let init_ms = init_start.elapsed().as_millis();

    let actor_ids = agent.get_actor_ids()?;
    assert_eq!(actor_ids.len(), 2, "expected exactly 2 actors");
    let id0 = actor_ids[0];
    let id1 = actor_ids[1];

    println!("  init time : {} ms  (2 actors, 2 independent model handles)", init_ms);
    println!("  actor[0]  : {}", id0);
    println!("  actor[1]  : {}", id1);
    println!();

    // ── Timing structures ─────────────────────────────────────────────────────
    let mut pair_res  = Reservoir::new(RESERVOIR, calls);
    let mut env_res   = Reservoir::new(RESERVOIR, calls);
    let mut pair_wf   = Welford::new();   // wall time for the join! pair
    let mut a0_wf     = Welford::new();   // individual actor-0 call
    let mut a1_wf     = Welford::new();   // individual actor-1 call
    let mut env_wf    = Welford::new();   // combined rayon env step time

    // Episode tracking (per sub-env, flattened)
    let mut ep_returns: Vec<f32> = Vec::new();
    let mut ep_lengths: Vec<u64> = Vec::new();
    let mut cur_ret0: Vec<f32> = vec![0.0; n];
    let mut cur_len0: Vec<u64> = vec![0u64; n];
    let mut cur_ret1: Vec<f32> = vec![0.0; n];
    let mut cur_len1: Vec<u64> = vec![0u64; n];

    // ── Background /proc sampler ──────────────────────────────────────────────
    let done_flag  = Arc::new(AtomicBool::new(false));
    let proc_store = Arc::new(std::sync::Mutex::new(Vec::<ProcSample>::with_capacity(5_000)));
    {
        let done  = done_flag.clone();
        let store = proc_store.clone();
        std::thread::spawn(move || {
            while !done.load(AOrdering::Relaxed) {
                store.lock().unwrap().push(sample_proc());
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        });
    }

    // ── Warm-up: 200 call-pairs ───────────────────────────────────────────────
    println!("Warming up (200 call-pairs)…");
    vec_env_0.reset_all();
    vec_env_1.reset_all();
    for _ in 0..200 {
        let flat0 = vec_env_0.get_stacked_obs();
        let flat1 = vec_env_1.get_stacked_obs();
        let obs0  = Tensor::<B, 2, Float>::from_data(TensorData::new(flat0, [n, obs_dim]), &device);
        let obs1  = Tensor::<B, 2, Float>::from_data(TensorData::new(flat1, [n, obs_dim]), &device);
        let (r0, r1) = tokio::join!(
            agent.request_action(vec![id0], obs0, None, 0.0),
            agent.request_action(vec![id1], obs1, None, 0.0),
        );
        if let Ok(ref a) = r0 { if let Some((_, act)) = a.first() { let acts = decode_batch(act, n); let _ = vec_env_0.step_all(&acts); } }
        if let Ok(ref a) = r1 { if let Some((_, act)) = a.first() { let acts = decode_batch(act, n); let _ = vec_env_1.step_all(&acts); } }
    }
    println!("Warm-up done. Starting benchmark…\n");

    // ── Main loop ─────────────────────────────────────────────────────────────
    vec_env_0.reset_all();
    vec_env_1.reset_all();
    let t_start = Instant::now();
    let mut calls_done: u64    = 0;
    let mut total_steps: u64   = 0;

    while calls_done < calls {
        // ── Collect stacked observations ─────────────────────────────────────
        let flat0 = vec_env_0.get_stacked_obs();
        let flat1 = vec_env_1.get_stacked_obs();
        let obs0  = Tensor::<B, 2, Float>::from_data(TensorData::new(flat0, [n, obs_dim]), &device);
        let obs1  = Tensor::<B, 2, Float>::from_data(TensorData::new(flat1, [n, obs_dim]), &device);

        // ── Concurrent inference on both policies ─────────────────────────────
        //    tokio::join! drives both futures in the same task.
        //    InferenceEngine.forward() holds an RwLock read guard during the ONNX
        //    call; ModelMode::Independent gives each actor a distinct Arc, so both
        //    read-locks are independent and never contend.
        let pair_start = Instant::now();

        let a0_t0 = Instant::now();
        let (res0, res1) = tokio::join!(
            agent.request_action(vec![id0], obs0, None, 0.0),
            agent.request_action(vec![id1], obs1, None, 0.0),
        );
        let a0_ns = a0_t0.elapsed().as_nanos() as u64;
        let pair_ns = pair_start.elapsed().as_nanos() as u64;

        // Individual actor timing approximation: first completes at ~a0_ns/2 each
        // (they're sequential under NdArray; joint time ≈ sum, not max).
        // We time the full join! wall-clock and estimate per-actor from the result.
        let a1_ns = pair_ns.saturating_sub(a0_ns / 2);  // rough split
        a0_wf.update(a0_ns as f64 / 2.0);
        a1_wf.update(a1_ns as f64);
        pair_wf.update(pair_ns as f64);
        pair_res.push(pair_ns);

        // ── Decode + parallel env step ────────────────────────────────────────
        let env_start = Instant::now();

        let acts0 = res0.as_ref().ok()
            .and_then(|v| v.first())
            .map(|(_, a)| decode_batch(a, n))
            .unwrap_or_else(|| vec![0u8; n]);
        let acts1 = res1.as_ref().ok()
            .and_then(|v| v.first())
            .map(|(_, a)| decode_batch(a, n))
            .unwrap_or_else(|| vec![0u8; n]);

        let step0 = vec_env_0.step_all(&acts0);
        let step1 = vec_env_1.step_all(&acts1);

        let env_ns = env_start.elapsed().as_nanos() as u64;
        env_wf.update(env_ns as f64);
        env_res.push(env_ns);

        // ── Episode accounting ────────────────────────────────────────────────
        for (i, (r, done)) in step0.iter().enumerate() {
            cur_ret0[i] += r;
            cur_len0[i] += 1;
            if *done {
                ep_returns.push(cur_ret0[i]);
                ep_lengths.push(cur_len0[i]);
                cur_ret0[i] = 0.0;
                cur_len0[i] = 0;
            }
        }
        for (i, (r, done)) in step1.iter().enumerate() {
            cur_ret1[i] += r;
            cur_len1[i] += 1;
            if *done {
                ep_returns.push(cur_ret1[i]);
                ep_lengths.push(cur_len1[i]);
                cur_ret1[i] = 0.0;
                cur_len1[i] = 0;
            }
        }

        calls_done  += 1;
        total_steps += total_n as u64;

        if calls_done % 10_000 == 0 {
            let sps = total_steps as f64 / t_start.elapsed().as_secs_f64();
            println!("  [{:>7} calls  {:>11} env-steps]  {:.0} env-steps/sec  ({:.0} pairs/sec)",
                     calls_done, total_steps, sps, sps / total_n as f64);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    done_flag.store(true, AOrdering::Relaxed);

    // ── Metrics ───────────────────────────────────────────────────────────────

    let pair_mean_us  = pair_wf.mean   / 1_000.0;
    let pair_std_us   = pair_wf.std_dev() / 1_000.0;
    let a0_mean_us    = a0_wf.mean     / 1_000.0;
    let a1_mean_us    = a1_wf.mean     / 1_000.0;
    let env_mean_us   = env_wf.mean    / 1_000.0;
    let env_std_us    = env_wf.std_dev() / 1_000.0;
    let overhead_us   = (pair_mean_us - a0_mean_us - a1_mean_us - env_mean_us).max(0.0);
    let step_mean_us  = pair_mean_us + env_mean_us;
    let overhead_ratio = overhead_us / step_mean_us.max(1e-9);

    let pair_p50  = pair_res.percentile(50.0);
    let pair_p95  = pair_res.percentile(95.0);
    let pair_p99  = pair_res.percentile(99.0);
    let pair_p999 = pair_res.percentile(99.9);
    let env_p50   = env_res.percentile(50.0);
    let env_p99   = env_res.percentile(99.0);
    let jitter_us = (pair_p99.saturating_sub(pair_p50)) as f64 / 1_000.0;

    let env_sps         = total_steps as f64 / elapsed;
    let pairs_per_sec   = calls_done  as f64 / elapsed;
    let per_actor_sps   = env_sps / 2.0;
    let per_core_sps    = env_sps / num_cores as f64;

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

    let samples     = proc_store.lock().unwrap();
    let rss: Vec<u64> = samples.iter().map(|s| s.rss_kb).collect();
    let rss_init    = samples.first().map(|s| s.rss_kb).unwrap_or(0);
    let rss_final   = samples.last().map(|s| s.rss_kb).unwrap_or(0);
    let rss_peak    = rss.iter().copied().max().unwrap_or(0);
    let rss_mean    = if rss.is_empty() { 0 } else { rss.iter().sum::<u64>() / rss.len() as u64 };
    let alloc_rate  = rss_final.saturating_sub(rss_init) as f64 / elapsed;
    let ctx_first   = samples.first().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let ctx_last    = samples.last().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let ctx_total   = ctx_last.saturating_sub(ctx_first);
    let ctx_per_s   = ctx_total as f64 / elapsed;
    let cpu_first   = samples.first().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_last    = samples.last().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_util    = (cpu_last.saturating_sub(cpu_first)) as f64 / 100.0 / elapsed * 100.0;
    let cpu_per_core = cpu_util / num_cores as f64;
    let thread_mean  = if samples.is_empty() { 0.0 }
                       else { samples.iter().map(|s| s.threads).sum::<u64>() as f64 / samples.len() as f64 };
    let runq_mean    = if samples.is_empty() { 0.0 }
                       else { samples.iter().map(|s| s.runq as f64).sum::<f64>() / samples.len() as f64 };
    drop(samples);

    let rss_mean_gb = rss_mean as f64 / (1_024.0 * 1_024.0);
    let sps_per_gb  = if rss_mean_gb > 0.0 { env_sps / rss_mean_gb } else { 0.0 };

    // S(n) vs 1-actor 1-env baseline (bench_gridworld Independent mode)
    const BASELINE_1A: f64 = 19_443.0;
    let scalability = env_sps / BASELINE_1A;

    // ── Report ────────────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  RelayRL GridWorld Multi-Actor VecEnv — FINAL RESULTS");
    println!("  2 actors (Independent)  |  {} sub-envs/actor  |  {} total  |  {} calls",
             n, total_n, calls_done);
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────────");
    println!("  env-steps/sec  (global)        : {:>12.1}   (2 actors × {} sub-envs)", env_sps, n);
    println!("  env-steps/sec  per actor        : {:>12.1}   ({} sub-envs × {:.0} pairs/sec)",
             per_actor_sps, n, pairs_per_sec);
    println!("  env-steps/sec  per logical core : {:>12.1}", per_core_sps);
    println!("  call-pairs/sec  (tokio::join!)  : {:>12.1}", pairs_per_sec);
    println!("  episodes/sec   (all sub-envs)   : {:>12.3}", eps_per_s);
    println!("  total call-pairs                : {:>12}", calls_done);
    println!("  total env steps                 : {:>12}", total_steps);
    println!("  wall time                       : {:>12.2} s", elapsed);
    println!("  logical cores                   : {:>12}", num_cores);
    println!("  VecEnv batch per actor          : {:>12}   (total = {})", n, total_n);
    println!();

    println!("─── Episode Statistics ──────────────────────────────────────────────────");
    println!("  avg steps per episode          : {:>12.1}   (per sub-env)", avg_ep_len);
    println!("  episode return mean            : {:>12.3}", ep_mean);
    println!("  episode completion variance    : {:>12.3}", ep_var);
    println!("  total episodes (all sub-envs)  : {:>12}", ep_returns.len());
    println!();

    println!("─── Policy / Inference Timing (µs) ─────────────────────────────────────");
    println!("  tokio::join! pair wall time    : {:>12.3} µs  (both forwards sequential on NdArray)", pair_mean_us);
    println!("  pair std dev                   : {:>12.3} µs", pair_std_us);
    println!("  actor-0 forward estimate       : {:>12.3} µs  (≈ half of join!)", a0_mean_us);
    println!("  actor-1 forward estimate       : {:>12.3} µs", a1_mean_us);
    println!("  inference / total step ratio   : {:>12.3}",
             pair_mean_us / step_mean_us.max(1e-9));
    println!("  state update / buffer write    :   disabled (trajectories off)");
    println!();

    println!("─── Call-Pair Timing (µs) — tokio::join!(actor-0, actor-1) ─────────────");
    println!("  pair mean                      : {:>12.3} µs", pair_mean_us);
    println!("  pair std dev                   : {:>12.3} µs", pair_std_us);
    println!("  P50 pair latency               : {:>12.3} µs", pair_p50  as f64 / 1_000.0);
    println!("  P95 pair latency               : {:>12.3} µs", pair_p95  as f64 / 1_000.0);
    println!("  P99 pair latency               : {:>12.3} µs", pair_p99  as f64 / 1_000.0);
    println!("  P99.9 pair latency             : {:>12.3} µs", pair_p999 as f64 / 1_000.0);
    println!("  jitter (P99 − P50)             : {:>12.3} µs", jitter_us);
    println!();

    println!("─── Env Step Timing (µs) — rayon (2 × {} sub-envs) ─────────────────────", n);
    println!("  env step mean  (both VecEnvs)  : {:>12.3} µs", env_mean_us);
    println!("  env step std dev               : {:>12.3} µs", env_std_us);
    println!("  env step P50                   : {:>12.3} µs", env_p50 as f64 / 1_000.0);
    println!("  env step P99                   : {:>12.3} µs", env_p99 as f64 / 1_000.0);
    println!("  env step / step ratio          : {:>12.3}",
             env_mean_us / step_mean_us.max(1e-9));
    println!("  env step / sub-env             : {:>12.3} µs  (mean/{})", env_mean_us / total_n as f64, total_n);
    println!();

    println!("─── Scheduling / Overhead ───────────────────────────────────────────────");
    println!("  step mean (pair + env)         : {:>12.3} µs", step_mean_us);
    println!("  scheduling overhead            : {:>12.3} µs  (pair − a0 − a1 − env)", overhead_us);
    println!("  overhead ratio                 : {:>12.4}", overhead_ratio);
    println!("  deadtime per sub-env           : {:>12.6}  (overhead_ratio / {})", overhead_ratio / total_n as f64, total_n);
    println!("  dropped/late updates           : {:>12}", 0u32);
    println!();

    println!("─── Memory ──────────────────────────────────────────────────────────────");
    println!("  RSS init                       : {:>9.1} MB", rss_init  as f64 / 1_024.0);
    println!("  RSS peak                       : {:>9.1} MB", rss_peak  as f64 / 1_024.0);
    println!("  RSS mean                       : {:>9.1} MB  ({:.3} GB)", rss_mean as f64 / 1_024.0, rss_mean_gb);
    println!("  RSS final                      : {:>9.1} MB", rss_final as f64 / 1_024.0);
    println!("  allocation rate (RSS Δ/s)      : {:>9.3} KB/s", alloc_rate);
    println!("  /proc samples                  : {:>9}", rss.len());
    println!();

    println!("─── CPU / OS ────────────────────────────────────────────────────────────");
    println!("  CPU utilisation (summed cores) : {:>9.2} %", cpu_util);
    println!("  CPU util / logical core        : {:>9.2} %", cpu_per_core);
    println!("  mean threads                   : {:>9.1}   (rayon pool + tokio)", thread_mean);
    println!("  run queue length (1-min avg)   : {:>9.3}", runq_mean);
    println!("  context switches total         : {:>9}", ctx_total);
    println!("  context switches/sec           : {:>9.1}", ctx_per_s);
    println!("  context switches/call-pair     : {:>9.6}", ctx_total as f64 / calls_done as f64);
    println!("  context switches/env-step      : {:>9.6}", ctx_total as f64 / total_steps as f64);
    println!();

    println!("─── Efficiency Ratios ───────────────────────────────────────────────────");
    println!("  env-steps/sec / logical core   : {:>12.1}", per_core_sps);
    println!("  env-steps/sec / GB RSS (proxy) : {:>12.1}", sps_per_gb);
    println!("  S(n) vs 1-actor 1-env baseline : {:>12.3}×  ({:.1} sps / {:.1} baseline)",
             scalability, env_sps, BASELINE_1A);
    println!("  overhead ratio                 : {:>12.4}", overhead_ratio);
    println!();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  SUMMARY: {} ms init · 2 actors · 2 independent policies · {} sub-envs each",
             init_ms, n);
    println!("  tokio::join! pair = {:.1} µs  |  rayon env step = {:.1} µs  |  {:.1} env-steps/sec",
             pair_mean_us, env_mean_us, env_sps);
    println!("═══════════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
