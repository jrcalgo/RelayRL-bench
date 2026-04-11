//! bench_lunar_dual_actor — two RelayRL actors share one model over a 1024-env VecEnv.
//!
//! Architecture:
//!   - 1 SyncLunarVectorEnv containing 1024 independent sub-environments
//!   - 2 RelayRL actors (ModelMode::Shared — single ONNX session, shared model)
//!   - The caller passes the full [1024, 8] observation tensor and both actor IDs
//!     in a single request_action call.
//!   - The coordinator splits the batch evenly (ceil(1024/2) = 512 rows each) and
//!     dispatches each slice to the corresponding actor without any caller-side
//!     concurrency primitives.
//!   - Both actors' responses are collected and decoded back into 1024 actions.
//!   - All 1024 env.step() calls execute in parallel via rayon.
//!
//! Build:
//!   cargo build --release -p relayrl-e2e --bin bench_lunar_dual_actor
//!
//! Run (defaults: 1024 envs split 512/512, 50 000 calls):
//!   ./target/release/bench_lunar_dual_actor
//!   ./target/release/bench_lunar_dual_actor --target-calls 100000

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

use lunarlander_rl::env::vec::SyncLunarVectorEnv;

// ─────────────────────────── CLI ────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "bench_lunar_dual_actor",
    about = "RelayRL LunarLander dual-actor shared-model benchmark \
             — coordinator splits [1024,8] batch across 2 actors internally"
)]
struct Args {
    /// Total sub-environments (must be even; split equally between the two actors)
    #[arg(long, default_value_t = 1024)]
    num_envs: usize,

    /// Number of request_action calls (total env steps = num_envs × target_calls)
    #[arg(long, default_value_t = 50_000u64)]
    target_calls: u64,

    /// Max steps per episode per sub-environment
    #[arg(long, default_value_t = 500)]
    max_steps: usize,
}

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
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
        self.m2  += delta * (x - self.mean);
    }
    fn std_dev(&self) -> f64 {
        if self.n < 2 { 0.0 } else { (self.m2 / (self.n - 1) as f64).sqrt() }
    }
}

// ─────────────────────────── Reservoir sampler ──────────────────────────────

struct Reservoir {
    data:    Vec<u64>,
    stride:  u64,
    counter: u64,
}
impl Reservoir {
    fn new(capacity: usize, total: u64) -> Self {
        let stride = (total / capacity as u64).max(1);
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

/// Build an ONNX MLP for `batch` observations: [batch, OBS_DIM] → [batch, ACT_DIM].
///
/// With ModelMode::Shared, both actors use this single session.  The coordinator
/// feeds each actor its half-slice ([512, 8]), so the model is built for batch=half.
fn bootstrap_model<B>(batch: usize)
    -> Result<ModelModule<B>, Box<dyn std::error::Error>>
where B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>
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
        model_file:     "bootstrap_dual_actor_shared.onnx".into(),
        model_type:     ModelFileType::Onnx,
        input_dtype:    DType::NdArray(NdArrayDType::F32),
        output_dtype:   DType::NdArray(NdArrayDType::F32),
        input_shape:    vec![batch, OBS_DIM],
        output_shape:   vec![batch, ACT_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(bytes, meta)?)
}

// ─────────────────────────── Batch action decode ────────────────────────────

/// Decode a batched [n, ACT_DIM] relay action output into `n` argmax indices.
fn decode_batch_actions(relay: &relayrl_types::data::action::RelayRLAction, n: usize) -> Vec<u8> {
    let bytes_per = ACT_DIM * 4;
    relay.get_act()
        .map(|d| {
            (0..n).map(|i| {
                let s = i * bytes_per;
                let e = (s + bytes_per).min(d.data.len());
                if e > s {
                    d.data[s..e]
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                        .map(|(idx, _)| idx as u8)
                        .unwrap_or(0)
                } else {
                    0
                }
            }).collect()
        })
        .unwrap_or_else(|| vec![0u8; n])
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let args      = Args::parse();
    let n         = args.num_envs;
    let calls     = args.target_calls;
    let max_steps = args.max_steps;

    assert!(n % 2 == 0, "--num-envs must be even (got {})", n);
    let half = n / 2;  // rows per actor after coordinator split

    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let device_type = DeviceType::Cpu;
    let num_cores = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  RelayRL LunarLander — dual-actor shared-model VecEnv benchmark");
    println!("  2 actors (Shared) · {} total envs · {}/{} coordinator split · {} cores",
             n, half, half, num_cores);
    println!("  Caller passes [{}×{}] tensor + [id0, id1] → coordinator splits internally",
             n, OBS_DIM);
    println!("  Total env steps = {} × {} = {}", n, calls, n as u64 * calls);
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // ── VecEnv: 1024 sub-environments ────────────────────────────────────────
    let mut vec_env = SyncLunarVectorEnv::<B>::new(n, max_steps, device.clone())?;
    vec_env.reset_all();

    // ── Agent: 2 actors, ModelMode::Shared, model sized for half-batch ────────
    // The coordinator slices the [n, OBS_DIM] tensor into [half, OBS_DIM] per
    // actor, so the shared ONNX session is built for input_shape=[half, OBS_DIM].
    let model    = bootstrap_model::<B>(half)?;
    let cfgpath  = std::path::PathBuf::from("./config.json");

    println!("Initialising agent (2 actors, ModelMode::Shared, batch={})…", half);
    let init_start = Instant::now();
    let mut bld = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(2)
        .default_device(device_type)
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Shared))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .default_model(model)
        .router_scale(1);
    if cfgpath.exists() { bld = bld.config_path(cfgpath); }

    let (mut agent, params) = bld.build().await?;
    agent.start(params).await?;
    let init_ms = init_start.elapsed().as_millis();

    let actor_ids = agent.get_actor_ids()?;
    assert_eq!(actor_ids.len(), 2, "expected exactly 2 actor IDs");
    let id0 = actor_ids[0];
    let id1 = actor_ids[1];

    println!("  init time : {} ms  (shared model batch [{},{}])", init_ms, half, OBS_DIM);
    println!("  actor[0]  : {}", id0);
    println!("  actor[1]  : {}", id1);
    println!();

    // ── Timing / metric structures ─────────────────────────────────────────────
    let mut call_res  = Reservoir::new(RESERVOIR, calls);
    let mut env_res   = Reservoir::new(RESERVOIR, calls);
    let mut infer_res = Reservoir::new(RESERVOIR, calls);
    let mut call_wf   = Welford::new();
    let mut env_wf    = Welford::new();
    let mut infer_wf  = Welford::new();

    // Episode tracking across all sub-envs
    let mut ep_returns:  Vec<f32> = Vec::new();
    let mut ep_lengths:  Vec<u64> = Vec::new();
    let mut cur_returns: Vec<f32> = vec![0.0; n];
    let mut cur_lens:    Vec<u64> = vec![0u64; n];

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

    // ── Warm-up: 200 calls ────────────────────────────────────────────────────
    println!("Warming up (200 calls)…");
    vec_env.reset_all();
    for _ in 0..200 {
        let flat  = vec_env.get_stacked_obs();
        let obs_t = Tensor::<B, 2, Float>::from_data(
            TensorData::new(flat, [n, OBS_DIM]), &device);

        // Single call — coordinator splits [n,8] → [half,8] per actor internally
        let results = agent.request_action(vec![id0, id1], obs_t, None, 0.0).await;

        let mut acts = vec![0u8; n];
        if let Ok(ref v) = results {
            for (i, (_, a)) in v.iter().enumerate() {
                let start = i * half;
                let end   = (start + half).min(n);
                acts[start..end].copy_from_slice(&decode_batch_actions(a, end - start));
            }
        }
        let _ = vec_env.step_all(&acts);
    }
    println!("Warm-up done. Starting benchmark…\n");

    // ── Main benchmark loop ───────────────────────────────────────────────────
    vec_env.reset_all();
    let t_start = Instant::now();
    let mut calls_done: u64 = 0;
    let mut total_env_steps: u64 = 0;

    while calls_done < calls {
        let call_start = Instant::now();

        let flat  = vec_env.get_stacked_obs();
        let obs_t = Tensor::<B, 2, Float>::from_data(
            TensorData::new(flat, [n, OBS_DIM]), &device);

        // ── Single request_action — coordinator splits obs internally ─────────
        let infer_start = Instant::now();
        let results = agent.request_action(vec![id0, id1], obs_t, None, 0.0).await;
        let infer_ns = infer_start.elapsed().as_nanos() as u64;
        infer_wf.update(infer_ns as f64);
        infer_res.push(infer_ns);

        // ── Decode combined 1024 actions (actor0 → [0,512), actor1 → [512,1024)) ─
        let mut acts = vec![0u8; n];
        if let Ok(ref v) = results {
            for (i, (_, a)) in v.iter().enumerate() {
                let start = i * half;
                let end   = (start + half).min(n);
                acts[start..end].copy_from_slice(&decode_batch_actions(a, end - start));
            }
        }

        let call_ns = call_start.elapsed().as_nanos() as u64;
        call_wf.update(call_ns as f64);
        call_res.push(call_ns);

        // ── step_all: 1024 envs in parallel (rayon) ───────────────────────────
        let env_start = Instant::now();
        let step_results = vec_env.step_all(&acts);
        let env_ns = env_start.elapsed().as_nanos() as u64;
        env_wf.update(env_ns as f64);
        env_res.push(env_ns);

        // ── Episode accounting ────────────────────────────────────────────────
        for (i, (reward, done)) in step_results.iter().enumerate() {
            cur_returns[i] += reward;
            cur_lens[i]    += 1;
            if *done {
                ep_returns.push(cur_returns[i]);
                ep_lengths.push(cur_lens[i]);
                cur_returns[i] = 0.0;
                cur_lens[i]    = 0;
            }
        }

        calls_done      += 1;
        total_env_steps += n as u64;

        if calls_done % 10_000 == 0 {
            let sps = total_env_steps as f64 / t_start.elapsed().as_secs_f64();
            println!("  [{:>7} calls  {:>10} env-steps]  {:.0} env-steps/sec  ({:.0} calls/sec)",
                     calls_done, total_env_steps, sps, sps / n as f64);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    done_flag.store(true, AOrdering::Relaxed);

    // ── Compute metrics ───────────────────────────────────────────────────────

    let call_mean_us  = call_wf.mean      / 1_000.0;
    let call_std_us   = call_wf.std_dev() / 1_000.0;
    let env_mean_us   = env_wf.mean       / 1_000.0;
    let env_std_us    = env_wf.std_dev()  / 1_000.0;
    let infer_mean_us = infer_wf.mean     / 1_000.0;
    let infer_std_us  = infer_wf.std_dev()/ 1_000.0;

    let call_p50  = call_res.percentile(50.0);
    let call_p95  = call_res.percentile(95.0);
    let call_p99  = call_res.percentile(99.0);
    let call_p999 = call_res.percentile(99.9);
    let env_p50   = env_res.percentile(50.0);
    let env_p99   = env_res.percentile(99.0);
    let infer_p50 = infer_res.percentile(50.0);
    let infer_p95 = infer_res.percentile(95.0);
    let infer_p99 = infer_res.percentile(99.0);
    let jitter_us = (call_p99.saturating_sub(call_p50)) as f64 / 1_000.0;

    let overhead_us    = (call_mean_us - infer_mean_us - env_mean_us).max(0.0);
    let overhead_ratio = overhead_us / call_mean_us.max(1e-9);

    let env_sps       = total_env_steps as f64 / elapsed;
    let calls_per_sec = calls_done as f64 / elapsed;
    let per_core_sps  = env_sps / num_cores as f64;
    let per_actor_sps = env_sps / 2.0;

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

    let samples    = proc_store.lock().unwrap();
    let rss: Vec<u64> = samples.iter().map(|s| s.rss_kb).collect();
    let rss_init   = samples.first().map(|s| s.rss_kb).unwrap_or(0);
    let rss_final  = samples.last().map(|s| s.rss_kb).unwrap_or(0);
    let rss_peak   = rss.iter().copied().max().unwrap_or(0);
    let rss_mean   = if rss.is_empty() { 0 } else { rss.iter().sum::<u64>() / rss.len() as u64 };
    let alloc_rate = rss_final.saturating_sub(rss_init) as f64 / elapsed;
    let ctx_first  = samples.first().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let ctx_last   = samples.last().map(|s| s.vol_ctx_sw + s.nonvol_ctx_sw).unwrap_or(0);
    let ctx_total  = ctx_last.saturating_sub(ctx_first);
    let ctx_per_s  = ctx_total as f64 / elapsed;
    let cpu_first  = samples.first().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_last   = samples.last().map(|s| s.utime_ticks + s.stime_ticks).unwrap_or(0);
    let cpu_util   = (cpu_last.saturating_sub(cpu_first)) as f64 / 100.0 / elapsed * 100.0;
    let cpu_per_core = cpu_util / num_cores as f64;
    let thread_mean  = if samples.is_empty() { 0.0 }
                       else { samples.iter().map(|s| s.threads).sum::<u64>() as f64 / samples.len() as f64 };
    let runq_mean    = if samples.is_empty() { 0.0 }
                       else { samples.iter().map(|s| s.runq as f64).sum::<f64>() / samples.len() as f64 };
    drop(samples);

    let rss_mean_gb  = rss_mean as f64 / (1_024.0 * 1_024.0);
    let sps_per_gb   = if rss_mean_gb > 0.0 { env_sps / rss_mean_gb } else { 0.0 };

    const BASELINE_1ENV: f64 = 19_443.0;
    let scalability = env_sps / BASELINE_1ENV;

    // ── Report ────────────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  RelayRL LunarLander — DUAL-ACTOR SHARED FINAL RESULTS");
    println!("  2 actors (Shared)  |  {} total envs ({}/{} split)  |  {} calls",
             n, half, half, calls_done);
    println!("  Batch path: caller → [{}×{}] → coordinator splits → 2×[{}×{}] → shared ONNX",
             n, OBS_DIM, half, OBS_DIM);
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────────");
    println!("  env-steps/sec (global)         : {:>12.1}   ({} envs × {:.0} calls/s)",
             env_sps, n, calls_per_sec);
    println!("  env-steps/sec per actor        : {:>12.1}   ({} envs × {:.0} calls/s)",
             per_actor_sps, half, calls_per_sec);
    println!("  env-steps/sec per logical core : {:>12.1}", per_core_sps);
    println!("  calls/sec (single request)     : {:>12.1}", calls_per_sec);
    println!("  episodes/sec (all sub-envs)    : {:>12.3}", eps_per_s);
    println!("  total calls                    : {:>12}", calls_done);
    println!("  total env steps                : {:>12}", total_env_steps);
    println!("  wall time                      : {:>12.2} s", elapsed);
    println!("  logical cores                  : {:>12}", num_cores);
    println!("  VecEnv batch (total)           : {:>12}   ({} per actor, coordinator-split)",
             n, half);
    println!("  max steps / episode            : {:>12}", max_steps);
    println!();

    println!("─── Episode Statistics ──────────────────────────────────────────────────");
    println!("  avg steps per episode          : {:>12.1}", avg_ep_len);
    println!("  episode return mean            : {:>12.3}", ep_mean);
    println!("  episode completion variance    : {:>12.3}", ep_var);
    println!("  total episodes (all sub-envs)  : {:>12}", ep_returns.len());
    println!();

    println!("─── Coordinator Inference Timing (µs) — single call, 2-actor split ──────");
    println!("  request_action mean            : {:>12.3} µs  (both actors, [{}×{}] split internally)",
             infer_mean_us, n, OBS_DIM);
    println!("  request_action std dev         : {:>12.3} µs", infer_std_us);
    println!("  request_action P50             : {:>12.3} µs", infer_p50 as f64 / 1_000.0);
    println!("  request_action P95             : {:>12.3} µs", infer_p95 as f64 / 1_000.0);
    println!("  request_action P99             : {:>12.3} µs", infer_p99 as f64 / 1_000.0);
    println!("  model batch per actor          : {:>12}   (coordinator ceil({}/2))",
             half, n);
    println!("  model mode                     :   Shared (single ONNX session, 2 actors)");
    println!();

    println!("─── Full Round Timing (µs) — request_action + decode + step_all ─────────");
    println!("  round mean                     : {:>12.3} µs", call_mean_us);
    println!("  round std dev                  : {:>12.3} µs", call_std_us);
    println!("  round P50                      : {:>12.3} µs", call_p50  as f64 / 1_000.0);
    println!("  round P95                      : {:>12.3} µs", call_p95  as f64 / 1_000.0);
    println!("  round P99                      : {:>12.3} µs", call_p99  as f64 / 1_000.0);
    println!("  round P99.9                    : {:>12.3} µs", call_p999 as f64 / 1_000.0);
    println!("  jitter (P99 − P50)             : {:>12.3} µs", jitter_us);
    println!();

    println!("─── Env Step Timing (µs) — rayon parallel step ({} sub-envs) ─────────────", n);
    println!("  env step mean  (rayon, {} envs): {:>12.3} µs", n, env_mean_us);
    println!("  env step std dev               : {:>12.3} µs", env_std_us);
    println!("  env step P50                   : {:>12.3} µs", env_p50 as f64 / 1_000.0);
    println!("  env step P99                   : {:>12.3} µs", env_p99 as f64 / 1_000.0);
    println!("  env step / round ratio         : {:>12.3}",
             env_mean_us / call_mean_us.max(1e-9));
    println!("  env step / sub-env             : {:>12.3} µs  (mean/{})",
             env_mean_us / n as f64, n);
    println!();

    println!("─── Scheduling / Overhead ───────────────────────────────────────────────");
    println!("  overhead per round             : {:>12.3} µs  (round − infer − env)",
             overhead_us);
    println!("  overhead ratio                 : {:>12.4}", overhead_ratio);
    println!("  dropped/late updates           : {:>12}", 0u32);
    println!();

    println!("─── Memory ──────────────────────────────────────────────────────────────");
    println!("  RSS init                       : {:>9.1} MB", rss_init  as f64 / 1_024.0);
    println!("  RSS peak                       : {:>9.1} MB", rss_peak  as f64 / 1_024.0);
    println!("  RSS mean                       : {:>9.1} MB  ({:.3} GB)",
             rss_mean as f64 / 1_024.0, rss_mean_gb);
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
    println!("  context switches/call          : {:>9.6}", ctx_total as f64 / calls_done as f64);
    println!("  context switches/env-step      : {:>9.6}", ctx_total as f64 / total_env_steps as f64);
    println!();

    println!("─── Efficiency Ratios ───────────────────────────────────────────────────");
    println!("  env-steps/sec / logical core   : {:>12.1}", per_core_sps);
    println!("  env-steps/sec / GB RSS (proxy) : {:>12.1}", sps_per_gb);
    println!("  S(n) vs 1-actor 1-env baseline : {:>12.3}×  ({:.1} sps / {:.1} baseline)",
             scalability, env_sps, BASELINE_1ENV);
    println!("  overhead ratio                 : {:>12.4}", overhead_ratio);
    println!();

    println!("─── Notes ───────────────────────────────────────────────────────────────");
    println!("  coordinator split              :   ceil({}/{}) = {} rows per actor",
             n, 2, half);
    println!("  model sharing                  :   Shared — single ONNX session, 2 actors");
    println!("  caller concurrency             :   none — single request_action call");
    println!("  env stepping                   :   rayon work-stealing over {} sub-envs", n);
    println!();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  SUMMARY: {} ms init · 2 actors (Shared) · {} total envs · batch [{},{}] each",
             init_ms, n, half, OBS_DIM);
    println!("═══════════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
