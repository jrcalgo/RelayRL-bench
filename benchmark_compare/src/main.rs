//! bench_gridworld_vecenv — relayrl_framework 0.5.0-beta.2 comparison binary.
//!
//! Same benchmark logic as crates/relayrl-e2e/src/bin/bench_gridworld_vecenv.rs,
//! running against the unpatched 0.5.0-beta.2 release from crates.io.
//!
//! 1 actor · N parallel GridWorld sub-envs · batched ONNX inference + rayon stepping.
//!
//! Build:
//!   cargo build --release --manifest-path benchmark_compare/Cargo.toml
//!
//! Run:
//!   ./target/release/bench_gridworld_vecenv_beta2 --num-envs 32

mod vec_env;

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

use vec_env::SyncVectorEnv;

// ─────────────────────────────── CLI ────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "bench_gridworld_vecenv_beta2",
    about = "RelayRL GridWorld VecEnv benchmark — relayrl_framework 0.5.0-beta.2"
)]
struct Args {
    #[arg(long, default_value_t = 64)]
    num_envs: usize,

    #[arg(long, default_value_t = 20_000u64)]
    target_calls: u64,

    #[arg(long, default_value_t = 10)]
    grid_size: usize,
}

// ─────────────────────────────── Constants ──────────────────────────────────

const ACT_DIM:   usize = 4;
const RESERVOIR: usize = 100_000;

// ─────────────────────────────── /proc helpers ──────────────────────────────

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
                "VmRSS"                       => s.rss_kb        = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"     => s.vol_ctx_sw    = v.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches"  => s.nonvol_ctx_sw = v.parse().unwrap_or(0),
                "Threads"                     => s.threads        = v.parse().unwrap_or(0),
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

// ─────────────────────────────── Welford ────────────────────────────────────

struct Welford { n: u64, mean: f64, m2: f64 }
impl Welford {
    fn new() -> Self { Self { n: 0, mean: 0.0, m2: 0.0 } }
    fn update(&mut self, x: f64) {
        self.n   += 1;
        let d     = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2  += d * (x - self.mean);
    }
    fn std_dev(&self) -> f64 {
        if self.n < 2 { 0.0 } else { (self.m2 / (self.n - 1) as f64).sqrt() }
    }
}

// ─────────────────────────────── Reservoir ──────────────────────────────────

struct Reservoir { data: Vec<u64>, stride: u64, counter: u64 }
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

// ─────────────────────────────── Bootstrap model ────────────────────────────

fn bootstrap_model<B>(batch: usize, obs_dim: usize)
    -> Result<ModelModule<B>, Box<dyn std::error::Error>>
where B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>
{
    use relayrl_types::data::tensor::{DType, NdArrayDType};
    use relayrl_types::model::{ModelFileType, ModelMetadata};
    let specs: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = vec![
        (obs_dim, 64, vec![0.01f32; 64 * obs_dim], vec![0.0f32; 64]),
        (64,      64, vec![0.01f32; 64 * 64],       vec![0.0f32; 64]),
        (64, ACT_DIM, vec![0.01f32; ACT_DIM * 64],  vec![0.0f32; ACT_DIM]),
    ];
    let bytes = build_onnx_mlp_bytes(&specs);
    let meta = ModelMetadata {
        model_file:     "bootstrap_vec.onnx".into(),
        model_type:     ModelFileType::Onnx,
        input_dtype:    DType::NdArray(NdArrayDType::F32),
        output_dtype:   DType::NdArray(NdArrayDType::F32),
        input_shape:    vec![batch, obs_dim],
        output_shape:   vec![batch, ACT_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<B>::from_onnx_bytes(bytes, meta)?)
}

// ─────────────────────────────── Action decode ──────────────────────────────

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
                } else { 0 }
            }).collect()
        })
        .unwrap_or_else(|| vec![0u8; n])
}

// ─────────────────────────────── Main ───────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let args    = Args::parse();
    let n       = args.num_envs;
    let calls   = args.target_calls;
    let gs      = args.grid_size.max(3);
    let obs_dim = gs * gs;

    let device: <B as burn_tensor::backend::Backend>::Device = Default::default();
    let device_type = DeviceType::Cpu;
    let num_cores = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  RelayRL GridWorld VecEnv benchmark  [relayrl_framework 0.5.0-beta.2]");
    println!("  1 actor · {} sub-envs · {} calls · {}×{} grid · {} cores",
             n, calls, gs, gs, num_cores);
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let mut vec_env = SyncVectorEnv::<B>::new(n, gs, device.clone())?;
    vec_env.reset_all();

    let model   = bootstrap_model::<B>(n, obs_dim)?;
    let cfgpath = std::path::PathBuf::from("./config.json");

    println!("Initialising agent (1 actor, ModelMode::Shared)…");
    let init_start = Instant::now();
    let mut bld = AgentBuilder::<B, 2, 2, Float, Float>::builder()
        .actor_count(1)
        .default_device(device_type)
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Shared))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .default_model(model)
        .router_scale(1);
    if cfgpath.exists() { bld = bld.config_path(cfgpath); }

    let (mut agent, params) = bld.build().await?;
    agent.start(params).await?;
    let init_ms  = init_start.elapsed().as_millis();
    let actor_ids = agent.get_actor_ids()?;
    assert_eq!(actor_ids.len(), 1);
    println!("  init time : {} ms\n", init_ms);

    let mut call_res  = Reservoir::new(RESERVOIR, calls);
    let mut env_res   = Reservoir::new(RESERVOIR, calls);
    let mut call_wf   = Welford::new();
    let mut env_wf    = Welford::new();
    let mut infer_wf  = Welford::new();

    let mut ep_returns:  Vec<f32> = Vec::new();
    let mut ep_lengths:  Vec<u64> = Vec::new();
    let mut cur_returns: Vec<f32> = vec![0.0; n];
    let mut cur_lens:    Vec<u64> = vec![0u64; n];

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

    // Warm-up
    println!("Warming up (200 calls)…");
    vec_env.reset_all();
    for _ in 0..200 {
        let flat  = vec_env.get_stacked_obs();
        let obs_t = Tensor::<B, 2, Float>::from_data(TensorData::new(flat, [n, obs_dim]), &device);
        if let Ok(ref actions) = agent.request_action(actor_ids.clone(), obs_t, None, 0.0).await {
            if let Some((_, ra)) = actions.first() {
                let acts = decode_batch_actions(ra, n);
                let _ = vec_env.step_all(&acts);
            }
        }
    }
    println!("Warm-up done. Starting benchmark…\n");

    vec_env.reset_all();
    let t_start   = Instant::now();
    let mut calls_done: u64 = 0;
    let mut total_env_steps: u64 = 0;

    while calls_done < calls {
        let flat  = vec_env.get_stacked_obs();
        let obs_t = Tensor::<B, 2, Float>::from_data(TensorData::new(flat, [n, obs_dim]), &device);

        let call_start = Instant::now();
        let results = agent.request_action(actor_ids.clone(), obs_t, None, 0.0).await;
        let call_ns = call_start.elapsed().as_nanos() as u64;
        call_wf.update(call_ns as f64);
        call_res.push(call_ns);

        let env_start = Instant::now();
        let step_results: Vec<(f32, bool)> = if let Ok(ref actions) = results {
            if let Some((_, ra)) = actions.first() {
                let acts = decode_batch_actions(ra, n);
                vec_env.step_all(&acts)
            } else { vec![(0.0, false); n] }
        } else {
            let dummy = vec![0u8; n];
            vec_env.step_all(&dummy)
        };
        let env_ns = env_start.elapsed().as_nanos() as u64;
        env_wf.update(env_ns as f64);
        env_res.push(env_ns);

        infer_wf.update(call_ns.saturating_sub(env_ns) as f64);

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

        if calls_done % 5_000 == 0 {
            let sps = total_env_steps as f64 / t_start.elapsed().as_secs_f64();
            println!("  [{:>7} calls  {:>10} env-steps]  {:.0} env-steps/sec",
                     calls_done, total_env_steps, sps);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    done_flag.store(true, AOrdering::Relaxed);

    // ── Compute metrics ──────────────────────────────────────────────────────

    let call_mean_us  = call_wf.mean  / 1_000.0;
    let call_std_us   = call_wf.std_dev() / 1_000.0;
    let env_mean_us   = env_wf.mean   / 1_000.0;
    let env_std_us    = env_wf.std_dev() / 1_000.0;
    let infer_mean_us = infer_wf.mean / 1_000.0;

    let call_p50  = call_res.percentile(50.0);
    let call_p95  = call_res.percentile(95.0);
    let call_p99  = call_res.percentile(99.0);
    let call_p999 = call_res.percentile(99.9);
    let env_p50   = env_res.percentile(50.0);
    let env_p99   = env_res.percentile(99.0);
    let jitter_us = (call_p99.saturating_sub(call_p50)) as f64 / 1_000.0;

    let step_mean_us   = call_mean_us + env_mean_us;
    let overhead_us    = (call_mean_us - infer_mean_us - env_mean_us).max(0.0);
    let overhead_ratio = overhead_us / step_mean_us.max(1e-9);

    let env_sps        = total_env_steps as f64 / elapsed;
    let calls_per_sec  = calls_done as f64 / elapsed;
    let per_core_sps   = env_sps / num_cores as f64;

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
    let dispatch_lat = call_p50 as f64 / 1_000.0;

    const BASELINE_1ENV: f64 = 19_443.0;
    let scalability = env_sps / BASELINE_1ENV;

    // ── Report ────────────────────────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  relayrl_framework 0.5.0-beta.2 — FINAL RESULTS");
    println!("  1 actor  |  {} sub-envs  |  {} calls  |  batched ONNX + rayon step", n, calls_done);
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────────");
    println!("  env-steps/sec (global)         : {:>12.1}   ({} envs × {:.0} calls/s)",
             env_sps, n, calls_per_sec);
    println!("  env-steps/sec per sub-env      : {:>12.1}", calls_per_sec);
    println!("  env-steps/sec per logical core : {:>12.1}", per_core_sps);
    println!("  calls/sec (request_action)     : {:>12.1}", calls_per_sec);
    println!("  episodes/sec (all sub-envs)    : {:>12.3}", eps_per_s);
    println!("  total calls                    : {:>12}", calls_done);
    println!("  total env steps                : {:>12}", total_env_steps);
    println!("  wall time                      : {:>12.2} s", elapsed);
    println!("  logical cores                  : {:>12}", num_cores);
    println!("  VecEnv batch size              : {:>12}", n);
    println!();

    println!("─── Episode Statistics ──────────────────────────────────────────────────");
    println!("  avg steps per episode          : {:>12.1}", avg_ep_len);
    println!("  episode return mean            : {:>12.3}", ep_mean);
    println!("  episode completion variance    : {:>12.3}", ep_var);
    println!("  total episodes (all sub-envs)  : {:>12}", ep_returns.len());
    println!();

    println!("─── Inference Timing (µs) ───────────────────────────────────────────────");
    println!("  inference mean  (call − env)   : {:>12.3} µs  [{} obs]", infer_mean_us, n);
    println!("  inference / total call ratio   : {:>12.3}", infer_mean_us / call_mean_us.max(1e-9));
    println!();

    println!("─── Call Timing (µs) ────────────────────────────────────────────────────");
    println!("  call mean                      : {:>12.3} µs", call_mean_us);
    println!("  call std dev                   : {:>12.3} µs", call_std_us);
    println!("  P50                            : {:>12.3} µs", call_p50  as f64 / 1_000.0);
    println!("  P95                            : {:>12.3} µs", call_p95  as f64 / 1_000.0);
    println!("  P99                            : {:>12.3} µs", call_p99  as f64 / 1_000.0);
    println!("  P99.9                          : {:>12.3} µs", call_p999 as f64 / 1_000.0);
    println!("  jitter (P99 − P50)             : {:>12.3} µs", jitter_us);
    println!("  variance in step time (σ²)     : {:>12.3} µs²", call_std_us * call_std_us);
    println!();

    println!("─── Env Step Timing (µs) ────────────────────────────────────────────────");
    println!("  env step mean  ({} envs)      : {:>12.3} µs", n, env_mean_us);
    println!("  env step std dev               : {:>12.3} µs", env_std_us);
    println!("  env step P50                   : {:>12.3} µs", env_p50 as f64 / 1_000.0);
    println!("  env step P99                   : {:>12.3} µs", env_p99 as f64 / 1_000.0);
    println!("  env step / call ratio          : {:>12.3}", env_mean_us / step_mean_us.max(1e-9));
    println!("  env step / sub-env             : {:>12.3} µs  (mean/{})", env_mean_us / n as f64, n);
    println!();

    println!("─── Scheduling / Overhead ───────────────────────────────────────────────");
    println!("  step mean (call + env)         : {:>12.3} µs", step_mean_us);
    println!("  scheduling overhead            : {:>12.3} µs", overhead_us);
    println!("  overhead ratio                 : {:>12.4}", overhead_ratio);
    println!("  actor dispatch latency (P50)   : {:>12.3} µs", dispatch_lat);
    println!("  deadtime per sub-env           : {:>12.6}", overhead_ratio / n as f64);
    println!();

    println!("─── Memory ──────────────────────────────────────────────────────────────");
    println!("  RSS init                       : {:>9.1} MB", rss_init  as f64 / 1_024.0);
    println!("  RSS peak                       : {:>9.1} MB", rss_peak  as f64 / 1_024.0);
    println!("  RSS mean                       : {:>9.1} MB  ({:.3} GB)", rss_mean as f64 / 1_024.0, rss_mean_gb);
    println!("  RSS final                      : {:>9.1} MB", rss_final as f64 / 1_024.0);
    println!("  allocation rate (RSS Δ/s)      : {:>9.3} KB/s", alloc_rate);
    println!();

    println!("─── CPU / OS ────────────────────────────────────────────────────────────");
    println!("  CPU utilisation (summed cores) : {:>9.2} %", cpu_util);
    println!("  CPU util / logical core        : {:>9.2} %", cpu_per_core);
    println!("  mean threads                   : {:>9.1}", thread_mean);
    println!("  run queue length (1-min avg)   : {:>9.3}", runq_mean);
    println!("  context switches total         : {:>9}", ctx_total);
    println!("  context switches/sec           : {:>9.1}", ctx_per_s);
    println!("  context switches/call          : {:>9.6}", ctx_total as f64 / calls_done as f64);
    println!("  context switches/env-step      : {:>9.6}", ctx_total as f64 / total_env_steps as f64);
    println!();

    println!("─── Efficiency Ratios ───────────────────────────────────────────────────");
    println!("  env-steps/sec / logical core   : {:>12.1}", per_core_sps);
    println!("  env-steps/sec / GB RSS (proxy) : {:>12.1}", sps_per_gb);
    println!("  S(n) vs 1-env baseline         : {:>12.3}×  ({:.1} sps / {:.1} baseline)",
             scalability, env_sps, BASELINE_1ENV);
    println!("  overhead ratio                 : {:>12.4}", overhead_ratio);
    println!();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  INIT: {} ms  |  1 actor  |  {} envs  |  batch [{},{}]",
             init_ms, n, n, obs_dim);
    println!("═══════════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
