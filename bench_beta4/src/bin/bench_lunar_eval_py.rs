//! bench_lunar_eval_py — pure inference eval on LunarLander-v3 via Python/gymnasium.
//!
//! Loads the pre-trained lunarlander_policy.onnx (64×64, 8→4), wraps 64 parallel
//! gymnasium envs via the PyVectorEnv bridge, and runs agent.run_env() inference-only.
//! Reports steps/sec, /proc RSS, context switches, and page faults so the output can
//! also be piped through `perf stat` for hardware counter data.
//!
//! Build:
//!   cargo build --release -p bench-beta4 --bin bench_lunar_eval_py
//!
//! Run standalone:
//!   LD_LIBRARY_PATH=... ./target/release/bench_lunar_eval_py
//!
//! Run under perf stat:
//!   perf stat -e cycles,instructions,cache-misses,cache-references,branch-misses \
//!     ./target/release/bench_lunar_eval_py

use std::path::PathBuf;
use std::time::Instant;

use burn_ndarray::NdArray;

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode,
    RelayRLActorEnv, RelayRLAgentActors,
};
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;

use relayrl_types::data::tensor::{DType, NdArrayDType};
use relayrl_types::model::{ModelFileType, ModelMetadata, ModelModule};

use bench_beta4::py_env::make_lunar_lander_vec;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const ENV_COUNT: u32 = 1024;

// Warm-up: 500 steps × 64 envs = 32k transitions (amortises ONNX JIT / first-call overhead)
const WARMUP_STEPS: usize = 500;
// Timed run: 5000 steps × 64 envs = 320k transitions
const TIMED_STEPS: usize = 5_000;

// Path to the pre-trained LunarLander ONNX policy (8→64→64→4, 4996 params).
// Trained by the relayrl-e2e lunarlander suite.
const MODEL_DIR: &str = "/home/user/RelayRL-end2end/model_lunar";

// ─────────────────────────── /proc helpers ──────────────────────────────────

#[derive(Default)]
struct ProcStats {
    rss_kb: u64,
    vol_ctx: u64,
    nvol_ctx: u64,
    minor_faults: u64,
    major_faults: u64,
}

fn read_proc_stats() -> ProcStats {
    let mut s = ProcStats::default();
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            let mut it = line.splitn(2, ':');
            let k = it.next().unwrap_or("").trim();
            let v = it.next().unwrap_or("").trim();
            match k {
                "VmRSS"                      => s.rss_kb    = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"    => s.vol_ctx   = v.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => s.nvol_ctx  = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        let fields: Vec<&str> = stat.split_whitespace().collect();
        // field 9: minflt, field 11: majflt (0-indexed from 0)
        s.minor_faults = fields.get(9).and_then(|x| x.parse().ok()).unwrap_or(0);
        s.major_faults = fields.get(11).and_then(|x| x.parse().ok()).unwrap_or(0);
    }
    s
}

// ─────────────────────────── Model loader ───────────────────────────────────

fn load_lunarlander_model() -> Result<ModelModule<NdArray>, Box<dyn std::error::Error>> {
    let onnx_bytes = std::fs::read(
        PathBuf::from(MODEL_DIR).join("lunarlander_policy.onnx"),
    )?;
    let metadata = ModelMetadata {
        model_file:     "lunarlander_policy.onnx".to_string(),
        model_type:     ModelFileType::Onnx,
        input_dtype:    DType::NdArray(NdArrayDType::F32),
        output_dtype:   DType::NdArray(NdArrayDType::F32),
        input_shape:    vec![1, OBS_DIM],
        output_shape:   vec![1, ACT_DIM],
        default_device: Some(DeviceType::Cpu),
    };
    Ok(ModelModule::<NdArray>::from_onnx_bytes(onnx_bytes, metadata)?)
}

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.26.0",
    );

    type B = NdArray;

    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let model = load_lunarlander_model()?;
    println!("══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta4 — eval — LunarLander-v3 — Python/gymnasium — {} envs", ENV_COUNT);
    println!("  model  : {MODEL_DIR}/lunarlander_policy.onnx  (8→64→64→4, 4996 params)");
    println!("  backend: NdArray (ONNX-runtime inference, CPU)");
    println!("  warmup : {} steps × {} envs = {} transitions",
             WARMUP_STEPS, ENV_COUNT, WARMUP_STEPS * ENV_COUNT as usize);
    println!("  timed  : {} steps × {} envs = {} transitions",
             TIMED_STEPS, ENV_COUNT, TIMED_STEPS * ENV_COUNT as usize);
    println!("  cores  : {num_cores} logical");
    println!("══════════════════════════════════════════════════════════════════\n");

    // ── Agent setup ───────────────────────────────────────────────────────────
    let config_path = PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B, 2, 2>::builder()
        .actor_count(1)
        .default_device(DeviceType::Cpu)
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .default_model(model)
        .router_scale(1);
    if config_path.exists() {
        builder = builder.config_path(config_path);
    }

    let (mut agent, params) = builder.build().await?;
    agent.start(params).await?;
    let actor_ids = agent.get_actor_ids()?;
    let actor_id = actor_ids[0];

    // ── Python gymnasium env ──────────────────────────────────────────────────
    let py_env = make_lunar_lander_vec(ENV_COUNT as usize, OBS_DIM, ACT_DIM)
        .map_err(|e| format!("gymnasium env creation failed: {e}"))?;
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(py_env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!("set_env OK — {} gymnasium LunarLander-v3 sub-envs registered\n", ENV_COUNT);

    // ── Warm-up ───────────────────────────────────────────────────────────────
    println!("Warming up ({} steps × {} envs)…", WARMUP_STEPS, ENV_COUNT);
    let t_warmup = Instant::now();
    agent.run_env(actor_id, WARMUP_STEPS).await?;
    let warmup_wall = t_warmup.elapsed().as_secs_f64();
    println!("Warm-up done in {warmup_wall:.2}s  ({:.0} env transitions/sec)\n",
             (WARMUP_STEPS * ENV_COUNT as usize) as f64 / warmup_wall);

    // ── Baseline /proc snapshot (before timed run) ────────────────────────────
    let before = read_proc_stats();

    // ── Timed run ─────────────────────────────────────────────────────────────
    println!("Starting timed run ({} steps × {} envs)…", TIMED_STEPS, ENV_COUNT);
    let t0 = Instant::now();
    agent.run_env(actor_id, TIMED_STEPS).await?;
    let wall = t0.elapsed().as_secs_f64();

    // ── Post-run /proc snapshot ───────────────────────────────────────────────
    let after = read_proc_stats();

    // ── Derived metrics ───────────────────────────────────────────────────────
    let total_transitions = TIMED_STEPS * ENV_COUNT as usize;
    let steps_per_sec     = TIMED_STEPS as f64 / wall;
    let transitions_sec   = total_transitions as f64 / wall;
    let us_per_step       = 1_000_000.0 / steps_per_sec;
    let us_per_transition = 1_000_000.0 / transitions_sec;

    let vol_delta   = after.vol_ctx.saturating_sub(before.vol_ctx);
    let nvol_delta  = after.nvol_ctx.saturating_sub(before.nvol_ctx);
    let total_ctx   = vol_delta + nvol_delta;
    let minor_delta = after.minor_faults.saturating_sub(before.minor_faults);
    let major_delta = after.major_faults.saturating_sub(before.major_faults);

    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  RESULTS — LunarLander-v3 eval — Python/gymnasium — {} envs", ENV_COUNT);
    println!("══════════════════════════════════════════════════════════════════");

    println!();
    println!("─── Throughput ────────────────────────────────────────────────────");
    println!("  env count              : {:>10}", ENV_COUNT);
    println!("  loop steps (timed)     : {:>10}", TIMED_STEPS);
    println!("  total env transitions  : {:>10}", total_transitions);
    println!("  wall time              : {:>10.3} s", wall);
    println!("  steps / sec            : {:>10.1}", steps_per_sec);
    println!("  env transitions / sec  : {:>10.1}", transitions_sec);
    println!("  µs / step              : {:>10.3}", us_per_step);
    println!("  µs / env transition    : {:>10.3}", us_per_transition);

    println!();
    println!("─── Memory ────────────────────────────────────────────────────────");
    println!("  RSS (after run)        : {:>8.1} MB", after.rss_kb as f64 / 1024.0);
    println!("  minor page faults (Δ) : {:>10}", minor_delta);
    println!("  major page faults (Δ) : {:>10}", major_delta);

    println!();
    println!("─── OS scheduling ─────────────────────────────────────────────────");
    println!("  vol ctx switches  (Δ) : {:>10}", vol_delta);
    println!("  nvol ctx switches (Δ) : {:>10}", nvol_delta);
    println!("  total ctx switches(Δ) : {:>10}", total_ctx);
    println!("  ctx switches / step   : {:>10.4}", total_ctx as f64 / TIMED_STEPS as f64);
    println!("  logical cores         : {:>10}", num_cores);

    println!();
    println!("─── Timing breakdown ──────────────────────────────────────────────");
    println!("  warmup  ({:>5} steps): {:>8.2} s  ({:.0} transitions/sec)",
             WARMUP_STEPS, warmup_wall, (WARMUP_STEPS * ENV_COUNT as usize) as f64 / warmup_wall);
    println!("  timed   ({:>5} steps): {:>8.2} s  ({:.0} transitions/sec)",
             TIMED_STEPS, wall, transitions_sec);
    println!("══════════════════════════════════════════════════════════════════");
    println!();
    println!("TIP: run under perf stat for hardware counters:");
    println!("  perf stat -e cycles,instructions,cache-misses,cache-references,branch-misses \\");
    println!("    ./target/release/bench_lunar_eval_py");

    agent.shutdown().await?;
    Ok(())
}
