//! bench_lunar_eval_envpool_2actor — two concurrent actors on LunarLander-v3 via EnvPool.
//!
//! Creates two independent actors inside a single RelayRL agent.  Each actor owns a
//! separate N-env EnvPool instance (default 64; set via --envs).  Both eval loops run
//! concurrently via tokio::spawn so their block_in_place calls land on separate OS threads.
//!
//! Measures aggregate throughput (2 × N × steps transitions) and per-actor timing.
//!
//! Build:
//!   cargo build --release -p bench-beta5 --bin bench_lunar_eval_envpool_2actor
//!
//! Run:
//!   /usr/bin/time -v ./target/release/bench_lunar_eval_envpool_2actor
//!   /usr/bin/time -v ./target/release/bench_lunar_eval_envpool_2actor --envs 4096

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use burn_ndarray::NdArray;
use clap::Parser;

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode,
    RelayRLActorEnv, RelayRLAgentActors,
};
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;

use relayrl_types::data::tensor::{DType, NdArrayDType};
use relayrl_types::model::{ModelFileType, ModelMetadata, ModelModule};

use bench_beta5::py_env::make_envpool_lunar_lander_vec;

// ─────────────────────────── CLI ────────────────────────────────────────────

#[derive(Parser)]
struct Cli {
    /// Number of parallel environments per actor
    #[arg(long, default_value_t = 64)]
    envs: u32,
}

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;

const ACTOR_COUNT: u32 = 2;

const WARMUP_STEPS: usize = 500;
const TIMED_STEPS: usize = 5_000;

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
                "VmRSS"                      => s.rss_kb   = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0),
                "voluntary_ctxt_switches"    => s.vol_ctx  = v.parse().unwrap_or(0),
                "nonvoluntary_ctxt_switches" => s.nvol_ctx = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        let fields: Vec<&str> = stat.split_whitespace().collect();
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

    let cli = Cli::parse();
    let env_count = cli.envs as usize;

    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let model = load_lunarlander_model()?;
    let total_envs = ACTOR_COUNT as usize * env_count;

    println!("══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta5 — eval — LunarLander-v3 — EnvPool — 2 actors");
    println!("  model  : {MODEL_DIR}/lunarlander_policy.onnx  (8→64→64→4, 4996 params)");
    println!("  backend: NdArray (ONNX-runtime inference, CPU) + EnvPool C++ thread pool");
    println!("  actors : {} × {} envs = {} total envs", ACTOR_COUNT, env_count, total_envs);
    println!("  warmup : {} steps × {} total envs = {} transitions",
             WARMUP_STEPS, total_envs, WARMUP_STEPS * total_envs);
    println!("  timed  : {} steps × {} total envs = {} transitions",
             TIMED_STEPS, total_envs, TIMED_STEPS * total_envs);
    println!("  cores  : {num_cores} logical");
    println!("══════════════════════════════════════════════════════════════════\n");

    // ── Agent + actor setup ───────────────────────────────────────────────────
    let config_path = PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B>::builder()
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .default_model(model.clone())
        .router_scale(1);
    if config_path.exists() {
        builder = builder.config_path(config_path);
    }

    let (mut agent, params) = builder.build().await?;
    agent.start(params).await?;

    // Two actors — each gets its own cloned ONNX session via Some(model.clone())
    agent.new_actors::<OBS_DIM, ACT_DIM>(ACTOR_COUNT, DeviceType::Cpu, 0, Some(model.clone())).await?;
    let actor_ids = agent.get_actor_ids()?;
    let actor0_id = actor_ids[0];
    let actor1_id = actor_ids[1];

    // ── EnvPool envs ─────────────────────────────────────────────────────────
    // Each actor gets an independent EnvPool instance with env_count sub-envs.
    let ep_env0 = make_envpool_lunar_lander_vec(env_count, OBS_DIM, ACT_DIM)
        .map_err(|e| format!("EnvPool env0 creation failed: {e}"))?;
    let ep_env1 = make_envpool_lunar_lander_vec(env_count, OBS_DIM, ACT_DIM)
        .map_err(|e| format!("EnvPool env1 creation failed: {e}"))?;

    agent.set_env(actor0_id, Box::new(ep_env0), env_count as u32).await?;
    agent.set_env(actor1_id, Box::new(ep_env1), env_count as u32).await?;
    println!("set_env OK — actor0: {} envs | actor1: {} envs\n", env_count, env_count);

    // Wrap in Arc so both spawned tasks share the agent via &self (run_env_eval is &self)
    let agent = Arc::new(agent);

    // ── Concurrent warm-up ────────────────────────────────────────────────────
    println!("Warming up ({} steps × {} actors × {} envs)…", WARMUP_STEPS, ACTOR_COUNT, env_count);
    let t_warmup = Instant::now();
    {
        let a0 = Arc::clone(&agent);
        let a1 = Arc::clone(&agent);
        let (r0, r1) = tokio::join!(
            tokio::spawn(async move { a0.run_env_eval(actor0_id, WARMUP_STEPS).await }),
            tokio::spawn(async move { a1.run_env_eval(actor1_id, WARMUP_STEPS).await }),
        );
        r0??;
        r1??;
    }
    let warmup_wall = t_warmup.elapsed().as_secs_f64();
    let warmup_total = WARMUP_STEPS * total_envs;
    println!("Warm-up done in {warmup_wall:.2}s  ({:.0} total env transitions/sec)\n",
             warmup_total as f64 / warmup_wall);

    // ── Baseline /proc snapshot ───────────────────────────────────────────────
    let before = read_proc_stats();

    // ── Concurrent timed run ─────────────────────────────────────────────────
    println!("Starting timed run ({} steps × {} actors × {} envs)…", TIMED_STEPS, ACTOR_COUNT, env_count);
    let t0 = Instant::now();
    {
        let a0 = Arc::clone(&agent);
        let a1 = Arc::clone(&agent);
        let (r0, r1) = tokio::join!(
            tokio::spawn(async move { a0.run_env_eval(actor0_id, TIMED_STEPS).await }),
            tokio::spawn(async move { a1.run_env_eval(actor1_id, TIMED_STEPS).await }),
        );
        r0??;
        r1??;
    }
    let wall = t0.elapsed().as_secs_f64();

    // ── Post-run /proc snapshot ───────────────────────────────────────────────
    let after = read_proc_stats();

    // ── Derived metrics ───────────────────────────────────────────────────────
    let total_transitions = TIMED_STEPS * total_envs;
    let per_actor_transitions = TIMED_STEPS * env_count;
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
    println!("  RESULTS — LunarLander-v3 eval — RelayRL+EnvPool — 2 actors");
    println!("══════════════════════════════════════════════════════════════════");

    println!();
    println!("─── Setup ─────────────────────────────────────────────────────────");
    println!("  actors                 : {:>10}", ACTOR_COUNT);
    println!("  envs per actor         : {:>10}", env_count);
    println!("  total envs             : {:>10}", total_envs);

    println!();
    println!("─── Throughput (aggregate, both actors concurrent) ────────────────");
    println!("  loop steps (timed)     : {:>10}", TIMED_STEPS);
    println!("  transitions per actor  : {:>10}", per_actor_transitions);
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
             WARMUP_STEPS, warmup_wall, warmup_total as f64 / warmup_wall);
    println!("  timed   ({:>5} steps): {:>8.2} s  ({:.0} transitions/sec)",
             TIMED_STEPS, wall, transitions_sec);
    println!("══════════════════════════════════════════════════════════════════");

    // Unwrap Arc (unique after join scopes dropped) so we can call &mut self shutdown
    let mut agent = Arc::try_unwrap(agent)
        .expect("Arc should be unique after both spawned tasks complete");
    agent.shutdown().await?;
    Ok(())
}
