//! bench_lunar_set_env_scalar1 — set_env / run_env internal path, 1 scalar env, 100k steps. (beta.4)
//!
//! Direct apples-to-apples comparison with bench_lunar_direct_scalar1:
//!   - same env count (1), same step count (100k), same warmup (10k)
//!   - this path: framework owns the loop via run_env (perform_local_byte_inference inline)
//!   - other path: caller owns the loop via request_action + flag_last_action
//!
//! Beta.4 change: run_env() now takes an optional algorithm-training tuple.
//! Pass None with NoTrainKernel to run inference-only (no training).
//!
//! Build & run:
//!   cargo build --release -p bench-beta4 --bin bench_lunar_set_env_scalar1
//!   ORT_DYLIB_PATH=... perf stat -r 10 ./target/release/bench_lunar_set_env_scalar1

use std::collections::HashMap;
use std::time::Instant;

use burn_ndarray::NdArray;
use burn_tensor::{Float, Int, Tensor, TensorKind};

use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode,
    RelayRLActorEnv, RelayRLAgentActors,
};
use relayrl_framework::prelude::types::model::ModelModule;
use relayrl_framework::prelude::types::tensor::relayrl::{BackendMatcher, DeviceType};

use relayrl_algorithms::algorithms::onnx_builder::build_onnx_mlp_bytes;
use relayrl_algorithms::{
    DDPGKernelTrait, MultiagentDDPGKernelTrait, MultiagentKernelTrait, MultiagentPPOKernelTrait,
    MultiagentReinforceKernelTrait, MultiagentTD3KernelTrait, PPOKernelTrait, REINFORCEKernelTrait,
    StepKernelTrait, TD3KernelTrait, WeightProvider,
};

use lunarlander_rl::env::LunarLanderEnv;

// ─────────────────────────── NoTrainKernel ───────────────────────────────────
// Satisfies run_env(None) kernel bounds without instantiating any trainer.
// All training methods are unimplemented — they are never called when
// algorithm_cfg = None.

#[derive(Default)]
struct NoTrainKernel;

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    StepKernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn step<const IN_D: usize, const OUT_D: usize>(
        &self,
        _obs: Tensor<B, IN_D, InK>,
        _mask: Tensor<B, OUT_D, OutK>,
    ) -> Result<
        (relayrl_algorithms::templates::base_algorithm::StepAction<B>, HashMap<String, relayrl_types::prelude::tensor::relayrl::TensorData>),
        relayrl_types::prelude::tensor::relayrl::TensorError,
    > {
        unimplemented!("NoTrainKernel: run_env called with None algorithm_cfg; step() should not be invoked")
    }
    fn get_input_dim(&self) -> usize { 0 }
    fn get_output_dim(&self) -> usize { 0 }
}

impl WeightProvider for NoTrainKernel {
    fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> { None }
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    REINFORCEKernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn train_pi_step(
        &mut self, _obs: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _act: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _mask: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _adv: &[f32], _logp_old: &[relayrl_types::prelude::tensor::relayrl::TensorData],
    ) -> (f32, HashMap<String, f32>) { unimplemented!() }
    fn train_vf_step(
        &mut self, _obs: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _mask: &[relayrl_types::prelude::tensor::relayrl::TensorData], _ret: &[f32],
    ) -> f32 { unimplemented!() }
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    PPOKernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn new_for_actor(_obs_dim: usize, _act_dim: usize) -> Self { NoTrainKernel }
    fn ppo_pi_loss(
        &mut self, _obs: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _act: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _mask: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _adv: &[f32], _logp_old: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _clip_ratio: f32,
    ) -> (f32, HashMap<String, f32>) { unimplemented!() }
    fn ppo_vf_loss(
        &mut self, _obs: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _mask: &[relayrl_types::prelude::tensor::relayrl::TensorData], _ret: &[f32],
    ) -> f32 { unimplemented!() }
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    DDPGKernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn new_for_actor(_obs_dim: usize, _act_dim: usize) -> Self { NoTrainKernel }
    fn ddpg_train_step(
        &mut self, _obs: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _act: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _next_obs: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _rew: &[f32], _done: &[f32], _gamma: f32, _tau: f32,
    ) -> relayrl_algorithms::algorithms::DDPG::DDPGTrainMetrics { unimplemented!() }
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    TD3KernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn new_for_actor(_obs_dim: usize, _act_dim: usize) -> Self { NoTrainKernel }
    fn td3_train_step(
        &mut self, _obs: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _act: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _next_obs: &[relayrl_types::prelude::tensor::relayrl::TensorData],
        _rew: &[f32], _done: &[f32], _gamma: f32, _tau: f32,
        _policy_noise: f32, _noise_clip: f32, _policy_frequency: u32,
    ) -> relayrl_algorithms::algorithms::TD3::TD3TrainMetrics { unimplemented!() }
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    MultiagentKernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn register_agent(&mut self) {}
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    MultiagentReinforceKernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn train_epoch(
        &mut self,
        _batches: &[relayrl_algorithms::algorithms::REINFORCE::multiagent::kernel::AgentBatch],
    ) -> relayrl_algorithms::algorithms::REINFORCE::multiagent::kernel::MultiagentTrainMetrics {
        unimplemented!()
    }
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    MultiagentPPOKernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn train_epoch(
        &mut self,
        _batches: &[relayrl_algorithms::algorithms::PPO::multiagent::kernel::AgentBatch],
        _clip_ratio: f32, _target_kl: f32, _train_pi_iters: u64, _train_vf_iters: u64,
    ) -> relayrl_algorithms::algorithms::PPO::multiagent::kernel::MultiagentPPOTrainMetrics {
        unimplemented!()
    }
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    MultiagentDDPGKernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn train_epoch(
        &mut self,
        _batches: &[relayrl_algorithms::algorithms::DDPG::multiagent::kernel::AgentBatch],
        _gamma: f32, _tau: f32, _policy_frequency: usize,
    ) -> relayrl_algorithms::algorithms::DDPG::multiagent::kernel::MultiagentDDPGTrainMetrics {
        unimplemented!()
    }
}

impl<B: burn_tensor::backend::Backend + BackendMatcher<Backend = B>, InK: TensorKind<B>, OutK: TensorKind<B>>
    MultiagentTD3KernelTrait<B, InK, OutK> for NoTrainKernel
{
    fn train_epoch(
        &mut self,
        _batches: &[relayrl_algorithms::algorithms::TD3::multiagent::kernel::AgentBatch],
        _gamma: f32, _tau: f32, _policy_noise: f32, _noise_clip: f32, _policy_frequency: usize,
    ) -> relayrl_algorithms::algorithms::TD3::multiagent::kernel::MultiagentTD3TrainMetrics {
        unimplemented!()
    }
}

unsafe impl Send for NoTrainKernel {}

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM:      usize = 8;
const ACT_DIM:      usize = 4;
const MAX_STEPS:    usize = 500;
const ENV_COUNT:    u32   = 1;
const TARGET_STEPS: usize = 100_000;
const WARMUP_ITERS: usize = 10_000;

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

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = NdArray;

    let num_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.4-orig — set_env / run_env — ScalarLunarLander — {} env", ENV_COUNT);
    println!("  1 actor · {} loop iters · {} total transitions · {} logical cores",
             TARGET_STEPS, TARGET_STEPS * ENV_COUNT as usize, num_cores);
    println!("  path: framework-internal run_env (perform_local_byte_inference)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let initial_model = make_bootstrap_model::<B>()?;
    let config_path   = std::path::PathBuf::from("./config.json");
    // beta.4: AgentBuilder no longer has KindIn/KindOut type params
    let mut builder = AgentBuilder::<B, 2, 2>::builder()
        .actor_count(1)
        .default_device(DeviceType::Cpu)
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

    let env   = LunarLanderEnv::<B>::new(MAX_STEPS, Default::default());
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(env);

    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!("set_env OK — registered {} scalar LunarLander env with actor {}",
             ENV_COUNT, actor_id);

    // ── Warm-up ───────────────────────────────────────────────────────────────
    println!("Warming up ({WARMUP_ITERS} iters)…");
    // beta.4: run_env now takes an optional algorithm-training tuple; pass None for inference-only
    agent.run_env::<Float, Float, NoTrainKernel>(actor_id, WARMUP_ITERS, None).await?;
    println!("Warm-up done. Starting timed run ({TARGET_STEPS} loop iters)…\n");

    // ── Timed run ─────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    agent.run_env::<Float, Float, NoTrainKernel>(actor_id, TARGET_STEPS, None).await?;
    let wall = t0.elapsed().as_secs_f64();

    let iters_per_sec    = TARGET_STEPS as f64 / wall;
    let transitions_sec  = iters_per_sec * ENV_COUNT as f64;
    let total_transitions = TARGET_STEPS * ENV_COUNT as usize;
    let us_per_iter      = 1_000_000.0 / iters_per_sec;

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
    println!("  RelayRL beta.4-orig — set_env / run_env — FINAL RESULTS  ({ENV_COUNT} scalar env, {TARGET_STEPS} steps)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("─── Throughput ──────────────────────────────────────────────────────");
    println!("  env count                : {:>10}", ENV_COUNT);
    println!("  loop iterations          : {:>10}", TARGET_STEPS);
    println!("  total env transitions    : {:>10}", total_transitions);
    println!("  wall time                : {:>10.3} s", wall);
    println!("  iters/sec                : {:>10.0}", iters_per_sec);
    println!("  env transitions/sec      : {:>10.0}", transitions_sec);
    println!("  µs / iter                : {:>10.3}", us_per_iter);
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
