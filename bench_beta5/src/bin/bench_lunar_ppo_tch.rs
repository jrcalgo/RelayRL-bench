//! bench_lunar_ppo_tch — RelayRL PPO on LunarLander, 512 envs, LibTorch backend,
//! Sample-Factory-matched hyperparameters and environment.
//!
//! Trains against EnvPool's `LunarLander-v2` with `max_episode_steps=500`
//! (one envpool instance of 512 envs, GIL released during step) via
//! `bench_beta5::py_env::make_sf_matched_envpool_lunar_lander_vec` — the same
//! environment and conditions used by `scripts/sf_lunar_bench.py`'s single
//! envpool instance, so the RelayRL vs Sample Factory comparison is not
//! confounded by differences between the `lunarlander-rl` Rust port and
//! envpool's real Box2D physics.
//!
//! Hyperparameters mirror the Sample Factory LunarLander-v2 config used for the
//! comparison benchmark: pi_lr=vf_lr=2.5e-4, vf_coef=1.0, train_pi/vf_iters=4
//! (matches SF num_epochs=4), target_kl effectively disabled (matches SF having
//! no KL early-stop), mini_batch=46080 (= 512 envs x 90-step rollout, matches SF
//! batch_size), ent_coef=0.01, normalize_returns=true, traj_per_epoch=512,
//! total_steps=75_000 -> 38.4M env frames (same total budget as the 64-env config).
//!
//! Build & run:
//!   LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
//!     cargo build --release -p bench-beta5 --bin bench_lunar_ppo_tch
//!   LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib \
//!     ./target/release/bench_lunar_ppo_tch

use std::path::PathBuf;
use std::time::Instant;

use burn_tch::LibTorch;
use burn_tensor::Float;

use relayrl_algorithms::TrainerArgs;
use relayrl_algorithms::algorithms::PPO::kernel::{DiscretePPOPolicyHead, PPOPolicyHead};
use relayrl_algorithms::algorithms::PPO::{IPPOParams, PPONetworkArgs, PPOTrainerSpec};
use relayrl_algorithms::algorithms::{ActivationKind, GenericMlp, WeightProvider, acquire_model_module};
use relayrl_framework::prelude::network::{
    ActorInferenceMode, ActorTrainingDataMode, AgentBuilder, ModelMode, RelayRLActorEnv,
    RelayRLAgentActors, ReplayBufferSize,
};
use relayrl_framework::prelude::types::tensor::relayrl::DeviceType;
use relayrl_types::data::tensor::{DType, TchDType};

use bench_beta5::py_env::make_sf_matched_envpool_lunar_lander_vec;

// ─────────────────────────── Constants ──────────────────────────────────────

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const MAX_STEPS: usize = 500;
const ENV_COUNT: u32 = 512;
const SEED: u64 = 1;

const GAMMA: f32 = 0.999;
const LAM: f32 = 0.98;
const CLIP_RATIO: f32 = 0.2;
const PI_LR: f64 = 2.5e-4; // matches SF lr=2.5e-4
const VF_LR: f64 = 2.5e-4;
const VF_COEF: f32 = 1.0; // matches SF vf_coef default
const TRAIN_PI_ITERS: u64 = 4; // matches SF num_epochs=4
const TRAIN_VF_ITERS: u64 = 4;
const TARGET_KL: f32 = 1.0; // effectively disabled (SF has no KL early-stop)
const MINI_BATCH_SIZE: usize = 46_080; // matches SF batch_size = 512 envs x 90-step rollout
const ENT_COEF: f32 = 0.01;
const NORMALIZE_RETURNS: bool = true; // per-batch normalization (no persistent RunningMeanStd)
// matches SF's --policy_initialization=orthogonal --policy_init_gain=1.0 (default), applied
// uniformly to every pi/vf layer with zero bias.
const POLICY_INIT_GAIN: f64 = 1.0;

// 512 trajs/epoch x 512 envs -> ~90 loop iters/epoch
const TRAJ_PER_EPOCH: u64 = 512;
// Step-count epoch trigger: 512 envs x 90-step rollout = 46080, matches SF exactly
const MIN_STEPS_PER_EPOCH: u64 = MINI_BATCH_SIZE as u64; // 46080
// 2x drain-epoch cap: 2 x ~512 eps = 1024 eps max in buffer
const MAX_BUFFERED_EPISODES: u64 = 1024;
// 75_000 loop iterations x 512 envs ~= 38.4M total env frames (same budget as the
// 64-env config's 600_000 x 64)
const TOTAL_STEPS: usize = 75_000;
const BUFFER_SIZE: ReplayBufferSize = 4_000_000;

// ─────────────────────────── Main ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi/libonnxruntime.so.1.26.0",
    );

    type B = LibTorch;

    let num_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let total_env_frames = TOTAL_STEPS * ENV_COUNT as usize;
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RelayRL beta.5 — PPO — gymnasium LunarLander-v2 discrete — LibTorch — {ENV_COUNT} envs");
    println!("  obs={OBS_DIM}  act={ACT_DIM}  MLP=[128,128]  max_steps={MAX_STEPS}");
    println!("  loop_steps={TOTAL_STEPS}  env-frames={total_env_frames}");
    println!("  gamma={GAMMA}  lam={LAM}  clip={CLIP_RATIO}  pi_lr={PI_LR}  vf_lr={VF_LR}  vf_coef={VF_COEF}");
    println!(
        "  pi_iters={TRAIN_PI_ITERS}  vf_iters={TRAIN_VF_ITERS}  target_kl={TARGET_KL}  ent_coef={ENT_COEF}  traj/epoch={TRAJ_PER_EPOCH}  mb={MINI_BATCH_SIZE}  normalize_returns={NORMALIZE_RETURNS}  policy_init_gain={POLICY_INIT_GAIN}  adam_eps=1e-6"
    );
    println!("  {num_cores} logical cores");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // ── PPO networks: [128, 128] MLPs for obs=8, act=4 ──────────────────────
    let burn_device = <B as burn_tensor::backend::Backend>::Device::default();
    // Seed the burn backend so network initialization is reproducible across
    // runs (previously unseeded, contributing to large run-to-run MeanReturn
    // variance even with a fixed env seed).
    <B as burn_tensor::backend::Backend>::seed(&burn_device, SEED);
    let obs_dtype = DType::Tch(TchDType::F32);
    let act_dtype = DType::Tch(TchDType::F32);

    let pi_mlp = GenericMlp::<B, Float, Float>::new_orthogonal(
        OBS_DIM,
        obs_dtype.clone(),
        &[128, 128],
        ACT_DIM,
        act_dtype.clone(),
        ActivationKind::ReLU(burn_nn::activation::Relu::new()),
        POLICY_INIT_GAIN,
        &burn_device,
    );
    // Seed the actor with a TorchScript export of the freshly-initialized policy
    // so `new_actor` doesn't try (and fail) to load a model from local_model_path.
    let initial_model = acquire_model_module::<B>(
        "ppo_pi",
        pi_mlp.get_layer_specs(),
        obs_dtype.clone(),
        act_dtype.clone(),
        vec![1, OBS_DIM],
        vec![1, ACT_DIM],
        Some(DeviceType::Cpu),
    );
    let pi_head = PPOPolicyHead::Discrete(DiscretePPOPolicyHead::new(pi_mlp)?);
    let vf_mlp = GenericMlp::<B, Float, Float>::new_orthogonal(
        OBS_DIM,
        obs_dtype.clone(),
        &[128, 128],
        1,
        DType::Tch(TchDType::F32),
        ActivationKind::ReLU(burn_nn::activation::Relu::new()),
        POLICY_INIT_GAIN,
        &burn_device,
    );

    let trainer_args = TrainerArgs {
        env_dir: PathBuf::from("./envs/lunar_ppo_tch"),
        save_model_path: PathBuf::from("./models/lunar_ppo_tch"),
        obs_dim: OBS_DIM,
        obs_dtype,
        act_dim: ACT_DIM,
        act_dtype,
        buffer_size: BUFFER_SIZE,
        device: DeviceType::Cpu,
    };
    let hyperparams = IPPOParams {
        discrete: true,
        gamma: GAMMA,
        lam: LAM,
        clip_ratio: CLIP_RATIO,
        pi_lr: PI_LR as f32,
        vf_lr: VF_LR as f32,
        vf_coef: VF_COEF,
        train_pi_iters: TRAIN_PI_ITERS,
        train_vf_iters: TRAIN_VF_ITERS,
        target_kl: TARGET_KL,
        traj_per_epoch: TRAJ_PER_EPOCH,
        ent_coef: ENT_COEF,
        max_episode_steps: Some(MAX_STEPS),
        minibatch: Some(MINI_BATCH_SIZE),
        normalize_returns: NORMALIZE_RETURNS,
        min_steps_per_epoch: Some(MIN_STEPS_PER_EPOCH),
        max_buffered_episodes: Some(MAX_BUFFERED_EPISODES),
        rollout_len: Some(MINI_BATCH_SIZE / ENV_COUNT as usize),
        ..Default::default()
    };
    let trainer_spec = PPOTrainerSpec::ppo(
        trainer_args,
        Some(hyperparams),
        PPONetworkArgs { pi_head, vf_mlp },
    );

    // ── Agent setup ─────────────────────────────────────────────────────────
    let config_path = PathBuf::from("./config.json");
    let mut builder = AgentBuilder::<B>::builder()
        .actor_inference_mode(ActorInferenceMode::Local(ModelMode::Independent))
        .actor_training_data_mode(ActorTrainingDataMode::Disabled)
        .router_scale(1);
    if config_path.exists() {
        builder = builder.config_path(config_path);
    }

    let (mut agent, params) = builder.build().await?;
    agent.start(params).await?;
    agent
        .new_actor::<OBS_DIM, ACT_DIM>(DeviceType::Cpu, MAX_STEPS, initial_model)
        .await?;
    let actor_ids = agent.get_actor_ids()?;
    let actor_id = actor_ids[0];

    // ── Environment ──────────────────────────────────────────────────────────
    // EnvPool LunarLander-v2, max_episode_steps=500, one envpool instance of
    // ENV_COUNT envs — matches scripts/sf_lunar_bench.py's single envpool instance.
    let py_env = make_sf_matched_envpool_lunar_lander_vec(ENV_COUNT as usize, OBS_DIM, ACT_DIM)
        .map_err(|e| format!("envpool env creation failed: {e}"))?;
    let boxed: Box<dyn relayrl_env_trait::Environment> = Box::new(py_env);
    agent.set_env(actor_id, boxed, ENV_COUNT).await?;
    println!(
        "set_env OK — registered {} EnvPool LunarLander-v2 env(s) with actor {}\n",
        ENV_COUNT, actor_id
    );

    // ── PPO training ─────────────────────────────────────────────────────────
    println!("Starting PPO training ({TOTAL_STEPS} steps)...\n");
    println!(
        "{:>12}  {:>8}  {:>14}  {:>10}",
        "epoch", "episodes", "mean_ret(100)", "last_ep"
    );
    println!("{}", "─".repeat(52));

    let t0 = Instant::now();
    agent
        .run_env_with_ppo::<Float, Float, _>(actor_id, TOTAL_STEPS, MAX_STEPS, trainer_spec)
        .await?;
    let wall = t0.elapsed().as_secs_f64();

    // ── Final stats ──────────────────────────────────────────────────────────
    let loop_steps_per_sec = TOTAL_STEPS as f64 / wall;
    let env_frames_per_sec = loop_steps_per_sec * ENV_COUNT as f64;

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  PPO training complete (LibTorch backend)");
    println!("  wall time         : {:.2}s", wall);
    println!("  loop steps/sec    : {:.0}  (each step = {} env transitions)", loop_steps_per_sec, ENV_COUNT);
    println!("  env-frames/sec    : {:.0}", env_frames_per_sec);
    println!("═══════════════════════════════════════════════════════════════════");

    agent.shutdown().await?;
    Ok(())
}
