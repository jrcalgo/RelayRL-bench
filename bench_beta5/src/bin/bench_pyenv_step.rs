//! bench_pyenv_step — isolated throughput micro-benchmark for `PyVectorEnv::step_bytes`.
//!
//! Measures the raw Rust↔Python marshaling cost of stepping the gymnasium
//! LunarLander-v2 SyncVectorEnv bridge used by `bench_lunar_ppo_tch`, with no
//! inference/training overhead. Used to evaluate `py_env.rs` marshaling
//! optimizations in isolation.
//!
//! Build & run:
//!   LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
//!     cargo build --release -p bench-beta5 --bin bench_pyenv_step
//!   ./target/release/bench_pyenv_step

use std::time::Instant;

use bench_beta5::py_env::make_sf_matched_lunar_lander_vec;
use relayrl_env_trait::VectorEnvironment;

const OBS_DIM: usize = 8;
const ACT_DIM: usize = 4;
const ENV_COUNT: usize = 64;
const WARMUP_VEC_STEPS: usize = 200;
const TIMED_VEC_STEPS: usize = 5_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = make_sf_matched_lunar_lander_vec(ENV_COUNT, OBS_DIM, ACT_DIM)
        .map_err(|e| format!("gymnasium env creation failed: {e}"))?;

    let ids = env.init_num_envs(ENV_COUNT)?;
    env.reset(&ids)?;

    // Fixed pseudo-random discrete action stream (0..ACT_DIM), reused every step.
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut next_byte = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % ACT_DIM as u64) as u8
    };
    let actions: Vec<u8> = (0..ENV_COUNT).map(|_| next_byte()).collect();

    println!("══════════════════════════════════════════════════════════════════");
    println!("  bench_pyenv_step — PyVectorEnv::step_bytes throughput");
    println!("  env: gymnasium LunarLander-v2, sync, {ENV_COUNT} envs");
    println!("  warmup: {WARMUP_VEC_STEPS} vec-steps   timed: {TIMED_VEC_STEPS} vec-steps");
    println!("══════════════════════════════════════════════════════════════════\n");

    for _ in 0..WARMUP_VEC_STEPS {
        env.step_bytes(&actions).expect("step_bytes returned None");
    }

    let t0 = Instant::now();
    for _ in 0..TIMED_VEC_STEPS {
        env.step_bytes(&actions).expect("step_bytes returned None");
    }
    let wall = t0.elapsed().as_secs_f64();

    let vec_steps_per_sec = TIMED_VEC_STEPS as f64 / wall;
    let env_frames_per_sec = vec_steps_per_sec * ENV_COUNT as f64;

    println!("  wall time         : {:.3}s", wall);
    println!("  vec-steps/sec     : {:.1}", vec_steps_per_sec);
    println!("  env-frames/sec    : {:.0}", env_frames_per_sec);
    println!("  us/vec-step       : {:.2}", 1_000_000.0 / vec_steps_per_sec);

    Ok(())
}
