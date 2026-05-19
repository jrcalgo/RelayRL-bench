//! rlox-bench — PPO on GridWorld-10x10 using rlox-core + rlox-burn.
//!
//! Uses the published rlox crate ecosystem (rlox-core, rlox-nn, rlox-burn)
//! to run a 100k step IPPO benchmark on the same 10×10 GridWorld used in
//! the RelayRL e2e benchmarks.

use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;

use rlox_burn::actor_critic::BurnActorCritic;
use rlox_core::buffer::columnar::ExperienceTable;
use rlox_core::buffer::ExperienceRecord;
use rlox_core::env::spaces::{Action, ActionSpace, ObsSpace, Observation};
use rlox_core::env::{RLEnv, Transition};
use rlox_core::error::RloxError;
use rlox_core::training::gae::compute_gae;
use rlox_nn::{ActorCritic, PPOStepConfig, TensorData};

use std::collections::HashMap;

// ── Backend ───────────────────────────────────────────────────────────────────

type TB = Autodiff<NdArray>;

// ── Hyperparameters ───────────────────────────────────────────────────────────

const GRID:            usize = 10;
const OBS_DIM:         usize = GRID * GRID; // 100-dim one-hot
const ACT_DIM:         usize = 4;
const HIDDEN:          usize = 64;
const GAMMA:           f64   = 0.99;
const LAM:             f64   = 0.97;
const PI_ITERS:        usize = 5;
const MAX_EP_STEPS:    usize = 200;
const STEPS_PER_EPOCH: usize = 1_600;
const TARGET_STEPS:    usize = 100_800; // 63 epochs

// ── GridWorld environment ─────────────────────────────────────────────────────

struct GridWorld {
    pos:          (isize, isize),
    walls:        Vec<(isize, isize)>,
    steps:        usize,
    done:         bool,
    last_reward:  f64,
    action_space: ActionSpace,
    obs_space:    ObsSpace,
    rng:          u64, // xorshift64 state (for reset seed)
}

impl GridWorld {
    fn new() -> Self {
        let walls = vec![
            (2,1),(2,2),(2,3),(2,4),
            (3,4),(4,4),(5,4),(6,4),(7,4),
            (2,6),(2,7),(2,8),
        ];
        let low  = vec![0.0f32; OBS_DIM];
        let high = vec![1.0f32; OBS_DIM];
        GridWorld {
            pos:          (0, 0),
            walls,
            steps:        0,
            done:         false,
            last_reward:  0.0,
            action_space: ActionSpace::Discrete(ACT_DIM),
            obs_space:    ObsSpace::Box { low, high, shape: vec![OBS_DIM] },
            rng:          0x1234_5678_DEAD_BEEFu64,
        }
    }

    fn obs_vec(&self) -> Vec<f32> {
        let mut o = vec![0.0f32; OBS_DIM];
        o[self.pos.0 as usize * GRID + self.pos.1 as usize] = 1.0;
        o
    }

    fn xorshift(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.rng = x;
        x
    }
}

impl RLEnv for GridWorld {
    fn reset(&mut self, seed: Option<u64>) -> Result<Observation, RloxError> {
        if let Some(s) = seed { self.rng = s; }
        self.pos   = (0, 0);
        self.steps = 0;
        self.done  = false;
        self.last_reward = 0.0;
        Ok(Observation(self.obs_vec()))
    }

    fn step(&mut self, action: &Action) -> Result<Transition, RloxError> {
        if self.done {
            return Err(RloxError::EnvError("step() called on done env".into()));
        }
        let act_idx = match action {
            Action::Discrete(a) => *a as usize,
            _ => return Err(RloxError::InvalidAction("expected Discrete".into())),
        };

        self.steps += 1;
        let (r, c) = self.pos;
        let next = match act_idx {
            0 => (r - 1, c),
            1 => (r + 1, c),
            2 => (r, c - 1),
            _ => (r, c + 1),
        };

        let in_bounds = next.0 >= 0 && next.0 < GRID as isize
                     && next.1 >= 0 && next.1 < GRID as isize;
        let is_wall   = self.walls.contains(&next);

        let (reward, terminated) = if !in_bounds || is_wall {
            (-1.0, false)
        } else if next == (9, 9) {
            self.pos = next;
            (10.0, true)
        } else {
            self.pos = next;
            (-0.01, false)
        };

        let truncated = !terminated && self.steps >= MAX_EP_STEPS;
        self.done = terminated || truncated;
        self.last_reward = reward;

        Ok(Transition {
            obs: Observation(self.obs_vec()),
            reward,
            terminated,
            truncated,
            info: HashMap::new(),
        })
    }

    fn action_space(&self) -> &ActionSpace { &self.action_space }
    fn obs_space(&self)    -> &ObsSpace    { &self.obs_space    }
}

// ── RSS helper ────────────────────────────────────────────────────────────────

fn read_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok()))
        .unwrap_or(0)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!("rlox-bench — PPO on GridWorld-10x10 (rlox-core + rlox-burn)");
    println!("OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM}, HIDDEN={HIDDEN}");
    println!("STEPS_PER_EPOCH={STEPS_PER_EPOCH}, TARGET={TARGET_STEPS}");
    println!();

    let device = <TB as Backend>::Device::default();
    let mut agent = BurnActorCritic::<TB>::new(OBS_DIM, ACT_DIM, HIDDEN, 3e-4_f32, device, 42);

    let mut env = GridWorld::new();
    let mut current_obs = env.reset(Some(42)).unwrap().0;

    let t_start        = std::time::Instant::now();
    let mut total_steps: u64 = 0;
    let mut epoch       = 0usize;
    let mut rss_samples = Vec::<u64>::new();
    let mut ep_returns: Vec<f32> = Vec::new();
    let mut ep_return   = 0.0f32;
    let mut ep_len      = 0usize;

    let ppo_config = PPOStepConfig::default();

    while total_steps < TARGET_STEPS as u64 {
        // ── Collect rollout ──────────────────────────────────────────────────
        let mut rollout  = ExperienceTable::new(OBS_DIM, 1);
        let mut old_logps: Vec<f32> = Vec::with_capacity(STEPS_PER_EPOCH);

        for _ in 0..STEPS_PER_EPOCH {
            // Act
            let obs_td = TensorData::new(current_obs.clone(), vec![1, OBS_DIM]);
            let out    = agent.act(&obs_td).expect("act failed");
            let action_idx = out.actions.data[0] as u32;
            let log_prob   = out.log_probs.data[0];

            // Step env
            let transition = match env.step(&Action::Discrete(action_idx)) {
                Ok(t)  => t,
                Err(_) => {
                    // Env is done (shouldn't happen if we reset properly), recover
                    current_obs = env.reset(None).unwrap().0;
                    continue;
                }
            };

            let next_obs = transition.obs.0.clone();
            ep_return   += transition.reward as f32;
            ep_len      += 1;

            // Store transition
            rollout.push(ExperienceRecord {
                obs:        current_obs.clone(),
                action:     vec![action_idx as f32],
                reward:     transition.reward as f32,
                terminated: transition.terminated,
                truncated:  transition.truncated,
            }).expect("rollout push failed");
            old_logps.push(log_prob);

            let episode_ended = transition.terminated || transition.truncated;
            if episode_ended || ep_len >= MAX_EP_STEPS {
                ep_returns.push(ep_return);
                ep_return  = 0.0;
                ep_len     = 0;
                current_obs = env.reset(None).unwrap().0;
            } else {
                current_obs = next_obs;
            }
        }

        total_steps += STEPS_PER_EPOCH as u64;
        let n = rollout.len();

        // ── Get values for rollout ───────────────────────────────────────────
        let obs_data = TensorData::new(
            rollout.observations_raw().to_vec(),
            vec![n, OBS_DIM],
        );
        let values_td = agent.value(&obs_data).expect("value failed");

        // Bootstrap value for last state
        let last_val = if env.done {
            0.0f64
        } else {
            let last_obs_td = TensorData::new(current_obs.clone(), vec![1, OBS_DIM]);
            let v = agent.value(&last_obs_td).expect("value bootstrap failed");
            v.data[0] as f64
        };

        // ── GAE ──────────────────────────────────────────────────────────────
        let rewards_f64: Vec<f64> = rollout.rewards_raw().iter().map(|&r| r as f64).collect();
        let values_f64:  Vec<f64> = values_td.data.iter().map(|&v| v as f64).collect();
        let dones_f64:   Vec<f64> = rollout.terminated().iter().zip(rollout.truncated())
            .map(|(&t, &tr)| if t || tr { 1.0 } else { 0.0 })
            .collect();

        let (adv_f64, ret_f64) = compute_gae(&rewards_f64, &values_f64, &dones_f64, last_val, GAMMA, LAM);

        // Normalise advantages
        let mean = adv_f64.iter().sum::<f64>() / n as f64;
        let std  = (adv_f64.iter().map(|&x| (x-mean).powi(2)).sum::<f64>() / n as f64 + 1e-8).sqrt();
        let adv_norm: Vec<f32>  = adv_f64.iter().map(|&x| ((x - mean) / std) as f32).collect();
        let returns_f32: Vec<f32> = ret_f64.iter().map(|&r| r as f32).collect();

        let actions_data    = TensorData::new(rollout.actions_raw().to_vec(), vec![n]);
        let old_logp_data   = TensorData::new(old_logps, vec![n]);
        let old_values_data = values_td; // keep original pre-update values
        let adv_data        = TensorData::new(adv_norm, vec![n]);
        let ret_data        = TensorData::new(returns_f32, vec![n]);

        // ── PPO update (PI_ITERS passes) ─────────────────────────────────────
        for _ in 0..PI_ITERS {
            agent.ppo_step(
                &obs_data,
                &actions_data,
                &old_logp_data,
                &adv_data,
                &ret_data,
                &old_values_data,
                &ppo_config,
            ).expect("ppo_step failed");
        }

        epoch += 1;
        rss_samples.push(read_rss_kb());

        if epoch % 10 == 0 || total_steps >= TARGET_STEPS as u64 {
            let elapsed  = t_start.elapsed().as_secs_f64();
            let sps      = total_steps as f64 / elapsed;
            let mean_ret = if ep_returns.is_empty() { 0.0f32 }
                           else { ep_returns.iter().sum::<f32>() / ep_returns.len() as f32 };
            let rss_kb   = rss_samples.last().copied().unwrap_or(0);
            println!(
                "epoch {:4} | steps {:7} | sps {:6.0} | mean_ep_ret {:7.2} | RSS {} KB",
                epoch, total_steps, sps, mean_ret, rss_kb
            );
            ep_returns.clear();
        }
    }

    let elapsed  = t_start.elapsed().as_secs_f64();
    let sps      = total_steps as f64 / elapsed;
    let rss_mean = rss_samples.iter().sum::<u64>() as f64 / rss_samples.len().max(1) as f64;
    let rss_max  = rss_samples.iter().copied().max().unwrap_or(0);

    println!();
    println!("══════════════════════════════════════════════════");
    println!("rlox-bench PPO GridWorld-10x10 — FINAL RESULTS");
    println!("══════════════════════════════════════════════════");
    println!("Total steps  : {total_steps}");
    println!("Elapsed      : {elapsed:.2}s");
    println!("Steps/sec    : {sps:.0}");
    println!("RSS mean     : {:.1} KB  ({:.1} MB)", rss_mean, rss_mean / 1024.0);
    println!("RSS peak     : {} KB  ({:.1} MB)", rss_max, rss_max as f64 / 1024.0);
    println!("══════════════════════════════════════════════════");
}
