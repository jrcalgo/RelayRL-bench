//! rlox — standalone synchronous PPO on a 10×10 GridWorld using Burn.
//!
//! No RelayRL framework, no async, no channels, no ONNX.
//! Designed as a clean baseline to compare against the full RelayRL stack.

#![allow(dead_code)]

extern crate burn_core as burn;

use burn::module::{AutodiffModule, Module};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Float, Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use burn_nn::{Linear, LinearConfig, Relu};
use burn_optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor};
use burn_tensor::backend::Backend;

// ── Backend types ────────────────────────────────────────────────────────────

type TB  = Autodiff<NdArray>;   // training backend (with autodiff)
type IB  = NdArray;             // inference backend (no autodiff)
type Dev = <TB as Backend>::Device;

// ── Hyperparameters ──────────────────────────────────────────────────────────

const GRID:         usize = 10;
const OBS_DIM:      usize = GRID * GRID;   // 100-dim one-hot
const ACT_DIM:      usize = 4;             // up/down/left/right
const HIDDEN:       usize = 64;
const GAMMA:        f32   = 0.99;
const LAM:          f32   = 0.97;
const CLIP:         f32   = 0.2;
const PI_LR:        f64   = 3e-4;
const VF_LR:        f64   = 3e-4;
const PI_ITERS:     usize = 5;
const VF_ITERS:     usize = 5;
const MAX_EP_STEPS:  usize = 200;
const STEPS_PER_EPOCH: usize = 1_600;  // 8 trajectories × 200 steps
const TARGET_STEPS: usize = 100_800;   // ~63 epochs

// ── GridWorld ────────────────────────────────────────────────────────────────

/// Row-major (row, col) positions in a GRID×GRID maze.
struct GridWorld {
    pos:      (isize, isize),
    walls:    Vec<(isize, isize)>,
    goal:     (isize, isize),
    steps:    usize,
    done:     bool,
    last_rwd: f32,
    // xorshift64 RNG state
    rng:      u64,
}

impl GridWorld {
    fn new(seed: u64) -> Self {
        let walls = vec![
            (2,1),(2,2),(2,3),(2,4),
            (3,4),(4,4),(5,4),(6,4),(7,4),
            (2,6),(2,7),(2,8),
        ];
        let mut gw = GridWorld {
            pos: (0, 0),
            walls,
            goal: (9, 9),
            steps: 0,
            done: false,
            last_rwd: 0.0,
            rng: if seed == 0 { 12345678901234567 } else { seed },
        };
        gw.reset();
        gw
    }

    fn xorshift(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    fn reset(&mut self) {
        self.pos = (0, 0);
        self.steps = 0;
        self.done = false;
        self.last_rwd = 0.0;
    }

    fn obs(&self) -> [f32; OBS_DIM] {
        let mut o = [0.0f32; OBS_DIM];
        let idx = self.pos.0 as usize * GRID + self.pos.1 as usize;
        o[idx] = 1.0;
        o
    }

    /// actions: 0=up, 1=down, 2=left, 3=right
    fn step(&mut self, action: usize) -> (f32, bool) {
        if self.done { return (0.0, true); }
        self.steps += 1;
        let (r, c) = self.pos;
        let next = match action {
            0 => (r - 1, c),
            1 => (r + 1, c),
            2 => (r, c - 1),
            _ => (r, c + 1),
        };
        let in_bounds = next.0 >= 0 && next.0 < GRID as isize
                     && next.1 >= 0 && next.1 < GRID as isize;
        let is_wall   = self.walls.contains(&next);

        let (reward, done) = if !in_bounds || is_wall {
            (-1.0, false)
        } else if next == self.goal {
            self.pos = next;
            (10.0, true)
        } else {
            self.pos = next;
            (-0.01, false)
        };

        let done = done || self.steps >= MAX_EP_STEPS;
        self.done = done;
        self.last_rwd = reward;
        (reward, done)
    }
}

// ── MLP (shared by policy and value function) ────────────────────────────────

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> Mlp<B> {
    fn new(in_dim: usize, hidden: usize, out_dim: usize, dev: &B::Device) -> Self {
        Mlp {
            l1:   LinearConfig::new(in_dim, hidden).init(dev),
            l2:   LinearConfig::new(hidden, hidden).init(dev),
            l3:   LinearConfig::new(hidden, out_dim).init(dev),
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let x = self.relu.forward(self.l1.forward(x));
        let x = self.relu.forward(self.l2.forward(x));
        self.l3.forward(x)
    }
}

// ── Actor (policy) ───────────────────────────────────────────────────────────

struct Actor {
    net:  Option<Mlp<TB>>,
    opt:  OptimizerAdaptor<Adam, Mlp<TB>, TB>,
    dev:  Dev,
}

impl Actor {
    fn new() -> Self {
        let dev  = Dev::default();
        let net  = Mlp::<TB>::new(OBS_DIM, HIDDEN, ACT_DIM, &dev);
        let opt  = AdamConfig::new().init::<TB, Mlp<TB>>();
        Actor { net: Some(net), opt, dev }
    }

    /// Inference: returns (action, logp) using the non-autodiff path.
    fn act(&self, obs: &[f32; OBS_DIM], rng: &mut u64) -> (usize, f32) {
        let net_tb = self.net.as_ref().unwrap();
        // Clone into NdArray backend for forward-pass without autodiff overhead.
        let net_ib: Mlp<IB> = net_tb.valid();
        let dev_ib = <IB as Backend>::Device::default();

        let obs_t = Tensor::<IB, 2, Float>::from_data(
            TensorData::new(obs.to_vec(), [1, OBS_DIM]),
            &dev_ib,
        );
        let logits = net_ib.forward(obs_t);
        let log_probs: Vec<f32> = log_softmax(logits, 1)
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![-1.386; ACT_DIM]);

        // Convert log-probs → probs, then sample categorically.
        let probs: Vec<f32> = log_probs.iter().map(|lp| lp.exp()).collect();
        let sum: f32 = probs.iter().sum();
        let mut xv = xorshift64(rng) as f32 / u64::MAX as f32;
        xv *= sum;
        let mut action = ACT_DIM - 1;
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if xv <= cum { action = i; break; }
        }
        (action, log_probs[action])
    }

    /// PPO policy update.
    fn update(
        &mut self,
        obs_flat: &[f32],       // [n, OBS_DIM]
        actions:  &[i64],       // [n]
        logp_old: &[f32],       // [n]
        adv:      &[f32],       // [n]
    ) {
        let n = actions.len();
        if n == 0 { return; }

        for _ in 0..PI_ITERS {
            let net = match self.net.take() { Some(x) => x, None => return };

            let obs_t = Tensor::<TB, 2, Float>::from_data(
                TensorData::new(obs_flat.to_vec(), [n, OBS_DIM]),
                &self.dev,
            );
            let logits = net.forward(obs_t);
            let log_probs = log_softmax(logits, 1);

            let act_t = Tensor::<TB, 2, Int>::from_data(
                TensorData::new(actions.to_vec(), [n, 1]),
                &self.dev,
            );
            let logp = log_probs.gather(1, act_t).reshape([n]);

            let logp_old_t = Tensor::<TB, 1, Float>::from_data(
                TensorData::new(logp_old.to_vec(), [n]),
                &self.dev,
            );
            let adv_t = Tensor::<TB, 1, Float>::from_data(
                TensorData::new(adv.to_vec(), [n]),
                &self.dev,
            );

            let ratio   = (logp.clone() - logp_old_t).exp();
            let clipped = ratio.clone().clamp(1.0 - CLIP, 1.0 + CLIP);
            let loss    = (ratio * adv_t.clone()).min_pair(clipped * adv_t).mean().neg();

            let grads  = loss.backward();
            let gp     = GradientsParams::from_grads(grads, &net);
            let net    = self.opt.step(PI_LR, net, gp);
            self.net   = Some(net);
        }
    }
}

// ── Critic (value function) ──────────────────────────────────────────────────

struct Critic {
    net: Option<Mlp<TB>>,
    opt: OptimizerAdaptor<Adam, Mlp<TB>, TB>,
    dev: Dev,
}

impl Critic {
    fn new() -> Self {
        let dev = Dev::default();
        let net = Mlp::<TB>::new(OBS_DIM, HIDDEN, 1, &dev);
        let opt = AdamConfig::new().init::<TB, Mlp<TB>>();
        Critic { net: Some(net), opt, dev }
    }

    /// Inference value for a single observation.
    fn value(&self, obs: &[f32; OBS_DIM]) -> f32 {
        let net_tb = self.net.as_ref().unwrap();
        let net_ib: Mlp<IB> = net_tb.valid();
        let dev_ib = <IB as Backend>::Device::default();

        let obs_t = Tensor::<IB, 2, Float>::from_data(
            TensorData::new(obs.to_vec(), [1, OBS_DIM]),
            &dev_ib,
        );
        net_ib.forward(obs_t)
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0])[0]
    }

    fn update(&mut self, obs_flat: &[f32], returns: &[f32]) {
        let n = returns.len();
        if n == 0 { return; }

        for _ in 0..VF_ITERS {
            let net = match self.net.take() { Some(x) => x, None => return };

            let obs_t = Tensor::<TB, 2, Float>::from_data(
                TensorData::new(obs_flat.to_vec(), [n, OBS_DIM]),
                &self.dev,
            );
            let v_pred = net.forward(obs_t).reshape([n]);
            let ret_t  = Tensor::<TB, 1, Float>::from_data(
                TensorData::new(returns.to_vec(), [n]),
                &self.dev,
            );
            let loss = (v_pred - ret_t).powf_scalar(2.0).mean();

            let grads = loss.backward();
            let gp    = GradientsParams::from_grads(grads, &net);
            let net   = self.opt.step(VF_LR, net, gp);
            self.net  = Some(net);
        }
    }
}

// ── GAE / rollout buffer ─────────────────────────────────────────────────────

fn compute_gae(rewards: &[f32], values: &[f32], last_val: f32, dones: &[bool]) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    let mut adv    = vec![0.0f32; n];
    let mut ret    = vec![0.0f32; n];
    let mut gae    = 0.0f32;
    let mut v_next = last_val;

    for t in (0..n).rev() {
        let mask   = if dones[t] { 0.0 } else { 1.0 };
        let delta  = rewards[t] + GAMMA * v_next * mask - values[t];
        gae        = delta + GAMMA * LAM * mask * gae;
        adv[t]     = gae;
        v_next     = values[t];
    }
    for t in 0..n {
        ret[t] = adv[t] + values[t];
    }
    (adv, ret)
}

// ── xorshift64 RNG ───────────────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ── RSS helper ───────────────────────────────────────────────────────────────

fn read_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

// ── Main training loop ───────────────────────────────────────────────────────

fn main() {
    println!("rlox — standalone synchronous PPO on GridWorld-10x10");
    println!("OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM}, HIDDEN={HIDDEN}");
    println!("STEPS_PER_EPOCH={STEPS_PER_EPOCH}, TARGET_STEPS={TARGET_STEPS}");
    println!();

    let mut actor  = Actor::new();
    let mut critic = Critic::new();
    let mut env    = GridWorld::new(0x1234_5678_DEAD_BEEFu64);
    let mut rng    = 0xABCD_1234_5678_9ABCu64;

    let t_start        = std::time::Instant::now();
    let mut total_steps: u64 = 0;
    let mut epoch       = 0usize;
    let mut rss_samples = Vec::<u64>::new();

    // Pre-allocate rollout buffer.
    let mut buf_obs:  Vec<[f32; OBS_DIM]> = Vec::with_capacity(STEPS_PER_EPOCH);
    let mut buf_act:  Vec<i64>             = Vec::with_capacity(STEPS_PER_EPOCH);
    let mut buf_logp: Vec<f32>             = Vec::with_capacity(STEPS_PER_EPOCH);
    let mut buf_rwd:  Vec<f32>             = Vec::with_capacity(STEPS_PER_EPOCH);
    let mut buf_val:  Vec<f32>             = Vec::with_capacity(STEPS_PER_EPOCH);
    let mut buf_done: Vec<bool>            = Vec::with_capacity(STEPS_PER_EPOCH);

    let mut ep_returns: Vec<f32> = Vec::new();
    let mut ep_return   = 0.0f32;
    let mut ep_len      = 0usize;

    while total_steps < TARGET_STEPS as u64 {
        // ── Collect rollout ──────────────────────────────────────────────────
        buf_obs.clear();  buf_act.clear();  buf_logp.clear();
        buf_rwd.clear();  buf_val.clear();  buf_done.clear();

        for _ in 0..STEPS_PER_EPOCH {
            let obs  = env.obs();
            let val  = critic.value(&obs);
            let (act, logp) = actor.act(&obs, &mut rng);

            let (reward, done) = env.step(act);
            ep_return += reward;
            ep_len    += 1;

            buf_obs.push(obs);
            buf_act.push(act as i64);
            buf_logp.push(logp);
            buf_rwd.push(reward);
            buf_val.push(val);
            buf_done.push(done);

            if done || ep_len >= MAX_EP_STEPS {
                ep_returns.push(ep_return);
                ep_return = 0.0;
                ep_len    = 0;
                env.reset();
            }
        }

        // Bootstrap value at end of rollout (0 if terminal).
        let last_val = if env.done { 0.0 } else { critic.value(&env.obs()) };

        total_steps += STEPS_PER_EPOCH as u64;

        // ── Compute advantages & returns ─────────────────────────────────────
        let (adv, ret) = compute_gae(&buf_rwd, &buf_val, last_val, &buf_done);

        // Normalise advantages.
        let mean_adv = adv.iter().sum::<f32>() / adv.len() as f32;
        let var_adv  = adv.iter().map(|&x| (x - mean_adv).powi(2)).sum::<f32>() / adv.len() as f32;
        let std_adv  = (var_adv + 1e-8).sqrt();
        let adv_norm: Vec<f32> = adv.iter().map(|&x| (x - mean_adv) / std_adv).collect();

        // Flatten obs into a single Vec<f32>.
        let obs_flat: Vec<f32> = buf_obs.iter().flat_map(|o| o.iter().copied()).collect();

        // ── Update networks ──────────────────────────────────────────────────
        actor.update(&obs_flat, &buf_act, &buf_logp, &adv_norm);
        critic.update(&obs_flat, &ret);

        epoch += 1;
        rss_samples.push(read_rss_kb());

        if epoch % 10 == 0 || total_steps >= TARGET_STEPS as u64 {
            let elapsed    = t_start.elapsed().as_secs_f64();
            let sps        = total_steps as f64 / elapsed;
            let mean_ret   = if ep_returns.is_empty() { 0.0 }
                             else { ep_returns.iter().sum::<f32>() / ep_returns.len() as f32 };
            let rss_kb     = rss_samples.last().copied().unwrap_or(0);
            println!(
                "epoch {:4} | steps {:7} | sps {:6.0} | mean_ep_ret {:7.2} | RSS {} KB",
                epoch, total_steps, sps, mean_ret, rss_kb
            );
            ep_returns.clear();
        }
    }

    // ── Final summary ────────────────────────────────────────────────────────
    let elapsed    = t_start.elapsed().as_secs_f64();
    let sps        = total_steps as f64 / elapsed;
    let rss_mean   = rss_samples.iter().sum::<u64>() as f64 / rss_samples.len().max(1) as f64;
    let rss_max    = rss_samples.iter().copied().max().unwrap_or(0);

    println!();
    println!("══════════════════════════════════════════════");
    println!("rlox PPO GridWorld Benchmark — FINAL RESULTS");
    println!("══════════════════════════════════════════════");
    println!("Total steps      : {total_steps}");
    println!("Elapsed          : {elapsed:.2}s");
    println!("Steps/sec        : {sps:.0}");
    println!("RSS mean         : {:.1} KB  ({:.1} MB)", rss_mean, rss_mean / 1024.0);
    println!("RSS peak         : {} KB  ({:.1} MB)", rss_max, rss_max as f64 / 1024.0);
    println!("══════════════════════════════════════════════");
}
