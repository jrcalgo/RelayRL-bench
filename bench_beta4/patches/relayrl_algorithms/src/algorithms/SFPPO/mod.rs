// Sample Factory–aligned PPO hyperparameters (APPO defaults).
//
// The loss formula is identical to IPPOAlgorithm — only the defaults differ:
//   clip=0.1, epochs=1, rollout=32, normalize_returns=true, ent_coef=0.001
// These match SF's `--sample_env_agents` workflow with LunarLander-style tasks.

use super::PPO::IPPOParams;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SFPPOParams {
    pub discrete: bool,
    pub gamma: f32,
    pub lam: f32,
    pub clip_ratio: f32,
    pub pi_lr: f32,
    pub vf_lr: f32,
    pub train_pi_iters: u64,
    pub train_vf_iters: u64,
    pub target_kl: f32,
    pub traj_per_epoch: u64,
    #[serde(default)]
    pub ent_coef: f32,
    #[serde(default)]
    pub max_episode_steps: Option<usize>,
    #[serde(default)]
    pub mini_batch_size: Option<usize>,
    #[serde(default = "default_vf_coef")]
    pub vf_coef: f32,
    #[serde(default)]
    pub min_steps_per_epoch: Option<u64>,
    #[serde(default)]
    pub max_buffered_episodes: Option<u64>,
    #[serde(default = "default_max_version_lag")]
    pub max_version_lag: i64,
    #[serde(default = "default_normalize_returns")]
    pub normalize_returns: bool,
    #[serde(default)]
    pub rollout_len: Option<usize>,
}

fn default_vf_coef() -> f32 { 0.5 }
fn default_max_version_lag() -> i64 { 1 }
fn default_normalize_returns() -> bool { true }

impl Default for SFPPOParams {
    fn default() -> Self {
        Self {
            discrete: true,
            gamma: 0.99,
            lam: 0.95,
            clip_ratio: 0.1,
            pi_lr: 1e-4,
            vf_lr: 1e-4,
            train_pi_iters: 1,
            train_vf_iters: 1,
            target_kl: 0.1,
            traj_per_epoch: 64,
            ent_coef: 0.001,
            max_episode_steps: Some(500),
            mini_batch_size: Some(2048),
            vf_coef: 0.5,
            min_steps_per_epoch: Some(2048),
            max_buffered_episodes: Some(128),
            max_version_lag: 1,
            normalize_returns: true,
            rollout_len: Some(32),
        }
    }
}

impl From<SFPPOParams> for IPPOParams {
    fn from(p: SFPPOParams) -> Self {
        IPPOParams {
            discrete: p.discrete,
            gamma: p.gamma,
            lam: p.lam,
            clip_ratio: p.clip_ratio,
            pi_lr: p.pi_lr,
            vf_lr: p.vf_lr,
            train_pi_iters: p.train_pi_iters,
            train_vf_iters: p.train_vf_iters,
            target_kl: p.target_kl,
            traj_per_epoch: p.traj_per_epoch,
            ent_coef: p.ent_coef,
            max_episode_steps: p.max_episode_steps,
            mini_batch_size: p.mini_batch_size,
            vf_coef: p.vf_coef,
            min_steps_per_epoch: p.min_steps_per_epoch,
            max_buffered_episodes: p.max_buffered_episodes,
            max_version_lag: p.max_version_lag,
            normalize_returns: p.normalize_returns,
            rollout_len: p.rollout_len,
        }
    }
}
