#![allow(non_upper_case_globals)]

use crate::algorithms::REINFORCE::{
    ActivationKind, BaselineValueNetwork, ContinuousPolicyNetwork, DiscretePolicyNetwork,
};
use crate::templates::base_algorithm::{StepAction, StepKernelTrait};

use std::collections::HashMap;
use std::sync::Arc;

use burn_tensor::activation::log_softmax;
use burn_tensor::backend::Backend;
use burn_tensor::{BasicOps, Float, Int, Tensor, TensorData as BurnTensorData, TensorKind};

use relayrl_types::data::tensor::{
    BackendMatcher, ConversionBurnTensor, DType, SupportedTensorBackend, TensorData, TensorError,
};

fn backend_f32_dtype<B: Backend + BackendMatcher>() -> Result<DType, TensorError> {
    match B::get_supported_backend() {
        #[cfg(feature = "ndarray-backend")]
        SupportedTensorBackend::NdArray => Ok(DType::NdArray(
            relayrl_types::data::tensor::NdArrayDType::F32,
        )),
        #[cfg(feature = "tch-backend")]
        SupportedTensorBackend::Tch => Ok(DType::Tch(relayrl_types::data::tensor::TchDType::F32)),
        _ => Err(TensorError::BackendError(
            "Unsupported backend for f32 TensorData conversion".to_string(),
        )),
    }
}

fn float_tensor_to_data<B: Backend + BackendMatcher, const D: usize>(
    tensor: Tensor<B, D, Float>,
) -> Result<TensorData, TensorError> {
    TensorData::try_from(ConversionBurnTensor {
        inner: Arc::new(tensor),
        conversion_dtype: backend_f32_dtype::<B>()?,
    })
}

#[allow(clippy::large_enum_variant)]
enum PolicyHead<B: Backend, InK: TensorKind<B>, OutK: TensorKind<B>>
where
    OutK: BasicOps<B>,
{
    Discrete(DiscretePolicyNetwork<B, InK, OutK>),
    Continuous(ContinuousPolicyNetwork<B, InK, OutK>),
}

pub trait PPOKernelTrait<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>:
    StepKernelTrait<B, InK, OutK>
{
    fn new_for_actor(obs_dim: usize, act_dim: usize) -> Self;

    fn ppo_pi_loss(
        &mut self,
        obs: &[TensorData],
        act: &[TensorData],
        mask: &[TensorData],
        adv: &[f32],
        logp_old: &[TensorData],
        clip_ratio: f32,
    ) -> (f32, HashMap<String, f32>);

    fn ppo_vf_loss(&mut self, obs: &[TensorData], mask: &[TensorData], ret: &[f32]) -> f32;

    fn value_forward_only(&self, obs: &[TensorData], mask: &[TensorData]) -> Vec<f32>;

    fn ppo_pi_loss_flat(
        &mut self,
        obs_flat: &[f32],
        obs_dim: usize,
        act_flat: &[i64],
        adv: &[f32],
        logp_old: &[f32],
        clip_ratio: f32,
        ent_coef: f32,
        compute_stats: bool,
    ) -> (f32, HashMap<String, f32>);

    fn ppo_vf_loss_flat(
        &mut self,
        obs_flat: &[f32],
        obs_dim: usize,
        ret: &[f32],
    ) -> f32;

    /// Combined pi+vf loss in one backward pass with vf_coef scaling.
    /// Returns (pi_loss, vf_loss, stats{kl, entropy, clipfrac}).
    fn ppo_combined_loss_flat(
        &mut self,
        obs_flat: &[f32],
        obs_dim: usize,
        act_flat: &[i64],
        adv: &[f32],
        logp_old: &[f32],
        ret: &[f32],
        clip_ratio: f32,
        ent_coef: f32,
        vf_coef: f32,
        compute_stats: bool,
    ) -> (f32, f32, HashMap<String, f32>);

    fn value_forward_only_flat(&self, obs_flat: &[f32], obs_dim: usize) -> Vec<f32>;

    fn get_pi_logprobs_flat(
        &self,
        obs_flat: &[f32],
        obs_dim: usize,
        act_flat: &[i64],
    ) -> Vec<f32>;
}

// Training module: compiled when either ndarray or tch backend is available.
#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
mod training {
    use super::*;

    extern crate burn_core as burn;

    use burn_autodiff::Autodiff;
    use burn_core::module::Module;
    use burn_core::module::Initializer;
    use burn_nn::{Linear, LinearConfig, Relu};
    use burn_optim::adaptor::OptimizerAdaptor;
    use burn_optim::grad_clipping::GradientClipping;
    use burn_optim::{Adam, AdamConfig, GradientsParams, Optimizer};

    // Use LibTorch autodiff when the tch-backend feature is active; fall back to NdArray.
    #[cfg(feature = "tch-backend")]
    use burn_tch::LibTorch;
    #[cfg(feature = "tch-backend")]
    pub type TB = Autodiff<LibTorch>;

    #[cfg(all(feature = "ndarray-backend", not(feature = "tch-backend")))]
    use burn_ndarray::NdArray;
    #[cfg(all(feature = "ndarray-backend", not(feature = "tch-backend")))]
    pub type TB = Autodiff<NdArray>;

    /// Combined actor-critic MLP: separate pi and vf layer stacks, shared obs encoder not used
    /// (each head has its own independent layers). Trained with one shared Adam optimizer.
    #[derive(Module, Debug)]
    pub struct ActorCriticMlp<B: burn_tensor::backend::Backend> {
        pub pi_layers: Vec<Linear<B>>,
        pub vf_layers: Vec<Linear<B>>,
        pub relu: Relu,
        pub obs_dim: usize,
        pub act_dim: usize,
    }

    impl<B: burn_tensor::backend::Backend> ActorCriticMlp<B> {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            device: &B::Device,
        ) -> Self {
            let mut pi_dims = vec![obs_dim];
            pi_dims.extend_from_slice(hidden_sizes);
            pi_dims.push(act_dim);
            let pi_n = pi_dims.len() - 1;
            let pi_layers = pi_dims
                .windows(2)
                .enumerate()
                .map(|(i, w)| {
                    let gain = if i < pi_n - 1 { 2.0f64.sqrt() } else { 0.01 };
                    // Bias must stay 1-D so we can't use Orthogonal directly on LinearConfig
                    // (it would also apply to the bias and panic). Initialize bias with Zeros,
                    // then overwrite the weight parameter with an orthogonal init.
                    let mut layer = LinearConfig::new(w[0], w[1])
                        .with_initializer(Initializer::Zeros)
                        .init(device);
                    layer.weight = Initializer::Orthogonal { gain }
                        .init_with([w[0], w[1]], Some(w[0]), Some(w[1]), device);
                    layer
                })
                .collect();

            let mut vf_dims = vec![obs_dim];
            vf_dims.extend_from_slice(hidden_sizes);
            vf_dims.push(1);
            let vf_n = vf_dims.len() - 1;
            let vf_layers = vf_dims
                .windows(2)
                .enumerate()
                .map(|(i, w)| {
                    let gain = if i < vf_n - 1 { 2.0f64.sqrt() } else { 1.0 };
                    let mut layer = LinearConfig::new(w[0], w[1])
                        .with_initializer(Initializer::Zeros)
                        .init(device);
                    layer.weight = Initializer::Orthogonal { gain }
                        .init_with([w[0], w[1]], Some(w[0]), Some(w[1]), device);
                    layer
                })
                .collect();

            Self { pi_layers, vf_layers, relu: Relu::new(), obs_dim, act_dim }
        }

        pub fn pi_forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
            let mut x = input;
            for (i, layer) in self.pi_layers.iter().enumerate() {
                x = layer.forward(x);
                if i < self.pi_layers.len() - 1 {
                    x = self.relu.forward(x);
                }
            }
            x
        }

        pub fn vf_forward(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
            let mut x = input;
            for (i, layer) in self.vf_layers.iter().enumerate() {
                x = layer.forward(x);
                if i < self.vf_layers.len() - 1 {
                    x = self.relu.forward(x);
                }
            }
            x
        }
    }

    pub struct ActorCriticTrainer {
        pub network: Option<ActorCriticMlp<TB>>,
        pub optimizer: OptimizerAdaptor<Adam, ActorCriticMlp<TB>, TB>,
        pub lr: f64,
        pub vf_coef: f32,
        pub lr_schedule_steps: Option<u64>,
        pub grad_step_count: u64,
    }

    impl ActorCriticTrainer {
        pub fn new(
            obs_dim: usize,
            hidden_sizes: &[usize],
            act_dim: usize,
            lr: f64,
            vf_coef: f32,
        ) -> Self {
            let device = <TB as burn_tensor::backend::Backend>::Device::default();
            let network = ActorCriticMlp::new(obs_dim, hidden_sizes, act_dim, &device);
            let optimizer = AdamConfig::new()
                .init::<TB, ActorCriticMlp<TB>>()
                .with_grad_clipping(GradientClipping::Norm(4.0));
            Self {
                network: Some(network),
                optimizer,
                lr,
                vf_coef,
                lr_schedule_steps: None,
                grad_step_count: 0,
            }
        }

        fn effective_lr(&self) -> f64 {
            match self.lr_schedule_steps {
                Some(total) if total > 0 => {
                    let frac = 1.0 - (self.grad_step_count as f64 / total as f64).min(1.0);
                    self.lr * frac.max(0.0)
                }
                _ => self.lr,
            }
        }

        /// Combined pi+vf forward+backward in one pass. Returns (pi_loss, vf_loss, stats).
        pub fn train_step_flat(
            &mut self,
            obs_flat: &[f32],
            obs_dim: usize,
            act_flat: &[i64],
            adv: &[f32],
            logp_old: &[f32],
            ret: &[f32],
            clip_ratio: f32,
            ent_coef: f32,
            compute_stats: bool,
        ) -> (f32, f32, HashMap<String, f32>) {
            let n = (obs_flat.len() / obs_dim.max(1))
                .min(act_flat.len())
                .min(adv.len())
                .min(logp_old.len())
                .min(ret.len());
            if n == 0 {
                return (0.0, 0.0, zero_pi_info().1);
            }
            let net = match self.network.take() {
                Some(net) => net,
                None => return (0.0, 0.0, zero_pi_info().1),
            };
            let device = <TB as burn_tensor::backend::Backend>::Device::default();

            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat[..n * obs_dim].to_vec(), [n, obs_dim]),
                &device,
            );

            // ── Policy head ───────────────────────────────────────────────
            let logits = net.pi_forward(obs.clone());
            let log_probs_full = log_softmax(logits, 1);
            let act = Tensor::<TB, 2, Int>::from_data(
                BurnTensorData::new(act_flat[..n].to_vec(), [n, 1]),
                &device,
            );
            let logp = log_probs_full.clone().gather(1, act).reshape([n]);
            let adv_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(adv[..n].to_vec(), [n]),
                &device,
            );
            let logp_old_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(logp_old[..n].to_vec(), [n]),
                &device,
            );
            let ratio = (logp.clone() - logp_old_tensor).exp();
            let clipped_ratio = ratio.clone().clamp(1.0 - clip_ratio, 1.0 + clip_ratio);
            let clip_obj = (ratio.clone() * adv_tensor.clone())
                .min_pair(clipped_ratio * adv_tensor)
                .mean();
            let entropy_t = (log_probs_full.clone().exp() * log_probs_full)
                .neg()
                .sum_dim(1)
                .reshape([n])
                .mean();
            let pi_loss_t = -(clip_obj + ent_coef * entropy_t.clone());

            // ── Value head ────────────────────────────────────────────────
            let v_pred = net.vf_forward(obs).reshape([n]);
            let ret_tensor = Tensor::<TB, 1, Float>::from_data(
                BurnTensorData::new(ret[..n].to_vec(), [n]),
                &device,
            );
            let vf_loss_t = (v_pred - ret_tensor).powf_scalar(2.0).mean();

            // ── Combined loss → single backward pass ──────────────────────
            let vf_coef_t = self.vf_coef;
            let total_loss = pi_loss_t.clone() + vf_loss_t.clone() * vf_coef_t;

            let pi_loss_val = scalar_from_tensor(&pi_loss_t);
            let vf_loss_val = scalar_from_tensor(&vf_loss_t);

            let grads = total_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &net);
            let lr = self.effective_lr();
            let net = self.optimizer.step(lr, net, grads_params);
            self.network = Some(net);
            self.grad_step_count += 1;

            if !compute_stats {
                return (pi_loss_val, vf_loss_val, HashMap::new());
            }

            let entropy_val = entropy_t.into_scalar();
            let approx_kl = ((ratio.clone() - 1.0) - ratio.clone().log()).mean().into_scalar();
            let ratio_values = ratio.into_data().to_vec::<f32>().unwrap_or_else(|_| vec![1.0; n]);
            let clipfrac = ratio_values
                .iter()
                .filter(|r| (**r - 1.0).abs() > clip_ratio)
                .count() as f32
                / n as f32;

            let mut info = HashMap::new();
            info.insert("kl".to_string(), approx_kl);
            info.insert("entropy".to_string(), entropy_val);
            info.insert("clipfrac".to_string(), clipfrac);
            (pi_loss_val, vf_loss_val, info)
        }

        /// Value-only forward — used for deferred GAE inference.
        pub fn value_forward_flat(&self, obs_flat: &[f32], obs_dim: usize) -> Vec<f32> {
            let n = if obs_dim > 0 { obs_flat.len() / obs_dim } else { 0 };
            if n == 0 {
                return Vec::new();
            }
            let net = match self.network.as_ref() {
                Some(net) => net,
                None => return vec![0.0; n],
            };
            let device = <TB as burn_tensor::backend::Backend>::Device::default();
            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat[..n * obs_dim].to_vec(), [n, obs_dim]),
                &device,
            );
            let v = net.vf_forward(obs);
            v.into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n])
        }

        /// Log-probs for (obs, act) — for logp_old recompute.
        pub fn logprobs_flat(
            &self,
            obs_flat: &[f32],
            obs_dim: usize,
            act_flat: &[i64],
        ) -> Vec<f32> {
            let n = (obs_flat.len() / obs_dim.max(1)).min(act_flat.len());
            if n == 0 {
                return Vec::new();
            }
            let net = match self.network.as_ref() {
                Some(net) => net,
                None => return vec![0.0; n],
            };
            let device = <TB as burn_tensor::backend::Backend>::Device::default();
            let obs = Tensor::<TB, 2, Float>::from_data(
                BurnTensorData::new(obs_flat[..n * obs_dim].to_vec(), [n, obs_dim]),
                &device,
            );
            let logits = net.pi_forward(obs);
            let log_probs = log_softmax(logits, 1);
            let act = Tensor::<TB, 2, Int>::from_data(
                BurnTensorData::new(act_flat[..n].to_vec(), [n, 1]),
                &device,
            );
            let logp = log_probs.gather(1, act).reshape([n]);
            logp.into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n])
        }
    }

    pub fn zero_pi_info() -> (f32, HashMap<String, f32>) {
        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        info.insert("clipfrac".to_string(), 0.0);
        (0.0, info)
    }

    fn scalar_from_tensor(t: &Tensor<TB, 1, Float>) -> f32 {
        t.clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0])[0]
    }

    fn obs_flat(obs_tensors: &[TensorData]) -> Vec<f32> {
        obs_tensors
            .iter()
            .flat_map(|tensor| bytemuck::cast_slice::<u8, f32>(&tensor.data).to_vec())
            .collect()
    }

    fn action_indices(act_tensors: &[TensorData]) -> Vec<i64> {
        act_tensors
            .iter()
            .map(|tensor| {
                if tensor.data.len() >= 8 {
                    bytemuck::cast_slice::<u8, i64>(&tensor.data[..8])
                        .first()
                        .copied()
                        .unwrap_or(0)
                } else if tensor.data.len() >= 4 {
                    bytemuck::cast_slice::<u8, f32>(&tensor.data[..4])
                        .first()
                        .copied()
                        .unwrap_or(0.0) as i64
                } else {
                    0
                }
            })
            .collect()
    }

    fn scalar_tensor_values(tensors: &[TensorData]) -> Vec<f32> {
        tensors
            .iter()
            .map(|tensor| {
                bytemuck::cast_slice::<u8, f32>(&tensor.data)
                    .first()
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

pub struct PPOPolicyWithBaseline<
    B: Backend + BackendMatcher,
    InK: TensorKind<B>,
    OutK: TensorKind<B>,
> where
    OutK: BasicOps<B>,
    InK: BasicOps<B>,
{
    policy: PolicyHead<B, InK, OutK>,
    pub baseline: BaselineValueNetwork<B, InK, OutK>,
    input_dim: usize,
    output_dim: usize,
    // Persistent return normalization (SF-aligned): running mean and variance of returns
    returns_mean: f32,
    returns_variance: f32,
    returns_count: u64,
    #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
    actor_critic_trainer: Option<training::ActorCriticTrainer>,
}

pub type DefaultPPOKernel<B, InK, OutK> = PPOPolicyWithBaseline<B, InK, OutK>;
pub type PPOKernel<B, InK, OutK> = PPOPolicyWithBaseline<B, InK, OutK>;
pub type PlaceholderPPOKernel<B, InK, OutK> = PPOPolicyWithBaseline<B, InK, OutK>;

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    pub fn new(
        obs_dim: usize,
        act_dim: usize,
        discrete: bool,
        hidden_sizes: &[usize],
        activation: ActivationKind,
        pi_lr: f64,
        vf_coef: f32,
        device: &B::Device,
    ) -> Self {
        Self::new_with_schedule(
            obs_dim, act_dim, discrete, hidden_sizes, activation, pi_lr, vf_coef, None, device,
        )
    }

    pub fn new_with_schedule(
        obs_dim: usize,
        act_dim: usize,
        discrete: bool,
        hidden_sizes: &[usize],
        activation: ActivationKind,
        pi_lr: f64,
        vf_coef: f32,
        lr_schedule_steps: Option<u64>,
        device: &B::Device,
    ) -> Self {
        let policy = if discrete {
            PolicyHead::Discrete(DiscretePolicyNetwork::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                device,
            ))
        } else {
            PolicyHead::Continuous(ContinuousPolicyNetwork::new(
                obs_dim,
                hidden_sizes,
                act_dim,
                device,
            ))
        };
        let baseline = BaselineValueNetwork::new(obs_dim, hidden_sizes, activation, device);

        #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
        let actor_critic_trainer = if discrete {
            let mut t = training::ActorCriticTrainer::new(obs_dim, hidden_sizes, act_dim, pi_lr, vf_coef);
            t.lr_schedule_steps = lr_schedule_steps;
            Some(t)
        } else {
            None
        };

        Self {
            policy,
            baseline,
            input_dim: obs_dim,
            output_dim: act_dim,
            returns_mean: 0.0,
            returns_variance: 1.0,
            returns_count: 0,
            #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
            actor_critic_trainer,
        }
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>> Default
    for PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: TensorKind<B> + BasicOps<B>,
    OutK: TensorKind<B> + BasicOps<B>,
{
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(
            1,
            1,
            true,
            &[64, 64],
            ActivationKind::ReLU,
            3e-4,
            0.5,
            &device,
        )
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B> + burn_tensor::Numeric<B>,
{
    pub fn value_forward_only<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, InK>,
        mask: Tensor<B, OUT_D, OutK>,
    ) -> Vec<f32> {
        let v = self.baseline.forward(obs, mask);
        v.into_data().to_vec::<f32>().unwrap_or_default()
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    StepKernelTrait<B, InK, OutK> for PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn step<const IN_D: usize, const OUT_D: usize>(
        &self,
        obs: Tensor<B, IN_D, InK>,
        mask: Tensor<B, OUT_D, OutK>,
    ) -> Result<(StepAction<B>, HashMap<String, TensorData>), TensorError> {
        if obs.dims()[IN_D - 1] != self.input_dim || mask.dims()[OUT_D - 1] != self.output_dim {
            return Err(TensorError::ShapeError(format!(
                "Expected obs/mask trailing dims ({}, {}), got ({}, {})",
                self.input_dim,
                self.output_dim,
                obs.dims()[IN_D - 1],
                mask.dims()[OUT_D - 1]
            )));
        }

        let mut data = HashMap::new();
        match &self.policy {
            PolicyHead::Discrete(policy) => {
                let (probs, logits) = policy.distribution(obs.clone(), mask.clone());
                let probs_rank2 = probs
                    .clone()
                    .reshape([probs.dims()[0], probs.dims()[OUT_D - 1]]);
                let act = policy.sample_for_action(probs_rank2);
                let device = logits.device();
                let act_out_d: Tensor<B, OUT_D, Int> =
                    Tensor::from_data(act.clone().into_data(), &device);
                let log_pmf = log_softmax(logits, 1);
                let logp_a: Tensor<B, OUT_D, Float> = log_pmf.gather(1, act_out_d);
                let v = self.baseline.forward(obs.clone(), mask.clone());
                let v_float: Tensor<B, OUT_D, Float> =
                    Tensor::from_data(v.into_data().convert::<f32>(), &device);
                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("val".to_string(), float_tensor_to_data(v_float)?);
                Ok((StepAction::Discrete(act), data))
            }
            PolicyHead::Continuous(policy) => {
                let (mean, std) = policy.distribution(obs.clone(), mask.clone());
                let act = policy.sample_for_action(mean.clone(), std.clone());
                let logp_a = policy.log_prob_from_distribution(mean, std, act.clone());
                let v = self.baseline.forward(obs.clone(), mask.clone());
                let act_for_step = act.clone().reshape([act.dims()[0], act.dims()[OUT_D - 1]]);
                let device = v.device();
                let v_float: Tensor<B, OUT_D, Float> =
                    Tensor::from_data(v.into_data().convert::<f32>(), &device);
                data.insert("logp_a".to_string(), float_tensor_to_data(logp_a)?);
                data.insert("val".to_string(), float_tensor_to_data(v_float)?);
                Ok((StepAction::Continuous(act_for_step), data))
            }
        }
    }

    fn get_input_dim(&self) -> usize {
        self.input_dim
    }

    fn get_output_dim(&self) -> usize {
        self.output_dim
    }
}

impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    PPOKernelTrait<B, InK, OutK> for PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn new_for_actor(obs_dim: usize, act_dim: usize) -> Self {
        let device = B::Device::default();
        Self::new(obs_dim, act_dim, true, &[64, 64], ActivationKind::ReLU, 3e-4, 0.5, &device)
    }

    fn ppo_pi_loss(
        &mut self,
        _obs: &[TensorData],
        _act: &[TensorData],
        _mask: &[TensorData],
        _adv: &[f32],
        _logp_old: &[TensorData],
        _clip_ratio: f32,
    ) -> (f32, HashMap<String, f32>) {
        // Superseded by ppo_combined_loss_flat; kept for trait compat.
        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        info.insert("clipfrac".to_string(), 0.0);
        (0.0, info)
    }

    fn ppo_vf_loss(&mut self, _obs: &[TensorData], _mask: &[TensorData], _ret: &[f32]) -> f32 {
        0.0
    }

    fn ppo_pi_loss_flat(
        &mut self,
        _obs_flat: &[f32],
        _obs_dim: usize,
        _act_flat: &[i64],
        _adv: &[f32],
        _logp_old: &[f32],
        _clip_ratio: f32,
        _ent_coef: f32,
        _compute_stats: bool,
    ) -> (f32, HashMap<String, f32>) {
        // Superseded by ppo_combined_loss_flat.
        let mut info = HashMap::new();
        info.insert("kl".to_string(), 0.0);
        info.insert("entropy".to_string(), 0.0);
        info.insert("clipfrac".to_string(), 0.0);
        (0.0, info)
    }

    fn ppo_vf_loss_flat(&mut self, _obs_flat: &[f32], _obs_dim: usize, _ret: &[f32]) -> f32 {
        0.0
    }

    fn ppo_combined_loss_flat(
        &mut self,
        obs_flat: &[f32],
        obs_dim: usize,
        act_flat: &[i64],
        adv: &[f32],
        logp_old: &[f32],
        ret: &[f32],
        clip_ratio: f32,
        ent_coef: f32,
        _vf_coef: f32, // stored on trainer; parameter kept for API symmetry
        compute_stats: bool,
    ) -> (f32, f32, HashMap<String, f32>) {
        #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
        if let Some(trainer) = &mut self.actor_critic_trainer {
            return trainer.train_step_flat(
                obs_flat, obs_dim, act_flat, adv, logp_old, ret,
                clip_ratio, ent_coef, compute_stats,
            );
        }
        (0.0, 0.0, HashMap::new())
    }

    fn get_pi_logprobs_flat(&self, obs_flat: &[f32], obs_dim: usize, act_flat: &[i64]) -> Vec<f32> {
        #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
        if let Some(trainer) = &self.actor_critic_trainer {
            return trainer.logprobs_flat(obs_flat, obs_dim, act_flat);
        }
        vec![0.0; act_flat.len()]
    }

    fn value_forward_only(&self, obs: &[TensorData], _mask: &[TensorData]) -> Vec<f32> {
        if obs.is_empty() {
            return Vec::new();
        }
        let n = obs.len();
        let obs_dim = obs[0].shape[0];
        let device = <B as burn_tensor::backend::Backend>::Device::default();
        let flat: Vec<f32> = obs.iter()
            .flat_map(|td| td.data.chunks(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])))
            .collect();
        let obs_t = burn_tensor::Tensor::<B, 2, InK>::from_data(
            burn_tensor::TensorData::new(flat, [n, obs_dim]),
            &device,
        );
        let mask_t = burn_tensor::Tensor::<B, 2, OutK>::ones([n, self.output_dim], &device);
        let v = self.baseline.forward(obs_t, mask_t);
        v.into_data().to_vec::<f32>().unwrap_or_default()
    }

    fn value_forward_only_flat(&self, obs_flat: &[f32], obs_dim: usize) -> Vec<f32> {
        // Use the trained VF network when available; self.baseline is never updated.
        #[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
        if let Some(trainer) = &self.actor_critic_trainer {
            return trainer.value_forward_flat(obs_flat, obs_dim);
        }
        let n = if obs_dim > 0 { obs_flat.len() / obs_dim } else { 0 };
        if n == 0 {
            return Vec::new();
        }
        let device = <B as burn_tensor::backend::Backend>::Device::default();
        let obs_t = burn_tensor::Tensor::<B, 2, InK>::from_data(
            burn_tensor::TensorData::new(obs_flat.to_vec(), [n, obs_dim]),
            &device,
        );
        let mask_t = burn_tensor::Tensor::<B, 2, OutK>::ones([n, self.output_dim], &device);
        let v = self.baseline.forward(obs_t, mask_t);
        v.into_data().to_vec::<f32>().unwrap_or_else(|_| vec![0.0; n])
    }
}

#[cfg(any(feature = "ndarray-backend", feature = "tch-backend"))]
impl<B: Backend + BackendMatcher, InK: TensorKind<B>, OutK: TensorKind<B>>
    crate::templates::base_algorithm::WeightProvider for PPOPolicyWithBaseline<B, InK, OutK>
where
    InK: BasicOps<B>,
    OutK: BasicOps<B>,
{
    fn get_pi_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        let trainer = self.actor_critic_trainer.as_ref()?;
        let network = trainer.network.as_ref()?;
        let mut specs = Vec::new();
        for layer in &network.pi_layers {
            let w = layer.weight.val();
            let dims = w.dims();
            let weights: Vec<f32> = w.into_data().to_vec::<f32>().unwrap_or_default();
            let biases: Vec<f32> = if let Some(bias_param) = &layer.bias {
                bias_param.val().into_data().to_vec::<f32>().unwrap_or_default()
            } else {
                vec![0.0; dims[1]]
            };
            specs.push((dims[0], dims[1], weights, biases));
        }
        Some(specs)
    }

    fn get_vf_layer_specs(&self) -> Option<Vec<(usize, usize, Vec<f32>, Vec<f32>)>> {
        let trainer = self.actor_critic_trainer.as_ref()?;
        let network = trainer.network.as_ref()?;
        let mut specs = Vec::new();
        for layer in &network.vf_layers {
            let w = layer.weight.val();
            let dims = w.dims();
            let weights: Vec<f32> = w.into_data().to_vec::<f32>().unwrap_or_default();
            let biases: Vec<f32> = if let Some(bias_param) = &layer.bias {
                bias_param.val().into_data().to_vec::<f32>().unwrap_or_default()
            } else {
                vec![0.0; dims[1]]
            };
            specs.push((dims[0], dims[1], weights, biases));
        }
        Some(specs)
    }
}
