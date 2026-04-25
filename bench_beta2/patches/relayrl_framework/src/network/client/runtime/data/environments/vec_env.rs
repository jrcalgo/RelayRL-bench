use crate::network::ENVIRONMENT_CONTEXT_PREFIX;
use active_uuid_registry::{
    ContextString, UuidPoolError,
    interface::{add_id, remove_id, reserve_id},
};
use burn_tensor::{BasicOps, Bool, Float, Int, Tensor, TensorKind, backend::Backend};
use relayrl_env_trait::*;
use relayrl_types::data::tensor::{AnyBurnTensor, BackendMatcher, DType, DeviceType};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VecEnvError {
    #[error("Invalid environment count: {0}")]
    InvalidEnvironmentCount(String),
    #[error("Unknown environment: {0}")]
    UnknownEnv(EnvironmentUuid),
    #[error(transparent)]
    UuidPoolError(#[from] UuidPoolError),
    #[error(transparent)]
    EnvironmentError(#[from] EnvironmentError),
    #[error("Tensor error: {0}")]
    TensorError(String),
}

#[derive(Debug, Clone)]
pub(crate) struct EnvResetRecord<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize> {
    pub env_id: EnvironmentUuid,
    pub observation: AnyBurnTensor<B, D_IN>,
    #[allow(dead_code)]
    pub info: Option<EnvInfo>,
}

#[derive(Debug, Clone)]
pub(crate) struct EnvStepRecord<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize> {
    pub env_id: EnvironmentUuid,
    pub observation: AnyBurnTensor<B, D_IN>,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    #[allow(dead_code)]
    pub info: Option<EnvInfo>,
}

fn any_tensor_data<B: Backend + BackendMatcher<Backend = B>, const D: usize>(
    any: &AnyBurnTensor<B, D>,
) -> burn_tensor::TensorData {
    match any {
        AnyBurnTensor::Float(w) => w.tensor.to_data(),
        AnyBurnTensor::Int(w) => w.tensor.to_data(),
        AnyBurnTensor::Bool(w) => w.tensor.to_data(),
    }
}

pub(crate) trait IntoAnyTensorKind<B: Backend + BackendMatcher<Backend = B>, const D: usize>:
    TensorKind<B>
{
    fn into_any(tensor: Tensor<B, D, Self>, dtype: DType) -> AnyBurnTensor<B, D>
    where
        Self: Sized;
}

impl<B: Backend + BackendMatcher<Backend = B>, const D: usize> IntoAnyTensorKind<B, D> for Float {
    fn into_any(tensor: Tensor<B, D, Self>, dtype: DType) -> AnyBurnTensor<B, D> {
        AnyBurnTensor::Float(relayrl_types::data::tensor::FloatBurnTensor {
            tensor: Arc::new(tensor),
            dtype,
        })
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D: usize> IntoAnyTensorKind<B, D> for Int {
    fn into_any(tensor: Tensor<B, D, Self>, dtype: DType) -> AnyBurnTensor<B, D> {
        AnyBurnTensor::Int(relayrl_types::data::tensor::IntBurnTensor {
            tensor: Arc::new(tensor),
            dtype,
        })
    }
}

impl<B: Backend + BackendMatcher<Backend = B>, const D: usize> IntoAnyTensorKind<B, D> for Bool {
    fn into_any(tensor: Tensor<B, D, Self>, dtype: DType) -> AnyBurnTensor<B, D> {
        AnyBurnTensor::Bool(relayrl_types::data::tensor::BoolBurnTensor {
            tensor: Arc::new(tensor),
            dtype,
        })
    }
}

/// Byte width of one element for a given NdArray dtype.
fn dtype_bytes_per_elem(dtype: &EnvNdArrayDType) -> usize {
    match dtype {
        EnvNdArrayDType::F16 | EnvNdArrayDType::I16 => 2,
        EnvNdArrayDType::F32 | EnvNdArrayDType::I32 => 4,
        EnvNdArrayDType::F64 | EnvNdArrayDType::I64 => 8,
        EnvNdArrayDType::I8 | EnvNdArrayDType::Bool  => 1,
    }
}

pub(crate) trait VecEnvTrait<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
>: Send + Sync
{
    fn get_env_count(&self) -> Result<usize, VecEnvError>;
    fn env_ids(&self) -> Vec<EnvironmentUuid>;
    fn resize(&mut self, count: usize) -> Result<(), VecEnvError>;
    fn reset_all(&mut self) -> Result<Vec<EnvResetRecord<B, D_IN>>, VecEnvError>;
    fn reset_where(
        &mut self,
        env_ids: &[EnvironmentUuid],
    ) -> Result<Vec<EnvResetRecord<B, D_IN>>, VecEnvError>;
    fn step(
        &mut self,
        actions: &[(EnvironmentUuid, AnyBurnTensor<B, D_OUT>)],
    ) -> Result<Vec<EnvStepRecord<B, D_IN>>, VecEnvError>;

    // ── Flat-buffer fast path (opt-in via VectorEnvironment defaults) ────────────
    /// Returns `(n_envs, obs_dim, act_dim)` if the underlying env supports the flat path.
    fn n_envs_dims(&self) -> Option<(usize, usize, usize)> { None }
    /// Current observations as a flat `[n_envs × obs_dim]` f32 Vec, or None.
    fn flat_obs_clone(&self) -> Option<Vec<f32>> { None }
    /// Step all sub-envs with discrete (argmax) integer actions; returns `(new_obs_flat, rewards, dones)`.
    fn step_flat_actions(&mut self, actions: &[u8]) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> { None }
    /// Step all sub-envs with continuous actions as typed raw bytes.
    /// `dtype` identifies the element type of `actions`.
    fn step_flat_actions_cont_bytes(
        &mut self,
        actions: &[u8],
        dtype: &EnvNdArrayDType,
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> { None }
    /// Stable env UUIDs in flat-path order, or None if fast path unsupported.
    fn flat_env_ids(&self) -> Option<Vec<EnvironmentUuid>> { None }
    /// `true` if the action space is discrete, `false` if continuous.
    fn action_is_discrete(&self) -> Option<bool> { None }
}

pub(crate) struct ScalarVecEnv<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B> + IntoAnyTensorKind<B, D_IN> + Send + Sync,
    KOutput: TensorKind<B> + BasicOps<B> + Send + Sync,
> {
    client_namespace: Arc<str>,
    env_context: ContextString,
    prototype: Box<dyn DynScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>>,
    envs: HashMap<EnvironmentUuid, Box<dyn DynScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>>>,
    // Fast-path: stable ordering and flat obs buffer
    ordered_ids: Vec<EnvironmentUuid>,
    obs_flat: Vec<f32>,
    obs_dim: usize,
    act_dim: usize,
    device: DeviceType,
    observation_dtype: DType,
    #[allow(dead_code)]
    action_dtype: DType,
    _phantom: PhantomData<(B, KInput, KOutput)>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B> + IntoAnyTensorKind<B, D_IN> + Send + Sync,
    KOutput: TensorKind<B> + BasicOps<B> + Send + Sync,
> ScalarVecEnv<B, D_IN, D_OUT, KInput, KOutput>
{
    pub(crate) fn init_boxed(
        client_namespace: Arc<str>,
        env: Box<dyn DynScalarEnvironment<B, D_IN, D_OUT, KInput, KOutput>>,
        count: usize,
        device: DeviceType,
        observation_dtype: DType,
        action_dtype: DType,
    ) -> Result<Self, VecEnvError> {
        if count == 0 {
            return Err(VecEnvError::InvalidEnvironmentCount(
                "count must be greater than zero".to_string(),
            ));
        }

        let env_context = format!("{}:scalar", ENVIRONMENT_CONTEXT_PREFIX);
        let mut envs = HashMap::with_capacity(count);
        let mut ordered_ids = Vec::with_capacity(count);
        for _ in 0..count {
            let env_id = reserve_id(client_namespace.as_ref(), env_context.as_ref())?;
            envs.insert(env_id, env.clone());
            ordered_ids.push(env_id);
        }

        // Probe fast-path support from the prototype env
        let obs_dim = env.dyn_flat_obs().map(|o| o.len()).unwrap_or(0);
        let act_dim = env.dyn_act_dim().unwrap_or(0);
        let obs_flat = if obs_dim > 0 && act_dim > 0 {
            let mut buf = Vec::with_capacity(count * obs_dim);
            for uuid in &ordered_ids {
                match envs[uuid].dyn_flat_obs() {
                    Some(obs) => buf.extend_from_slice(&obs),
                    None => buf.extend(std::iter::repeat(0.0f32).take(obs_dim)),
                }
            }
            buf
        } else {
            Vec::new()
        };

        Ok(Self {
            client_namespace,
            env_context,
            prototype: env,
            envs,
            ordered_ids,
            obs_flat,
            obs_dim,
            act_dim,
            device,
            observation_dtype,
            action_dtype,
            _phantom: PhantomData,
        })
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B> + IntoAnyTensorKind<B, D_IN> + Send + Sync,
    KOutput: TensorKind<B> + BasicOps<B> + Send + Sync,
> VecEnvTrait<B, D_IN, D_OUT> for ScalarVecEnv<B, D_IN, D_OUT, KInput, KOutput>
{
    fn get_env_count(&self) -> Result<usize, VecEnvError> {
        Ok(self.envs.len())
    }

    fn env_ids(&self) -> Vec<EnvironmentUuid> {
        self.envs.keys().copied().collect()
    }

    fn resize(&mut self, count: usize) -> Result<(), VecEnvError> {
        let current = self.envs.len();
        if count == current {
            return Ok(());
        }
        if count > current {
            for _ in 0..(count - current) {
                let env_id = reserve_id(self.client_namespace.as_ref(), self.env_context.as_ref())?;
                let new_env = self.prototype.clone();
                if self.obs_dim > 0 {
                    match new_env.dyn_flat_obs() {
                        Some(obs) => self.obs_flat.extend_from_slice(&obs),
                        None => self.obs_flat.extend(std::iter::repeat(0.0f32).take(self.obs_dim)),
                    }
                }
                self.envs.insert(env_id, new_env);
                self.ordered_ids.push(env_id);
            }
        } else {
            let removed: Vec<_> = self.ordered_ids.drain(count..).collect();
            if self.obs_dim > 0 {
                self.obs_flat.truncate(count * self.obs_dim);
            }
            for env_id in removed {
                self.envs.remove(&env_id);
                remove_id(
                    self.client_namespace.as_ref(),
                    self.env_context.as_ref(),
                    env_id,
                )?;
            }
        }
        Ok(())
    }

    fn reset_all(&mut self) -> Result<Vec<EnvResetRecord<B, D_IN>>, VecEnvError> {
        let ids = self.env_ids();
        self.reset_where(&ids)
    }

    fn reset_where(
        &mut self,
        env_ids: &[EnvironmentUuid],
    ) -> Result<Vec<EnvResetRecord<B, D_IN>>, VecEnvError> {
        let dtype = self.observation_dtype.clone();
        let obs_dim = self.obs_dim;
        env_ids
            .iter()
            .map(|env_id| {
                let env = self
                    .envs
                    .get(env_id)
                    .ok_or_else(|| VecEnvError::UnknownEnv(*env_id))?;
                let reset = env.reset()?;
                // Keep obs_flat in sync when fast path is active
                if obs_dim > 0 {
                    if let Some(idx) = self.ordered_ids.iter().position(|id| id == env_id) {
                        if let Some(obs) = env.dyn_flat_obs() {
                            self.obs_flat[idx * obs_dim..(idx + 1) * obs_dim]
                                .copy_from_slice(&obs);
                        }
                    }
                }
                Ok(EnvResetRecord {
                    env_id: *env_id,
                    observation: KInput::into_any(reset.observation, dtype.clone()),
                    info: reset.info,
                })
            })
            .collect()
    }

    fn step(
        &mut self,
        actions: &[(EnvironmentUuid, AnyBurnTensor<B, D_OUT>)],
    ) -> Result<Vec<EnvStepRecord<B, D_IN>>, VecEnvError> {
        let device =
            B::get_device(&self.device).map_err(|e| VecEnvError::TensorError(e.to_string()))?;
        let dtype = self.observation_dtype.clone();
        actions
            .iter()
            .map(|(env_id, action)| {
                let env = self
                    .envs
                    .get(env_id)
                    .ok_or_else(|| VecEnvError::UnknownEnv(*env_id))?;
                let action =
                    Tensor::<B, D_OUT, KOutput>::from_data(any_tensor_data(action), &device);
                let step = env.step(action)?;
                Ok(EnvStepRecord {
                    env_id: *env_id,
                    observation: KInput::into_any(step.observation, dtype.clone()),
                    reward: step.reward,
                    terminated: step.terminated,
                    truncated: step.truncated,
                    info: step.info,
                })
            })
            .collect()
    }

    // ── Fast-path overrides ──────────────────────────────────────────────────────

    fn n_envs_dims(&self) -> Option<(usize, usize, usize)> {
        if self.obs_dim == 0 || self.act_dim == 0 {
            return None;
        }
        let n = self.ordered_ids.len();
        if n == 0 { None } else { Some((n, self.obs_dim, self.act_dim)) }
    }

    fn flat_obs_clone(&self) -> Option<Vec<f32>> {
        if self.obs_dim == 0 { return None; }
        Some(self.obs_flat.clone())
    }

    fn step_flat_actions(&mut self, actions: &[u8]) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> {
        if self.obs_dim == 0 { return None; }
        let n = self.ordered_ids.len();
        let obs_dim = self.obs_dim;
        let mut rewards = Vec::with_capacity(n);
        let mut dones = Vec::with_capacity(n);

        for (i, uuid) in self.ordered_ids.iter().enumerate() {
            let env = self.envs.get(uuid)?;
            let (obs, reward, done) = env.dyn_step_discrete(actions[i])?;
            self.obs_flat[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&obs);
            rewards.push(reward);
            dones.push(done);
        }
        Some((self.obs_flat.clone(), rewards, dones))
    }

    fn step_flat_actions_cont_bytes(
        &mut self,
        actions: &[u8],
        dtype: &EnvNdArrayDType,
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> {
        if self.obs_dim == 0 { return None; }
        let n = self.ordered_ids.len();
        let obs_dim = self.obs_dim;
        let act_dim = self.act_dim;
        let bytes_per_elem = dtype_bytes_per_elem(dtype);
        let act_bytes = act_dim * bytes_per_elem;
        let mut rewards = Vec::with_capacity(n);
        let mut dones = Vec::with_capacity(n);

        for (i, uuid) in self.ordered_ids.iter().enumerate() {
            let env = self.envs.get(uuid)?;
            let env_act = &actions[i * act_bytes..(i + 1) * act_bytes];
            let (obs, reward, done) = env.dyn_step_continuous_bytes(env_act, dtype)?;
            self.obs_flat[i * obs_dim..(i + 1) * obs_dim].copy_from_slice(&obs);
            rewards.push(reward);
            dones.push(done);
        }
        Some((self.obs_flat.clone(), rewards, dones))
    }

    fn flat_env_ids(&self) -> Option<Vec<EnvironmentUuid>> {
        if self.obs_dim == 0 { return None; }
        Some(self.ordered_ids.clone())
    }

    fn action_is_discrete(&self) -> Option<bool> {
        if self.obs_dim == 0 { return None; }
        self.ordered_ids.first()
            .and_then(|uuid| self.envs.get(uuid))
            .and_then(|env| env.dyn_action_is_discrete())
            .or(Some(true))
    }
}

pub(crate) struct BatchVecEnv<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B> + IntoAnyTensorKind<B, D_IN> + Send + Sync,
    KOutput: TensorKind<B> + BasicOps<B> + Send + Sync,
> {
    client_namespace: Arc<str>,
    env_context: ContextString,
    env: Box<DynVectorEnv<B, D_IN, D_OUT, KInput, KOutput>>,
    env_ids: Vec<EnvironmentUuid>,
    device: DeviceType,
    observation_dtype: DType,
    #[allow(dead_code)]
    action_dtype: DType,
    _phantom: PhantomData<(B, KInput, KOutput)>,
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B> + IntoAnyTensorKind<B, D_IN> + Send + Sync,
    KOutput: TensorKind<B> + BasicOps<B> + Send + Sync,
> BatchVecEnv<B, D_IN, D_OUT, KInput, KOutput>
{
    pub(crate) fn init_boxed(
        client_namespace: Arc<str>,
        env: Box<DynVectorEnv<B, D_IN, D_OUT, KInput, KOutput>>,
        count: usize,
        device: DeviceType,
        observation_dtype: DType,
        action_dtype: DType,
    ) -> Result<Self, VecEnvError> {
        if count == 0 {
            return Err(VecEnvError::InvalidEnvironmentCount(
                "count must be greater than zero".to_string(),
            ));
        }

        let env_context = format!("{}:vector", ENVIRONMENT_CONTEXT_PREFIX);
        let env_ids = env.init_num_envs(count)?;
        for env_id in &env_ids {
            add_id(client_namespace.as_ref(), env_context.as_ref(), *env_id)?;
        }

        Ok(Self {
            client_namespace,
            env_context,
            env,
            env_ids,
            device,
            observation_dtype,
            action_dtype,
            _phantom: PhantomData,
        })
    }
}

impl<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
    KInput: TensorKind<B> + BasicOps<B> + IntoAnyTensorKind<B, D_IN> + Send + Sync,
    KOutput: TensorKind<B> + BasicOps<B> + Send + Sync,
> VecEnvTrait<B, D_IN, D_OUT> for BatchVecEnv<B, D_IN, D_OUT, KInput, KOutput>
{
    fn get_env_count(&self) -> Result<usize, VecEnvError> {
        Ok(self.env_ids.len())
    }

    fn env_ids(&self) -> Vec<EnvironmentUuid> {
        self.env_ids.clone()
    }

    fn resize(&mut self, count: usize) -> Result<(), VecEnvError> {
        let current = self.env_ids.len();
        if count == current {
            return Ok(());
        }

        if count > current {
            let new_ids = self.env.init_num_envs(count - current)?;
            for env_id in &new_ids {
                add_id(
                    self.client_namespace.as_ref(),
                    self.env_context.as_ref(),
                    *env_id,
                )?;
            }
            self.env_ids.extend(new_ids);
        } else {
            let removed = self.env_ids.split_off(count);
            for env_id in removed {
                remove_id(
                    self.client_namespace.as_ref(),
                    self.env_context.as_ref(),
                    env_id,
                )?;
            }
        }
        Ok(())
    }

    fn reset_all(&mut self) -> Result<Vec<EnvResetRecord<B, D_IN>>, VecEnvError> {
        let ids = self.env_ids.clone();
        self.reset_where(&ids)
    }

    fn reset_where(
        &mut self,
        env_ids: &[EnvironmentUuid],
    ) -> Result<Vec<EnvResetRecord<B, D_IN>>, VecEnvError> {
        let dtype = self.observation_dtype.clone();
        self.env
            .reset(env_ids)?
            .into_iter()
            .map(|reset| {
                Ok(EnvResetRecord {
                    env_id: reset.env_id,
                    observation: KInput::into_any(reset.observation, dtype.clone()),
                    info: reset.info,
                })
            })
            .collect()
    }

    fn step(
        &mut self,
        actions: &[(EnvironmentUuid, AnyBurnTensor<B, D_OUT>)],
    ) -> Result<Vec<EnvStepRecord<B, D_IN>>, VecEnvError> {
        let device =
            B::get_device(&self.device).map_err(|e| VecEnvError::TensorError(e.to_string()))?;
        let dtype = self.observation_dtype.clone();
        let typed_actions: Vec<_> = actions
            .iter()
            .map(|(env_id, action)| {
                (
                    *env_id,
                    Tensor::<B, D_OUT, KOutput>::from_data(any_tensor_data(action), &device),
                )
            })
            .collect();

        self.env
            .step(&typed_actions)?
            .into_iter()
            .map(|step| {
                Ok(EnvStepRecord {
                    env_id: step.env_id,
                    observation: KInput::into_any(step.observation, dtype.clone()),
                    reward: step.reward,
                    terminated: step.terminated,
                    truncated: step.truncated,
                    info: step.info,
                })
            })
            .collect()
    }

    fn n_envs_dims(&self) -> Option<(usize, usize, usize)> {
        let n   = self.env.n_envs();
        let obs = self.env.obs_dim();
        let act = self.env.act_dim();
        if n == 0 { None } else { Some((n, obs, act)) }
    }

    fn flat_obs_clone(&self) -> Option<Vec<f32>> {
        self.env.flat_obs()
    }

    fn step_flat_actions(&mut self, actions: &[u8]) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> {
        self.env.step_raw_actions(actions)
    }

    fn step_flat_actions_cont_bytes(
        &mut self,
        actions: &[u8],
        dtype: &EnvNdArrayDType,
    ) -> Option<(Vec<f32>, Vec<f32>, Vec<bool>)> {
        self.env.step_raw_actions_cont_bytes(actions, dtype)
    }

    fn flat_env_ids(&self) -> Option<Vec<EnvironmentUuid>> {
        let n = self.env.n_envs();
        if n == 0 { None } else { Some(self.env_ids.clone()) }
    }

    fn action_is_discrete(&self) -> Option<bool> {
        self.env.action_is_discrete().or(Some(true))
    }
}
