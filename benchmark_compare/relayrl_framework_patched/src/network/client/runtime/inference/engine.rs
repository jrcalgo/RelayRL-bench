//! Policy-partitioned batched inference engine.
//!
//! `InferenceEngine` owns all `HotReloadableModel` handles keyed by policy and
//! executes forward passes **directly in the coordinator's async task** — with no
//! channel dispatch, no per-actor message round-trip, and no oneshot allocation on
//! the hot path.
//!
//! # Design invariant
//!
//! Actors **never** participate in inference execution after this refactor.  Only the
//! coordinator (via `InferenceEngine`) issues `model.forward` calls.  Actors retain
//! their model handles solely for the handshake / hot-reload write path; because
//! handles are `Arc<ArcSwapOption<HotReloadableModel<B>>>`, any write through an
//! actor is immediately visible here via the lock-free swap.

use crate::network::client::runtime::actor::LocalModelHandle;

use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::tensor::{AnyBurnTensor, BackendMatcher};

use active_uuid_registry::registry_uuid::Uuid;
use burn_tensor::backend::Backend;
use dashmap::DashMap;
use std::marker::PhantomData;
use std::sync::Arc;

// ─── PolicyId ────────────────────────────────────────────────────────────────

/// A policy identifier.  Currently one-to-one with `ActorUuid`.
pub(crate) type PolicyId = Uuid;

// ─── InferenceEngine ─────────────────────────────────────────────────────────

/// Owns all model handles keyed by policy and executes forward passes directly.
///
/// Handles are the **same `Arc`** clones that actors hold for model updates, so a
/// hot-reload swap by any actor is immediately visible here without extra synchronisation.
pub(crate) struct InferenceEngine<
    B: Backend + BackendMatcher<Backend = B>,
    const D_IN: usize,
    const D_OUT: usize,
> {
    /// Maps `PolicyId → LocalModelHandle<B>`.
    pub(crate) registry: DashMap<PolicyId, LocalModelHandle<B>>,
    _d: PhantomData<([(); D_IN], [(); D_OUT])>,
}

impl<B: Backend + BackendMatcher<Backend = B>, const D_IN: usize, const D_OUT: usize>
    InferenceEngine<B, D_IN, D_OUT>
{
    pub(crate) fn new() -> Self {
        Self {
            registry: DashMap::new(),
            _d: PhantomData,
        }
    }

    /// Register a policy's model handle.
    pub(crate) fn register(&self, policy_id: PolicyId, handle: LocalModelHandle<B>) {
        self.registry.insert(policy_id, handle);
    }

    /// Deregister a policy when the corresponding actor is removed.
    pub(crate) fn deregister(&self, policy_id: &PolicyId) {
        self.registry.remove(policy_id);
    }

    /// Execute a forward pass for `policy_id` directly in the caller's task.
    ///
    /// Uses `ArcSwapOption::load()` (lock-free) to access the model snapshot,
    /// then calls `model.forward` inline.
    ///
    /// Returns `None` if the policy has no registered handle or the model is
    /// not yet loaded (handshake pending).
    pub(crate) async fn forward(
        &self,
        policy_id: PolicyId,
        obs: Arc<AnyBurnTensor<B, D_IN>>,
        mask: Option<Arc<AnyBurnTensor<B, D_OUT>>>,
        reward: f32,
    ) -> Option<RelayRLAction> {
        let handle = self.registry.get(&policy_id)?;
        let guard = handle.load();
        let model = match &*guard {
            Some(m) => m,
            None => return None,
        };
        model
            .forward::<D_IN, D_OUT>(obs, mask, reward, policy_id)
            .ok()
    }
}
