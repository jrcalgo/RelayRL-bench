use crate::network::client::runtime::router::buffer::TrajectorySinkError;
use crate::network::client::runtime::router::filter::FilterError;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
use crate::network::client::runtime::router::receiver::TransportReceiverError;

use relayrl_types::data::action::RelayRLAction;
use relayrl_types::data::trajectory::RelayRLTrajectory;

use active_uuid_registry::registry_uuid::Uuid;

use std::sync::Arc;
use thiserror::Error;
use tokio::sync::oneshot;

pub(crate) mod buffer;
pub(crate) mod filter;
#[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
pub(crate) mod receiver;
pub(crate) mod router_dispatcher;

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
pub enum RouterError {
    #[error(transparent)]
    FilterError(#[from] FilterError),
    #[error(transparent)]
    #[cfg(any(feature = "nats-transport", feature = "zmq-transport"))]
    TransportReceiverError(#[from] TransportReceiverError),
    #[error(transparent)]
    TrajectorySinkError(#[from] TrajectorySinkError),
}

pub(crate) struct RoutedMessage {
    pub actor_id: Uuid,
    pub protocol: RoutingProtocol,
    pub payload: RoutedPayload,
}

pub(crate) enum RoutingProtocol {
    ModelHandshake,
    FlagLastInference,
    ModelVersion,
    ModelUpdate,
    SendTrajectory,
    RecordAction,
    Shutdown,
}

pub(crate) enum RoutedPayload {
    ModelHandshake,
    FlagLastInference {
        reward: f32,
    },
    ModelVersion {
        reply_to: oneshot::Sender<i64>,
    },
    ModelUpdate {
        model_bytes: Vec<u8>,
        version: i64,
    },
    SendTrajectory {
        timestamp: (u128, u128),
        trajectory: RelayRLTrajectory,
    },
    RecordAction(Arc<RelayRLAction>),
    Shutdown,
}
