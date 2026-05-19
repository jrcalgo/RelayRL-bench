mod kernel;
mod replay_buffer;

pub mod independent;
pub mod multiagent;

pub use independent::kernel::*;
pub use independent::replay_buffer::*;
pub use independent::{
    EpochTrainOutput, IPPOAlgorithm, IPPOParams, IndependentPPOAlgorithm, PPOAlgorithm, PPOParams,
    SlotTrainResult,
};
pub use multiagent::kernel::MultiagentPPOKernel;
pub use multiagent::replay_buffer::MultiagentPPOReplayBuffer;
pub use multiagent::{
    MAPPOAlgorithm, MAPPOParams, MultiagentPPOAlgorithm, MultiagentPPOKernelTrait,
};
