mod coordinator;
mod model_runner;
mod iroh_executor;
mod plan;
mod swarm_planner;
pub mod rpc;
pub mod tensor_frame;
pub mod kv_cache;

#[cfg(feature = "burn")]
pub mod llm_shard;

pub use model_runner::{ModelShardRunner, ModelShardRunnerConfig};
pub use coordinator::{
    Coordinator, EchoExecutor, InferenceRequest, InferenceResult, ShardExecutor, ShardStep,
};
pub use iroh_executor::IrohDistributedExecutor;
pub use plan::{RoundRobinPlanner, ShardPlan, ShardSpec};
pub use swarm_planner::SwarmPlanner;
pub use tensor_frame::TensorFrame;

#[cfg(feature = "burn")]
pub use llm_shard::{LlamaShard, LlamaShardConfig};

#[cfg(feature = "burn")]
pub use kv_cache::KvCacheStore;

/// The GPU backend used for production inference (WGPU).
#[cfg(feature = "burn")]
pub type InferenceBackend = burn::backend::Wgpu;
