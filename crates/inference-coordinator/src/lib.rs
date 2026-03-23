mod coordinator;
mod model_runner;
mod iroh_executor;
mod plan;
mod swarm_planner;
pub mod rpc;
pub mod tensor_frame;
pub mod kv_cache;
pub mod weight_cache;
pub mod activation_cache;

#[cfg(feature = "burn")]
pub mod llm_shard;

#[cfg(feature = "burn")]
pub mod qwen35_cache;

#[cfg(feature = "burn")]
pub mod qwen35_shard;

pub use model_runner::{ModelShardRunner, ModelShardRunnerConfig};
pub use coordinator::{
    Coordinator, EchoExecutor, InferenceRequest, InferenceResult, ShardExecutor, ShardStep,
};
pub use iroh_executor::IrohDistributedExecutor;
pub use plan::{RoundRobinPlanner, ShardPlan, ShardSpec};
pub use swarm_planner::SwarmPlanner;
pub use tensor_frame::TensorFrame;
pub use weight_cache::WeightCache;
pub use activation_cache::{ActivationRingBuffer, CachedActivation, OutputRingBuffer, CachedOutput};

#[cfg(feature = "burn")]
pub use llm_shard::{LlamaShard, LlamaShardConfig};

#[cfg(feature = "burn")]
pub use kv_cache::KvCacheStore;

#[cfg(feature = "burn")]
pub use qwen35_shard::{Qwen35Shard, Qwen35ShardConfig};

#[cfg(feature = "burn")]
pub use qwen35_cache::Qwen35CacheStore;

/// The GPU backend used for production inference (WGPU).
#[cfg(feature = "burn")]
pub type InferenceBackend = burn::backend::Wgpu;
