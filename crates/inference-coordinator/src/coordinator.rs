use anyhow::Result;
use async_trait::async_trait;

use crate::plan::{ShardPlan, ShardSpec};

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub request_id: String,
    pub input: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub output: String,
    pub steps: Vec<ShardStep>,
}

#[derive(Debug, Clone)]
pub struct ShardStep {
    pub shard_index: usize,
    pub peer_id: String,
    pub output: String,
}

#[async_trait]
pub trait ShardExecutor: Send + Sync {
    async fn run_shard(
        &self,
        shard: &ShardSpec,
        input: &str,
        request: &InferenceRequest,
    ) -> Result<String>;
}

pub struct Coordinator<E> {
    pub plan: ShardPlan,
    pub executor: E,
}

impl<E: ShardExecutor> Coordinator<E> {
    pub fn new(plan: ShardPlan, executor: E) -> Self {
        Self { plan, executor }
    }

    pub async fn run(&self, request: InferenceRequest) -> Result<InferenceResult> {
        let mut current = request.input.clone();
        let mut steps = Vec::new();

        for (idx, shard) in self.plan.shards.iter().enumerate() {
            let output = self.executor.run_shard(shard, &current, &request).await?;
            current = output.clone();
            steps.push(ShardStep {
                shard_index: idx,
                peer_id: shard.peer_id.clone(),
                output,
            });
        }

        Ok(InferenceResult {
            output: current,
            steps,
        })
    }
}

#[derive(Debug, Default)]
pub struct EchoExecutor;

#[async_trait]
impl ShardExecutor for EchoExecutor {
    async fn run_shard(
        &self,
        shard: &ShardSpec,
        input: &str,
        request: &InferenceRequest,
    ) -> Result<String> {
        Ok(format!(
            "{} [peer:{} layers:{}-{} max_tokens:{} temp:{:.2}]",
            input,
            shard.peer_id,
            shard.layer_start,
            shard.layer_end,
            request.max_tokens,
            request.temperature
        ))
    }
}
