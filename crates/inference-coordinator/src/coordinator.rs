use anyhow::Result;
use async_trait::async_trait;

use crate::plan::{ShardPlan, ShardSpec};
use crate::tensor_frame::TensorFrame;

// ── Request / Result types ────────────────────────────────────────────────────

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

// ── Executor trait ────────────────────────────────────────────────────────────

/// Runs a single activation frame through one peer's assigned layers.
///
/// Implementations:
///   - `EchoExecutor`           — echo/debug, no real inference
///   - `IrohDistributedExecutor` — opens a QUIC stream to a remote peer
#[async_trait]
pub trait ShardExecutor: Send + Sync {
    async fn run_shard(
        &self,
        shard: &ShardSpec,
        input: &str,
        request: &InferenceRequest,
    ) -> Result<String>;

    /// Tensor-aware variant used by the real generate loop.
    ///
    /// The default implementation is a forwarding shim that packs/unpacks the
    /// token-ID → string representation so legacy executors still work.
    async fn run_shard_tensor(
        &self,
        shard: &ShardSpec,
        frame: TensorFrame,
        request: &InferenceRequest,
    ) -> Result<TensorFrame> {
        // Encode token IDs as a comma-separated string for the legacy path
        let token_str: String = frame
            .to_f32_vec()?
            .iter()
            .map(|f| (*f as u32).to_string())
            .collect::<Vec<_>>()
            .join(",");
        let out = self.run_shard(shard, &token_str, request).await?;
        // Parse the response back as a single token ID
        let token_id: f32 = out.trim().parse().unwrap_or(0.0);
        Ok(TensorFrame::from_f32(&[token_id], vec![1, 1], frame.seq_offset + 1))
    }
}

// ── Coordinator ───────────────────────────────────────────────────────────────

pub struct Coordinator<E> {
    pub plan: ShardPlan,
    pub executor: E,
}

impl<E: ShardExecutor> Coordinator<E> {
    pub fn new(plan: ShardPlan, executor: E) -> Self {
        Self { plan, executor }
    }

    /// Legacy string-based pipeline (used by `dist-run --executor echo|candle`).
    pub async fn run(&self, request: InferenceRequest) -> Result<InferenceResult> {
        let mut current = request.input.clone();
        let mut steps = Vec::new();

        for (idx, shard) in self.plan.shards.iter().enumerate() {
            let output = self
                .executor
                .run_shard(shard, &current, &request)
                .await?;
            current = output.clone();
            steps.push(ShardStep {
                shard_index: idx,
                peer_id: shard.peer_id.clone(),
                output,
            });
        }

        Ok(InferenceResult { output: current, steps })
    }

    /// Real autoregressive generation using typed tensor frames.
    ///
    /// 1. Tokenizes `request.input` using a simple byte-level vocabulary
    ///    (token id = byte value, vocab_size = 256).  Replace with a proper
    ///    `tokenizers` tokenizer when model weights ship with one.
    /// 2. Sends a prefill frame (all prompt tokens) through every shard.
    /// 3. Loops up to `max_tokens` steps: each iteration sends a single token
    ///    through every shard and collects the next predicted token.
    /// 4. Stops on EOS (token id 0 used as sentinel) or `max_tokens`.
    pub async fn generate(&self, request: InferenceRequest) -> Result<String> {
        // Tokenize: byte-level (byte → u32 id)
        let prompt_ids: Vec<f32> = request.input.bytes().map(|b| b as f32).collect();
        let prompt_len = prompt_ids.len();

        // ── Prefill ──────────────────────────────────────────────────────────
        let mut frame = TensorFrame::from_f32(
            &prompt_ids,
            vec![1, prompt_len],
            0,
        );

        for shard in &self.plan.shards {
            frame = self
                .executor
                .run_shard_tensor(shard, frame, &request)
                .await?;
        }

        // After the final shard, `frame` contains the first next-token ID: [1,1]
        let mut generated_ids: Vec<u32> = Vec::new();
        let first_id = frame.to_f32_vec()?.first().copied().unwrap_or(0.0) as u32;
        if first_id != 0 {
            generated_ids.push(first_id);
        }

        // ── Decode loop ──────────────────────────────────────────────────────
        for _ in 1..request.max_tokens {
            let last = *generated_ids.last().unwrap_or(&0);
            if last == 0 {
                break; // EOS
            }

            let mut frame = TensorFrame::from_f32(
                &[last as f32],
                vec![1, 1],
                prompt_len + generated_ids.len() - 1,
            );

            for shard in &self.plan.shards {
                frame = self
                    .executor
                    .run_shard_tensor(shard, frame, &request)
                    .await?;
            }

            let next_id = frame.to_f32_vec()?.first().copied().unwrap_or(0.0) as u32;
            if next_id == 0 {
                break;
            }
            generated_ids.push(next_id);
        }

        // Decode byte-level token IDs back to UTF-8 (best-effort)
        let bytes: Vec<u8> = generated_ids.iter().map(|&id| id as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

// ── EchoExecutor ─────────────────────────────────────────────────────────────

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
