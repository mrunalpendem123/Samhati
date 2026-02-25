use anyhow::{anyhow, Result};
use async_trait::async_trait;
use iroh::{
    endpoint::{Connection, RecvStream, SendStream},
    Endpoint, EndpointId,
};
use std::str::FromStr;
use tokio::time::{timeout, Duration};

use crate::plan::ShardSpec;
use crate::rpc::{RpcRequest, RpcResponse, INFERENCE_ALPN};
use crate::tensor_frame::TensorFrame;
use crate::{InferenceRequest, ShardExecutor};

/// Implements `ShardExecutor` by opening a direct iroh QUIC stream to a peer,
/// forwarding the activation `TensorFrame`, and awaiting the response.
///
/// Both the request and response are size-prefixed (4-byte big-endian length)
/// bincode payloads, identical to what `InferenceServer` expects.
#[derive(Debug, Clone)]
pub struct IrohDistributedExecutor {
    endpoint: Endpoint,
    /// Timeout (seconds) for the initial QUIC connection.
    connect_timeout_secs: u64,
    /// Timeout (seconds) for the full RPC round-trip.
    rpc_timeout_secs: u64,
}

impl IrohDistributedExecutor {
    pub fn new(endpoint: Endpoint) -> Self {
        Self { endpoint, connect_timeout_secs: 10, rpc_timeout_secs: 120 }
    }

    pub fn with_timeouts(mut self, connect_secs: u64, rpc_secs: u64) -> Self {
        self.connect_timeout_secs = connect_secs;
        self.rpc_timeout_secs = rpc_secs;
        self
    }

    /// Low-level helper: serialise `req`, send over a bi-directional QUIC
    /// stream to `peer_id`, and deserialise the response.
    async fn call_peer(
        &self,
        peer_id: &str,
        req: &RpcRequest,
    ) -> Result<RpcResponse> {
        let target_id = EndpointId::from_str(peer_id)
            .map_err(|e| anyhow!("invalid EndpointId '{}': {}", peer_id, e))?;

        // 1. Connect
        let conn: Connection = timeout(
            Duration::from_secs(self.connect_timeout_secs),
            self.endpoint.connect(target_id, INFERENCE_ALPN),
        )
        .await
        .map_err(|_| anyhow!("timeout connecting to peer {}", peer_id))??;

        let (mut send, mut recv): (SendStream, RecvStream) = conn.open_bi().await?;

        // 2. Serialise and send request with 4-byte size prefix
        let req_bytes = req.to_bytes()?;
        let size = (req_bytes.len() as u32).to_be_bytes();
        send.write_all(&size).await?;
        send.write_all(&req_bytes).await?;
        send.finish()?;

        // 3. Read size-prefixed response
        let mut size_buf = [0u8; 4];
        recv.read_exact(&mut size_buf).await?;
        let resp_size = u32::from_be_bytes(size_buf) as usize;
        let mut resp_bytes = vec![0u8; resp_size];
        recv.read_exact(&mut resp_bytes).await?;

        // 4. Deserialise
        let resp = RpcResponse::from_bytes(&resp_bytes)?;
        if let Some(err) = &resp.error {
            return Err(anyhow!("peer {} reported error: {}", peer_id, err));
        }
        Ok(resp)
    }
}

// ── ShardExecutor impl ────────────────────────────────────────────────────────

#[async_trait]
impl ShardExecutor for IrohDistributedExecutor {
    /// Legacy string-based call (wraps a single token ID string in a frame).
    async fn run_shard(
        &self,
        shard: &ShardSpec,
        input: &str,
        request: &InferenceRequest,
    ) -> Result<String> {
        // Encode the text prompt as a byte-level token-ID frame for the peer
        let token_ids: Vec<f32> = input.bytes().map(|b| b as f32).collect();
        let frame = TensorFrame::from_f32(&token_ids, vec![1, token_ids.len().max(1)], 0);

        let resp = self.rpc(shard, frame, request).await?;
        // Interpret the returned token IDs as bytes → text
        let decoded: String = resp
            .to_f32_vec()?
            .iter()
            .map(|&f| f as u8)
            .take_while(|&b| b != 0)
            .collect::<Vec<u8>>()
            .into_iter()
            .map(|b| b as char)
            .collect();
        Ok(decoded)
    }

    /// Tensor-native call — used by `Coordinator::generate()`.
    async fn run_shard_tensor(
        &self,
        shard: &ShardSpec,
        frame: TensorFrame,
        request: &InferenceRequest,
    ) -> Result<TensorFrame> {
        self.rpc(shard, frame, request).await
    }
}

impl IrohDistributedExecutor {
    async fn rpc(
        &self,
        shard: &ShardSpec,
        frame: TensorFrame,
        request: &InferenceRequest,
    ) -> Result<TensorFrame> {
        let req = RpcRequest {
            session_id: request.request_id.clone(),
            layer_start: shard.layer_start as u32,
            layer_end: shard.layer_end as u32,
            total_layers: self.plan_total_layers(shard),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            tensor: frame,
        };

        let resp = timeout(
            Duration::from_secs(self.rpc_timeout_secs),
            self.call_peer(&shard.peer_id, &req),
        )
        .await
        .map_err(|_| anyhow!("RPC timeout to peer {}", shard.peer_id))??;

        Ok(resp.tensor)
    }

    /// We don't have a back-reference to the ShardPlan here, so we infer
    /// `total_layers` from shard metadata (coordinators set this on ShardSpec).
    fn plan_total_layers(&self, shard: &ShardSpec) -> u32 {
        // ShardSpec.total_layers was added; fall back to layer_end if absent.
        shard.total_layers.unwrap_or(shard.layer_end) as u32
    }
}
