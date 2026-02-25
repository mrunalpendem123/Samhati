use serde::{Deserialize, Serialize};

use crate::tensor_frame::TensorFrame;

/// ALPN identifier for the mesh inference QUIC protocol.
pub const INFERENCE_ALPN: &[u8] = b"mesh-inference/2";

pub type SessionId = String;

/// Request sent over the QUIC stream from coordinator → shard peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    pub session_id: SessionId,
    /// Index of the first layer this peer should execute.
    pub layer_start: u32,
    /// One-past the last layer this peer should execute.
    pub layer_end: u32,
    /// Total layers in the model (lets the peer detect first / last shard).
    pub total_layers: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Activation tensor to pass through.
    ///
    /// * First shard  → token IDs encoded as f32, shape [1, seq_len].
    /// * Other shards → hidden states, shape [1, seq_len, hidden_size].
    pub tensor: TensorFrame,
}

/// Response sent back from a shard peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    /// Output tensor from this shard.
    ///
    /// * Non-final shards → hidden states [1, seq_len, hidden_size].
    /// * Final shard      → next token ID as f32, shape [1, 1].
    pub tensor: TensorFrame,
    pub error: Option<String>,
}

impl RpcRequest {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

impl RpcResponse {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

// ── Backwards-compat shims ────────────────────────────────────────────────────
// mesh-node/src/server.rs imports these names; keep them alive.

/// Legacy byte-blob request — used by `server.rs` mock path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyRpcRequest {
    pub session_id: SessionId,
    pub layer_start: u32,
    pub layer_end: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    #[serde(with = "serde_bytes")]
    pub tensor_bytes: Vec<u8>,
}

/// Legacy byte-blob response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyRpcResponse {
    #[serde(with = "serde_bytes")]
    pub tensor_bytes: Vec<u8>,
    pub error: Option<String>,
}

impl LegacyRpcRequest {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

impl LegacyRpcResponse {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}
