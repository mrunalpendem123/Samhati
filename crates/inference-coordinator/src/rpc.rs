use bincode::Options;
use serde::{Deserialize, Serialize};

use crate::tensor_frame::TensorFrame;

/// ALPN identifier for the mesh inference QUIC protocol.
pub const INFERENCE_ALPN: &[u8] = b"mesh-inference/2";

/// Maximum allowed size for a single bincode-deserialized message (256 MB).
/// Prevents OOM from maliciously crafted Vec length fields.
const MAX_MESSAGE_SIZE: u64 = 256 * 1024 * 1024;

fn bincode_options() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_limit(MAX_MESSAGE_SIZE)
        .with_fixint_encoding()
        .allow_trailing_bytes()
}

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
        Ok(bincode_options().deserialize(bytes)?)
    }
}

impl RpcResponse {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode_options().deserialize(bytes)?)
    }
}

// ── Replay RPC (fault recovery) ───────────────────────────────────────────────
//
// When a shard peer fails mid-generation, the client replays its cached
// activations to the replacement node so it can rebuild its KV cache and
// resume without a full restart.

/// Request to replay cached activations through a replacement shard node.
///
/// The replacement node processes each frame sequentially through its layers,
/// building up the KV cache as if those tokens had been generated normally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub session_id: SessionId,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    /// Ordered sequence of activation frames to replay, oldest first.
    pub frames: Vec<TensorFrame>,
}

/// Response after replay completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResponse {
    /// Output tensor from the final replayed frame.
    pub tensor: TensorFrame,
    /// Number of frames successfully replayed.
    pub frames_replayed: u32,
    pub error: Option<String>,
}

impl ReplayRequest {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode_options().deserialize(bytes)?)
    }
}

impl ReplayResponse {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode_options().deserialize(bytes)?)
    }
}

/// Message type discriminator for multiplexing over INFERENCE_ALPN.
///
/// Prepended as the first byte of the stream to distinguish between
/// normal inference requests and replay requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RpcMessageType {
    Inference = 0x00,
    Replay = 0x01,
}

impl RpcMessageType {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::Inference),
            0x01 => Some(Self::Replay),
            _ => None,
        }
    }

    pub fn to_byte(self) -> u8 {
        self as u8
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
        Ok(bincode_options().deserialize(bytes)?)
    }
}

impl LegacyRpcResponse {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode_options().deserialize(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_frame::TensorFrame;

    #[test]
    fn rpc_request_roundtrip() {
        let req = RpcRequest {
            session_id: "test-session".to_string(),
            layer_start: 0,
            layer_end: 16,
            total_layers: 32,
            max_tokens: 128,
            temperature: 0.7,
            tensor: TensorFrame::from_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4], 0),
        };
        let bytes = req.to_bytes().unwrap();
        let decoded = RpcRequest::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.session_id, "test-session");
        assert_eq!(decoded.layer_start, 0);
        assert_eq!(decoded.layer_end, 16);
    }

    #[test]
    fn rpc_response_roundtrip() {
        let resp = RpcResponse {
            tensor: TensorFrame::from_f32(&[42.0], vec![1, 1], 0),
            error: None,
        };
        let bytes = resp.to_bytes().unwrap();
        let decoded = RpcResponse::from_bytes(&bytes).unwrap();
        assert!(decoded.error.is_none());
        assert_eq!(decoded.tensor.shape, vec![1, 1]);
    }

    #[test]
    fn bincode_rejects_crafted_oversized_payload() {
        // A small garbage payload should fail deserialization, not panic.
        let garbage = vec![0xFF; 32];
        assert!(RpcRequest::from_bytes(&garbage).is_err());
    }
}
