use proximity_router::swarm::LayerRange;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeRole {
    Inference,
    Routing,
    Cache,
}

/// A layer-range announcement embedded in a capability payload.
/// Mirrors `proximity_router::LayerRange` but derives Serialize/Deserialize
/// for gossip wire encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostedLayerRange {
    pub model: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub total_layers: usize,
}

impl From<HostedLayerRange> for LayerRange {
    fn from(h: HostedLayerRange) -> Self {
        LayerRange {
            model: h.model,
            layer_start: h.layer_start,
            layer_end: h.layer_end,
            total_layers: h.total_layers,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPayload {
    pub node_id: String,
    pub free_vram_gb: f64,
    pub bandwidth_mbps: f64,
    pub reliability: f64,
    pub gpu_capacity_score: f64,
    pub rtt_ms: f64,
    pub kv_bits: u8,
    pub context: usize,
    pub quant_bits: u8,
    pub role: NodeRole,
    /// Layer ranges this node is currently serving (empty if not an inference node).
    #[serde(default)]
    pub layers_hosted: Vec<HostedLayerRange>,
    /// Normalised load: 0 = idle, 1 = saturated. Used by SwarmRegistry scoring.
    #[serde(default)]
    pub load_score: f64,
    /// Self-reported uptime in seconds.
    #[serde(default)]
    pub uptime_secs: u64,
    /// blake3 hashes of cached weight shards (hex-encoded).
    /// Peers can request these shards via iroh-blobs or the shard-store protocol.
    #[serde(default)]
    pub cached_shard_hashes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub name: String,
    pub params_b: f64,
    pub quant_bits: Option<u8>,
    pub kv_bits: Option<u8>,
    pub max_context: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsAnnouncement {
    pub node_id: String,
    pub models: Vec<ModelSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingMessage {
    pub from_id: String,
    pub target_id: String,
    pub nonce: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PongMessage {
    pub from_id: String,
    pub target_id: String,
    pub nonce: String,
}

pub fn parse_capability(body: &str) -> Option<CapabilityPayload> {
    serde_json::from_str(body).ok()
}

pub fn parse_models(body: &str) -> Option<ModelsAnnouncement> {
    serde_json::from_str(body).ok()
}

pub fn parse_ping(body: &str) -> Option<PingMessage> {
    serde_json::from_str(body).ok()
}

pub fn parse_pong(body: &str) -> Option<PongMessage> {
    serde_json::from_str(body).ok()
}

// RPC ALPN constant lives in inference-coordinator to avoid a circular dependency.
pub use inference_coordinator::rpc::INFERENCE_ALPN;
