use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeRole {
    Inference,
    Routing,
    Cache,
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

pub const INFERENCE_ALPN: &[u8] = b"mesh-inference";
pub type SessionId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub session_id: SessionId,
    pub layer_start: u32,
    pub layer_end: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    #[serde(with = "serde_bytes")]
    pub tensor_bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    #[serde(with = "serde_bytes")]
    pub tensor_bytes: Vec<u8>,
    pub error: Option<String>,
}

impl InferenceRequest {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let bytes = bincode::serialize(self)?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let req = bincode::deserialize(bytes)?;
        Ok(req)
    }
}

impl InferenceResponse {
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let bytes = bincode::serialize(self)?;
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let res = bincode::deserialize(bytes)?;
        Ok(res)
    }
}
