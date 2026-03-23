//! Type definitions for the Samhati SDK, OpenAI-compatible with Samhati extensions.

use serde::{Deserialize, Serialize};

// ── Request Types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    /// Swarm routing mode: "quick" (N=3), "best" (N=7), "local" (N=1)
    #[serde(default = "default_mode")]
    pub samhati_mode: String,
    /// Domain hint for specialist routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samhati_domain: Option<String>,
    /// Include TOPLOC proof in response
    #[serde(default)]
    pub samhati_proof: bool,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_max_tokens() -> u32 {
    2048
}
fn default_mode() -> String {
    "best".to_string()
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            model: "samhati-general-3b".to_string(),
            messages: Vec::new(),
            temperature: 0.7,
            max_tokens: 2048,
            stream: false,
            samhati_mode: "best".to_string(),
            samhati_domain: None,
            samhati_proof: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

// ── Response Types ──

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
    // Samhati extensions
    pub samhati_node_id: Option<String>,
    pub samhati_confidence: Option<f64>,
    pub samhati_proof: Option<String>,
    pub samhati_n_nodes: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming Types ──

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    // Samhati extensions (typically on final chunk)
    pub samhati_node_id: Option<String>,
    pub samhati_confidence: Option<f64>,
    pub samhati_proof: Option<String>,
    pub samhati_n_nodes: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeltaMessage {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ── Model / Health Types ──

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<Model>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub node_id: Option<String>,
    pub version: Option<String>,
    pub models_loaded: Option<Vec<String>>,
    pub peers_connected: Option<u32>,
}

// ── API Error Body ──

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ApiErrorBody {
    pub error: Option<ApiErrorDetail>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ApiErrorDetail {
    pub message: Option<String>,
}
