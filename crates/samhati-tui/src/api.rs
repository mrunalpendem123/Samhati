use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::app::ModelInfo;

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatRequestMessage>,
}

#[derive(Serialize)]
struct ChatRequestMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub n_nodes: Option<usize>,
}

#[derive(Deserialize)]
pub struct ChatChoice {
    pub message: ChatChoiceMessage,
}

#[derive(Deserialize)]
pub struct ChatChoiceMessage {
    pub content: String,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    domain: Option<String>,
    #[serde(default)]
    size_gb: Option<f32>,
    #[serde(default)]
    smti_bonus: Option<String>,
    #[serde(default)]
    installed: Option<bool>,
    #[serde(default)]
    active: Option<bool>,
}

pub struct ApiClient {
    base_url: String,
    client: reqwest::Client,
}

impl ApiClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
        }
    }

    pub async fn chat(&self, message: &str, model: &str) -> Result<ChatResponse> {
        let req = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatRequestMessage {
                role: "user".into(),
                content: message.into(),
            }],
        };

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json::<ChatResponse>()
            .await?;

        Ok(resp)
    }

    pub async fn health(&self) -> Result<bool> {
        let resp = self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .await;
        Ok(resp.is_ok())
    }

    pub async fn models(&self) -> Result<Vec<ModelInfo>> {
        let resp = self
            .client
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .json::<ModelsResponse>()
            .await?;

        Ok(resp
            .data
            .into_iter()
            .map(|m| ModelInfo {
                name: m.id,
                domain: m.domain.unwrap_or_else(|| "General".into()),
                size_gb: m.size_gb.unwrap_or(0.0),
                smti_bonus: m.smti_bonus.unwrap_or_else(|| "1.0x".into()),
                installed: m.installed.unwrap_or(false),
                active: m.active.unwrap_or(false),
            })
            .collect())
    }
}
