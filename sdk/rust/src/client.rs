//! Samhati client — OpenAI-compatible Rust client.

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use reqwest::StatusCode;

use crate::error::{Result, SamhatiError};
use crate::streaming::SseStream;
use crate::types::*;

const DEFAULT_BASE_URL: &str = "http://localhost:8000";
const DEFAULT_TIMEOUT_SECS: u64 = 60;
const DEFAULT_MAX_RETRIES: u32 = 2;

/// OpenAI-compatible client for Samhati decentralized AI network.
pub struct SamhatiClient {
    base_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
    max_retries: u32,
}

impl SamhatiClient {
    /// Create a new client pointing at the given base URL.
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
            client,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Create a client with the default local URL.
    pub fn default_local() -> Self {
        Self::new(DEFAULT_BASE_URL)
    }

    /// Set the API key (builder pattern).
    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Set a custom timeout in seconds (builder pattern).
    pub fn with_timeout(self, timeout_secs: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to build HTTP client");

        Self { client, ..self }
    }

    /// Set max retries (builder pattern).
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    // ── Private helpers ──

    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("samhati-rust/0.1.0"),
        );
        if let Some(ref key) = self.api_key {
            if let Ok(val) = HeaderValue::from_str(&format!("Bearer {key}")) {
                headers.insert(AUTHORIZATION, val);
            }
        }
        headers
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    async fn handle_error_response(
        &self,
        status: StatusCode,
        response: reqwest::Response,
    ) -> SamhatiError {
        let body: Option<serde_json::Value> = response.json().await.ok();
        let message = body
            .as_ref()
            .and_then(|b| b.get("error"))
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error")
            .to_string();

        match status.as_u16() {
            401 => SamhatiError::Authentication(message),
            429 => SamhatiError::RateLimit(message),
            _ => SamhatiError::Api {
                message,
                status_code: status.as_u16(),
                body,
            },
        }
    }

    // ── Public API ──

    /// Create a chat completion (non-streaming).
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatCompletion> {
        let mut last_err = None;

        for attempt in 0..=self.max_retries {
            let resp = self
                .client
                .post(self.url("/v1/chat/completions"))
                .headers(self.headers())
                .json(&request)
                .send()
                .await;

            match resp {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return response
                            .json::<ChatCompletion>()
                            .await
                            .map_err(SamhatiError::Http);
                    }
                    if status.as_u16() == 429 && attempt < self.max_retries {
                        last_err = Some(SamhatiError::RateLimit("Rate limited".into()));
                        continue;
                    }
                    return Err(self.handle_error_response(status, response).await);
                }
                Err(e) => {
                    if e.is_timeout() {
                        last_err = Some(SamhatiError::Timeout(e.to_string()));
                    } else if e.is_connect() {
                        last_err = Some(SamhatiError::Connection(e.to_string()));
                    } else {
                        return Err(SamhatiError::Http(e));
                    }
                    if attempt >= self.max_retries {
                        return Err(last_err.unwrap());
                    }
                }
            }
        }

        Err(last_err.unwrap_or(SamhatiError::Connection("Unknown error".into())))
    }

    /// Create a streaming chat completion. Returns an SSE stream.
    pub async fn chat_stream(&self, mut request: ChatRequest) -> Result<SseStream> {
        request.stream = true;

        let response = self
            .client
            .post(self.url("/v1/chat/completions"))
            .headers(self.headers())
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    SamhatiError::Timeout(e.to_string())
                } else if e.is_connect() {
                    SamhatiError::Connection(e.to_string())
                } else {
                    SamhatiError::Http(e)
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(self.handle_error_response(status, response).await);
        }

        Ok(SseStream::new(response))
    }

    /// List available models.
    pub async fn models(&self) -> Result<Vec<Model>> {
        let response = self
            .client
            .get(self.url("/v1/models"))
            .headers(self.headers())
            .send()
            .await
            .map_err(SamhatiError::Http)?;

        let status = response.status();
        if !status.is_success() {
            return Err(self.handle_error_response(status, response).await);
        }

        let list: ModelList = response.json().await.map_err(SamhatiError::Http)?;
        Ok(list.data)
    }

    /// Check node health.
    pub async fn health(&self) -> Result<HealthStatus> {
        let response = self
            .client
            .get(self.url("/v1/health"))
            .headers(self.headers())
            .send()
            .await
            .map_err(SamhatiError::Http)?;

        let status = response.status();
        if !status.is_success() {
            return Err(self.handle_error_response(status, response).await);
        }

        response.json().await.map_err(SamhatiError::Http)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn chat_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "samhati-general-3b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "samhati_node_id": "node-xyz",
            "samhati_confidence": 0.92,
            "samhati_n_nodes": 7
        })
    }

    fn models_response_json() -> serde_json::Value {
        serde_json::json!({
            "object": "list",
            "data": [
                {"id": "samhati-general-3b", "object": "model", "created": 1700000000, "owned_by": "samhati"},
                {"id": "samhati-code-7b", "object": "model", "created": 1700000000, "owned_by": "samhati"}
            ]
        })
    }

    fn health_response_json() -> serde_json::Value {
        serde_json::json!({
            "status": "ok",
            "node_id": "node-xyz",
            "version": "0.2.0",
            "models_loaded": ["samhati-general-3b"],
            "peers_connected": 12
        })
    }

    #[tokio::test]
    async fn test_chat_completion() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response_json()))
            .mount(&server)
            .await;

        let client = SamhatiClient::new(&server.uri());
        let req = ChatRequest {
            messages: vec![ChatMessage::user("Hi")],
            ..Default::default()
        };

        let result = client.chat(req).await.unwrap();
        assert_eq!(result.id, "chatcmpl-abc123");
        assert_eq!(
            result.choices[0].message.content.as_deref(),
            Some("Hello!")
        );
        assert_eq!(result.samhati_node_id.as_deref(), Some("node-xyz"));
        assert_eq!(result.samhati_confidence, Some(0.92));
        assert_eq!(result.samhati_n_nodes, Some(7));
    }

    #[tokio::test]
    async fn test_list_models() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(models_response_json()))
            .mount(&server)
            .await;

        let client = SamhatiClient::new(&server.uri());
        let models = client.models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "samhati-general-3b");
    }

    #[tokio::test]
    async fn test_health() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(health_response_json()))
            .mount(&server)
            .await;

        let client = SamhatiClient::new(&server.uri());
        let health = client.health().await.unwrap();
        assert_eq!(health.status, "ok");
        assert_eq!(health.peers_connected, Some(12));
    }

    #[tokio::test]
    async fn test_authentication_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(401)
                    .set_body_json(serde_json::json!({"error": {"message": "Invalid API key"}})),
            )
            .mount(&server)
            .await;

        let client = SamhatiClient::new(&server.uri());
        let req = ChatRequest {
            messages: vec![ChatMessage::user("Hi")],
            ..Default::default()
        };

        let err = client.chat(req).await.unwrap_err();
        assert!(matches!(err, SamhatiError::Authentication(_)));
    }

    #[tokio::test]
    async fn test_api_error_500() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(500)
                    .set_body_json(serde_json::json!({"error": {"message": "Internal error"}})),
            )
            .mount(&server)
            .await;

        let client = SamhatiClient::new(&server.uri()).with_max_retries(0);
        let req = ChatRequest {
            messages: vec![ChatMessage::user("Hi")],
            ..Default::default()
        };

        let err = client.chat(req).await.unwrap_err();
        assert!(matches!(
            err,
            SamhatiError::Api {
                status_code: 500,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_streaming() {
        let server = MockServer::start().await;

        let sse_body = [
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1700000000,"model":"samhati-general-3b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            "",
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1700000000,"model":"samhati-general-3b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            "",
            r#"data: {"id":"c1","object":"chat.completion.chunk","created":1700000000,"model":"samhati-general-3b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"samhati_node_id":"node-xyz"}"#,
            "",
            "data: [DONE]",
            "",
        ]
        .join("\n");

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let client = SamhatiClient::new(&server.uri());
        let req = ChatRequest {
            messages: vec![ChatMessage::user("Hi")],
            stream: true,
            ..Default::default()
        };

        let stream = client.chat_stream(req).await.unwrap();

        use futures::StreamExt;
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].is_ok());
        let first = chunks[0].as_ref().unwrap();
        assert_eq!(first.choices[0].delta.role.as_deref(), Some("assistant"));

        let second = chunks[1].as_ref().unwrap();
        assert_eq!(second.choices[0].delta.content.as_deref(), Some("Hello"));

        let third = chunks[2].as_ref().unwrap();
        assert_eq!(third.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(third.samhati_node_id.as_deref(), Some("node-xyz"));
    }
}
