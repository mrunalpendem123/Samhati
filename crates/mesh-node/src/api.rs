use anyhow::Result;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::inference::{infer_http, infer_local_exec, ChatMessage, LocalExecConfig};

#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub bind: String,
    pub infer_base: Option<String>,
    pub infer_key: Option<String>,
    pub default_model: Option<String>,
    pub local_exec: Option<LocalExecConfig>,
    pub api_auth: Option<String>,
    pub allow_models: Vec<String>,
    pub rate_limit_rps: Option<f64>,
    pub rate_limit_burst: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ApiState {
    pub infer_base: Option<String>,
    pub infer_key: Option<String>,
    pub default_model: Option<String>,
    pub local_exec: Option<LocalExecConfig>,
    pub api_auth: Option<String>,
    pub allow_models: Option<HashSet<String>>,
    rate_limiter: Option<Arc<Mutex<RateLimiter>>>,
}

pub async fn serve(config: ApiConfig) -> Result<()> {
    let state = ApiState {
        infer_base: config.infer_base,
        infer_key: config.infer_key,
        default_model: config.default_model,
        local_exec: config.local_exec,
        api_auth: config.api_auth,
        allow_models: if config.allow_models.is_empty() {
            None
        } else {
            Some(config.allow_models.into_iter().collect())
        },
        rate_limiter: build_rate_limiter(config.rate_limit_rps, config.rate_limit_burst),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(Arc::new(state));

    let listener = tokio::net::TcpListener::bind(&config.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
struct ModelEntry {
    id: String,
    object: String,
}

async fn list_models(
    State(state): State<Arc<ApiState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    if let Err(resp) = check_auth(&state, &headers).await {
        return resp;
    }

    let data = if let Some(allow) = &state.allow_models {
        allow
            .iter()
            .cloned()
            .map(|id| ModelEntry {
                id,
                object: "model".to_string(),
            })
            .collect()
    } else {
        Vec::new()
    };

    let resp = ModelsResponse {
        object: "list".to_string(),
        data,
    };
    Json(resp).into_response()
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<ChatMessageInput>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessageInput {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatMessageInput,
    finish_reason: String,
}

async fn chat_completions(
    State(state): State<Arc<ApiState>>,
    headers: HeaderMap,
    Json(_req): Json<ChatRequest>,
) -> impl IntoResponse {
    if let Err(resp) = check_auth(&state, &headers).await {
        return resp;
    }
    if let Err(resp) = check_rate_limit(&state).await {
        return resp;
    }
    if state.infer_base.is_none() && state.local_exec.is_none() {
        let body = json!({
            "error": {
                "message": "inference backend not configured (set --infer-base or --local-bin/--local-args/--local-model)",
                "type": "not_configured",
            }
        });
        return (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response();
    }

    let mut req = _req;
    if req.messages.is_empty() {
        let body = json!({
            "error": {
                "message": "messages must be non-empty",
                "type": "invalid_request",
            }
        });
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }
    if req.stream.unwrap_or(false) {
        let body = json!({
            "error": {
                "message": "streaming not supported yet",
                "type": "not_supported",
            }
        });
        return (StatusCode::NOT_IMPLEMENTED, Json(body)).into_response();
    }

    let model = req.model.take().or_else(|| state.default_model.clone());
    let model = match model {
        Some(m) => m,
        None => {
            let body = json!({
                "error": {
                    "message": "model is required",
                    "type": "invalid_request",
                }
            });
            return (StatusCode::BAD_REQUEST, Json(body)).into_response();
        }
    };

    if let Some(allow) = &state.allow_models {
        if !allow.contains(&model) {
            let body = json!({
                "error": {
                    "message": "model not allowed",
                    "type": "forbidden",
                }
            });
            return (StatusCode::FORBIDDEN, Json(body)).into_response();
        }
    }

    let messages = req
        .messages
        .into_iter()
        .map(|m| ChatMessage {
            role: m.role,
            content: m.content,
        })
        .collect::<Vec<_>>();

    let max_tokens = req.max_tokens.unwrap_or(128);
    let temperature = req.temperature.unwrap_or(0.2);
    let content = if let Some(local) = state.local_exec.clone() {
        match infer_local_exec(&local, messages, max_tokens, temperature).await {
            Ok(c) => c,
            Err(e) => {
                let body = json!({
                    "error": {
                        "message": format!("local inference error: {e}"),
                        "type": "backend_error",
                    }
                });
                return (StatusCode::BAD_GATEWAY, Json(body)).into_response();
            }
        }
    } else {
        let infer_base = state.infer_base.clone().unwrap();
        let infer_key = state.infer_key.clone();
        match infer_http(
            &infer_base,
            infer_key,
            model.clone(),
            messages,
            max_tokens,
            temperature,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                let body = json!({
                    "error": {
                        "message": format!("inference error: {e}"),
                        "type": "backend_error",
                    }
                });
                return (StatusCode::BAD_GATEWAY, Json(body)).into_response();
            }
        }
    };

    let response = ChatCompletionResponse {
        id: format!("cmpl-{}", chrono_id()),
        object: "chat.completion".to_string(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageInput {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: "stop".to_string(),
        }],
    };

    Json(response).into_response()
}

fn chrono_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos}")
}

async fn check_auth(state: &ApiState, headers: &HeaderMap) -> Result<(), axum::response::Response> {
    let required = match &state.api_auth {
        Some(key) => key,
        None => return Ok(()),
    };

    let auth = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let api_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let bearer_ok = auth.strip_prefix("Bearer ").map(|v| v == required).unwrap_or(false);
    let key_ok = api_key == required;

    if bearer_ok || key_ok {
        Ok(())
    } else {
        let body = json!({
            "error": {
                "message": "unauthorized",
                "type": "unauthorized",
            }
        });
        Err((StatusCode::UNAUTHORIZED, Json(body)).into_response())
    }
}

async fn check_rate_limit(state: &ApiState) -> Result<(), axum::response::Response> {
    let limiter = match &state.rate_limiter {
        Some(l) => l,
        None => return Ok(()),
    };
    let mut guard = limiter.lock().await;
    if guard.allow() {
        Ok(())
    } else {
        let body = json!({
            "error": {
                "message": "rate limit exceeded",
                "type": "rate_limited",
            }
        });
        Err((StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response())
    }
}

fn build_rate_limiter(rps: Option<f64>, burst: Option<u32>) -> Option<Arc<Mutex<RateLimiter>>> {
    let rps = rps?;
    if rps <= 0.0 {
        return None;
    }
    let burst = burst.unwrap_or_else(|| rps.ceil() as u32).max(1);
    Some(Arc::new(Mutex::new(RateLimiter::new(rps, burst))))
}

#[derive(Debug)]
struct RateLimiter {
    capacity: f64,
    tokens: f64,
    refill_per_sec: f64,
    last_refill: Instant,
}

impl RateLimiter {
    fn new(rps: f64, burst: u32) -> Self {
        let capacity = burst as f64;
        Self {
            capacity,
            tokens: capacity,
            refill_per_sec: rps,
            last_refill: Instant::now(),
        }
    }

    fn allow(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.last_refill = now;
        self.tokens = (self.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}
