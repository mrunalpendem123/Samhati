use anyhow::Result;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{sse::{Event, KeepAlive, Sse}, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use futures_util::{stream::BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};

use inference_coordinator::{Coordinator, IrohDistributedExecutor, InferenceRequest, SwarmPlanner};
use proximity_router::SwarmRegistry;

use crate::inference::{infer_http, infer_http_stream, infer_local_exec, words_stream, ChatMessage, LocalExecConfig};

pub struct SwarmHandle {
    pub endpoint: iroh::Endpoint,
    pub registry: Arc<RwLock<SwarmRegistry>>,
    pub model_name: String,
    pub total_layers: usize,
    pub layers_per_shard: usize,
    pub max_retries: usize,
}

impl SwarmHandle {
    pub async fn run(&self, req: InferenceRequest) -> anyhow::Result<String> {
        let planner = SwarmPlanner::new(self.layers_per_shard);
        let plan = {
            let reg = self.registry.read().await;
            planner.plan(&self.model_name, self.total_layers, &reg)?
        };
        let executor = IrohDistributedExecutor::new(self.endpoint.clone())
            .with_registry(self.registry.clone(), &self.model_name, self.max_retries);
        Coordinator::new(plan, executor).generate(req).await
    }
}

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
    pub swarm: Option<Arc<SwarmHandle>>,
}

pub struct ApiState {
    pub infer_base: Option<String>,
    pub infer_key: Option<String>,
    pub default_model: Option<String>,
    pub local_exec: Option<LocalExecConfig>,
    pub api_auth: Option<String>,
    pub allow_models: Option<HashSet<String>>,
    rate_limiter: Option<Arc<Mutex<RateLimiter>>>,
    pub swarm: Option<Arc<SwarmHandle>>,
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
        swarm: config.swarm,
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
    Json(req): Json<ChatRequest>,
) -> axum::response::Response {
    if let Err(resp) = check_auth(&state, &headers).await {
        return resp;
    }
    if let Err(resp) = check_rate_limit(&state).await {
        return resp;
    }
    if state.infer_base.is_none() && state.local_exec.is_none() && state.swarm.is_none() {
        return api_error(
            StatusCode::NOT_IMPLEMENTED,
            "not_configured",
            "inference backend not configured (set --infer-base, --local-bin/--local-args/--local-model, or --topic for swarm)",
        );
    }
    if req.messages.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "invalid_request", "messages must be non-empty");
    }

    let model = match req.model.or_else(|| state.default_model.clone()) {
        Some(m) => m,
        None => return api_error(StatusCode::BAD_REQUEST, "invalid_request", "model is required"),
    };
    if let Some(allow) = &state.allow_models {
        if !allow.contains(&model) {
            return api_error(StatusCode::FORBIDDEN, "forbidden", "model not allowed");
        }
    }

    let messages = req
        .messages
        .into_iter()
        .map(|m| ChatMessage { role: m.role, content: m.content })
        .collect::<Vec<_>>();
    let max_tokens = req.max_tokens.unwrap_or(128);
    let temperature = req.temperature.unwrap_or(0.2);
    let completion_id = chrono_id();

    // ── Streaming path ────────────────────────────────────────────────────────
    if req.stream.unwrap_or(false) {
        let content_stream: BoxStream<'static, String> =
            if let Some(swarm) = state.swarm.clone() {
                let prompt = messages
                    .iter()
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                let infer_req = InferenceRequest {
                    request_id: completion_id.clone(),
                    input: prompt,
                    max_tokens,
                    temperature,
                };
                match swarm.run(infer_req).await {
                    Ok(text) => Box::pin(words_stream(text)),
                    Err(e) => {
                        return api_error(
                            StatusCode::BAD_GATEWAY,
                            "swarm_error",
                            &e.to_string(),
                        );
                    }
                }
            } else if let Some(local) = state.local_exec.clone() {
                match infer_local_exec(&local, messages, max_tokens, temperature).await {
                    Ok(text) => Box::pin(words_stream(text)),
                    Err(e) => {
                        return api_error(
                            StatusCode::BAD_GATEWAY,
                            "backend_error",
                            &format!("local inference error: {e}"),
                        );
                    }
                }
            } else {
                let infer_base = state.infer_base.clone().unwrap();
                match infer_http_stream(
                    &infer_base,
                    state.infer_key.clone(),
                    model.clone(),
                    messages,
                    max_tokens,
                    temperature,
                )
                .await
                {
                    Ok(s) => Box::pin(s),
                    Err(e) => {
                        return api_error(
                            StatusCode::BAD_GATEWAY,
                            "backend_error",
                            &format!("inference error: {e}"),
                        );
                    }
                }
            };
        return sse_response(completion_id, model, content_stream);
    }

    // ── Non-streaming path ────────────────────────────────────────────────────
    let content = if let Some(swarm) = state.swarm.clone() {
        let prompt = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");
        let infer_req = InferenceRequest {
            request_id: completion_id.clone(),
            input: prompt,
            max_tokens,
            temperature,
        };
        match swarm.run(infer_req).await {
            Ok(c) => c,
            Err(e) => {
                return api_error(StatusCode::BAD_GATEWAY, "swarm_error", &e.to_string());
            }
        }
    } else if let Some(local) = state.local_exec.clone() {
        match infer_local_exec(&local, messages, max_tokens, temperature).await {
            Ok(c) => c,
            Err(e) => {
                return api_error(
                    StatusCode::BAD_GATEWAY,
                    "backend_error",
                    &format!("local inference error: {e}"),
                );
            }
        }
    } else {
        let infer_base = state.infer_base.clone().unwrap();
        match infer_http(&infer_base, state.infer_key.clone(), model.clone(), messages, max_tokens, temperature).await {
            Ok(c) => c,
            Err(e) => {
                return api_error(
                    StatusCode::BAD_GATEWAY,
                    "backend_error",
                    &format!("inference error: {e}"),
                );
            }
        }
    };

    let response = ChatCompletionResponse {
        id: format!("cmpl-{completion_id}"),
        object: "chat.completion".to_string(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageInput { role: "assistant".to_string(), content },
            finish_reason: "stop".to_string(),
        }],
    };
    Json(response).into_response()
}

/// Build a standard OpenAI-style error JSON response.
fn api_error(status: StatusCode, error_type: &str, message: &str) -> axum::response::Response {
    let body = json!({ "error": { "message": message, "type": error_type } });
    (status, Json(body)).into_response()
}

/// Serialise one SSE chunk in OpenAI `chat.completion.chunk` format.
/// Fields are omitted when `None` so the wire format matches upstream conventions:
/// - role event: `delta: {"role": "assistant"}`
/// - content event: `delta: {"content": "..."}`
/// - stop event: `delta: {}`, `finish_reason: "stop"`
fn make_chunk_event(
    id: &str,
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish_reason: Option<&str>,
) -> Event {
    let mut delta = serde_json::Map::new();
    if let Some(r) = role {
        delta.insert("role".into(), json!(r));
    }
    if let Some(c) = content {
        delta.insert("content".into(), json!(c));
    }
    let data = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
    });
    Event::default().data(data.to_string())
}

/// Wraps a content-delta stream in an SSE response.
///
/// Emits: role announcement → content chunks → stop chunk → `[DONE]`.
fn sse_response(
    id: String,
    model: String,
    content_stream: BoxStream<'static, String>,
) -> axum::response::Response {
    // Build the three bookend events before moving id/model.
    let role_event  = make_chunk_event(&id, &model, Some("assistant"), None, None);
    let stop_event  = make_chunk_event(&id, &model, None, None, Some("stop"));

    // Content events: one SSE chunk per delta string.
    let content_events = content_stream.map(move |text| {
        Ok::<Event, Infallible>(make_chunk_event(&id, &model, None, Some(&text), None))
    });

    let full_stream = futures_util::stream::once(async move {
        Ok::<Event, Infallible>(role_event)
    })
    .chain(content_events)
    .chain(futures_util::stream::once(async move {
        Ok::<Event, Infallible>(stop_event)
    }))
    .chain(futures_util::stream::once(async {
        Ok::<Event, Infallible>(Event::default().data("[DONE]"))
    }));

    Sse::new(full_stream).keep_alive(KeepAlive::default()).into_response()
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
