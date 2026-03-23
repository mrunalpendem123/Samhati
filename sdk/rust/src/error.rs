//! Custom errors for the Samhati SDK.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SamhatiError {
    #[error("API error (status {status_code}): {message}")]
    Api {
        message: String,
        status_code: u16,
        body: Option<serde_json::Value>,
    },

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    #[error("Request timed out: {0}")]
    Timeout(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("SSE parse error: {0}")]
    SseParse(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, SamhatiError>;
