//! SSE streaming handler for the Samhati SDK.

use futures::stream::Stream;
use reqwest::Response;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::error::{Result, SamhatiError};
use crate::types::ChatCompletionChunk;

/// A stream of `ChatCompletionChunk` parsed from an SSE response.
pub struct SseStream {
    inner: Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>,
}

impl SseStream {
    pub fn new(response: Response) -> Self {
        let stream = async_stream::stream! {
            let mut buffer = String::new();
            let mut bytes_stream = response.bytes_stream();

            use futures::StreamExt;
            while let Some(result) = bytes_stream.next().await {
                let bytes = match result {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(SamhatiError::Http(e));
                        return;
                    }
                };

                let text = match std::str::from_utf8(&bytes) {
                    Ok(t) => t.to_string(),
                    Err(e) => {
                        yield Err(SamhatiError::SseParse(e.to_string()));
                        return;
                    }
                };

                buffer.push_str(&text);

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with(':') {
                        continue;
                    }

                    if let Some(payload) = trimmed.strip_prefix("data: ") {
                        let payload = payload.trim();
                        if payload == "[DONE]" {
                            return;
                        }

                        match serde_json::from_str::<ChatCompletionChunk>(payload) {
                            Ok(chunk) => yield Ok(chunk),
                            Err(e) => {
                                yield Err(SamhatiError::SseParse(format!(
                                    "Failed to parse chunk: {e}"
                                )));
                                return;
                            }
                        }
                    }
                }
            }
        };

        Self {
            inner: Box::pin(stream),
        }
    }
}

impl Stream for SseStream {
    type Item = Result<ChatCompletionChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
