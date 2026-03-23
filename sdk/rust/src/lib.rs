//! Samhati SDK — OpenAI-compatible client for the Samhati decentralized AI network.
//!
//! # Example
//!
//! ```no_run
//! use samhati_sdk::{SamhatiClient, ChatRequest, ChatMessage};
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = SamhatiClient::default_local();
//!     let request = ChatRequest {
//!         messages: vec![ChatMessage::user("Explain photosynthesis")],
//!         ..Default::default()
//!     };
//!     let response = client.chat(request).await.unwrap();
//!     println!("{}", response.choices[0].message.content.as_deref().unwrap_or(""));
//! }
//! ```

pub mod client;
pub mod error;
pub mod streaming;
pub mod types;

pub use client::SamhatiClient;
pub use error::{Result, SamhatiError};
pub use streaming::SseStream;
pub use types::*;
