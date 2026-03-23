use serde::{Deserialize, Serialize};

/// Default number of top logits to capture per token.
pub const DEFAULT_TOP_K: usize = 8;

/// Default number of tokens per proof chunk.
pub const DEFAULT_CHUNK_SIZE: usize = 32;

/// Default number of calibration prompts.
pub const DEFAULT_CALIBRATION_PROMPTS: usize = 10;

/// A model identifier with its precomputed BLAKE3 hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub model_hash: [u8; 32],
}

impl ModelInfo {
    /// Compute model hash from the model identifier string.
    pub fn from_id(model_id: &str) -> Self {
        let model_hash = blake3::hash(model_id.as_bytes());
        Self {
            model_id: model_id.to_string(),
            model_hash: *model_hash.as_bytes(),
        }
    }
}

/// Compute the BLAKE3 hash of a model identifier string.
pub fn model_id_to_hash(model_id: &str) -> [u8; 32] {
    *blake3::hash(model_id.as_bytes()).as_bytes()
}
