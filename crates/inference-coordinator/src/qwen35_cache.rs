//! Per-session state for Qwen3.5 hybrid attention/linear-attention inference.
//!
//! Qwen3.5 alternates between full-attention layers (which need KV caches) and
//! Gated DeltaNet layers (which carry recurrent state + conv1d buffers).
//! `Qwen35Session` holds the per-layer state for whichever type each layer is,
//! and `Qwen35CacheStore` provides thread-safe session management with TTL.

#[cfg(feature = "burn")]
use std::collections::HashMap;
#[cfg(feature = "burn")]
use std::sync::Arc;
#[cfg(feature = "burn")]
use std::time::Instant;
#[cfg(feature = "burn")]
use tokio::sync::RwLock;

#[cfg(feature = "burn")]
use burn::tensor::{backend::Backend, Tensor};

#[cfg(feature = "burn")]
use crate::kv_cache::LayerKv;

// ── Recurrent state for Gated DeltaNet layers ────────────────────────────────

/// Recurrent state carried between decode steps for one linear-attention layer.
#[cfg(feature = "burn")]
pub struct SsmState<B: Backend> {
    /// Hidden state matrix: `[batch, num_heads, key_dim, value_dim]`.
    pub h: Option<Tensor<B, 4>>,
    /// Causal conv1d state buffer: `[batch, conv_channels, kernel_size - 1]`.
    pub conv_state: Option<Tensor<B, 3>>,
}

#[cfg(feature = "burn")]
impl<B: Backend> SsmState<B> {
    pub fn new() -> Self {
        Self { h: None, conv_state: None }
    }
}

// ── Per-layer state enum ─────────────────────────────────────────────────────

/// State for a single Qwen3.5 layer — either KV cache (full attention) or
/// recurrent state (Gated DeltaNet).
#[cfg(feature = "burn")]
pub enum LayerState<B: Backend> {
    FullAttention(LayerKv<B>),
    LinearAttention(SsmState<B>),
}

// ── Session ──────────────────────────────────────────────────────────────────

/// Per-session state for a Qwen3.5 shard.
#[cfg(feature = "burn")]
pub struct Qwen35Session<B: Backend> {
    /// One state per local layer (in shard order).
    pub layers: Vec<LayerState<B>>,
    /// Total tokens processed so far.
    pub seq_pos: usize,
    pub last_access: Instant,
}

#[cfg(feature = "burn")]
impl<B: Backend> Qwen35Session<B> {
    /// Create a new session.
    ///
    /// `layer_types` lists whether each local layer is full-attention (`true`)
    /// or linear-attention (`false`), in order.
    pub fn new(layer_types: &[bool]) -> Self {
        let layers = layer_types
            .iter()
            .map(|&is_full| {
                if is_full {
                    LayerState::FullAttention(LayerKv::new())
                } else {
                    LayerState::LinearAttention(SsmState::new())
                }
            })
            .collect();
        Self {
            layers,
            seq_pos: 0,
            last_access: Instant::now(),
        }
    }
}

// ── Store ────────────────────────────────────────────────────────────────────

/// Thread-safe map of `session_id → Qwen35Session<B>`.
#[cfg(feature = "burn")]
#[derive(Clone)]
pub struct Qwen35CacheStore<B: Backend> {
    pub inner: Arc<RwLock<HashMap<String, Qwen35Session<B>>>>,
    ttl_secs: u64,
    /// Which local layers are full-attention (`true`) vs linear (`false`).
    layer_types: Vec<bool>,
}

#[cfg(feature = "burn")]
impl<B: Backend> Qwen35CacheStore<B> {
    pub fn new(ttl_secs: u64, layer_types: Vec<bool>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            ttl_secs,
            layer_types,
        }
    }

    pub async fn evict_idle(&self) {
        let now = Instant::now();
        let ttl = self.ttl_secs;
        let mut map = self.inner.write().await;
        map.retain(|_, sess| now.duration_since(sess.last_access).as_secs() < ttl);
    }

    pub async fn touch(&self, session_id: &str) {
        let mut map = self.inner.write().await;
        let sess = map
            .entry(session_id.to_string())
            .or_insert_with(|| Qwen35Session::new(&self.layer_types));
        sess.last_access = Instant::now();
    }

    pub async fn remove(&self, session_id: &str) {
        self.inner.write().await.remove(session_id);
    }
}
