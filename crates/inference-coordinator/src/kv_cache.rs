//! Per-session KV cache for transformer inference.
//!
//! Each shard node holds a `KvCacheStore` that maps `session_id →` the
//! accumulated key/value tensors for every layer it owns.  Sessions are
//! automatically evicted after a configurable idle TTL.

// ── Burn-gated types ──────────────────────────────────────────────────────────

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

/// Accumulated K and V vectors for one attention layer.
///
/// Each chunk is `[batch, n_kv_heads, chunk_seq, head_dim]`.
#[cfg(feature = "burn")]
pub struct LayerKv<B: Backend> {
    pub k_chunks: Vec<Tensor<B, 4>>,
    pub v_chunks: Vec<Tensor<B, 4>>,
}

#[cfg(feature = "burn")]
impl<B: Backend> LayerKv<B> {
    pub fn new() -> Self {
        Self { k_chunks: Vec::new(), v_chunks: Vec::new() }
    }

    /// Append a new `(key, value)` chunk and return the full concatenated
    /// `(K, V)` tensors along the sequence dimension (dim 2).
    pub fn append(
        &mut self,
        k_new: Tensor<B, 4>,
        v_new: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        self.k_chunks.push(k_new);
        self.v_chunks.push(v_new);
        let k_all = Tensor::cat(self.k_chunks.clone(), 2);
        let v_all = Tensor::cat(self.v_chunks.clone(), 2);
        (k_all, v_all)
    }

    /// Total sequence length accumulated so far in this layer.
    pub fn seq_len(&self) -> usize {
        self.k_chunks.iter().map(|t| t.dims()[2]).sum()
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
pub struct SessionKv<B: Backend> {
    /// One `LayerKv` per transformer layer owned by this shard.
    pub layers: Vec<LayerKv<B>>,
    /// Total tokens fed into this session (across all decode steps).
    pub seq_pos: usize,
    pub last_access: Instant,
}

#[cfg(feature = "burn")]
impl<B: Backend> SessionKv<B> {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerKv::new()).collect(),
            seq_pos: 0,
            last_access: Instant::now(),
        }
    }
}

// ── Store ─────────────────────────────────────────────────────────────────────

/// Thread-safe map of `session_id → SessionKv<B>`.
#[cfg(feature = "burn")]
#[derive(Clone)]
pub struct KvCacheStore<B: Backend> {
    pub inner: Arc<RwLock<HashMap<String, SessionKv<B>>>>,
    ttl_secs: u64,
}

#[cfg(feature = "burn")]
impl<B: Backend> KvCacheStore<B> {
    /// Create a store where idle sessions are evicted after `ttl_secs`.
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            ttl_secs,
        }
    }

    /// Evict sessions that have been idle longer than the configured TTL.
    pub async fn evict_idle(&self) {
        let now = Instant::now();
        let ttl = self.ttl_secs;
        let mut map = self.inner.write().await;
        map.retain(|_, sess| now.duration_since(sess.last_access).as_secs() < ttl);
    }

    /// Ensure a session exists (creating it with `n_layers` layer slots if
    /// needed) and update its `last_access` time.
    pub async fn touch(&self, session_id: &str, n_layers: usize) {
        let mut map = self.inner.write().await;
        let sess = map
            .entry(session_id.to_string())
            .or_insert_with(|| SessionKv::new(n_layers));
        sess.last_access = Instant::now();
    }

    /// Remove a session entirely (e.g. after the final decode step).
    pub async fn remove(&self, session_id: &str) {
        self.inner.write().await.remove(session_id);
    }
}
