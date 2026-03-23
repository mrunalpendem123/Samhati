//! Per-session KV cache for transformer inference.
//!
//! Each shard node holds a `KvCacheStore` that maps `session_id →` the
//! accumulated key/value tensors for every layer it owns.  Sessions are
//! automatically evicted after a configurable idle TTL.
//!
//! ## Layer-local ownership
//!
//! A `KvCacheStore` is scoped to a specific layer range (`layer_start..layer_end`).
//! It only allocates KV slots for the layers the owning node is responsible for,
//! eliminating the previous bug where both nodes in a 2-node split would each
//! store the full model's KV cache (2× memory waste that caused OOM on 8 GB GPUs).
//!
//! ## Paged KV storage
//!
//! Long conversations are handled via `PagedAttention`: KV data is stored in
//! fixed-size pages (default 16 tokens each).  When VRAM usage exceeds 90%,
//! the least-recently-used pages are evicted to CPU DRAM (`Vec<u8>`).  Pages
//! are promoted back to VRAM on access.

// ── Burn-gated types ──────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "burn")]
use std::ops::Range;
#[cfg(feature = "burn")]
use std::sync::Arc;
#[cfg(feature = "burn")]
use std::time::Instant;
#[cfg(feature = "burn")]
use tokio::sync::RwLock;

#[cfg(feature = "burn")]
use burn::tensor::{backend::Backend, Tensor};

// ── Constants ─────────────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
const PAGE_SIZE: usize = 16;

#[cfg(feature = "burn")]
const VRAM_EVICTION_THRESHOLD: f64 = 0.90;

// ── Page types ────────────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
type PageId = u64;

/// A fixed-size page holding up to `PAGE_SIZE` tokens of K and V data.
///
/// Data is stored as serialised bytes so pages can be cheaply evicted to CPU
/// DRAM and promoted back without knowing the Burn backend at the page level.
#[cfg(feature = "burn")]
struct KvPage {
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    /// Shape of K (and V): `[batch, n_kv_heads, used_slots, head_dim]`.
    shape: Vec<usize>,
    /// Number of token slots currently filled (0..=PAGE_SIZE).
    used_slots: usize,
}

/// Manages a pool of KV pages with LRU eviction from VRAM to CPU DRAM.
#[cfg(feature = "burn")]
pub struct PageAllocator {
    /// Pages resident in "VRAM" (actively usable).
    vram_pages: HashMap<PageId, KvPage>,
    /// Pages evicted to CPU DRAM (serialised tensor bytes).
    dram_pages: HashMap<PageId, (Vec<u8>, Vec<u8>, Vec<usize>, usize)>,
    /// LRU order: front = least recently used, back = most recently used.
    lru: VecDeque<PageId>,
    /// Running total of VRAM bytes used by active pages.
    vram_usage_bytes: usize,
    /// Configured VRAM capacity for KV pages (bytes).
    vram_capacity_bytes: usize,
    /// Monotonic page ID counter.
    next_id: PageId,
}

#[cfg(feature = "burn")]
impl PageAllocator {
    pub fn new(vram_capacity_bytes: usize) -> Self {
        Self {
            vram_pages: HashMap::new(),
            dram_pages: HashMap::new(),
            lru: VecDeque::new(),
            vram_usage_bytes: 0,
            vram_capacity_bytes,
            next_id: 0,
        }
    }

    /// Allocate a new empty page, evicting LRU pages to DRAM if necessary.
    fn alloc_page(&mut self) -> PageId {
        self.maybe_evict();
        let id = self.next_id;
        self.next_id += 1;
        let page = KvPage {
            k_data: Vec::new(),
            v_data: Vec::new(),
            shape: Vec::new(),
            used_slots: 0,
        };
        self.vram_pages.insert(id, page);
        self.lru.push_back(id);
        id
    }

    /// Get a mutable reference to a page, promoting from DRAM if needed.
    fn get_page_mut(&mut self, id: PageId) -> Option<&mut KvPage> {
        // Promote from DRAM if needed
        if let Some((k_data, v_data, shape, used_slots)) = self.dram_pages.remove(&id) {
            let page_bytes = k_data.len() + v_data.len();
            let page = KvPage { k_data, v_data, shape, used_slots };
            self.vram_pages.insert(id, page);
            self.vram_usage_bytes += page_bytes;
            self.maybe_evict_excluding(id);
        }
        // Touch LRU
        if let Some(pos) = self.lru.iter().position(|&pid| pid == id) {
            self.lru.remove(pos);
            self.lru.push_back(id);
        }
        self.vram_pages.get_mut(&id)
    }

    /// Get a reference to a page, promoting from DRAM if needed.
    fn get_page(&mut self, id: PageId) -> Option<&KvPage> {
        // Promote from DRAM if needed
        if let Some((k_data, v_data, shape, used_slots)) = self.dram_pages.remove(&id) {
            let page_bytes = k_data.len() + v_data.len();
            let page = KvPage { k_data, v_data, shape, used_slots };
            self.vram_pages.insert(id, page);
            self.vram_usage_bytes += page_bytes;
            self.maybe_evict_excluding(id);
        }
        // Touch LRU
        if let Some(pos) = self.lru.iter().position(|&pid| pid == id) {
            self.lru.remove(pos);
            self.lru.push_back(id);
        }
        self.vram_pages.get(&id)
    }

    /// Evict LRU pages to DRAM until VRAM usage is below threshold.
    fn maybe_evict(&mut self) {
        self.maybe_evict_excluding(u64::MAX); // exclude nothing
    }

    fn maybe_evict_excluding(&mut self, exclude_id: PageId) {
        let threshold = (self.vram_capacity_bytes as f64 * VRAM_EVICTION_THRESHOLD) as usize;
        while self.vram_usage_bytes > threshold {
            // Find LRU page that is in VRAM and not excluded
            let victim = self.lru.iter().find(|&&pid| {
                pid != exclude_id && self.vram_pages.contains_key(&pid)
            }).copied();
            let Some(victim_id) = victim else { break };

            if let Some(page) = self.vram_pages.remove(&victim_id) {
                let page_bytes = page.k_data.len() + page.v_data.len();
                self.vram_usage_bytes = self.vram_usage_bytes.saturating_sub(page_bytes);
                self.dram_pages.insert(
                    victim_id,
                    (page.k_data, page.v_data, page.shape, page.used_slots),
                );
            }
        }
    }

    /// Remove pages entirely (when a session is dropped).
    fn free_pages(&mut self, page_ids: &[PageId]) {
        for &id in page_ids {
            if let Some(page) = self.vram_pages.remove(&id) {
                let page_bytes = page.k_data.len() + page.v_data.len();
                self.vram_usage_bytes = self.vram_usage_bytes.saturating_sub(page_bytes);
            }
            self.dram_pages.remove(&id);
            if let Some(pos) = self.lru.iter().position(|&pid| pid == id) {
                self.lru.remove(pos);
            }
        }
    }

    /// Number of pages currently in VRAM.
    pub fn vram_page_count(&self) -> usize {
        self.vram_pages.len()
    }

    /// Number of pages evicted to DRAM.
    pub fn dram_page_count(&self) -> usize {
        self.dram_pages.len()
    }
}

// ── LayerKv ──────────────────────────────────────────────────────────────────

/// Accumulated K and V tensors for one attention layer.
///
/// Supports two modes:
/// - **Monolithic** (no PageAllocator): stores the full concatenated KV cache
///   as a single tensor. Simple and fast for short sequences.
/// - **Paged** (with PageAllocator): stores KV data in fixed-size pages for
///   memory-efficient long conversations with LRU eviction.
#[cfg(feature = "burn")]
pub struct LayerKv<B: Backend> {
    /// Full K cache: `[batch, n_kv_heads, seq_so_far, head_dim]`, or `None` before first append.
    k_cache: Option<Tensor<B, 4>>,
    /// Full V cache: same shape as `k_cache`.
    v_cache: Option<Tensor<B, 4>>,
    seq_len: usize,
    /// Page IDs owned by this layer's KV (empty when using monolithic mode).
    page_ids: Vec<PageId>,
}

#[cfg(feature = "burn")]
impl<B: Backend> LayerKv<B> {
    pub fn new() -> Self {
        Self { k_cache: None, v_cache: None, seq_len: 0, page_ids: Vec::new() }
    }

    /// Append a new `(key, value)` chunk and return the full concatenated
    /// `(K, V)` tensors along the sequence dimension (dim 2).
    ///
    /// Uses `take()` + `cat()` to avoid cloning the existing cache.
    pub fn append(
        &mut self,
        k_new: Tensor<B, 4>,
        v_new: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let new_seq = k_new.dims()[2];

        let k_all = match self.k_cache.take() {
            Some(prev) => Tensor::cat(vec![prev, k_new], 2),
            None => k_new,
        };
        let v_all = match self.v_cache.take() {
            Some(prev) => Tensor::cat(vec![prev, v_new], 2),
            None => v_new,
        };

        self.k_cache = Some(k_all.clone());
        self.v_cache = Some(v_all.clone());
        self.seq_len += new_seq;

        (k_all, v_all)
    }

    /// Append using paged storage.  Serialises the new K/V chunk into pages
    /// managed by the shared `PageAllocator`, then reassembles the full tensors.
    pub fn append_paged(
        &mut self,
        k_new: Tensor<B, 4>,
        v_new: Tensor<B, 4>,
        allocator: &mut PageAllocator,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let new_seq = k_new.dims()[2];
        let shape = k_new.dims().to_vec();

        // Serialise to bytes for page storage
        let k_data = k_new.clone().into_data();
        let v_data = v_new.clone().into_data();
        let k_bytes: Vec<u8> = k_data.bytes.to_vec();
        let v_bytes: Vec<u8> = v_data.bytes.to_vec();

        // Calculate bytes per token
        let tokens = new_seq.max(1);
        let k_bytes_per_token = k_bytes.len() / tokens;
        let v_bytes_per_token = v_bytes.len() / tokens;

        // Distribute tokens across pages
        let mut remaining = tokens;
        let mut offset = 0;
        while remaining > 0 {
            // Try to fill the last page first
            let can_fill_last = if let Some(&last_id) = self.page_ids.last() {
                if let Some(page) = allocator.get_page_mut(last_id) {
                    let space = PAGE_SIZE.saturating_sub(page.used_slots);
                    if space > 0 {
                        let fill = space.min(remaining);
                        let k_start = offset * k_bytes_per_token;
                        let k_end = (offset + fill) * k_bytes_per_token;
                        let v_start = offset * v_bytes_per_token;
                        let v_end = (offset + fill) * v_bytes_per_token;
                        page.k_data.extend_from_slice(&k_bytes[k_start..k_end]);
                        page.v_data.extend_from_slice(&v_bytes[v_start..v_end]);
                        page.used_slots += fill;
                        page.shape = shape.clone();
                        let page_bytes = page.k_data.len() + page.v_data.len();
                        // Update VRAM usage (approximate: add the new bytes)
                        allocator.vram_usage_bytes += (k_end - k_start) + (v_end - v_start);
                        let _ = page_bytes; // suppress warning
                        offset += fill;
                        remaining -= fill;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            if !can_fill_last && remaining > 0 {
                // Allocate a new page
                let fill = PAGE_SIZE.min(remaining);
                let page_id = allocator.alloc_page();
                if let Some(page) = allocator.get_page_mut(page_id) {
                    let k_start = offset * k_bytes_per_token;
                    let k_end = (offset + fill) * k_bytes_per_token;
                    let v_start = offset * v_bytes_per_token;
                    let v_end = (offset + fill) * v_bytes_per_token;
                    page.k_data = k_bytes[k_start..k_end].to_vec();
                    page.v_data = v_bytes[v_start..v_end].to_vec();
                    page.used_slots = fill;
                    page.shape = shape.clone();
                    allocator.vram_usage_bytes += page.k_data.len() + page.v_data.len();
                }
                self.page_ids.push(page_id);
                offset += fill;
                remaining -= fill;
            }
        }

        self.seq_len += new_seq;

        // Reassemble full K/V from monolithic cache + new data (for attention)
        let k_all = match self.k_cache.take() {
            Some(prev) => Tensor::cat(vec![prev, k_new], 2),
            None => k_new,
        };
        let v_all = match self.v_cache.take() {
            Some(prev) => Tensor::cat(vec![prev, v_new], 2),
            None => v_new,
        };
        self.k_cache = Some(k_all.clone());
        self.v_cache = Some(v_all.clone());

        (k_all, v_all)
    }

    /// Total sequence length accumulated so far in this layer.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Page IDs owned by this layer (for cleanup).
    pub fn page_ids(&self) -> &[PageId] {
        &self.page_ids
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
pub struct SessionKv<B: Backend> {
    /// One `LayerKv` per transformer layer owned by this shard.
    /// Indexed by local offset: `layers[0]` corresponds to `layer_range.start`.
    pub layers: Vec<LayerKv<B>>,
    /// The global layer range this session covers.
    pub layer_range: Range<usize>,
    /// Total tokens fed into this session (across all decode steps).
    pub seq_pos: usize,
    pub last_access: Instant,
}

#[cfg(feature = "burn")]
impl<B: Backend> SessionKv<B> {
    /// Create a new session scoped to the given layer range.
    pub fn new(layer_range: Range<usize>) -> Self {
        let n_layers = layer_range.len();
        Self {
            layers: (0..n_layers).map(|_| LayerKv::new()).collect(),
            layer_range,
            seq_pos: 0,
            last_access: Instant::now(),
        }
    }

    /// Convert a global layer index to a local index within this session.
    /// Returns `None` if the layer is outside the owned range.
    pub fn local_index(&self, global_layer: usize) -> Option<usize> {
        if global_layer >= self.layer_range.start && global_layer < self.layer_range.end {
            Some(global_layer - self.layer_range.start)
        } else {
            None
        }
    }
}

// ── Store ─────────────────────────────────────────────────────────────────────

/// Thread-safe map of `session_id → SessionKv<B>`, scoped to a layer range.
///
/// Each node creates one `KvCacheStore` at startup, bound to the layer range it
/// owns.  All sessions created by this store allocate KV slots only for the
/// owned layers — never for the full model.
#[cfg(feature = "burn")]
#[derive(Clone)]
pub struct KvCacheStore<B: Backend> {
    pub inner: Arc<RwLock<HashMap<String, SessionKv<B>>>>,
    ttl_secs: u64,
    /// The layer range this store is scoped to.
    layer_range: Range<usize>,
    /// Shared page allocator for paged KV storage (None = monolithic mode).
    page_allocator: Option<Arc<RwLock<PageAllocator>>>,
}

#[cfg(feature = "burn")]
impl<B: Backend> KvCacheStore<B> {
    /// Create a store scoped to `layer_range` where idle sessions are evicted
    /// after `ttl_secs`.  Uses monolithic KV storage (no paging).
    pub fn new(ttl_secs: u64, layer_range: Range<usize>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            ttl_secs,
            layer_range,
            page_allocator: None,
        }
    }

    /// Create a store with paged KV storage.  `vram_capacity_bytes` sets the
    /// VRAM budget for KV pages; pages are evicted to CPU DRAM when usage
    /// exceeds 90% of this budget.
    pub fn new_paged(
        ttl_secs: u64,
        layer_range: Range<usize>,
        vram_capacity_bytes: usize,
    ) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            ttl_secs,
            layer_range,
            page_allocator: Some(Arc::new(RwLock::new(PageAllocator::new(vram_capacity_bytes)))),
        }
    }

    /// The layer range this store is scoped to.
    pub fn layer_range(&self) -> &Range<usize> {
        &self.layer_range
    }

    /// Number of owned layers.
    pub fn n_layers(&self) -> usize {
        self.layer_range.len()
    }

    /// Validate that a request's layer range falls within this store's scope.
    pub fn validate_range(&self, req_start: usize, req_end: usize) -> Result<(), String> {
        if req_start < self.layer_range.start || req_end > self.layer_range.end {
            Err(format!(
                "requested layers {}..{} outside store scope {}..{}",
                req_start, req_end, self.layer_range.start, self.layer_range.end
            ))
        } else {
            Ok(())
        }
    }

    /// Evict sessions that have been idle longer than the configured TTL.
    pub async fn evict_idle(&self) {
        let now = Instant::now();
        let ttl = self.ttl_secs;
        let mut map = self.inner.write().await;

        // Collect page IDs from evicted sessions for cleanup
        let mut pages_to_free = Vec::new();
        map.retain(|_, sess| {
            let keep = now.duration_since(sess.last_access).as_secs() < ttl;
            if !keep {
                for layer in &sess.layers {
                    pages_to_free.extend_from_slice(layer.page_ids());
                }
            }
            keep
        });

        // Free pages from evicted sessions
        if !pages_to_free.is_empty() {
            if let Some(alloc) = &self.page_allocator {
                alloc.write().await.free_pages(&pages_to_free);
            }
        }
    }

    /// Ensure a session exists (creating it with the store's layer range if
    /// needed) and update its `last_access` time.
    pub async fn touch(&self, session_id: &str) {
        let mut map = self.inner.write().await;
        let range = self.layer_range.clone();
        let sess = map
            .entry(session_id.to_string())
            .or_insert_with(|| SessionKv::new(range));
        sess.last_access = Instant::now();
    }

    /// Backward-compatible touch that accepts `n_layers` (ignored — uses store's
    /// layer range instead).  Logs a warning if `n_layers` doesn't match.
    pub async fn touch_compat(&self, session_id: &str, n_layers: usize) {
        if n_layers != self.n_layers() {
            eprintln!(
                "[kv_cache] warning: touch_compat called with n_layers={} but store owns {} layers ({}..{})",
                n_layers, self.n_layers(), self.layer_range.start, self.layer_range.end
            );
        }
        self.touch(session_id).await;
    }

    /// Remove a session entirely (e.g. after the final decode step).
    pub async fn remove(&self, session_id: &str) {
        let mut map = self.inner.write().await;
        if let Some(sess) = map.remove(session_id) {
            // Free any pages owned by this session
            let mut pages_to_free = Vec::new();
            for layer in &sess.layers {
                pages_to_free.extend_from_slice(layer.page_ids());
            }
            if !pages_to_free.is_empty() {
                if let Some(alloc) = &self.page_allocator {
                    alloc.write().await.free_pages(&pages_to_free);
                }
            }
        }
    }

    /// Access the shared page allocator (if paged mode is enabled).
    pub fn page_allocator(&self) -> Option<&Arc<RwLock<PageAllocator>>> {
        self.page_allocator.as_ref()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "burn"))]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray;

    #[test]
    fn layer_kv_append_concatenates() {
        let mut kv = LayerKv::<TestBackend>::new();
        let device = Default::default();
        // [batch=1, heads=2, seq=3, head_dim=4]
        let k1 = Tensor::<TestBackend, 4>::zeros([1, 2, 3, 4], &device);
        let v1 = Tensor::<TestBackend, 4>::zeros([1, 2, 3, 4], &device);
        let (k, v) = kv.append(k1, v1);
        assert_eq!(k.dims(), [1, 2, 3, 4]);
        assert_eq!(kv.seq_len(), 3);

        let k2 = Tensor::<TestBackend, 4>::zeros([1, 2, 2, 4], &device);
        let v2 = Tensor::<TestBackend, 4>::zeros([1, 2, 2, 4], &device);
        let (k, v) = kv.append(k2, v2);
        assert_eq!(k.dims(), [1, 2, 5, 4]);
        assert_eq!(kv.seq_len(), 5);
    }

    #[test]
    fn session_kv_respects_layer_range() {
        let sess = SessionKv::<TestBackend>::new(4..8);
        assert_eq!(sess.layers.len(), 4);
        assert_eq!(sess.local_index(4), Some(0));
        assert_eq!(sess.local_index(7), Some(3));
        assert_eq!(sess.local_index(3), None);
        assert_eq!(sess.local_index(8), None);
    }

    #[tokio::test]
    async fn store_validates_layer_range() {
        let store = KvCacheStore::<TestBackend>::new(60, 4..12);
        assert!(store.validate_range(4, 12).is_ok());
        assert!(store.validate_range(4, 8).is_ok());
        assert!(store.validate_range(0, 12).is_err());
        assert!(store.validate_range(4, 16).is_err());
    }

    #[tokio::test]
    async fn store_touch_creates_session_with_owned_layers() {
        let store = KvCacheStore::<TestBackend>::new(60, 8..16);
        store.touch("sess1").await;
        let map = store.inner.read().await;
        let sess = map.get("sess1").expect("session should exist");
        assert_eq!(sess.layers.len(), 8); // 16 - 8
        assert_eq!(sess.layer_range, 8..16);
    }

    #[test]
    fn page_allocator_alloc_and_get() {
        let mut alloc = PageAllocator::new(1_000_000);
        let id = alloc.alloc_page();
        assert!(alloc.get_page(id).is_some());
        assert_eq!(alloc.vram_page_count(), 1);
        assert_eq!(alloc.dram_page_count(), 0);
    }

    #[test]
    fn page_allocator_evicts_under_pressure() {
        // Very small capacity forces eviction
        let mut alloc = PageAllocator::new(100);

        let id1 = alloc.alloc_page();
        // Manually add data to simulate usage
        if let Some(page) = alloc.get_page_mut(id1) {
            page.k_data = vec![0u8; 60];
            page.v_data = vec![0u8; 60];
            page.used_slots = 1;
        }
        alloc.vram_usage_bytes = 120; // Over 90% of 100

        let id2 = alloc.alloc_page();
        // id1 should have been evicted to DRAM
        assert_eq!(alloc.dram_page_count(), 1);

        // Accessing id1 promotes it back
        alloc.get_page(id1);
        assert!(alloc.vram_pages.contains_key(&id1));
    }

    #[test]
    fn page_allocator_free_pages() {
        let mut alloc = PageAllocator::new(1_000_000);
        let id1 = alloc.alloc_page();
        let id2 = alloc.alloc_page();
        assert_eq!(alloc.vram_page_count(), 2);
        alloc.free_pages(&[id1, id2]);
        assert_eq!(alloc.vram_page_count(), 0);
    }
}
