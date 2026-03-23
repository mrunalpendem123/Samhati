//! Weight cache integration — bridges `shard_store::ShardStore` into the inference path.
//!
//! At node startup, weights for the assigned layer range are loaded into the
//! `ShardStore` (content-addressed, blake3-hashed).  Subsequent inference
//! requests read weights from the store (memory-mapped) instead of re-reading
//! from the original safetensors files on every forward pass.

use anyhow::{anyhow, Result};
use shard_store::{Hash, ShardStore};
use std::path::{Path, PathBuf};

/// A wrapper around `ShardStore` tailored for the inference hot-path.
///
/// Provides a simple `get_or_load()` API: if the shard is already cached by
/// hash, return the on-disk path for mmap; otherwise, ingest the bytes, cache
/// them, and return the path.
pub struct WeightCache {
    store: ShardStore,
}

impl WeightCache {
    /// Open (or create) a weight cache at `cache_dir`.
    pub fn open(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let store = ShardStore::open(cache_dir)
            .map_err(|e| anyhow!("failed to open weight cache: {e}"))?;
        Ok(Self { store })
    }

    /// Return the on-disk path for a cached shard, loading it first if needed.
    ///
    /// `data_fn` is called lazily only when the shard is not yet cached.
    pub fn get_or_load<F>(
        &self,
        model: &str,
        layer_start: u32,
        layer_end: u32,
        data_fn: F,
    ) -> Result<PathBuf>
    where
        F: FnOnce() -> Result<Vec<u8>>,
    {
        // Check if already cached by model + layer range.
        if let Some(meta) = self.store.find(model, layer_start, layer_end)? {
            if let Some(path) = self.store.shard_file_path(&meta.hash) {
                return Ok(path);
            }
        }

        // Not cached — load bytes and ingest.
        let data = data_fn()?;
        let hash = self.store.add_shard(&data, model, layer_start, layer_end)?;
        self.store
            .shard_file_path(&hash)
            .ok_or_else(|| anyhow!("shard file missing immediately after add"))
    }

    /// Retrieve raw bytes by hash (for non-mmap use cases).
    pub fn get_bytes(&self, hash: &Hash) -> Result<Option<Vec<u8>>> {
        self.store
            .get(hash)
            .map_err(|e| anyhow!("weight cache read error: {e}"))
    }

    /// Return the file path for a cached shard by hash (for mmap).
    pub fn shard_path(&self, hash: &Hash) -> Option<PathBuf> {
        self.store.shard_file_path(hash)
    }

    /// Check if a shard is cached.
    pub fn has(&self, hash: &Hash) -> bool {
        self.store.has(hash)
    }

    /// List all cached shard hashes (hex strings) for gossip announcements.
    pub fn cached_hashes(&self) -> Vec<String> {
        self.store
            .list()
            .unwrap_or_default()
            .iter()
            .map(|m| m.hash.to_hex())
            .collect()
    }

    /// Access the underlying store.
    pub fn inner(&self) -> &ShardStore {
        &self.store
    }

    /// Cache directory path.
    pub fn cache_dir(&self) -> &Path {
        self.store.cache_dir()
    }
}
