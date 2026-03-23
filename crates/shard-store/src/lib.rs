//! Content-addressed on-disk cache for model weight shards.
//!
//! Each shard is stored as `{cache_dir}/{hex_hash}.shard`.
//! Uses blake3 for content addressing — the same hash function used by iroh-blobs —
//! so hashes produced here are directly compatible with iroh-blobs if added later.
//!
//! # Design
//!
//! - `ShardStore::add(data)` → hashes bytes, writes file, returns hash
//! - `ShardStore::get(hash)` → returns cached bytes or None
//! - `ShardStore::has(hash)` → quick existence check
//! - `ShardRegistry` → tracks which layer ranges are cached (model → shard list)

use std::{
    io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

/// 32-byte blake3 hash, hex-encoded as the filename stem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hash([u8; 32]);

impl Hash {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Decode a 64-character lowercase hex string into a `Hash`.
    pub fn from_hex(s: &str) -> anyhow::Result<Self> {
        let bytes = hex::decode(s)
            .map_err(|e| anyhow::anyhow!("invalid hex string: {e}"))?;
        if bytes.len() != 32 {
            return Err(anyhow::anyhow!(
                "expected 32-byte (64-char hex) hash, got {} bytes",
                bytes.len()
            ));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Hash(arr))
    }
}

impl std::fmt::Display for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl From<blake3::Hash> for Hash {
    fn from(h: blake3::Hash) -> Self {
        Hash(*h.as_bytes())
    }
}

/// Metadata for a single stored shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMeta {
    pub hash: Hash,
    pub model: String,
    pub layer_start: u32,
    pub layer_end: u32,
    /// Uncompressed size in bytes.
    pub size_bytes: usize,
}

/// Content-addressed on-disk cache for model weight shards.
///
/// Files are stored as `{cache_dir}/{hex_hash}.shard`.
/// A JSON index at `{cache_dir}/index.json` tracks shard metadata.
pub struct ShardStore {
    cache_dir: PathBuf,
}

impl ShardStore {
    /// Open (or create) a shard store rooted at `cache_dir`.
    pub fn open(cache_dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = cache_dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { cache_dir: dir })
    }

    /// Store raw bytes and return their blake3 hash.
    ///
    /// If the content is already cached (same hash → same file exists) this is a no-op.
    pub fn add(&self, data: &[u8]) -> io::Result<Hash> {
        let hash = Hash::from(blake3::hash(data));
        let path = self.shard_path(&hash);
        if !path.exists() {
            std::fs::write(&path, data)?;
        }
        Ok(hash)
    }

    /// Store a shard with full metadata (model name, layer range).
    /// Updates the index so the shard can be looked up by model + layer range later.
    pub fn add_shard(
        &self,
        data: &[u8],
        model: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> anyhow::Result<Hash> {
        let hash = self.add(data)?;
        let meta = ShardMeta {
            hash,
            model: model.to_string(),
            layer_start,
            layer_end,
            size_bytes: data.len(),
        };
        self.write_meta(&meta)?;
        Ok(hash)
    }

    /// Retrieve bytes by hash. Returns `None` if not cached locally.
    pub fn get(&self, hash: &Hash) -> io::Result<Option<Vec<u8>>> {
        let path = self.shard_path(hash);
        if path.exists() {
            Ok(Some(std::fs::read(path)?))
        } else {
            Ok(None)
        }
    }

    /// Return the on-disk path for a cached shard, if it exists.
    ///
    /// Callers can mmap this path directly instead of loading into memory via `get()`.
    pub fn shard_file_path(&self, hash: &Hash) -> Option<std::path::PathBuf> {
        let path = self.shard_path(hash);
        if path.exists() { Some(path) } else { None }
    }

    /// Returns `true` if the shard for `hash` is already cached locally.
    pub fn has(&self, hash: &Hash) -> bool {
        self.shard_path(hash).exists()
    }

    /// Look up shards by model and layer range.
    pub fn find(
        &self,
        model: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> anyhow::Result<Option<ShardMeta>> {
        let index = self.load_index()?;
        Ok(index
            .into_iter()
            .find(|m| m.model == model && m.layer_start == layer_start && m.layer_end == layer_end))
    }

    /// List all shard metadata entries in the index.
    pub fn list(&self) -> anyhow::Result<Vec<ShardMeta>> {
        self.load_index()
    }

    /// Remove a shard from disk and the index.
    pub fn evict(&self, hash: &Hash) -> anyhow::Result<()> {
        let path = self.shard_path(hash);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        let mut index = self.load_index()?;
        index.retain(|m| &m.hash != hash);
        self.save_index(&index)?;
        Ok(())
    }

    /// Total bytes used by cached shards (sum of file sizes on disk).
    pub fn disk_usage_bytes(&self) -> io::Result<u64> {
        let mut total = 0u64;
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) == Some("shard") {
                total += entry.metadata()?.len();
            }
        }
        Ok(total)
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    // --- internal helpers ---

    fn shard_path(&self, hash: &Hash) -> PathBuf {
        self.cache_dir.join(format!("{}.shard", hash.to_hex()))
    }

    fn index_path(&self) -> PathBuf {
        self.cache_dir.join("index.json")
    }

    fn load_index(&self) -> anyhow::Result<Vec<ShardMeta>> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    fn save_index(&self, index: &[ShardMeta]) -> anyhow::Result<()> {
        let raw = serde_json::to_string_pretty(index)?;
        std::fs::write(self.index_path(), raw)?;
        Ok(())
    }

    fn write_meta(&self, meta: &ShardMeta) -> anyhow::Result<()> {
        let mut index = self.load_index()?;
        // Replace existing entry for same model+layers, or append.
        if let Some(existing) = index
            .iter_mut()
            .find(|m| m.model == meta.model && m.layer_start == meta.layer_start && m.layer_end == meta.layer_end)
        {
            *existing = meta.clone();
        } else {
            index.push(meta.clone());
        }
        self.save_index(&index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn round_trip_add_get() {
        let dir = std::env::temp_dir().join("shard_store_test_rt");
        let _ = fs::remove_dir_all(&dir);
        let store = ShardStore::open(&dir).unwrap();

        let data = b"fake model weights for layer 0-3";
        let hash = store.add(data).unwrap();

        assert!(store.has(&hash));
        let retrieved = store.get(&hash).unwrap().unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn dedup_identical_content() {
        let dir = std::env::temp_dir().join("shard_store_test_dedup");
        let _ = fs::remove_dir_all(&dir);
        let store = ShardStore::open(&dir).unwrap();

        let data = b"identical bytes";
        let h1 = store.add(data).unwrap();
        let h2 = store.add(data).unwrap(); // second call — same hash, no duplicate file
        assert_eq!(h1, h2);

        let count = fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .and_then(|x| x.to_str())
                    == Some("shard")
            })
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn index_find_and_evict() {
        let dir = std::env::temp_dir().join("shard_store_test_index");
        let _ = fs::remove_dir_all(&dir);
        let store = ShardStore::open(&dir).unwrap();

        let data = b"layer weights";
        let hash = store.add_shard(data, "my-model", 0, 7).unwrap();

        let found = store.find("my-model", 0, 7).unwrap().unwrap();
        assert_eq!(found.hash, hash);
        assert_eq!(found.layer_start, 0);
        assert_eq!(found.layer_end, 7);

        store.evict(&hash).unwrap();
        assert!(!store.has(&hash));
        assert!(store.find("my-model", 0, 7).unwrap().is_none());
    }

    #[test]
    fn disk_usage() {
        let dir = std::env::temp_dir().join("shard_store_test_usage");
        let _ = fs::remove_dir_all(&dir);
        let store = ShardStore::open(&dir).unwrap();

        let data = vec![0u8; 1024];
        store.add(&data).unwrap();

        let usage = store.disk_usage_bytes().unwrap();
        assert_eq!(usage, 1024);
    }
}
