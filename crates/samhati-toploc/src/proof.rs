use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Serde helper for `[u8; 64]` (Ed25519 signatures).
/// Serializes as a hex string in human-readable formats, raw bytes otherwise.
mod serde_sig {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            serializer.serialize_str(&hex)
        } else {
            serializer.serialize_bytes(bytes)
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let s = String::deserialize(deserializer)?;
            let bytes = hex_decode(&s).map_err(serde::de::Error::custom)?;
            if bytes.len() != 64 {
                return Err(serde::de::Error::custom(format!(
                    "expected 64 bytes, got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0u8; 64];
            arr.copy_from_slice(&bytes);
            Ok(arr)
        } else {
            let bytes = <Vec<u8>>::deserialize(deserializer)?;
            if bytes.len() != 64 {
                return Err(serde::de::Error::custom(format!(
                    "expected 64 bytes, got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0u8; 64];
            arr.copy_from_slice(&bytes);
            Ok(arr)
        }
    }

    fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
        if s.len() % 2 != 0 {
            return Err("odd-length hex string".to_string());
        }
        (0..s.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&s[i..i + 2], 16)
                    .map_err(|e| format!("invalid hex at position {}: {}", i, e))
            })
            .collect()
    }
}

/// Top-K logits captured from a single token generation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogits {
    /// The token that was actually generated (argmax or sampled).
    pub token_id: u32,
    /// (token_id, logit_value) pairs sorted descending by logit value, K=8 by default.
    pub top_k: Vec<(u32, f32)>,
}

impl TokenLogits {
    /// Serialize deterministically for hashing.
    /// Sorts by token_id (not logit value) so the representation is canonical,
    /// and uses big-endian fixed-width encoding for reproducibility.
    pub fn to_deterministic_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Write the generated token id
        buf.extend_from_slice(&self.token_id.to_be_bytes());
        // Sort top_k by token_id for determinism
        let mut sorted: Vec<(u32, f32)> = self.top_k.clone();
        sorted.sort_by_key(|(tid, _)| *tid);
        // Write count
        buf.extend_from_slice(&(sorted.len() as u32).to_be_bytes());
        for (tid, logit) in &sorted {
            buf.extend_from_slice(&tid.to_be_bytes());
            buf.extend_from_slice(&logit.to_be_bytes());
        }
        buf
    }
}

/// A TOPLOC proof for a chunk of 32 tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitChunkProof {
    /// Index of this chunk (0-based).
    pub chunk_index: u32,
    /// BLAKE3 hash of the serialized top-K logits for the tokens in this chunk.
    pub hash: [u8; 32],
}

/// Complete TOPLOC proof for an inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToplocProof {
    /// BLAKE3 hash of the model weights identifier.
    pub model_hash: [u8; 32],
    /// Number of tokens this proof covers.
    pub token_count: u32,
    /// Per-chunk logit hashes.
    pub chunk_proofs: Vec<LogitChunkProof>,
    /// Unix timestamp (seconds since epoch).
    pub timestamp: u64,
    /// 32-byte Ed25519 public key of the node that produced this proof.
    /// Binds the proof to a specific node identity for verification.
    #[serde(default)]
    pub node_pubkey: [u8; 32],
    /// Ed25519 signature over (model_hash || token_count || chunk hashes || timestamp || node_pubkey).
    #[serde(with = "serde_sig")]
    pub node_signature: [u8; 64],
}

impl ToplocProof {
    /// Serialize to bytes for wire transport.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ToplocProof serialization should not fail")
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).context("failed to deserialize ToplocProof")
    }

    /// Size in bytes of the serialized form.
    pub fn byte_size(&self) -> usize {
        self.to_bytes().len()
    }

    /// Compute the proof hash (for on-chain storage).
    /// This is the BLAKE3 hash of the canonical byte representation.
    pub fn proof_hash(&self) -> [u8; 32] {
        *blake3::hash(&self.to_bytes()).as_bytes()
    }

    /// Produce the signable message: the bytes that are signed / verified.
    pub fn signable_message(&self) -> Vec<u8> {
        let mut msg = Vec::new();
        msg.extend_from_slice(&self.model_hash);
        msg.extend_from_slice(&self.token_count.to_be_bytes());
        for chunk in &self.chunk_proofs {
            msg.extend_from_slice(&chunk.chunk_index.to_be_bytes());
            msg.extend_from_slice(&chunk.hash);
        }
        msg.extend_from_slice(&self.timestamp.to_be_bytes());
        msg.extend_from_slice(&self.node_pubkey);
        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_logits_deterministic_bytes_is_stable() {
        let tl = TokenLogits {
            token_id: 42,
            top_k: vec![(10, 3.5), (5, 4.0), (20, 2.0)],
        };
        let b1 = tl.to_deterministic_bytes();
        let b2 = tl.to_deterministic_bytes();
        assert_eq!(b1, b2);
    }

    #[test]
    fn token_logits_deterministic_regardless_of_order() {
        let tl_a = TokenLogits {
            token_id: 42,
            top_k: vec![(10, 3.5), (5, 4.0), (20, 2.0)],
        };
        let tl_b = TokenLogits {
            token_id: 42,
            top_k: vec![(20, 2.0), (10, 3.5), (5, 4.0)],
        };
        assert_eq!(tl_a.to_deterministic_bytes(), tl_b.to_deterministic_bytes());
    }

    #[test]
    fn proof_serialization_roundtrip() {
        let proof = ToplocProof {
            model_hash: [1u8; 32],
            token_count: 64,
            chunk_proofs: vec![
                LogitChunkProof { chunk_index: 0, hash: [2u8; 32] },
                LogitChunkProof { chunk_index: 1, hash: [3u8; 32] },
            ],
            timestamp: 1700000000,
            node_pubkey: [5u8; 32],
            node_signature: [4u8; 64],
        };
        let bytes = proof.to_bytes();
        let decoded = ToplocProof::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.model_hash, proof.model_hash);
        assert_eq!(decoded.token_count, proof.token_count);
        assert_eq!(decoded.chunk_proofs.len(), 2);
        assert_eq!(decoded.timestamp, proof.timestamp);
        assert_eq!(decoded.node_pubkey, proof.node_pubkey);
        assert_eq!(decoded.node_signature, proof.node_signature);
    }

    #[test]
    fn proof_hash_is_deterministic() {
        let proof = ToplocProof {
            model_hash: [1u8; 32],
            token_count: 32,
            chunk_proofs: vec![LogitChunkProof { chunk_index: 0, hash: [9u8; 32] }],
            timestamp: 1700000000,
            node_pubkey: [0u8; 32],
            node_signature: [0u8; 64],
        };
        let h1 = proof.proof_hash();
        let h2 = proof.proof_hash();
        assert_eq!(h1, h2);
    }
}
