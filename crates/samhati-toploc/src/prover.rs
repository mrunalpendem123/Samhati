use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use ed25519_dalek::{SigningKey, Signer};
use tracing::debug;

use crate::proof::{LogitChunkProof, TokenLogits, ToplocProof};
use crate::types::{model_id_to_hash, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K};

/// Proof generator — runs on the inference node during token generation.
pub struct ToplocProver {
    model_hash: [u8; 32],
    top_k: usize,
    chunk_size: usize,
    signing_key: SigningKey,
    /// Accumulated logits for the current inference run.
    pending_logits: Vec<TokenLogits>,
}

impl ToplocProver {
    /// Create a new prover for the given model.
    ///
    /// `model_id` — human-readable model identifier (e.g. "llama-3-8b").
    /// `signing_key_bytes` — 32-byte Ed25519 private key seed.
    pub fn new(model_id: &str, signing_key_bytes: [u8; 32]) -> Self {
        let model_hash = model_id_to_hash(model_id);
        let signing_key = SigningKey::from_bytes(&signing_key_bytes);
        debug!(model_id, "ToplocProver created");
        Self {
            model_hash,
            top_k: DEFAULT_TOP_K,
            chunk_size: DEFAULT_CHUNK_SIZE,
            signing_key,
            pending_logits: Vec::new(),
        }
    }

    /// Override the default top-K value.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Override the default chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Record the top-K logits produced after generating a single token.
    /// Call this once per forward-pass step.
    pub fn record_token(&mut self, token_logits: TokenLogits) {
        self.pending_logits.push(token_logits);
    }

    /// Finalize the current inference run and produce a signed TOPLOC proof.
    pub fn finalize(&mut self) -> Result<ToplocProof> {
        if self.pending_logits.is_empty() {
            bail!("no token logits recorded — cannot produce proof");
        }

        let token_count = self.pending_logits.len() as u32;

        // Build chunk proofs
        let mut chunk_proofs = Vec::new();
        for (i, chunk) in self.pending_logits.chunks(self.chunk_size).enumerate() {
            let hash = Self::hash_chunk(chunk);
            chunk_proofs.push(LogitChunkProof {
                chunk_index: i as u32,
                hash,
            });
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_secs();

        // Build an unsigned proof to compute the signable message
        let mut proof = ToplocProof {
            model_hash: self.model_hash,
            token_count,
            chunk_proofs,
            timestamp,
            node_signature: [0u8; 64],
        };

        // Sign
        let msg = proof.signable_message();
        let sig = self.signing_key.sign(&msg);
        proof.node_signature = sig.to_bytes();

        debug!(token_count, chunks = proof.chunk_proofs.len(), "TOPLOC proof finalized");
        Ok(proof)
    }

    /// Discard accumulated logits and prepare for the next inference run.
    pub fn reset(&mut self) {
        self.pending_logits.clear();
    }

    /// Number of tokens recorded so far.
    pub fn recorded_count(&self) -> usize {
        self.pending_logits.len()
    }

    /// Compute the BLAKE3 hash for a chunk of token logits.
    /// Uses deterministic serialization (token_id–sorted, big-endian, fixed-width f32).
    pub(crate) fn hash_chunk(logits: &[TokenLogits]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for tl in logits {
            hasher.update(&tl.to_deterministic_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    fn random_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        key
    }

    fn make_token_logits(token_id: u32) -> TokenLogits {
        TokenLogits {
            token_id,
            top_k: (0..8)
                .map(|i| (token_id + i, 10.0 - i as f32))
                .collect(),
        }
    }

    #[test]
    fn prover_64_tokens_produces_2_chunks() {
        let key = random_key();
        let mut prover = ToplocProver::new("test-model", key);
        for i in 0..64 {
            prover.record_token(make_token_logits(i));
        }
        let proof = prover.finalize().unwrap();
        assert_eq!(proof.token_count, 64);
        assert_eq!(proof.chunk_proofs.len(), 2);
        assert_eq!(proof.chunk_proofs[0].chunk_index, 0);
        assert_eq!(proof.chunk_proofs[1].chunk_index, 1);
    }

    #[test]
    fn prover_33_tokens_produces_2_chunks() {
        let key = random_key();
        let mut prover = ToplocProver::new("test-model", key);
        for i in 0..33 {
            prover.record_token(make_token_logits(i));
        }
        let proof = prover.finalize().unwrap();
        assert_eq!(proof.token_count, 33);
        assert_eq!(proof.chunk_proofs.len(), 2); // ceil(33/32) = 2
    }

    #[test]
    fn prover_finalize_empty_fails() {
        let key = random_key();
        let mut prover = ToplocProver::new("test-model", key);
        assert!(prover.finalize().is_err());
    }

    #[test]
    fn prover_reset_clears_state() {
        let key = random_key();
        let mut prover = ToplocProver::new("test-model", key);
        prover.record_token(make_token_logits(0));
        assert_eq!(prover.recorded_count(), 1);
        prover.reset();
        assert_eq!(prover.recorded_count(), 0);
    }

    #[test]
    fn chunk_hash_is_deterministic() {
        let logits: Vec<TokenLogits> = (0..32).map(make_token_logits).collect();
        let h1 = ToplocProver::hash_chunk(&logits);
        let h2 = ToplocProver::hash_chunk(&logits);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_logits_produce_different_hashes() {
        let logits_a: Vec<TokenLogits> = (0..32).map(make_token_logits).collect();
        let logits_b: Vec<TokenLogits> = (100..132).map(make_token_logits).collect();
        assert_ne!(
            ToplocProver::hash_chunk(&logits_a),
            ToplocProver::hash_chunk(&logits_b)
        );
    }

    #[test]
    fn proof_model_hash_matches_expected() {
        let key = random_key();
        let model_id = "llama-3-8b";
        let mut prover = ToplocProver::new(model_id, key);
        for i in 0..32 {
            prover.record_token(make_token_logits(i));
        }
        let proof = prover.finalize().unwrap();
        let expected_hash = crate::types::model_id_to_hash(model_id);
        assert_eq!(proof.model_hash, expected_hash);
    }

    #[test]
    fn proof_signature_is_non_zero() {
        let key = random_key();
        let mut prover = ToplocProver::new("m", key);
        for i in 0..32 {
            prover.record_token(make_token_logits(i));
        }
        let proof = prover.finalize().unwrap();
        assert_ne!(proof.node_signature, [0u8; 64]);
    }
}
