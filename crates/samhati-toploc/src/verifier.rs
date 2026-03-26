use std::collections::HashMap;

use ed25519_dalek::{Signature, VerifyingKey, Verifier};
use serde::{Deserialize, Serialize};

use crate::proof::ToplocProof;
use crate::types::model_id_to_hash;

/// Result of verifying a TOPLOC proof.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationResult {
    Valid,
    InvalidModelHash {
        expected: [u8; 32],
        got: [u8; 32],
    },
    TokenCountMismatch {
        expected: u32,
        got: u32,
    },
    InvalidSignature,
    /// No public keys have been registered — cannot verify signatures.
    NoKeysRegistered,
    MissingChunks {
        expected: usize,
        got: usize,
    },
    UnknownModel(String),
    InvalidProofStructure(String),
    /// Proof timestamp is too far from current time.
    StaleProof {
        proof_ts: u64,
        now_ts: u64,
        max_age_secs: u64,
    },
    /// Proof node_pubkey does not match any registered key.
    UnknownNode,
}

impl VerificationResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, VerificationResult::Valid)
    }
}

/// Trait for TOPLOC verification — allows mocking in tests.
pub trait ToplocVerifier: Send + Sync {
    /// Verify a proof is structurally valid and matches the claimed model.
    fn verify(
        &self,
        proof: &ToplocProof,
        claimed_model_id: &str,
        output_token_count: u32,
    ) -> VerificationResult;
}

/// Production verifier that checks model hash, token count, chunk count, and Ed25519 signature.
pub struct ToplocVerifierImpl {
    /// Known model hashes: model_id -> expected BLAKE3 hash.
    known_models: HashMap<String, [u8; 32]>,
    /// Known node public keys: keyed by 32-byte pubkey for direct lookup.
    known_public_keys: HashMap<[u8; 32], VerifyingKey>,
    /// Maximum allowed age of a proof in seconds (0 = no freshness check).
    max_proof_age_secs: u64,
}

impl ToplocVerifierImpl {
    /// Create a new verifier with no known models or keys.
    pub fn new() -> Self {
        Self {
            known_models: HashMap::new(),
            known_public_keys: HashMap::new(),
            max_proof_age_secs: 300, // 5 minutes default
        }
    }

    /// Set the maximum allowed age of a proof (0 = disable freshness check).
    pub fn with_max_proof_age(mut self, secs: u64) -> Self {
        self.max_proof_age_secs = secs;
        self
    }

    /// Register a model so its hash is accepted during verification.
    pub fn register_model(&mut self, model_id: &str) {
        let hash = model_id_to_hash(model_id);
        self.known_models.insert(model_id.to_string(), hash);
    }

    /// Register a node public key for signature verification.
    pub fn register_public_key(&mut self, public_key: VerifyingKey) {
        self.known_public_keys.insert(*public_key.as_bytes(), public_key);
    }

    /// Register a node public key from raw 32 bytes.
    pub fn register_public_key_bytes(&mut self, bytes: [u8; 32]) -> Result<(), ed25519_dalek::SignatureError> {
        let vk = VerifyingKey::from_bytes(&bytes)?;
        self.known_public_keys.insert(bytes, vk);
        Ok(())
    }

    /// Verify the Ed25519 signature on a proof against the node's declared public key.
    fn verify_signature(&self, proof: &ToplocProof) -> VerificationResult {
        // Look up the specific key declared in the proof
        let vk = match self.known_public_keys.get(&proof.node_pubkey) {
            Some(vk) => vk,
            None => return VerificationResult::UnknownNode,
        };
        let msg = proof.signable_message();
        let sig = Signature::from_bytes(&proof.node_signature);
        if vk.verify(&msg, &sig).is_ok() {
            VerificationResult::Valid
        } else {
            VerificationResult::InvalidSignature
        }
    }
}

impl Default for ToplocVerifierImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl ToplocVerifier for ToplocVerifierImpl {
    fn verify(
        &self,
        proof: &ToplocProof,
        claimed_model_id: &str,
        output_token_count: u32,
    ) -> VerificationResult {
        // 1. Reject if no public keys registered — cannot verify anything
        if self.known_public_keys.is_empty() {
            return VerificationResult::NoKeysRegistered;
        }

        // 2. Check model is known
        let expected_hash = match self.known_models.get(claimed_model_id) {
            Some(h) => *h,
            None => return VerificationResult::UnknownModel(claimed_model_id.to_string()),
        };

        // 3. Check model hash
        if proof.model_hash != expected_hash {
            return VerificationResult::InvalidModelHash {
                expected: expected_hash,
                got: proof.model_hash,
            };
        }

        // 4. Check token count
        if proof.token_count != output_token_count {
            return VerificationResult::TokenCountMismatch {
                expected: output_token_count,
                got: proof.token_count,
            };
        }

        // 5. Check chunk count = ceil(token_count / 32)
        let chunk_size = 32usize;
        let expected_chunks = (proof.token_count as usize + chunk_size - 1) / chunk_size;
        if proof.chunk_proofs.len() != expected_chunks {
            return VerificationResult::MissingChunks {
                expected: expected_chunks,
                got: proof.chunk_proofs.len(),
            };
        }

        // 6. Validate chunk indices are sequential
        for (i, chunk) in proof.chunk_proofs.iter().enumerate() {
            if chunk.chunk_index != i as u32 {
                return VerificationResult::InvalidProofStructure(format!(
                    "chunk index mismatch: expected {}, got {}",
                    i, chunk.chunk_index
                ));
            }
        }

        // 7. Check proof freshness
        if self.max_proof_age_secs > 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let age = now.saturating_sub(proof.timestamp);
            if age > self.max_proof_age_secs {
                return VerificationResult::StaleProof {
                    proof_ts: proof.timestamp,
                    now_ts: now,
                    max_age_secs: self.max_proof_age_secs,
                };
            }
        }

        // 8. Verify Ed25519 signature against the proof's declared node_pubkey
        let sig_result = self.verify_signature(proof);
        if !sig_result.is_valid() {
            return sig_result;
        }

        VerificationResult::Valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::TokenLogits;
    use crate::prover::ToplocProver;
    use ed25519_dalek::{Signer, SigningKey};
    use rand::RngCore;

    fn random_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        key
    }

    fn make_token_logits(token_id: u32) -> TokenLogits {
        TokenLogits {
            token_id,
            top_k: (0..8).map(|i| (token_id + i, 10.0 - i as f32)).collect(),
        }
    }

    fn make_valid_proof(model_id: &str, key_bytes: [u8; 32], n_tokens: u32) -> ToplocProof {
        let mut prover = ToplocProver::new(model_id, key_bytes);
        for i in 0..n_tokens {
            prover.record_token(make_token_logits(i));
        }
        prover.finalize().unwrap()
    }

    #[test]
    fn valid_proof_passes_verification() {
        let key_bytes = random_key();
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        let model_id = "test-model";
        let proof = make_valid_proof(model_id, key_bytes, 64);

        let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(0); // disable freshness for unit test
        verifier.register_model(model_id);
        verifier.register_public_key(verifying_key);

        let result = verifier.verify(&proof, model_id, 64);
        assert_eq!(result, VerificationResult::Valid);
    }

    #[test]
    fn wrong_model_hash_fails() {
        let key_bytes = random_key();
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        let proof = make_valid_proof("model-a", key_bytes, 32);

        let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(0);
        verifier.register_model("model-a");
        verifier.register_model("model-b");
        verifier.register_public_key(verifying_key);

        // Claim it was model-b but proof was generated for model-a
        let result = verifier.verify(&proof, "model-b", 32);
        assert!(matches!(result, VerificationResult::InvalidModelHash { .. }));
    }

    #[test]
    fn wrong_token_count_fails() {
        let key_bytes = random_key();
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        let proof = make_valid_proof("m", key_bytes, 64);

        let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(0);
        verifier.register_model("m");
        verifier.register_public_key(verifying_key);

        // Claim 128 tokens but proof has 64
        let result = verifier.verify(&proof, "m", 128);
        assert!(matches!(result, VerificationResult::TokenCountMismatch { .. }));
    }

    #[test]
    fn unknown_model_fails() {
        let key_bytes = random_key();
        let signing_key = SigningKey::from_bytes(&key_bytes);

        let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(0);
        verifier.register_public_key(signing_key.verifying_key());
        // no models registered

        let proof = make_valid_proof("m", key_bytes, 32);
        let result = verifier.verify(&proof, "m", 32);
        assert!(matches!(result, VerificationResult::UnknownModel(_)));
    }

    #[test]
    fn invalid_signature_fails() {
        let key_bytes = random_key();
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let mut proof = make_valid_proof("m", key_bytes, 32);

        // Tamper with the signature
        proof.node_signature[0] ^= 0xFF;
        proof.node_signature[1] ^= 0xFF;

        let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(0);
        verifier.register_model("m");
        verifier.register_public_key(signing_key.verifying_key());

        let result = verifier.verify(&proof, "m", 32);
        assert_eq!(result, VerificationResult::InvalidSignature);
    }

    #[test]
    fn unknown_node_pubkey_fails() {
        let key_bytes = random_key();
        let proof = make_valid_proof("m", key_bytes, 32);

        // Register a different key than the one that signed the proof
        let other_key = random_key();
        let other_signing = SigningKey::from_bytes(&other_key);

        let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(0);
        verifier.register_model("m");
        verifier.register_public_key(other_signing.verifying_key());

        let result = verifier.verify(&proof, "m", 32);
        assert_eq!(result, VerificationResult::UnknownNode);
    }

    #[test]
    fn missing_chunks_fails() {
        let key_bytes = random_key();
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        let mut proof = make_valid_proof("m", key_bytes, 64);
        // Remove a chunk
        proof.chunk_proofs.pop();

        let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(0);
        verifier.register_model("m");
        verifier.register_public_key(verifying_key);

        let result = verifier.verify(&proof, "m", 64);
        assert!(matches!(result, VerificationResult::MissingChunks { .. }));
    }

    #[test]
    fn verification_without_public_keys_rejects() {
        let key_bytes = random_key();
        let proof = make_valid_proof("m", key_bytes, 32);

        let mut verifier = ToplocVerifierImpl::new();
        verifier.register_model("m");
        // No public keys registered — must reject

        let result = verifier.verify(&proof, "m", 32);
        assert_eq!(result, VerificationResult::NoKeysRegistered);
    }

    #[test]
    fn stale_proof_rejected() {
        let key_bytes = random_key();
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        let mut proof = make_valid_proof("m", key_bytes, 32);
        // Set timestamp to 10 minutes ago
        proof.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 600;
        // Re-sign with the tampered timestamp
        let msg = proof.signable_message();
        let sig = signing_key.sign(&msg);
        proof.node_signature = sig.to_bytes();

        let mut verifier = ToplocVerifierImpl::new().with_max_proof_age(300);
        verifier.register_model("m");
        verifier.register_public_key(verifying_key);

        let result = verifier.verify(&proof, "m", 32);
        assert!(matches!(result, VerificationResult::StaleProof { .. }));
    }
}
