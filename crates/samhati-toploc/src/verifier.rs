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
    MissingChunks {
        expected: usize,
        got: usize,
    },
    UnknownModel(String),
    InvalidProofStructure(String),
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
    /// Known node public keys: for signature verification.
    /// Maps a public key bytes to the verifying key. In production this would be
    /// looked up by node id; here we try all known keys.
    known_public_keys: Vec<VerifyingKey>,
}

impl ToplocVerifierImpl {
    /// Create a new verifier with no known models or keys.
    pub fn new() -> Self {
        Self {
            known_models: HashMap::new(),
            known_public_keys: Vec::new(),
        }
    }

    /// Register a model so its hash is accepted during verification.
    pub fn register_model(&mut self, model_id: &str) {
        let hash = model_id_to_hash(model_id);
        self.known_models.insert(model_id.to_string(), hash);
    }

    /// Register a node public key for signature verification.
    pub fn register_public_key(&mut self, public_key: VerifyingKey) {
        self.known_public_keys.push(public_key);
    }

    /// Register a node public key from raw 32 bytes.
    pub fn register_public_key_bytes(&mut self, bytes: [u8; 32]) -> Result<(), ed25519_dalek::SignatureError> {
        let vk = VerifyingKey::from_bytes(&bytes)?;
        self.known_public_keys.push(vk);
        Ok(())
    }

    /// Verify the Ed25519 signature on a proof using any of the known public keys.
    fn verify_signature(&self, proof: &ToplocProof) -> bool {
        let msg = proof.signable_message();
        let sig = match Signature::from_bytes(&proof.node_signature) {
            sig => sig,
        };
        for vk in &self.known_public_keys {
            if vk.verify(&msg, &sig).is_ok() {
                return true;
            }
        }
        false
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
        // 1. Check model is known
        let expected_hash = match self.known_models.get(claimed_model_id) {
            Some(h) => *h,
            None => return VerificationResult::UnknownModel(claimed_model_id.to_string()),
        };

        // 2. Check model hash
        if proof.model_hash != expected_hash {
            return VerificationResult::InvalidModelHash {
                expected: expected_hash,
                got: proof.model_hash,
            };
        }

        // 3. Check token count
        if proof.token_count != output_token_count {
            return VerificationResult::TokenCountMismatch {
                expected: output_token_count,
                got: proof.token_count,
            };
        }

        // 4. Check chunk count = ceil(token_count / 32)
        let chunk_size = 32usize;
        let expected_chunks = (proof.token_count as usize + chunk_size - 1) / chunk_size;
        if proof.chunk_proofs.len() != expected_chunks {
            return VerificationResult::MissingChunks {
                expected: expected_chunks,
                got: proof.chunk_proofs.len(),
            };
        }

        // 5. Validate chunk indices are sequential
        for (i, chunk) in proof.chunk_proofs.iter().enumerate() {
            if chunk.chunk_index != i as u32 {
                return VerificationResult::InvalidProofStructure(format!(
                    "chunk index mismatch: expected {}, got {}",
                    i, chunk.chunk_index
                ));
            }
        }

        // 6. Verify Ed25519 signature
        if !self.known_public_keys.is_empty() && !self.verify_signature(proof) {
            return VerificationResult::InvalidSignature;
        }

        VerificationResult::Valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::TokenLogits;
    use crate::prover::ToplocProver;
    use ed25519_dalek::SigningKey;
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

        let mut verifier = ToplocVerifierImpl::new();
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

        let mut verifier = ToplocVerifierImpl::new();
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

        let mut verifier = ToplocVerifierImpl::new();
        verifier.register_model("m");
        verifier.register_public_key(verifying_key);

        // Claim 128 tokens but proof has 64
        let result = verifier.verify(&proof, "m", 128);
        assert!(matches!(result, VerificationResult::TokenCountMismatch { .. }));
    }

    #[test]
    fn unknown_model_fails() {
        let key_bytes = random_key();
        let proof = make_valid_proof("m", key_bytes, 32);

        let verifier = ToplocVerifierImpl::new(); // no models registered
        let result = verifier.verify(&proof, "m", 32);
        assert!(matches!(result, VerificationResult::UnknownModel(_)));
    }

    #[test]
    fn invalid_signature_fails() {
        let key_bytes = random_key();
        let mut proof = make_valid_proof("m", key_bytes, 32);

        // Tamper with the signature
        proof.node_signature[0] ^= 0xFF;
        proof.node_signature[1] ^= 0xFF;

        // Use a different key for the verifier
        let other_key = random_key();
        let other_signing = SigningKey::from_bytes(&other_key);

        let mut verifier = ToplocVerifierImpl::new();
        verifier.register_model("m");
        verifier.register_public_key(other_signing.verifying_key());

        let result = verifier.verify(&proof, "m", 32);
        assert_eq!(result, VerificationResult::InvalidSignature);
    }

    #[test]
    fn missing_chunks_fails() {
        let key_bytes = random_key();
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();

        let mut proof = make_valid_proof("m", key_bytes, 64);
        // Remove a chunk
        proof.chunk_proofs.pop();

        let mut verifier = ToplocVerifierImpl::new();
        verifier.register_model("m");
        verifier.register_public_key(verifying_key);

        let result = verifier.verify(&proof, "m", 64);
        assert!(matches!(result, VerificationResult::MissingChunks { .. }));
    }

    #[test]
    fn verification_without_public_keys_skips_signature_check() {
        let key_bytes = random_key();
        let proof = make_valid_proof("m", key_bytes, 32);

        let mut verifier = ToplocVerifierImpl::new();
        verifier.register_model("m");
        // No public keys registered — signature check is skipped

        let result = verifier.verify(&proof, "m", 32);
        assert_eq!(result, VerificationResult::Valid);
    }
}
