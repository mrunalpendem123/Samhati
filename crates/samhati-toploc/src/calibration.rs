use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::proof::ToplocProof;
use crate::verifier::{ToplocVerifier, VerificationResult};

/// A single calibration prompt to be answered by the candidate node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPrompt {
    /// The prompt text the node must run inference on.
    pub prompt: String,
    /// The model the node must use.
    pub model_id: String,
    /// Expected proof hash (BLAKE3 of the serialized proof).
    /// `None` on the very first calibration — the proof will be recorded as ground truth.
    pub expected_proof_hash: Option<[u8; 32]>,
}

/// The outcome of a complete calibration round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Whether the node passed calibration.
    pub passed: bool,
    /// How many prompts the node answered correctly.
    pub prompts_passed: usize,
    /// Total number of prompts.
    pub prompts_total: usize,
    /// Collected proofs (for on-chain submission / future reference).
    pub proofs: Vec<ToplocProof>,
    /// Details of each failure: (prompt_index, reason).
    pub failures: Vec<(usize, VerificationResult)>,
}

/// A calibration round that a new node must complete before receiving real work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationRound {
    /// The prompts the node must answer.
    pub prompts: Vec<CalibrationPrompt>,
    /// How many prompts must pass (default: all).
    pub required_pass_count: usize,
}

/// Predefined calibration prompt templates.
/// In production these would be model-specific; here we use generic reasoning prompts.
const CALIBRATION_TEMPLATES: &[&str] = &[
    "What is the capital of France? Answer in one word.",
    "Compute 17 * 23 and give only the number.",
    "Translate 'hello world' into Spanish.",
    "What is the boiling point of water in Celsius?",
    "Name the largest planet in our solar system.",
    "What programming language is known for the borrow checker?",
    "How many bits in a byte?",
    "What is the chemical formula for water?",
    "Name one primary color.",
    "What year did the Berlin Wall fall?",
    "What is the speed of light in m/s (approximate)?",
    "What is the square root of 144?",
    "Name the author of 'A Brief History of Time'.",
    "What is the smallest prime number?",
    "What does HTTP stand for?",
    "How many continents are there?",
    "What gas do plants absorb from the atmosphere?",
    "Name the closest star to Earth.",
    "What is absolute zero in Kelvin?",
    "What data structure uses FIFO ordering?",
];

impl CalibrationRound {
    /// Generate a set of calibration prompts for a given model.
    ///
    /// Randomly selects `n_prompts` from the template pool.
    /// `expected_proof_hash` is set to `None` (first-time calibration).
    pub fn generate(model_id: &str, n_prompts: usize) -> Self {
        let n = n_prompts.min(CALIBRATION_TEMPLATES.len());
        let mut rng = rand::thread_rng();
        let mut templates: Vec<&str> = CALIBRATION_TEMPLATES.to_vec();
        templates.shuffle(&mut rng);

        let prompts = templates[..n]
            .iter()
            .map(|&t| CalibrationPrompt {
                prompt: t.to_string(),
                model_id: model_id.to_string(),
                expected_proof_hash: None,
            })
            .collect();

        Self {
            prompts,
            required_pass_count: n,
        }
    }

    /// Generate a calibration round with pre-computed expected hashes.
    pub fn with_expected_hashes(
        model_id: &str,
        prompts_and_hashes: Vec<(String, [u8; 32])>,
    ) -> Self {
        let n = prompts_and_hashes.len();
        let prompts = prompts_and_hashes
            .into_iter()
            .map(|(prompt, hash)| CalibrationPrompt {
                prompt,
                model_id: model_id.to_string(),
                expected_proof_hash: Some(hash),
            })
            .collect();
        Self {
            prompts,
            required_pass_count: n,
        }
    }

    /// Verify a node's calibration responses.
    ///
    /// `responses` is a list of (output_text, proof) pairs, one per prompt, in order.
    pub fn verify(
        &self,
        responses: &[(String, ToplocProof)],
        verifier: &dyn ToplocVerifier,
    ) -> CalibrationResult {
        let mut passed_count = 0usize;
        let mut failures = Vec::new();
        let mut proofs = Vec::new();

        for (i, prompt) in self.prompts.iter().enumerate() {
            if i >= responses.len() {
                failures.push((
                    i,
                    VerificationResult::InvalidProofStructure("missing response".to_string()),
                ));
                continue;
            }

            let (ref _output_text, ref proof) = responses[i];
            proofs.push(proof.clone());

            // Structural + signature verification
            let result = verifier.verify(proof, &prompt.model_id, proof.token_count);
            if !result.is_valid() {
                failures.push((i, result));
                continue;
            }

            // If we have an expected proof hash, check it
            if let Some(expected_hash) = prompt.expected_proof_hash {
                let actual_hash = proof.proof_hash();
                if actual_hash != expected_hash {
                    failures.push((
                        i,
                        VerificationResult::InvalidProofStructure(format!(
                            "proof hash mismatch: expected {:?}, got {:?}",
                            hex_short(&expected_hash),
                            hex_short(&actual_hash),
                        )),
                    ));
                    continue;
                }
            }

            passed_count += 1;
        }

        CalibrationResult {
            passed: passed_count >= self.required_pass_count,
            prompts_passed: passed_count,
            prompts_total: self.prompts.len(),
            proofs,
            failures,
        }
    }

    /// Number of prompts in this calibration round.
    pub fn prompt_count(&self) -> usize {
        self.prompts.len()
    }
}

fn hex_short(bytes: &[u8; 32]) -> String {
    format!("{}...", hex::encode(&bytes[..4]))
}

/// Minimal hex encoder (avoids pulling in the `hex` crate at runtime).
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::ToplocProver;
    use crate::proof::TokenLogits;
    use crate::verifier::ToplocVerifierImpl;
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

    fn make_proof_for_model(model_id: &str, key_bytes: [u8; 32], n_tokens: u32) -> ToplocProof {
        let mut prover = ToplocProver::new(model_id, key_bytes);
        for i in 0..n_tokens {
            prover.record_token(make_token_logits(i));
        }
        prover.finalize().unwrap()
    }

    fn setup_verifier(model_id: &str, key_bytes: [u8; 32]) -> ToplocVerifierImpl {
        let signing_key = SigningKey::from_bytes(&key_bytes);
        let mut verifier = ToplocVerifierImpl::new();
        verifier.register_model(model_id);
        verifier.register_public_key(signing_key.verifying_key());
        verifier
    }

    #[test]
    fn calibration_generate_produces_correct_count() {
        let round = CalibrationRound::generate("test-model", 10);
        assert_eq!(round.prompts.len(), 10);
        assert_eq!(round.required_pass_count, 10);
        for p in &round.prompts {
            assert_eq!(p.model_id, "test-model");
            assert!(p.expected_proof_hash.is_none());
        }
    }

    #[test]
    fn calibration_generate_caps_at_template_count() {
        let round = CalibrationRound::generate("m", 1000);
        assert_eq!(round.prompts.len(), CALIBRATION_TEMPLATES.len());
    }

    #[test]
    fn calibration_all_pass() {
        let model_id = "cal-model";
        let key = random_key();
        let verifier = setup_verifier(model_id, key);

        let round = CalibrationRound::generate(model_id, 5);
        let responses: Vec<(String, ToplocProof)> = round
            .prompts
            .iter()
            .map(|_| {
                let proof = make_proof_for_model(model_id, key, 32);
                ("some output".to_string(), proof)
            })
            .collect();

        let result = round.verify(&responses, &verifier);
        assert!(result.passed);
        assert_eq!(result.prompts_passed, 5);
        assert_eq!(result.prompts_total, 5);
        assert!(result.failures.is_empty());
        assert_eq!(result.proofs.len(), 5);
    }

    #[test]
    fn calibration_partial_fail() {
        let model_id = "cal-model";
        let key = random_key();
        let verifier = setup_verifier(model_id, key);

        let round = CalibrationRound::generate(model_id, 5);
        let mut responses: Vec<(String, ToplocProof)> = round
            .prompts
            .iter()
            .map(|_| {
                let proof = make_proof_for_model(model_id, key, 32);
                ("output".to_string(), proof)
            })
            .collect();

        // Sabotage one response: use wrong model
        let bad_key = random_key();
        let bad_proof = make_proof_for_model("wrong-model", bad_key, 32);
        responses[2] = ("bad output".to_string(), bad_proof);

        let result = round.verify(&responses, &verifier);
        assert!(!result.passed); // required all 5
        assert_eq!(result.prompts_passed, 4);
        assert_eq!(result.failures.len(), 1);
        assert_eq!(result.failures[0].0, 2); // prompt index 2 failed
    }

    #[test]
    fn calibration_total_fail_wrong_model() {
        let model_id = "cal-model";
        let key = random_key();
        let verifier = setup_verifier(model_id, key);

        let round = CalibrationRound::generate(model_id, 3);
        // All responses use wrong model
        let bad_key = random_key();
        let responses: Vec<(String, ToplocProof)> = round
            .prompts
            .iter()
            .map(|_| {
                let proof = make_proof_for_model("totally-wrong", bad_key, 32);
                ("output".to_string(), proof)
            })
            .collect();

        let result = round.verify(&responses, &verifier);
        assert!(!result.passed);
        assert_eq!(result.prompts_passed, 0);
        assert_eq!(result.failures.len(), 3);
    }

    #[test]
    fn calibration_missing_responses() {
        let model_id = "m";
        let key = random_key();
        let verifier = setup_verifier(model_id, key);

        let round = CalibrationRound::generate(model_id, 5);
        // Only provide 3 responses
        let responses: Vec<(String, ToplocProof)> = (0..3)
            .map(|_| {
                let proof = make_proof_for_model(model_id, key, 32);
                ("out".to_string(), proof)
            })
            .collect();

        let result = round.verify(&responses, &verifier);
        assert!(!result.passed);
        assert_eq!(result.prompts_passed, 3);
        assert_eq!(result.failures.len(), 2); // 2 missing
    }

    #[test]
    fn calibration_with_expected_hashes_match() {
        let model_id = "m";
        let key = random_key();
        let verifier = setup_verifier(model_id, key);

        // Generate a proof and record its hash
        let proof1 = make_proof_for_model(model_id, key, 32);
        let hash1 = proof1.proof_hash();

        let round = CalibrationRound::with_expected_hashes(
            model_id,
            vec![("prompt1".to_string(), hash1)],
        );

        // The same proof should pass hash check
        let result = round.verify(&[("out".to_string(), proof1)], &verifier);
        assert!(result.passed);
    }

    #[test]
    fn calibration_with_expected_hashes_mismatch() {
        let model_id = "m";
        let key = random_key();
        let verifier = setup_verifier(model_id, key);

        let round = CalibrationRound::with_expected_hashes(
            model_id,
            vec![("prompt1".to_string(), [0xAB; 32])], // wrong expected hash
        );

        let proof = make_proof_for_model(model_id, key, 32);
        let result = round.verify(&[("out".to_string(), proof)], &verifier);
        assert!(!result.passed);
        assert_eq!(result.failures.len(), 1);
    }
}
