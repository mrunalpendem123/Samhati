//! # samhati-toploc
//!
//! **TOPLOC** (Token-Level Proof of Compute) — cryptographic verification layer
//! that proves a node ran real inference with a specific model.
//!
//! Based on arXiv:2501.16007.
//!
//! ## Overview
//!
//! TOPLOC works by committing to the top-K logit values produced during inference.
//! These logits are internal activation states that are mathematically impossible to
//! produce without running the actual forward pass with the actual model weights.
//!
//! ## Modules
//!
//! - [`proof`] — Core data structures: `TokenLogits`, `LogitChunkProof`, `ToplocProof`
//! - [`prover`] — Proof generation (runs on inference node)
//! - [`verifier`] — Proof verification (CPU-only, no GPU needed)
//! - [`calibration`] — Calibration round for new node registration
//! - [`types`] — Shared types and constants

pub mod calibration;
pub mod proof;
pub mod prover;
pub mod types;
pub mod verifier;

// Re-exports for convenience
pub use proof::{LogitChunkProof, TokenLogits, ToplocProof};
pub use prover::ToplocProver;
pub use verifier::{ToplocVerifier, ToplocVerifierImpl, VerificationResult};
pub use calibration::{CalibrationPrompt, CalibrationResult, CalibrationRound};
pub use types::{ModelInfo, model_id_to_hash};
