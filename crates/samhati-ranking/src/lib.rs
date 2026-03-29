//! # samhati-ranking
//!
//! BradleyTerry pairwise ranking, EMA reputation, and reward distribution
//! for Samhati's decentralized swarm inference quality measurement.
//!
//! Reputation system combines:
//!   - Fortytwo-style dual-track EMA (answer quality + ranking quality)
//!   - TrueSkill-inspired uncertainty (σ) tracking
//!   - Hyperbolic-style adaptive validation rates
//!   - Domain-specific tracking (Samhati original)
//!
//! This crate is pure computation — no async, no networking.

pub mod bradley_terry;
pub mod elo;
pub mod reputation;
pub mod rewards;
pub mod types;

pub use bradley_terry::{BradleyTerryEngine, PairwiseComparison, RankingResult};
pub use elo::{EloRating, EloStore, RoundOutcome};
pub use reputation::{NodeReputation, ReputationStore, ReputationRoundOutcome};
pub use rewards::{RewardCalculator, RewardDistribution};
pub use types::NodeId;
