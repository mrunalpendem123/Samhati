//! # samhati-ranking
//!
//! BradleyTerry pairwise ranking aggregation and ELO management for
//! Samhati's decentralized swarm inference quality measurement.
//!
//! This crate is pure computation — no async, no networking.

pub mod bradley_terry;
pub mod elo;
pub mod rewards;
pub mod types;

pub use bradley_terry::{BradleyTerryEngine, PairwiseComparison, RankingResult};
pub use elo::{EloRating, EloStore, RoundOutcome};
pub use rewards::{RewardCalculator, RewardDistribution};
pub use types::NodeId;
