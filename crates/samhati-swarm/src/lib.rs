//! `samhati-swarm` — Core orchestration layer for Samhati's swarm inference protocol.
//!
//! Implements the full inference lifecycle:
//! 1. Complexity classification (prompt → N nodes)
//! 2. Node selection from the model registry (ELO + domain matching)
//! 3. Fan-out: parallel dispatch to N nodes via iroh QUIC
//! 4. Fan-in: collect responses, verify TOPLOC proofs
//! 5. Peer ranking: C(N,2) pairwise comparisons with reasoning chains
//! 6. BradleyTerry aggregation → winner selection
//! 7. ELO updates + training example emission

pub mod classifier;
pub mod config;
pub mod coordinator;
pub mod registry;
pub mod types;

// Re-export primary public types for ergonomic imports.
pub use classifier::{ComplexityClassifier, ComplexityResult};
pub use config::SwarmConfigBuilder;
pub use coordinator::{
    BradleyTerryEngine, NodeDispatcher, NoopToplocVerifier, PeerRanker, StubNodeDispatcher,
    StubPeerRanker, SwarmCoordinator, ToplocVerifier,
};
pub use registry::{ModelRegistry, NodeInfo};
pub use types::{
    Complexity, InferenceRequest, NodeId, NodeResponse, PairwiseRanking, SwarmConfig, SwarmResult,
    TrainingExample, VerifiedResponse,
};
