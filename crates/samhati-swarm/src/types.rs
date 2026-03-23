use serde::{Deserialize, Serialize};
use std::time::Duration;

/// 32-byte node identifier (typically an iroh NodeId or ed25519 public key).
pub type NodeId = [u8; 32];

/// Complexity tier — drives the swarm size N.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Complexity {
    Trivial,        // N = 1
    Conversational, // N = 3
    Reasoning,      // N = 5
    Hard,           // N = 7
    Expert,         // N = 9
}

impl Complexity {
    /// Number of swarm nodes for this complexity tier.
    pub fn n_nodes(self) -> usize {
        match self {
            Complexity::Trivial => 1,
            Complexity::Conversational => 3,
            Complexity::Reasoning => 5,
            Complexity::Hard => 7,
            Complexity::Expert => 9,
        }
    }
}

/// Incoming inference request from a user / upstream caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    /// User can override the classifier and force a specific swarm size.
    pub user_override_n: Option<usize>,
    pub preferred_domains: Vec<String>,
}

/// Raw response from a single swarm node (before TOPLOC verification).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResponse {
    pub node_id: NodeId,
    pub answer: String,
    pub tokens_generated: usize,
    pub generation_time_ms: u64,
    /// TOPLOC proof blob — ~258 bytes per 32 output tokens.
    pub toploc_proof: Option<Vec<u8>>,
    pub model_name: String,
}

/// Response after TOPLOC verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedResponse {
    pub node_id: NodeId,
    pub answer: String,
    pub tokens_generated: usize,
    pub generation_time_ms: u64,
    pub proof_valid: bool,
    pub model_name: String,
}

/// A single pairwise comparison produced during Phase 2 peer-ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseRanking {
    pub ranker_node_id: NodeId,
    pub node_a: NodeId,
    pub node_b: NodeId,
    /// P(A beats B) in [0.0, 1.0].
    pub preference: f32,
    /// 50–100 token reasoning chain justifying the preference.
    pub reasoning: String,
    pub domain_tags: Vec<String>,
}

/// Final output of a swarm inference round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResult {
    pub winner_node_id: NodeId,
    pub winning_answer: String,
    /// BradleyTerry posterior probability for the winner.
    pub confidence: f32,
    pub all_responses: Vec<VerifiedResponse>,
    pub rankings: Vec<PairwiseRanking>,
    pub elo_deltas: Vec<(NodeId, i32)>,
    pub n_nodes: usize,
    pub total_time_ms: u64,
    pub complexity: Complexity,
    pub domain_tags: Vec<String>,
    /// Emitted when confidence >= training_confidence_threshold for self-evolving pipeline.
    pub training_example: Option<TrainingExample>,
}

/// Data collected for the self-evolving fine-tuning pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub prompt: String,
    pub winning_answer: String,
    pub losing_answers: Vec<String>,
    pub reasoning_chains: Vec<String>,
    pub domain_tags: Vec<String>,
    pub confidence: f32,
}

/// Global configuration for a swarm round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Maximum wall-clock time to wait for the slowest node during fan-out.
    #[serde(with = "duration_millis")]
    pub max_fan_out_timeout: Duration,
    /// Minimum number of valid (proof-verified) responses needed to continue.
    pub min_responses: usize,
    /// If true, discard any response without a valid TOPLOC proof.
    pub toploc_required: bool,
    /// Minimum BradleyTerry winner confidence to emit a training example.
    pub training_confidence_threshold: f32,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            max_fan_out_timeout: Duration::from_secs(30),
            min_responses: 1,
            toploc_required: false,
            training_confidence_threshold: 0.70,
        }
    }
}

/// Serde helper: serialize/deserialize Duration as milliseconds (u64).
mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_millis() as u64)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let ms = u64::deserialize(d)?;
        Ok(Duration::from_millis(ms))
    }
}
