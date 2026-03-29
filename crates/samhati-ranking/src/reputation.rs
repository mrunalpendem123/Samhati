//! EMA-based dual-track reputation system (inspired by Fortytwo, extended with domains).
//!
//! Tracks two independent dimensions:
//!   R_generation — how often your answers win consensus
//!   R_ranking    — how well your rankings align with BradleyTerry consensus
//!
//! Combined: R = α × R_ranking + (1-α) × R_generation
//!
//! Per-domain tracking: separate R_generation per domain (Code, Math, Reasoning, General).
//! Inactivity decay: R *= (1 - δ) per round of inactivity.
//! Slash: drop to R_MIN if proof verification fails.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::types::NodeId;

/// Reputation configuration.
const R_INITIAL: f64 = 0.5;
const R_MIN: f64 = 0.0;
const R_MAX: f64 = 1.0;
const ALPHA: f64 = 0.4;              // weight of ranking quality (40%)
const BETA_GEN: f64 = 0.15;          // EMA smoothing for generation wins
const BETA_RANK: f64 = 0.15;         // EMA smoothing for ranking accuracy
const DECAY_RATE: f64 = 0.02;        // 2% decay per inactive round
const SLASH_VALUE: f64 = 0.0;        // reputation after slash (must recalibrate)
const RECALIBRATION_THRESHOLD: f64 = 0.1; // below this, node must recalibrate
const SIGMA_INITIAL: f64 = 0.4;          // high uncertainty for new nodes
const SIGMA_MIN: f64 = 0.05;             // minimum uncertainty (very established)
const SIGMA_DECAY: f64 = 0.95;           // σ *= this each round (shrinks over time)
const VALIDATION_MAX: f64 = 0.50;        // 50% audit rate for new/unproven nodes
const VALIDATION_MIN: f64 = 0.02;        // 2% audit rate for highly trusted nodes

/// Per-node reputation state.
///
/// Combines:
///   - Fortytwo-style dual-track EMA (answer quality + ranking quality)
///   - TrueSkill-inspired uncertainty (σ) that shrinks as the node proves itself
///   - Hyperbolic-style adaptive validation rate (new nodes audited more)
///   - Domain-specific tracking (Samhati original)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeReputation {
    /// EMA of generation wins (did your answer win?)
    pub r_generation: f64,
    /// EMA of ranking accuracy (did your rankings agree with consensus?)
    pub r_ranking: f64,
    /// Combined reputation score
    pub r_combined: f64,
    /// Uncertainty (σ) — starts high, shrinks with rounds. TrueSkill-inspired.
    /// High σ = we don't know this node well yet.
    /// Low σ = this node's reputation is reliable.
    pub uncertainty: f64,
    /// Validation rate — probability this node gets audited per round.
    /// New/low-rep nodes: ~50%. High-rep nodes: ~2%. Hyperbolic-inspired.
    pub validation_rate: f64,
    /// Per-domain generation reputation
    pub domain_reputation: HashMap<String, f64>,
    /// Total rounds participated
    pub total_rounds: u64,
    /// Total rounds won
    pub rounds_won: u64,
    /// Rounds since last participation (for decay)
    pub rounds_inactive: u64,
    /// Whether node needs recalibration (slashed or below threshold)
    pub needs_recalibration: bool,
}

impl NodeReputation {
    fn new() -> Self {
        Self {
            r_generation: R_INITIAL,
            r_ranking: R_INITIAL,
            r_combined: R_INITIAL,
            uncertainty: SIGMA_INITIAL,
            validation_rate: VALIDATION_MAX,
            domain_reputation: HashMap::new(),
            total_rounds: 0,
            rounds_won: 0,
            rounds_inactive: 0,
            needs_recalibration: false,
        }
    }

    /// Recompute combined reputation and adaptive rates.
    fn update_combined(&mut self) {
        self.r_combined = (ALPHA * self.r_ranking + (1.0 - ALPHA) * self.r_generation)
            .clamp(R_MIN, R_MAX);

        // Shrink uncertainty — each round gives us more confidence in this node's rating
        self.uncertainty = (self.uncertainty * SIGMA_DECAY).max(SIGMA_MIN);

        // Adaptive validation rate — inversely proportional to reputation × certainty
        // High rep + low uncertainty = rarely audited
        // Low rep or high uncertainty = frequently audited
        let trust = self.r_combined * (1.0 - self.uncertainty);
        self.validation_rate = VALIDATION_MAX - (VALIDATION_MAX - VALIDATION_MIN) * trust;
        self.validation_rate = self.validation_rate.clamp(VALIDATION_MIN, VALIDATION_MAX);
    }

    /// Get domain-specific reputation, falling back to global.
    pub fn domain_score(&self, domain: &str) -> f64 {
        self.domain_reputation.get(domain).copied().unwrap_or(self.r_generation)
    }

    /// Should this node be audited this round?
    /// Returns true with probability = validation_rate.
    pub fn should_audit(&self, round_seed: u64) -> bool {
        // Deterministic from seed so auditing is verifiable
        let hash = ((round_seed ^ 0xDEADBEEF).wrapping_mul(2654435761)) as f64 / u64::MAX as f64;
        hash < self.validation_rate
    }
}

/// Outcome of a round, used to update reputations.
#[derive(Debug, Clone)]
pub struct ReputationRoundOutcome {
    /// All nodes that participated.
    pub participants: Vec<NodeId>,
    /// The winning node.
    pub winner: NodeId,
    /// For each node that judged: did their ranking agree with BT consensus?
    /// Maps judge_node_id → accuracy (0.0 = completely wrong, 1.0 = perfectly aligned).
    pub ranking_accuracy: HashMap<NodeId, f64>,
    /// Optional domain.
    pub domain: Option<String>,
}

/// In-memory reputation store.
#[derive(Debug, Clone, Default)]
pub struct ReputationStore {
    reputations: HashMap<NodeId, NodeReputation>,
}

impl ReputationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, node_id: &NodeId) -> Option<&NodeReputation> {
        self.reputations.get(node_id)
    }

    pub fn register(&mut self, node_id: NodeId) {
        self.reputations.entry(node_id).or_insert_with(NodeReputation::new);
    }

    pub fn ratings_mut(&mut self) -> &mut HashMap<NodeId, NodeReputation> {
        &mut self.reputations
    }

    /// Update reputations after a round.
    /// Returns Vec<(NodeId, f64)> of new combined reputation scores.
    pub fn update_round(&mut self, outcome: &ReputationRoundOutcome) -> Vec<(NodeId, f64)> {
        let mut updates = Vec::new();

        for &participant in &outcome.participants {
            let rep = self.reputations.entry(participant).or_insert_with(NodeReputation::new);

            // Reset inactivity
            rep.rounds_inactive = 0;
            rep.total_rounds += 1;

            // ── R_generation: did this node's answer win? ──
            let won = participant == outcome.winner;
            let gen_signal = if won { 1.0 } else { 0.0 };
            rep.r_generation = (1.0 - BETA_GEN) * rep.r_generation + BETA_GEN * gen_signal;

            if won {
                rep.rounds_won += 1;
            }

            // ── R_ranking: did this node's rankings agree with consensus? ──
            if let Some(&accuracy) = outcome.ranking_accuracy.get(&participant) {
                rep.r_ranking = (1.0 - BETA_RANK) * rep.r_ranking + BETA_RANK * accuracy;
            }
            // If node didn't judge this round, r_ranking stays unchanged.

            // ── Domain-specific R_generation ──
            if let Some(ref domain) = outcome.domain {
                let domain_r = rep.domain_reputation.entry(domain.clone()).or_insert(R_INITIAL);
                *domain_r = (1.0 - BETA_GEN) * *domain_r + BETA_GEN * gen_signal;
            }

            // ── Recompute combined ──
            rep.update_combined();

            // ── Check recalibration threshold ──
            if rep.r_combined < RECALIBRATION_THRESHOLD {
                rep.needs_recalibration = true;
            }

            updates.push((participant, rep.r_combined));
        }

        // ── Decay inactive nodes ──
        // (nodes NOT in this round's participants)
        for (node_id, rep) in &mut self.reputations {
            if !outcome.participants.contains(node_id) {
                rep.rounds_inactive += 1;
                rep.r_generation *= 1.0 - DECAY_RATE;
                rep.r_ranking *= 1.0 - DECAY_RATE;
                rep.update_combined();
            }
        }

        updates
    }

    /// Slash a node's reputation (failed proof verification).
    pub fn slash(&mut self, node_id: &NodeId) -> Option<f64> {
        if let Some(rep) = self.reputations.get_mut(node_id) {
            rep.r_generation = SLASH_VALUE;
            rep.r_ranking = SLASH_VALUE;
            rep.r_combined = SLASH_VALUE;
            rep.needs_recalibration = true;
            // Clear domain reputations too
            for v in rep.domain_reputation.values_mut() {
                *v = SLASH_VALUE;
            }
            Some(SLASH_VALUE)
        } else {
            None
        }
    }

    /// Get the top N nodes by combined reputation, optionally filtered by domain.
    pub fn top_nodes(&self, n: usize, domain: Option<&str>) -> Vec<(NodeId, f64)> {
        let mut scored: Vec<(NodeId, f64)> = self.reputations.iter()
            .filter(|(_, rep)| !rep.needs_recalibration)
            .map(|(&id, rep)| {
                let score = match domain {
                    Some(d) => rep.domain_score(d),
                    None => rep.r_combined,
                };
                (id, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::node_id_from_byte;

    #[test]
    fn new_node_starts_at_initial() {
        let mut store = ReputationStore::new();
        let n = node_id_from_byte(1);
        store.register(n);
        let rep = store.get(&n).unwrap();
        assert_eq!(rep.r_combined, R_INITIAL);
        assert_eq!(rep.r_generation, R_INITIAL);
        assert_eq!(rep.r_ranking, R_INITIAL);
    }

    #[test]
    fn winner_gains_generation_reputation() {
        let mut store = ReputationStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        store.register(a);
        store.register(b);

        let outcome = ReputationRoundOutcome {
            participants: vec![a, b],
            winner: a,
            ranking_accuracy: HashMap::new(),
            domain: None,
        };
        store.update_round(&outcome);

        let ra = store.get(&a).unwrap();
        let rb = store.get(&b).unwrap();
        assert!(ra.r_generation > rb.r_generation, "Winner should have higher r_generation");
        assert!(ra.r_combined > rb.r_combined, "Winner should have higher combined");
    }

    #[test]
    fn good_judge_gains_ranking_reputation() {
        let mut store = ReputationStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        store.register(a);
        store.register(b);

        let outcome = ReputationRoundOutcome {
            participants: vec![a, b],
            winner: a,
            ranking_accuracy: [(a, 0.9), (b, 0.3)].into(),
            domain: None,
        };
        store.update_round(&outcome);

        let ra = store.get(&a).unwrap();
        let rb = store.get(&b).unwrap();
        assert!(ra.r_ranking > rb.r_ranking, "Good judge should have higher r_ranking");
    }

    #[test]
    fn domain_reputation_tracked_separately() {
        let mut store = ReputationStore::new();
        let a = node_id_from_byte(1);
        store.register(a);

        // Win a Code round
        store.update_round(&ReputationRoundOutcome {
            participants: vec![a],
            winner: a,
            ranking_accuracy: HashMap::new(),
            domain: Some("Code".into()),
        });

        // Lose a Math round (add another node as winner)
        let b = node_id_from_byte(2);
        store.register(b);
        store.update_round(&ReputationRoundOutcome {
            participants: vec![a, b],
            winner: b,
            ranking_accuracy: HashMap::new(),
            domain: Some("Math".into()),
        });

        let ra = store.get(&a).unwrap();
        let code_r = ra.domain_reputation.get("Code").unwrap();
        let math_r = ra.domain_reputation.get("Math").unwrap();
        assert!(code_r > math_r, "Code reputation should be higher than Math");
    }

    #[test]
    fn inactive_nodes_decay() {
        let mut store = ReputationStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        store.register(a);
        store.register(b);

        // Only b participates, a is inactive
        let r_before = store.get(&a).unwrap().r_combined;
        store.update_round(&ReputationRoundOutcome {
            participants: vec![b],
            winner: b,
            ranking_accuracy: HashMap::new(),
            domain: None,
        });
        let r_after = store.get(&a).unwrap().r_combined;
        assert!(r_after < r_before, "Inactive node should decay");
    }

    #[test]
    fn slash_drops_to_zero() {
        let mut store = ReputationStore::new();
        let a = node_id_from_byte(1);
        store.register(a);

        store.slash(&a);
        let ra = store.get(&a).unwrap();
        assert_eq!(ra.r_combined, 0.0);
        assert!(ra.needs_recalibration);
    }

    #[test]
    fn top_nodes_filters_recalibration() {
        let mut store = ReputationStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        store.register(a);
        store.register(b);

        store.slash(&b); // b needs recalibration

        let top = store.top_nodes(10, None);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, a);
    }

    #[test]
    fn reputation_converges_over_rounds() {
        let mut store = ReputationStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        let c = node_id_from_byte(3);
        store.register(a);
        store.register(b);
        store.register(c);

        // a wins 8 out of 10 rounds
        for i in 0..10 {
            let winner = if i < 8 { a } else { b };
            store.update_round(&ReputationRoundOutcome {
                participants: vec![a, b, c],
                winner,
                ranking_accuracy: [(a, 0.8), (b, 0.5), (c, 0.3)].into(),
                domain: Some("Code".into()),
            });
        }

        let ra = store.get(&a).unwrap();
        let rb = store.get(&b).unwrap();
        let rc = store.get(&c).unwrap();

        assert!(ra.r_combined > rb.r_combined, "a should be highest");
        assert!(rb.r_combined > rc.r_combined, "b should be higher than c");
        assert!(ra.r_generation > 0.6, "a's generation rep should be high");
        assert!(ra.r_ranking > 0.6, "a's ranking rep should be high (good judge)");
        assert!(rc.r_ranking < 0.4, "c's ranking rep should be low (bad judge)");
    }
}
