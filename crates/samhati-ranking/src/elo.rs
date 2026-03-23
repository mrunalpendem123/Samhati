use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::types::NodeId;

/// ELO floor — no node's rating can drop below this.
const ELO_FLOOR: i32 = 100;

/// Default starting ELO for newly registered nodes.
const STARTING_ELO: i32 = 1500;

/// Slash penalty applied for failed TOPLOC verification.
const SLASH_PENALTY: i32 = 200;

/// K-factor for new nodes (fewer than this many rounds).
const K_FACTOR_THRESHOLD: u64 = 1000;
const K_FACTOR_NEW: f64 = 32.0;
const K_FACTOR_ESTABLISHED: f64 = 16.0;

/// Per-node ELO rating with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloRating {
    /// Current global ELO score.
    pub score: i32,
    /// Total rounds participated in.
    pub total_rounds: u64,
    /// Total rounds won.
    pub rounds_won: u64,
    /// Per-domain ELO tracking.
    pub domain_elos: HashMap<String, i32>,
}

impl EloRating {
    fn new() -> Self {
        Self {
            score: STARTING_ELO,
            total_rounds: 0,
            rounds_won: 0,
            domain_elos: HashMap::new(),
        }
    }
}

/// The outcome of a swarm inference round, used to update ELO scores.
#[derive(Debug, Clone)]
pub struct RoundOutcome {
    /// All nodes that participated in this round.
    pub participants: Vec<NodeId>,
    /// The winner of the round (from BradleyTerry).
    pub winner: NodeId,
    /// BradleyTerry strength estimates per node.
    pub strengths: HashMap<NodeId, f64>,
    /// Optional domain tag for domain-specific ELO tracking.
    pub domain: Option<String>,
}

/// In-memory ELO rating store for all nodes.
#[derive(Debug, Clone, Default)]
pub struct EloStore {
    ratings: HashMap<NodeId, EloRating>,
}

impl EloStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a node's ELO rating.
    pub fn get(&self, node_id: &NodeId) -> Option<&EloRating> {
        self.ratings.get(node_id)
    }

    /// Register a new node with the default starting ELO (1500).
    pub fn register(&mut self, node_id: NodeId) {
        self.ratings.entry(node_id).or_insert_with(EloRating::new);
    }

    /// Update ELO scores after a completed swarm round.
    ///
    /// The winner gets actual_score = 1.0.
    /// Non-winners get actual_score proportional to their BradleyTerry strength relative to the winner.
    /// Each participant's expected score is the average of their pairwise expected scores vs all others.
    ///
    /// Returns a Vec of (node_id, elo_delta) for each participant.
    pub fn update_round(&mut self, round: &RoundOutcome) -> Vec<(NodeId, i32)> {
        if round.participants.is_empty() {
            return Vec::new();
        }

        // Ensure all participants are registered.
        for &p in &round.participants {
            self.register(p);
        }

        // Compute actual scores.
        // Winner gets 1.0; others get their strength / winner's strength, clamped to [0, 1).
        let winner_strength = round.strengths.get(&round.winner).copied().unwrap_or(1.0);
        let actual_scores: HashMap<NodeId, f64> = round
            .participants
            .iter()
            .map(|&p| {
                if p == round.winner {
                    (p, 1.0)
                } else {
                    let s = round.strengths.get(&p).copied().unwrap_or(0.0);
                    // Convert log-strengths to ratio via exp if strengths are log-scale.
                    // The BT engine returns log-strengths (beta), so use exp for ratio.
                    let ratio = if winner_strength.is_finite() && s.is_finite() {
                        (s - winner_strength).exp()
                    } else {
                        0.0
                    };
                    (p, ratio.clamp(0.0, 0.99))
                }
            })
            .collect();

        let n = round.participants.len();
        let mut deltas = Vec::with_capacity(n);

        for &node in &round.participants {
            let node_rating = self.ratings[&node].score;
            let k = self.k_factor(&node);

            // expected_score_vs_field = average of expected_score(node, other) for all others.
            let expected: f64 = if n > 1 {
                round
                    .participants
                    .iter()
                    .filter(|&&other| other != node)
                    .map(|&other| {
                        let other_rating = self.ratings[&other].score;
                        Self::expected_score(node_rating, other_rating)
                    })
                    .sum::<f64>()
                    / (n - 1) as f64
            } else {
                0.5
            };

            let actual = actual_scores[&node];
            let delta = (k * (actual - expected)).round() as i32;
            deltas.push((node, delta));
        }

        // Apply deltas.
        for &(node, delta) in &deltas {
            let rating = self.ratings.get_mut(&node).unwrap();
            rating.score = (rating.score + delta).max(ELO_FLOOR);
            rating.total_rounds += 1;
            if node == round.winner {
                rating.rounds_won += 1;
            }

            // Update domain-specific ELO if domain is set.
            if let Some(ref domain) = round.domain {
                let domain_elo = rating.domain_elos.entry(domain.clone()).or_insert(STARTING_ELO);
                *domain_elo = (*domain_elo + delta).max(ELO_FLOOR);
            }

            debug!(
                node = ?node,
                delta,
                new_score = rating.score,
                "ELO updated"
            );
        }

        deltas
    }

    /// Apply a slash penalty of -200 ELO for failed TOPLOC verification.
    /// Returns the new ELO score.
    pub fn slash(&mut self, node_id: &NodeId) -> Result<i32> {
        let rating = self
            .ratings
            .get_mut(node_id)
            .ok_or_else(|| anyhow::anyhow!("node not found in ELO store"))?;

        rating.score = (rating.score - SLASH_PENALTY).max(ELO_FLOOR);

        debug!(
            node = ?node_id,
            new_score = rating.score,
            "TOPLOC slash applied"
        );

        Ok(rating.score)
    }

    /// K-factor: 32 for nodes with fewer than 1000 rounds, 16 for established nodes.
    fn k_factor(&self, node_id: &NodeId) -> f64 {
        match self.ratings.get(node_id) {
            Some(r) if r.total_rounds >= K_FACTOR_THRESHOLD => K_FACTOR_ESTABLISHED,
            _ => K_FACTOR_NEW,
        }
    }

    /// Mutable access to the underlying ratings map (for testing and administrative use).
    pub fn ratings_mut(&mut self) -> &mut HashMap<NodeId, EloRating> {
        &mut self.ratings
    }

    /// Expected score of player A vs player B.
    /// E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    pub fn expected_score(rating_a: i32, rating_b: i32) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) as f64 / 400.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::node_id_from_byte;

    fn make_round(participants: Vec<NodeId>, winner: NodeId) -> RoundOutcome {
        let mut strengths = HashMap::new();
        // Give winner the highest strength.
        for (i, &p) in participants.iter().enumerate() {
            if p == winner {
                strengths.insert(p, 1.0);
            } else {
                strengths.insert(p, -(i as f64) - 0.5);
            }
        }
        RoundOutcome {
            participants,
            winner,
            strengths,
            domain: None,
        }
    }

    #[test]
    fn test_winner_goes_up_losers_go_down() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        let c = node_id_from_byte(3);
        store.register(a);
        store.register(b);
        store.register(c);

        let round = make_round(vec![a, b, c], a);
        let deltas = store.update_round(&round);

        let winner_delta = deltas.iter().find(|(n, _)| *n == a).unwrap().1;
        assert!(winner_delta > 0, "winner delta should be positive: {winner_delta}");

        for &(node, delta) in &deltas {
            if node != a {
                assert!(delta <= 0, "loser delta should be <= 0: {delta}");
            }
        }

        assert!(store.get(&a).unwrap().score > STARTING_ELO);
    }

    #[test]
    fn test_elo_floor() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        store.register(a);

        // Manually set to a low score.
        store.ratings_mut().get_mut(&a).unwrap().score = 110;

        // Slash should respect the floor.
        let new_score = store.slash(&a).unwrap();
        assert_eq!(new_score, ELO_FLOOR);
    }

    #[test]
    fn test_slash_penalty() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        store.register(a);

        let before = store.get(&a).unwrap().score;
        let after = store.slash(&a).unwrap();
        assert_eq!(after, before - SLASH_PENALTY);
    }

    #[test]
    fn test_slash_unknown_node() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        assert!(store.slash(&a).is_err());
    }

    #[test]
    fn test_k_factor_new_vs_established() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        store.register(a);

        // New node: K=32.
        assert_eq!(store.k_factor(&a), K_FACTOR_NEW);

        // Simulate 1000 rounds.
        store.ratings_mut().get_mut(&a).unwrap().total_rounds = 1000;
        assert_eq!(store.k_factor(&a), K_FACTOR_ESTABLISHED);
    }

    #[test]
    fn test_k_factor_boundary() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        store.register(a);

        store.ratings_mut().get_mut(&a).unwrap().total_rounds = 999;
        assert_eq!(store.k_factor(&a), K_FACTOR_NEW);

        store.ratings_mut().get_mut(&a).unwrap().total_rounds = 1000;
        assert_eq!(store.k_factor(&a), K_FACTOR_ESTABLISHED);
    }

    #[test]
    fn test_expected_score_equal_ratings() {
        let e = EloStore::expected_score(1500, 1500);
        assert!((e - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_expected_score_higher_rated_favored() {
        let e = EloStore::expected_score(1700, 1500);
        assert!(e > 0.5);
        // Symmetric: lower rated has complementary probability.
        let e2 = EloStore::expected_score(1500, 1700);
        assert!((e + e2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_domain_elo_tracking() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        store.register(a);
        store.register(b);

        let mut round = make_round(vec![a, b], a);
        round.domain = Some("code".to_string());
        store.update_round(&round);

        let a_rating = store.get(&a).unwrap();
        assert!(a_rating.domain_elos.contains_key("code"));
        assert!(a_rating.domain_elos["code"] > STARTING_ELO);
    }

    #[test]
    fn test_rounds_counting() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        store.register(a);
        store.register(b);

        let round = make_round(vec![a, b], a);
        store.update_round(&round);

        assert_eq!(store.get(&a).unwrap().total_rounds, 1);
        assert_eq!(store.get(&a).unwrap().rounds_won, 1);
        assert_eq!(store.get(&b).unwrap().total_rounds, 1);
        assert_eq!(store.get(&b).unwrap().rounds_won, 0);
    }

    #[test]
    fn test_multiple_rounds_elo_diverges() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        store.register(a);
        store.register(b);

        // A wins 10 rounds in a row.
        for _ in 0..10 {
            let round = make_round(vec![a, b], a);
            store.update_round(&round);
        }

        let a_elo = store.get(&a).unwrap().score;
        let b_elo = store.get(&b).unwrap().score;
        assert!(a_elo > b_elo + 100, "after 10 wins, A should be far ahead: A={a_elo} B={b_elo}");
    }
}
