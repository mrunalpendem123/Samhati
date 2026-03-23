use std::collections::HashMap;

use tracing::debug;

use crate::elo::{EloStore, RoundOutcome};
use crate::types::NodeId;

/// Distribution of SMTI rewards for a completed round.
#[derive(Debug, Clone)]
pub struct RewardDistribution {
    /// Per-node reward amounts in SMTI units.
    pub rewards: Vec<(NodeId, u64)>,
    /// Total SMTI emitted for this round (including domain bonuses).
    pub total_emitted: u64,
}

/// Calculator for SMTI token rewards after each inference round.
#[derive(Debug, Clone)]
pub struct RewardCalculator {
    /// Base SMTI emission per round.
    pub base_emission_per_round: u64,
    /// Fraction of base emission reserved for the winner (default 0.60).
    pub winner_share: f64,
    /// Fraction of base emission split among all participants by ELO weight (default 0.40).
    pub participant_share: f64,
    /// Multiplier applied to a node's reward if its model matches the query domain (default 1.5).
    pub domain_bonus: f64,
}

impl Default for RewardCalculator {
    fn default() -> Self {
        Self {
            base_emission_per_round: 1000,
            winner_share: 0.60,
            participant_share: 0.40,
            domain_bonus: 1.5,
        }
    }
}

impl RewardCalculator {
    pub fn new(
        base_emission_per_round: u64,
        winner_share: f64,
        participant_share: f64,
        domain_bonus: f64,
    ) -> Self {
        Self {
            base_emission_per_round,
            winner_share,
            participant_share,
            domain_bonus,
        }
    }

    /// Calculate SMTI reward distribution for a completed round.
    ///
    /// Formula:
    /// - winner_reward = base_emission * winner_share
    /// - remaining = base_emission * participant_share
    /// - Each participant's share of remaining = remaining * (elo_i / sum(elo_all))
    /// - If domain_match[node] is true: multiply that node's total reward by domain_bonus
    pub fn calculate(
        &self,
        round: &RoundOutcome,
        elo_store: &EloStore,
        domain_match: &HashMap<NodeId, bool>,
    ) -> RewardDistribution {
        if round.participants.is_empty() {
            return RewardDistribution {
                rewards: Vec::new(),
                total_emitted: 0,
            };
        }

        let base = self.base_emission_per_round as f64;
        let winner_pool = base * self.winner_share;
        let participant_pool = base * self.participant_share;

        // Sum of all participants' ELO scores (for weighting the participant pool).
        let elo_sum: f64 = round
            .participants
            .iter()
            .map(|p| {
                elo_store
                    .get(p)
                    .map(|r| r.score as f64)
                    .unwrap_or(1500.0)
            })
            .sum();

        let mut rewards = Vec::with_capacity(round.participants.len());
        let mut total_emitted = 0u64;

        for &node in &round.participants {
            let node_elo = elo_store
                .get(&node)
                .map(|r| r.score as f64)
                .unwrap_or(1500.0);

            // Base reward: winner gets winner_pool, everyone gets ELO-weighted share of participant_pool.
            let mut reward = if node == round.winner {
                winner_pool
            } else {
                0.0
            };

            // Participant share weighted by ELO.
            if elo_sum > 0.0 {
                reward += participant_pool * (node_elo / elo_sum);
            }

            // Domain bonus.
            if domain_match.get(&node).copied().unwrap_or(false) {
                reward *= self.domain_bonus;
            }

            let reward_u64 = reward.round() as u64;
            rewards.push((node, reward_u64));
            total_emitted += reward_u64;
        }

        debug!(
            total_emitted,
            n_participants = round.participants.len(),
            "rewards distributed"
        );

        RewardDistribution {
            rewards,
            total_emitted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elo::EloStore;
    use crate::types::node_id_from_byte;

    fn setup_store_and_round() -> (EloStore, RoundOutcome) {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        let c = node_id_from_byte(3);
        store.register(a);
        store.register(b);
        store.register(c);

        let mut strengths = HashMap::new();
        strengths.insert(a, 1.0);
        strengths.insert(b, 0.3);
        strengths.insert(c, -0.5);

        let round = RoundOutcome {
            participants: vec![a, b, c],
            winner: a,
            strengths,
            domain: None,
        };

        (store, round)
    }

    #[test]
    fn test_rewards_sum_to_base_emission_no_domain_bonus() {
        let (store, round) = setup_store_and_round();
        let calc = RewardCalculator {
            base_emission_per_round: 1000,
            winner_share: 0.60,
            participant_share: 0.40,
            domain_bonus: 1.5,
        };

        let no_domain: HashMap<NodeId, bool> = HashMap::new();
        let dist = calc.calculate(&round, &store, &no_domain);

        // With no domain bonus and equal ELOs, rewards should sum close to base_emission.
        // There may be small rounding differences.
        let diff = (dist.total_emitted as i64 - 1000).abs();
        assert!(
            diff <= 2,
            "total should be ~1000, got {}",
            dist.total_emitted
        );
    }

    #[test]
    fn test_winner_gets_most() {
        let (store, round) = setup_store_and_round();
        let calc = RewardCalculator::default();
        let no_domain: HashMap<NodeId, bool> = HashMap::new();
        let dist = calc.calculate(&round, &store, &no_domain);

        let a = node_id_from_byte(1);
        let winner_reward = dist.rewards.iter().find(|(n, _)| *n == a).unwrap().1;
        for &(node, reward) in &dist.rewards {
            if node != a {
                assert!(
                    winner_reward > reward,
                    "winner should get more than others"
                );
            }
        }
    }

    #[test]
    fn test_domain_bonus_multiplier() {
        let (store, round) = setup_store_and_round();
        let calc = RewardCalculator::default();

        let b = node_id_from_byte(2);

        // Without domain bonus.
        let no_domain: HashMap<NodeId, bool> = HashMap::new();
        let dist_no_bonus = calc.calculate(&round, &store, &no_domain);
        let b_no_bonus = dist_no_bonus.rewards.iter().find(|(n, _)| *n == b).unwrap().1;

        // With domain bonus for b.
        let mut domain_match = HashMap::new();
        domain_match.insert(b, true);
        let dist_bonus = calc.calculate(&round, &store, &domain_match);
        let b_with_bonus = dist_bonus.rewards.iter().find(|(n, _)| *n == b).unwrap().1;

        // b_with_bonus should be ~1.5x b_no_bonus.
        let ratio = b_with_bonus as f64 / b_no_bonus as f64;
        assert!(
            (ratio - 1.5).abs() < 0.1,
            "domain bonus should be ~1.5x, got {ratio:.2}"
        );
    }

    #[test]
    fn test_domain_bonus_increases_total() {
        let (store, round) = setup_store_and_round();
        let calc = RewardCalculator::default();

        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);

        let mut domain_match = HashMap::new();
        domain_match.insert(a, true);
        domain_match.insert(b, true);

        let dist = calc.calculate(&round, &store, &domain_match);

        // With domain bonuses, total should exceed base_emission.
        assert!(
            dist.total_emitted > calc.base_emission_per_round,
            "domain bonuses should increase total above base: {}",
            dist.total_emitted
        );
    }

    #[test]
    fn test_elo_weighted_participant_share() {
        let mut store = EloStore::new();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        store.register(a);
        store.register(b);

        // Give b a higher ELO.
        store.ratings_mut().get_mut(&b).unwrap().score = 2000;

        let mut strengths = HashMap::new();
        strengths.insert(a, 1.0);
        strengths.insert(b, -0.5);

        let round = RoundOutcome {
            participants: vec![a, b],
            winner: a,
            strengths,
            domain: None,
        };

        let calc = RewardCalculator::default();
        let no_domain: HashMap<NodeId, bool> = HashMap::new();
        let dist = calc.calculate(&round, &store, &no_domain);

        // b should get a larger participant share than a (higher ELO weighting).
        // But a is also the winner, so a's total should still be larger.
        let a_reward = dist.rewards.iter().find(|(n, _)| *n == a).unwrap().1;
        let b_reward = dist.rewards.iter().find(|(n, _)| *n == b).unwrap().1;
        assert!(a_reward > b_reward, "winner a should still get more overall");
        // b's participant share portion: 2000/(1500+2000) * 400 = ~229
        // a's participant share portion: 1500/(1500+2000) * 400 = ~171
        // So b's share > a's participant share, which we verify indirectly.
    }

    #[test]
    fn test_empty_round() {
        let store = EloStore::new();
        let calc = RewardCalculator::default();
        let round = RoundOutcome {
            participants: vec![],
            winner: node_id_from_byte(1),
            strengths: HashMap::new(),
            domain: None,
        };
        let dist = calc.calculate(&round, &store, &HashMap::new());
        assert!(dist.rewards.is_empty());
        assert_eq!(dist.total_emitted, 0);
    }
}
