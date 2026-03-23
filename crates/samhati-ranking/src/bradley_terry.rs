use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use tracing::debug;

use crate::types::NodeId;

/// A single pairwise comparison between two nodes, as judged by a ranking node.
#[derive(Debug, Clone)]
pub struct PairwiseComparison {
    /// First node in the comparison.
    pub node_a: NodeId,
    /// Second node in the comparison.
    pub node_b: NodeId,
    /// Probability that node A beats node B, in [0, 1].
    pub prob_a_wins: f64,
    /// ELO rating of the ranking node that produced this comparison (used as weight).
    pub ranker_elo: i32,
}

/// Result of BradleyTerry aggregation over a set of pairwise comparisons.
#[derive(Debug, Clone)]
pub struct RankingResult {
    /// Estimated strength parameter beta_i for each node.
    pub strengths: HashMap<NodeId, f64>,
    /// Node with the highest strength (argmax).
    pub winner: NodeId,
    /// Win probability of the winner = exp(beta_winner) / sum(exp(beta_all)).
    pub win_probability: f64,
    /// Number of MM iterations used before convergence.
    pub iterations: usize,
}

/// BradleyTerry maximum-likelihood engine using the MM algorithm.
#[derive(Debug, Clone)]
pub struct BradleyTerryEngine {
    /// Maximum number of MM iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on max|beta_new - beta_old|.
    pub tolerance: f64,
}

impl Default for BradleyTerryEngine {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

impl BradleyTerryEngine {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Aggregate pairwise comparisons into strength estimates via MLE (MM algorithm).
    ///
    /// Each comparison is weighted by the ranker's ELO. The MM update for node i:
    ///   numerator   = sum of weighted wins for i
    ///   denominator = sum of weight / (exp(beta_i) + exp(beta_j)) for all comparisons involving i
    ///   beta_i_new  = ln(numerator / denominator)
    ///
    /// After each iteration, betas are mean-centered for identifiability.
    pub fn aggregate(&self, comparisons: &[PairwiseComparison]) -> Result<RankingResult> {
        if comparisons.is_empty() {
            bail!("no comparisons provided");
        }

        // Collect all unique node ids.
        let mut node_set = HashSet::new();
        for c in comparisons {
            node_set.insert(c.node_a);
            node_set.insert(c.node_b);
        }
        let nodes: Vec<NodeId> = node_set.into_iter().collect();

        if nodes.len() < 2 {
            bail!("need at least 2 distinct nodes, got {}", nodes.len());
        }

        // Initialize all beta_i = 0.
        let mut beta: HashMap<NodeId, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();

        let mut iterations = 0;

        for iter in 0..self.max_iterations {
            let mut new_beta: HashMap<NodeId, f64> = HashMap::new();

            for &node in &nodes {
                let mut numerator = 0.0f64;
                let mut denominator = 0.0f64;

                for c in comparisons {
                    let weight = c.ranker_elo.max(1) as f64;

                    if c.node_a == node {
                        // node is A: wins with prob_a_wins
                        numerator += weight * c.prob_a_wins;
                        let exp_i = beta[&c.node_a].exp();
                        let exp_j = beta[&c.node_b].exp();
                        denominator += weight / (exp_i + exp_j);
                    } else if c.node_b == node {
                        // node is B: wins with (1 - prob_a_wins)
                        numerator += weight * (1.0 - c.prob_a_wins);
                        let exp_i = beta[&c.node_b].exp();
                        let exp_j = beta[&c.node_a].exp();
                        denominator += weight / (exp_i + exp_j);
                    }
                }

                if denominator <= 0.0 || numerator <= 0.0 {
                    // Node has no wins or no comparisons — assign -inf effectively.
                    new_beta.insert(node, f64::NEG_INFINITY);
                } else {
                    new_beta.insert(node, (numerator / denominator).ln());
                }
            }

            // Normalize: subtract the mean of all finite betas.
            let finite_betas: Vec<f64> = new_beta.values().copied().filter(|v| v.is_finite()).collect();
            if finite_betas.is_empty() {
                bail!("all node strengths diverged to -infinity");
            }
            let mean = finite_betas.iter().sum::<f64>() / finite_betas.len() as f64;
            for v in new_beta.values_mut() {
                if v.is_finite() {
                    *v -= mean;
                }
            }

            // Check convergence.
            let max_delta = nodes
                .iter()
                .map(|n| {
                    let old = beta[n];
                    let new = new_beta[n];
                    if old.is_finite() && new.is_finite() {
                        (new - old).abs()
                    } else {
                        0.0
                    }
                })
                .fold(0.0f64, f64::max);

            beta = new_beta;
            iterations = iter + 1;

            debug!(iteration = iterations, max_delta, "BT MM iteration");

            if max_delta < self.tolerance {
                break;
            }
        }

        // Find winner = argmax(beta).
        let (&winner, &winner_beta) = beta
            .iter()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .ok_or_else(|| anyhow::anyhow!("no finite strengths"))?;

        // Win probability = softmax of winner.
        let max_beta = winner_beta; // for numerical stability
        let exp_sum: f64 = beta
            .values()
            .map(|&b| {
                if b.is_finite() {
                    (b - max_beta).exp()
                } else {
                    0.0
                }
            })
            .sum();
        let win_probability = 1.0 / exp_sum;

        debug!(
            ?winner,
            win_probability, iterations, "BT aggregation complete"
        );

        Ok(RankingResult {
            strengths: beta,
            winner,
            win_probability,
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::node_id_from_byte;

    #[test]
    fn test_two_nodes_clear_winner() {
        let engine = BradleyTerryEngine::default();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);

        // A beats B with high probability across multiple rankers.
        let comparisons = vec![
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.9, ranker_elo: 1500 },
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.85, ranker_elo: 1600 },
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.95, ranker_elo: 1400 },
        ];

        let result = engine.aggregate(&comparisons).unwrap();
        assert_eq!(result.winner, a);
        assert!(result.strengths[&a] > result.strengths[&b]);
        assert!(result.win_probability > 0.5);
    }

    #[test]
    fn test_two_nodes_close_match() {
        let engine = BradleyTerryEngine::default();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);

        let comparisons = vec![
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.51, ranker_elo: 1500 },
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.49, ranker_elo: 1500 },
        ];

        let result = engine.aggregate(&comparisons).unwrap();
        // Strengths should be close together.
        let diff = (result.strengths[&a] - result.strengths[&b]).abs();
        assert!(diff < 0.5, "diff should be small for a close match: {diff}");
    }

    #[test]
    fn test_three_nodes() {
        let engine = BradleyTerryEngine::default();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);
        let c = node_id_from_byte(3);

        let comparisons = vec![
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.8, ranker_elo: 1500 },
            PairwiseComparison { node_a: a, node_b: c, prob_a_wins: 0.9, ranker_elo: 1500 },
            PairwiseComparison { node_a: b, node_b: c, prob_a_wins: 0.7, ranker_elo: 1500 },
        ];

        let result = engine.aggregate(&comparisons).unwrap();
        assert_eq!(result.winner, a);
        assert!(result.strengths[&a] > result.strengths[&b]);
        assert!(result.strengths[&b] > result.strengths[&c]);
    }

    #[test]
    fn test_five_nodes() {
        let engine = BradleyTerryEngine::default();
        let nodes: Vec<NodeId> = (1..=5).map(node_id_from_byte).collect();

        // Node 1 is strongest, then 2, 3, 4, 5.
        let mut comparisons = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                // Lower index = stronger, so prob of i beating j increases with gap.
                let prob = 0.5 + 0.08 * (j - i) as f64;
                comparisons.push(PairwiseComparison {
                    node_a: nodes[i],
                    node_b: nodes[j],
                    prob_a_wins: prob.min(0.99),
                    ranker_elo: 1500,
                });
            }
        }

        let result = engine.aggregate(&comparisons).unwrap();
        assert_eq!(result.winner, nodes[0]);

        // Verify ordering.
        for i in 0..4 {
            assert!(
                result.strengths[&nodes[i]] > result.strengths[&nodes[i + 1]],
                "node {} should be stronger than node {}",
                i + 1,
                i + 2
            );
        }
    }

    #[test]
    fn test_convergence() {
        let engine = BradleyTerryEngine::new(1000, 1e-10);
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);

        let comparisons = vec![
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.7, ranker_elo: 1500 },
        ];

        let result = engine.aggregate(&comparisons).unwrap();
        // Should converge well within max_iterations.
        assert!(result.iterations < 1000);
    }

    #[test]
    fn test_ranker_elo_weighting() {
        let engine = BradleyTerryEngine::default();
        let a = node_id_from_byte(1);
        let b = node_id_from_byte(2);

        // Low-ELO ranker says B wins, high-ELO ranker says A wins.
        let comparisons = vec![
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.2, ranker_elo: 100 },
            PairwiseComparison { node_a: a, node_b: b, prob_a_wins: 0.9, ranker_elo: 2000 },
        ];

        let result = engine.aggregate(&comparisons).unwrap();
        // The high-ELO ranker's opinion should dominate.
        assert_eq!(result.winner, a);
    }

    #[test]
    fn test_empty_comparisons() {
        let engine = BradleyTerryEngine::default();
        assert!(engine.aggregate(&[]).is_err());
    }

    #[test]
    fn test_single_node_errors() {
        let engine = BradleyTerryEngine::default();
        let a = node_id_from_byte(1);
        let comparisons = vec![
            PairwiseComparison { node_a: a, node_b: a, prob_a_wins: 0.5, ranker_elo: 1500 },
        ];
        // Only one distinct node — should error.
        assert!(engine.aggregate(&comparisons).is_err());
    }
}
