use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::classifier::ComplexityClassifier;
use crate::registry::{ModelRegistry, NodeInfo};
use crate::types::*;

// ---------------------------------------------------------------------------
// Placeholder traits — will be replaced by real crate deps later
// ---------------------------------------------------------------------------

/// Verifies TOPLOC proofs attached to node responses.
#[async_trait]
pub trait ToplocVerifier: Send + Sync {
    /// Returns `true` if the proof is valid for the given answer.
    async fn verify(&self, answer: &str, proof: &[u8]) -> bool;
}

/// Placeholder TOPLOC verifier that accepts all proofs.
pub struct NoopToplocVerifier;

#[async_trait]
impl ToplocVerifier for NoopToplocVerifier {
    async fn verify(&self, _answer: &str, _proof: &[u8]) -> bool {
        true
    }
}

/// BradleyTerry aggregation engine: takes pairwise rankings and produces
/// per-node strength estimates.
pub struct BradleyTerryEngine;

impl BradleyTerryEngine {
    pub fn new() -> Self {
        Self
    }

    /// Run BradleyTerry MLE on pairwise rankings.
    /// Returns a map of node_id → estimated strength (probability of being the best).
    ///
    /// Uses iterative proportional fitting (IPF):
    ///   p_i ← (wins_i) / Σ_j (games_ij / (p_i + p_j))
    /// Normalised so Σ p_i = 1.
    pub fn aggregate(
        &self,
        rankings: &[PairwiseRanking],
        node_ids: &[NodeId],
    ) -> Vec<(NodeId, f32)> {
        if node_ids.is_empty() {
            return vec![];
        }
        if node_ids.len() == 1 {
            return vec![(node_ids[0], 1.0)];
        }

        let n = node_ids.len();
        // Index lookup
        let idx = |id: &NodeId| -> Option<usize> { node_ids.iter().position(|x| x == id) };

        // Accumulate wins and games matrices.
        let mut wins = vec![0.0f64; n];
        let mut games = vec![vec![0.0f64; n]; n];

        for r in rankings {
            let Some(ia) = idx(&r.node_a) else { continue };
            let Some(ib) = idx(&r.node_b) else { continue };
            wins[ia] += r.preference as f64;
            wins[ib] += (1.0 - r.preference) as f64;
            games[ia][ib] += 1.0;
            games[ib][ia] += 1.0;
        }

        // IPF iterations.
        let mut p = vec![1.0 / n as f64; n];
        for _ in 0..50 {
            let mut new_p = vec![0.0f64; n];
            for i in 0..n {
                let mut denom = 0.0f64;
                for j in 0..n {
                    if i == j || games[i][j] == 0.0 {
                        continue;
                    }
                    denom += games[i][j] / (p[i] + p[j]);
                }
                new_p[i] = if denom > 0.0 {
                    wins[i] / denom
                } else {
                    p[i]
                };
            }
            // Normalise.
            let sum: f64 = new_p.iter().sum();
            if sum > 0.0 {
                for v in &mut new_p {
                    *v /= sum;
                }
            }
            p = new_p;
        }

        node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, p[i] as f32))
            .collect()
    }
}

/// Placeholder node dispatcher — sends a prompt to a node and returns its response.
/// In production this will open an iroh QUIC stream.
#[async_trait]
pub trait NodeDispatcher: Send + Sync {
    async fn dispatch(&self, node: &NodeInfo, prompt: &str, max_tokens: usize, temperature: f32) -> Result<NodeResponse>;
}

/// Placeholder dispatcher that returns a canned response (useful for testing).
pub struct StubNodeDispatcher;

#[async_trait]
impl NodeDispatcher for StubNodeDispatcher {
    async fn dispatch(&self, node: &NodeInfo, prompt: &str, _max_tokens: usize, _temperature: f32) -> Result<NodeResponse> {
        Ok(NodeResponse {
            node_id: node.node_id,
            answer: format!("Response from {} for prompt: {}", node.model_name, &prompt[..prompt.len().min(40)]),
            tokens_generated: 32,
            generation_time_ms: 100,
            toploc_proof: Some(vec![0u8; 258]),
            model_name: node.model_name.clone(),
        })
    }
}

/// Placeholder peer ranker — in production each node produces pairwise rankings
/// via an LLM call. This stub produces synthetic preferences based on answer length.
#[async_trait]
pub trait PeerRanker: Send + Sync {
    async fn rank_pair(
        &self,
        ranker: &NodeInfo,
        a: &VerifiedResponse,
        b: &VerifiedResponse,
        domain_tags: &[String],
    ) -> Result<PairwiseRanking>;
}

pub struct StubPeerRanker;

#[async_trait]
impl PeerRanker for StubPeerRanker {
    async fn rank_pair(
        &self,
        ranker: &NodeInfo,
        a: &VerifiedResponse,
        b: &VerifiedResponse,
        domain_tags: &[String],
    ) -> Result<PairwiseRanking> {
        // Simple heuristic: prefer longer, faster answers.
        let score_a = a.tokens_generated as f64 / (a.generation_time_ms.max(1) as f64);
        let score_b = b.tokens_generated as f64 / (b.generation_time_ms.max(1) as f64);
        let total = score_a + score_b;
        let pref = if total > 0.0 {
            (score_a / total) as f32
        } else {
            0.5
        };

        Ok(PairwiseRanking {
            ranker_node_id: ranker.node_id,
            node_a: a.node_id,
            node_b: b.node_id,
            preference: pref,
            reasoning: format!(
                "Node A throughput {:.1} tok/s vs Node B {:.1} tok/s",
                score_a * 1000.0,
                score_b * 1000.0
            ),
            domain_tags: domain_tags.to_vec(),
        })
    }
}

// ---------------------------------------------------------------------------
// SwarmCoordinator
// ---------------------------------------------------------------------------

/// Core orchestration layer for Samhati swarm inference.
pub struct SwarmCoordinator {
    model_registry: Arc<ModelRegistry>,
    classifier: ComplexityClassifier,
    toploc_verifier: Arc<dyn ToplocVerifier>,
    ranking_engine: BradleyTerryEngine,
    dispatcher: Arc<dyn NodeDispatcher>,
    peer_ranker: Arc<dyn PeerRanker>,
    config: SwarmConfig,
}

impl SwarmCoordinator {
    /// Create a new coordinator with all dependencies injected.
    pub fn new(
        model_registry: Arc<ModelRegistry>,
        toploc_verifier: Arc<dyn ToplocVerifier>,
        dispatcher: Arc<dyn NodeDispatcher>,
        peer_ranker: Arc<dyn PeerRanker>,
        config: SwarmConfig,
    ) -> Self {
        Self {
            model_registry,
            classifier: ComplexityClassifier::new(),
            toploc_verifier,
            ranking_engine: BradleyTerryEngine::new(),
            dispatcher,
            peer_ranker,
            config,
        }
    }

    /// Convenience constructor with all stub/noop implementations.
    pub fn with_defaults(model_registry: Arc<ModelRegistry>, config: SwarmConfig) -> Self {
        Self::new(
            model_registry,
            Arc::new(NoopToplocVerifier),
            Arc::new(StubNodeDispatcher),
            Arc::new(StubPeerRanker),
            config,
        )
    }

    /// Full inference lifecycle.
    pub async fn infer(&self, request: InferenceRequest) -> Result<SwarmResult> {
        let start = Instant::now();

        // Step 1: classify complexity.
        let classification = self.classifier.classify(&request.prompt);
        let n = request.user_override_n.unwrap_or(classification.n_nodes);
        let domain_tags = if request.preferred_domains.is_empty() {
            classification.domain_tags.clone()
        } else {
            request.preferred_domains.clone()
        };

        info!(
            n_nodes = n,
            complexity = ?classification.complexity,
            domains = ?domain_tags,
            "Classified prompt"
        );

        // Step 2: select nodes.
        let nodes = self.model_registry.select_nodes(n, &domain_tags);
        if nodes.is_empty() {
            bail!("No nodes available in the model registry");
        }
        let actual_n = nodes.len();
        debug!(actual_n, "Selected nodes from registry");

        // Step 3: fan-out.
        let raw_responses = self
            .fan_out(&request.prompt, &nodes, request.max_tokens, request.temperature)
            .await
            .context("fan_out failed")?;

        // Step 4: fan-in (verify TOPLOC proofs).
        let verified = self.fan_in(raw_responses).await?;
        if verified.len() < self.config.min_responses {
            bail!(
                "Only {} verified responses, need at least {}",
                verified.len(),
                self.config.min_responses
            );
        }

        // Step 5 & 6: peer ranking + aggregation (skip if only 1 response).
        let (rankings, elo_deltas, winner_idx, confidence) = if verified.len() == 1 {
            (vec![], vec![], 0usize, 1.0f32)
        } else {
            let rankings = self.peer_rank(&nodes, &verified, &domain_tags).await?;
            let node_ids: Vec<NodeId> = verified.iter().map(|v| v.node_id).collect();
            let strengths = self.ranking_engine.aggregate(&rankings, &node_ids);

            // Pick winner (highest strength).
            let (winner_idx, &(_, confidence)) = strengths
                .iter()
                .enumerate()
                .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
                .unwrap();

            // ELO deltas: +16 for winner, -16/(n-1) for others (simplified K=32 system).
            let elo_k: i32 = 32;
            let n_others = (verified.len() - 1).max(1) as i32;
            let mut elo_deltas: Vec<(NodeId, i32)> = Vec::new();
            for (i, v) in verified.iter().enumerate() {
                let delta = if i == winner_idx {
                    elo_k / 2
                } else {
                    -(elo_k / 2) / n_others
                };
                elo_deltas.push((v.node_id, delta));
            }

            (rankings, elo_deltas, winner_idx, confidence)
        };

        // Apply ELO updates.
        for &(ref nid, delta) in &elo_deltas {
            self.model_registry.update_elo(nid, delta);
        }

        // Build training example if confidence is high enough.
        let training_example = if confidence >= self.config.training_confidence_threshold {
            let losing_answers: Vec<String> = verified
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != winner_idx)
                .map(|(_, v)| v.answer.clone())
                .collect();
            let reasoning_chains: Vec<String> =
                rankings.iter().map(|r| r.reasoning.clone()).collect();

            Some(TrainingExample {
                prompt: request.prompt.clone(),
                winning_answer: verified[winner_idx].answer.clone(),
                losing_answers,
                reasoning_chains,
                domain_tags: domain_tags.clone(),
                confidence,
            })
        } else {
            None
        };

        let total_time_ms = start.elapsed().as_millis() as u64;
        info!(total_time_ms, winner_idx, confidence, "Swarm inference complete");

        Ok(SwarmResult {
            winner_node_id: verified[winner_idx].node_id,
            winning_answer: verified[winner_idx].answer.clone(),
            confidence,
            all_responses: verified,
            rankings,
            elo_deltas,
            n_nodes: actual_n,
            total_time_ms,
            complexity: classification.complexity,
            domain_tags,
            training_example,
        })
    }

    /// Fan-out: dispatch the prompt to N nodes in parallel via tokio JoinSet.
    async fn fan_out(
        &self,
        prompt: &str,
        nodes: &[NodeInfo],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<NodeResponse>> {
        let mut join_set = JoinSet::new();

        for node in nodes {
            let dispatcher = Arc::clone(&self.dispatcher);
            let node = node.clone();
            let prompt = prompt.to_owned();
            let timeout = self.config.max_fan_out_timeout;

            join_set.spawn(async move {
                match tokio::time::timeout(timeout, dispatcher.dispatch(&node, &prompt, max_tokens, temperature))
                    .await
                {
                    Ok(Ok(resp)) => Some(resp),
                    Ok(Err(e)) => {
                        warn!(node_id = ?node.node_id[..4], error = %e, "Node dispatch failed");
                        None
                    }
                    Err(_) => {
                        warn!(node_id = ?node.node_id[..4], "Node dispatch timed out");
                        None
                    }
                }
            });
        }

        let mut responses = Vec::with_capacity(nodes.len());
        while let Some(result) = join_set.join_next().await {
            if let Ok(Some(resp)) = result {
                responses.push(resp);
            }
        }

        debug!(count = responses.len(), "Fan-out collected responses");
        Ok(responses)
    }

    /// Fan-in: verify TOPLOC proofs on all responses.
    async fn fan_in(&self, responses: Vec<NodeResponse>) -> Result<Vec<VerifiedResponse>> {
        let mut verified = Vec::with_capacity(responses.len());

        for resp in responses {
            let proof_valid = match &resp.toploc_proof {
                Some(proof) => self.toploc_verifier.verify(&resp.answer, proof).await,
                None => !self.config.toploc_required,
            };

            if self.config.toploc_required && !proof_valid {
                warn!(node_id = ?resp.node_id[..4], "Discarding response: invalid TOPLOC proof");
                continue;
            }

            verified.push(VerifiedResponse {
                node_id: resp.node_id,
                answer: resp.answer,
                tokens_generated: resp.tokens_generated,
                generation_time_ms: resp.generation_time_ms,
                proof_valid,
                model_name: resp.model_name,
            });
        }

        debug!(count = verified.len(), "Fan-in verified responses");
        Ok(verified)
    }

    /// Phase 2: every node produces C(N,2) pairwise rankings.
    async fn peer_rank(
        &self,
        nodes: &[NodeInfo],
        responses: &[VerifiedResponse],
        domain_tags: &[String],
    ) -> Result<Vec<PairwiseRanking>> {
        let mut join_set = JoinSet::new();

        for node in nodes {
            for i in 0..responses.len() {
                for j in (i + 1)..responses.len() {
                    let ranker = Arc::clone(&self.peer_ranker);
                    let node = node.clone();
                    let a = responses[i].clone();
                    let b = responses[j].clone();
                    let tags = domain_tags.to_vec();

                    join_set.spawn(async move {
                        ranker.rank_pair(&node, &a, &b, &tags).await
                    });
                }
            }
        }

        let mut rankings = Vec::new();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok(ranking)) => rankings.push(ranking),
                Ok(Err(e)) => warn!(error = %e, "Pairwise ranking failed"),
                Err(e) => warn!(error = %e, "Ranking task panicked"),
            }
        }

        debug!(count = rankings.len(), "Peer ranking complete");
        Ok(rankings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::NodeInfo;

    fn test_registry(n: u8) -> Arc<ModelRegistry> {
        let reg = ModelRegistry::new();
        for i in 0..n {
            let mut node_id = [0u8; 32];
            node_id[0] = i;
            reg.register(NodeInfo {
                node_id,
                endpoint: format!("iroh://node-{i}"),
                model_name: format!("test-model-{i}"),
                model_size_b: 3,
                domain_tags: vec!["code".into()],
                tokens_per_sec: 50.0,
                elo_score: 1500 + (i as i32) * 10,
                last_seen: Instant::now(),
                solana_pubkey: None,
            });
        }
        Arc::new(reg)
    }

    #[tokio::test]
    async fn single_node_infer() {
        let reg = test_registry(1);
        let coord = SwarmCoordinator::with_defaults(reg, SwarmConfig::default());

        let result = coord
            .infer(InferenceRequest {
                prompt: "hello".into(),
                max_tokens: 100,
                temperature: 0.7,
                user_override_n: Some(1),
                preferred_domains: vec![],
            })
            .await
            .unwrap();

        assert_eq!(result.n_nodes, 1);
        assert_eq!(result.confidence, 1.0);
        assert!(!result.winning_answer.is_empty());
    }

    #[tokio::test]
    async fn multi_node_infer() {
        let reg = test_registry(5);
        let coord = SwarmCoordinator::with_defaults(reg, SwarmConfig::default());

        let result = coord
            .infer(InferenceRequest {
                prompt: "Explain step by step how to prove the Pythagorean theorem and solve for the hypotenuse given sides 3 and 4".into(),
                max_tokens: 256,
                temperature: 0.7,
                user_override_n: None,
                preferred_domains: vec!["math".into()],
            })
            .await
            .unwrap();

        assert!(result.n_nodes >= 3);
        assert!(result.all_responses.len() >= 1);
        assert!(!result.rankings.is_empty());
        assert!(!result.elo_deltas.is_empty());
    }

    #[tokio::test]
    async fn empty_registry_errors() {
        let reg = Arc::new(ModelRegistry::new());
        let coord = SwarmCoordinator::with_defaults(reg, SwarmConfig::default());

        let err = coord
            .infer(InferenceRequest {
                prompt: "hello".into(),
                max_tokens: 100,
                temperature: 0.7,
                user_override_n: None,
                preferred_domains: vec![],
            })
            .await;

        assert!(err.is_err());
    }

    #[test]
    fn bradley_terry_single_node() {
        let engine = BradleyTerryEngine::new();
        let nid = [1u8; 32];
        let result = engine.aggregate(&[], &[nid]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 1.0);
    }

    #[test]
    fn bradley_terry_two_nodes() {
        let engine = BradleyTerryEngine::new();
        let a = [1u8; 32];
        let b = [2u8; 32];
        // A always beats B
        let rankings = vec![
            PairwiseRanking {
                ranker_node_id: [0u8; 32],
                node_a: a,
                node_b: b,
                preference: 1.0,
                reasoning: "A is better".into(),
                domain_tags: vec![],
            },
            PairwiseRanking {
                ranker_node_id: [0u8; 32],
                node_a: a,
                node_b: b,
                preference: 0.9,
                reasoning: "A is better".into(),
                domain_tags: vec![],
            },
        ];
        let result = engine.aggregate(&rankings, &[a, b]);
        // A should have higher strength
        assert!(result[0].1 > result[1].1);
    }
}
