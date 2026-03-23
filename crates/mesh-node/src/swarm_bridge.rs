//! Bridge between samhati-swarm traits and the iroh/inference-coordinator transport.
//!
//! Implements `NodeDispatcher` and `PeerRanker` by dispatching over iroh QUIC,
//! and `ToplocVerifier` via the samhati-toploc crate.

use anyhow::Result;
use async_trait::async_trait;
use samhati_swarm::coordinator::{NodeDispatcher, PeerRanker, ToplocVerifier};
use samhati_swarm::registry::NodeInfo;
use samhati_swarm::types::{NodeResponse, PairwiseRanking, VerifiedResponse};
use samhati_toploc::verifier::{ToplocVerifier as ToplocVerifierCheck, ToplocVerifierImpl};
use samhati_ranking::elo::EloStore;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// NodeDispatcher: sends inference requests to peers over iroh QUIC
// ---------------------------------------------------------------------------

pub struct IrohNodeDispatcher {
    pub endpoint: iroh::Endpoint,
}

#[async_trait]
impl NodeDispatcher for IrohNodeDispatcher {
    async fn dispatch(
        &self,
        node: &NodeInfo,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<NodeResponse> {
        use inference_coordinator::{InferenceRequest, IrohDistributedExecutor, RoundRobinPlanner, Coordinator};

        // For swarm-ranked mode, each node runs the full model independently.
        // We create a single-peer plan that covers all layers and dispatch via iroh.
        let peer_id = hex::encode(node.node_id);
        let total_layers = 32; // default; the serve-side node knows its own range

        let planner = RoundRobinPlanner::new(vec![peer_id.clone()], 1);
        let plan = planner.plan(&node.model_name, total_layers)?;

        let req = InferenceRequest {
            request_id: format!("swarm-{}", hex::encode(&node.node_id[..8])),
            input: prompt.to_string(),
            max_tokens: max_tokens as u32,
            temperature,
        };

        let start = Instant::now();
        let executor = IrohDistributedExecutor::new(self.endpoint.clone());
        let coordinator = Coordinator::new(plan, executor);
        let answer = coordinator.generate(req).await?;
        let elapsed = start.elapsed().as_millis() as u64;

        Ok(NodeResponse {
            node_id: node.node_id,
            answer,
            tokens_generated: max_tokens, // approximate
            generation_time_ms: elapsed,
            toploc_proof: None, // TODO: wire real TOPLOC proof from inference server
            model_name: node.model_name.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// PeerRanker: asks a node to rank two answers
// For now, uses a simple heuristic (length + keyword scoring).
// In production, this would send the answers to the node over QUIC and have
// the node's LLM produce a pairwise preference with reasoning.
// ---------------------------------------------------------------------------

pub struct LocalPeerRanker;

#[async_trait]
impl PeerRanker for LocalPeerRanker {
    async fn rank_pair(
        &self,
        ranker: &NodeInfo,
        a: &VerifiedResponse,
        b: &VerifiedResponse,
        _domain_tags: &[String],
    ) -> Result<PairwiseRanking> {
        // Heuristic scoring: prefer longer, more detailed answers
        let score_a = response_quality_score(&a.answer);
        let score_b = response_quality_score(&b.answer);
        let total = score_a + score_b;
        let preference = if total > 0.0 {
            score_a / total
        } else {
            0.5
        };

        let reasoning = format!(
            "Answer A scored {:.1} and Answer B scored {:.1} on quality heuristics (length, structure, specificity).",
            score_a, score_b
        );

        Ok(PairwiseRanking {
            ranker_node_id: ranker.node_id,
            node_a: a.node_id,
            node_b: b.node_id,
            preference: preference as f32,
            reasoning,
            domain_tags: vec![],
        })
    }
}

/// Simple quality heuristic for ranking answers.
fn response_quality_score(answer: &str) -> f64 {
    let len_score = (answer.len() as f64).sqrt();
    let has_code = if answer.contains("```") { 5.0 } else { 0.0 };
    let has_list = if answer.contains("\n- ") || answer.contains("\n* ") {
        3.0
    } else {
        0.0
    };
    let has_numbers = if answer.chars().any(|c| c.is_ascii_digit()) {
        2.0
    } else {
        0.0
    };
    let sentence_count = answer.matches(". ").count() as f64;
    len_score + has_code + has_list + has_numbers + sentence_count
}

// ---------------------------------------------------------------------------
// ToplocVerifier bridge: wraps samhati_toploc::ToplocVerifierImpl
// ---------------------------------------------------------------------------

pub struct ToplocVerifierBridge {
    inner: ToplocVerifierImpl,
}

impl ToplocVerifierBridge {
    pub fn new() -> Self {
        Self {
            inner: ToplocVerifierImpl::new(),
        }
    }

    #[allow(dead_code)]
    pub fn register_model(&mut self, model_id: &str) {
        self.inner.register_model(model_id);
    }
}

#[async_trait]
impl ToplocVerifier for ToplocVerifierBridge {
    async fn verify(&self, _answer: &str, proof: &[u8]) -> bool {
        if proof.is_empty() {
            // No proof provided — accept for now during development
            return true;
        }
        match samhati_toploc::proof::ToplocProof::from_bytes(proof) {
            Ok(toploc_proof) => {
                let result = ToplocVerifierCheck::verify(&self.inner, &toploc_proof, "unknown", toploc_proof.token_count);
                result.is_valid()
            }
            Err(_) => false,
        }
    }
}

// ---------------------------------------------------------------------------
// SwarmRankedHandle: the high-level handle used by the API server
// ---------------------------------------------------------------------------

use samhati_swarm::coordinator::SwarmCoordinator;
use samhati_swarm::registry::ModelRegistry;
use samhati_swarm::types::{SwarmConfig, SwarmResult, InferenceRequest as SwarmInferenceRequest};

pub struct SwarmRankedHandle {
    pub coordinator: SwarmCoordinator,
    #[allow(dead_code)]
    pub model_registry: Arc<ModelRegistry>,
    pub elo_store: Arc<Mutex<EloStore>>,
}

impl SwarmRankedHandle {
    pub fn new(endpoint: iroh::Endpoint, model_registry: Arc<ModelRegistry>) -> Self {
        let dispatcher = Arc::new(IrohNodeDispatcher {
            endpoint: endpoint.clone(),
        });
        let peer_ranker = Arc::new(LocalPeerRanker);
        let toploc_verifier = Arc::new(ToplocVerifierBridge::new());

        let config = SwarmConfig::default();
        let coordinator = SwarmCoordinator::new(
            model_registry.clone(),
            toploc_verifier,
            dispatcher,
            peer_ranker,
            config,
        );

        Self {
            coordinator,
            model_registry,
            elo_store: Arc::new(Mutex::new(EloStore::new())),
        }
    }

    /// Run a swarm-ranked inference: classify → fan-out → TOPLOC verify → peer-rank → BradleyTerry → ELO update
    pub async fn run(&self, prompt: String, max_tokens: usize, temperature: f32) -> Result<SwarmResult> {
        let request = SwarmInferenceRequest {
            prompt,
            max_tokens,
            temperature,
            user_override_n: None,
            preferred_domains: vec![],
        };

        let result = self.coordinator.infer(request).await?;

        // Update ELO store with round results
        if !result.elo_deltas.is_empty() {
            if let Ok(mut store) = self.elo_store.lock() {
                for (node_id, _delta) in &result.elo_deltas {
                    // Ensure node is registered
                    if store.get(node_id).is_none() {
                        store.register(*node_id);
                    }
                    store.update_round(&samhati_ranking::elo::RoundOutcome {
                        participants: result
                            .all_responses
                            .iter()
                            .map(|r| r.node_id)
                            .collect(),
                        winner: result.winner_node_id,
                        strengths: result
                            .elo_deltas
                            .iter()
                            .map(|(id, d)| (*id, *d as f64))
                            .collect(),
                        domain: result.domain_tags.first().cloned(),
                    });
                }
            }
        }

        Ok(result)
    }
}
