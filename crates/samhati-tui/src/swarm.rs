use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// A node endpoint (llama-server instance)
#[derive(Debug, Clone)]
pub struct SwarmNode {
    pub id: String,
    pub url: String,
    pub model: String,
    pub elo: i32,
    pub rounds_played: u64,
    pub rounds_won: u64,
}

/// Result of a swarm inference round
#[derive(Debug, Clone)]
pub struct SwarmRoundResult {
    pub winner_id: String,
    pub winning_answer: String,
    pub confidence: f32,
    pub all_answers: Vec<NodeAnswer>,
    pub elo_updates: Vec<(String, i32)>,
    pub n_nodes: usize,
    pub total_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct NodeAnswer {
    pub node_id: String,
    pub answer: String,
    pub latency_ms: u64,
    pub model: String,
}

/// The swarm orchestrator — manages nodes and coordinates inference.
///
/// Uses `Arc<Mutex<..>>` for the node list so that `infer()` can take `&self`
/// (required because the orchestrator is held across `.await` points in main).
pub struct SwarmOrchestrator {
    nodes: Arc<Mutex<Vec<SwarmNode>>>,
    client: reqwest::Client,
}

impl SwarmOrchestrator {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(Mutex::new(Vec::new())),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Register a node endpoint.
    pub fn add_node(&self, id: String, url: String, model: String) {
        if let Ok(mut nodes) = self.nodes.lock() {
            nodes.push(SwarmNode {
                id,
                url,
                model,
                elo: 1500,
                rounds_played: 0,
                rounds_won: 0,
            });
        }
    }

    /// Remove a node by id.
    pub fn remove_node(&self, id: &str) {
        if let Ok(mut nodes) = self.nodes.lock() {
            nodes.retain(|n| n.id != id);
        }
    }

    /// Get number of active nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.lock().map(|n| n.len()).unwrap_or(0)
    }

    /// Snapshot of current nodes (for display).
    pub fn snapshot_nodes(&self) -> Vec<SwarmNode> {
        self.nodes.lock().map(|n| n.clone()).unwrap_or_default()
    }

    /// Run swarm inference: fan out to all nodes, collect answers, BradleyTerry rank.
    pub async fn infer(&self, prompt: &str) -> Result<SwarmRoundResult> {
        let node_snapshot = {
            let guard = self.nodes.lock().map_err(|e| anyhow::anyhow!("lock: {}", e))?;
            guard.clone()
        };

        if node_snapshot.is_empty() {
            return Err(anyhow::anyhow!("No nodes registered in swarm"));
        }

        let start = Instant::now();
        let n_nodes = node_snapshot.len();

        // Phase 1: Fan out — send prompt to all nodes in parallel
        let mut handles = Vec::new();
        for node in &node_snapshot {
            let client = self.client.clone();
            let url = node.url.clone();
            let node_id = node.id.clone();
            let model = node.model.clone();
            let prompt_owned = prompt.to_string();

            handles.push(tokio::spawn(async move {
                let node_start = Instant::now();
                let result = call_node(&client, &url, &prompt_owned).await;
                let latency = node_start.elapsed().as_millis() as u64;
                (node_id, model, result, latency)
            }));
        }

        // Collect all answers
        let mut answers: Vec<NodeAnswer> = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((node_id, model, Ok(answer), latency)) => {
                    answers.push(NodeAnswer {
                        node_id,
                        answer,
                        latency_ms: latency,
                        model,
                    });
                }
                Ok((node_id, _, Err(e), _)) => {
                    eprintln!("Node {} failed: {}", node_id, e);
                }
                Err(e) => {
                    eprintln!("Task join error: {}", e);
                }
            }
        }

        if answers.is_empty() {
            return Err(anyhow::anyhow!("All nodes failed to respond"));
        }

        // If only 1 answer, return it directly
        if answers.len() == 1 {
            let winner = &answers[0];
            return Ok(SwarmRoundResult {
                winner_id: winner.node_id.clone(),
                winning_answer: winner.answer.clone(),
                confidence: 1.0,
                all_answers: answers,
                elo_updates: vec![],
                n_nodes,
                total_time_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Phase 2: Score answers using quality heuristics
        let scores = rank_answers(&answers);

        // Phase 3: BradleyTerry aggregation — pick winner
        let (winner_idx, confidence) = bradley_terry_select(&scores);
        let winner = &answers[winner_idx];

        // Phase 4: ELO updates (mutate shared state)
        let elo_updates = {
            let mut guard = self.nodes.lock().map_err(|e| anyhow::anyhow!("lock: {}", e))?;
            update_elo(&mut guard, &answers, winner_idx)
        };

        let total_time = start.elapsed().as_millis() as u64;

        Ok(SwarmRoundResult {
            winner_id: winner.node_id.clone(),
            winning_answer: winner.answer.clone(),
            confidence,
            all_answers: answers,
            elo_updates,
            n_nodes,
            total_time_ms: total_time,
        })
    }
}

/// Call a single node's OpenAI-compatible chat endpoint.
async fn call_node(client: &reqwest::Client, base_url: &str, prompt: &str) -> Result<String> {
    #[derive(Serialize)]
    struct Req {
        model: String,
        messages: Vec<Msg>,
        max_tokens: u32,
        temperature: f32,
    }
    #[derive(Serialize)]
    struct Msg {
        role: String,
        content: String,
    }
    #[derive(Deserialize)]
    struct Resp {
        choices: Vec<Choice>,
    }
    #[derive(Deserialize)]
    struct Choice {
        message: ChoiceMsg,
    }
    #[derive(Deserialize)]
    struct ChoiceMsg {
        content: String,
    }

    let req = Req {
        model: "default".into(),
        messages: vec![Msg {
            role: "user".into(),
            content: prompt.into(),
        }],
        max_tokens: 1024,
        temperature: 0.7,
    };

    let resp: Resp = client
        .post(format!(
            "{}/v1/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .json(&req)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    resp.choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow::anyhow!("Empty response from node"))
}

/// Score each answer using quality heuristics.
fn rank_answers(answers: &[NodeAnswer]) -> Vec<f64> {
    answers
        .iter()
        .map(|a| {
            let text = &a.answer;
            let len_score = (text.len() as f64).sqrt();
            let has_code = if text.contains("```") { 10.0 } else { 0.0 };
            let has_list = if text.contains("\n- ") || text.contains("\n* ") || text.contains("\n1.")
            {
                5.0
            } else {
                0.0
            };
            let has_numbers = if text.chars().any(|c| c.is_ascii_digit()) {
                2.0
            } else {
                0.0
            };
            let sentences = text.matches(". ").count() as f64;
            let paragraphs = text.matches("\n\n").count() as f64 * 3.0;
            // Penalize very short answers
            let brevity_penalty = if text.len() < 50 { -10.0 } else { 0.0 };
            // Penalize latency (faster is better, mild)
            let latency_penalty = -(a.latency_ms as f64 / 10000.0);

            len_score
                + has_code
                + has_list
                + has_numbers
                + sentences
                + paragraphs
                + brevity_penalty
                + latency_penalty
        })
        .collect()
}

/// Bradley-Terry selection: convert scores to probabilities, pick winner.
fn bradley_terry_select(scores: &[f64]) -> (usize, f32) {
    if scores.is_empty() {
        return (0, 0.0);
    }

    // Convert to exponential scale (Bradley-Terry model)
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();

    let probabilities: Vec<f64> = exp_scores.iter().map(|e| e / sum).collect();

    // Winner = argmax
    let (winner_idx, &max_prob) = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    (winner_idx, max_prob as f32)
}

/// Update ELO scores for all participating nodes.
fn update_elo(
    nodes: &mut [SwarmNode],
    answers: &[NodeAnswer],
    winner_idx: usize,
) -> Vec<(String, i32)> {
    // Compute average elo across participating nodes only
    let participating_elos: Vec<f64> = answers
        .iter()
        .filter_map(|a| nodes.iter().find(|n| n.id == a.node_id).map(|n| n.elo as f64))
        .collect();
    let avg_elo = if participating_elos.is_empty() {
        1500.0
    } else {
        participating_elos.iter().sum::<f64>() / participating_elos.len() as f64
    };

    let mut updates = Vec::new();

    for (i, answer) in answers.iter().enumerate() {
        if let Some(node) = nodes.iter_mut().find(|n| n.id == answer.node_id) {
            node.rounds_played += 1;

            let k = if node.rounds_played < 30 { 32 } else { 16 };
            let actual = if i == winner_idx { 1.0 } else { 0.0 };

            // Expected score vs field average
            let expected = 1.0 / (1.0 + 10.0_f64.powf((avg_elo - node.elo as f64) / 400.0));

            let delta = (k as f64 * (actual - expected)).round() as i32;
            node.elo = (node.elo + delta).max(100); // Floor at 100

            if i == winner_idx {
                node.rounds_won += 1;
            }

            updates.push((node.id.clone(), node.elo));
        }
    }

    updates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bradley_terry_select_single() {
        let scores = vec![5.0];
        let (idx, conf) = bradley_terry_select(&scores);
        assert_eq!(idx, 0);
        assert!((conf - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bradley_terry_select_clear_winner() {
        let scores = vec![1.0, 10.0, 2.0];
        let (idx, conf) = bradley_terry_select(&scores);
        assert_eq!(idx, 1);
        assert!(conf > 0.5);
    }

    #[test]
    fn test_rank_answers_prefers_longer() {
        let short = NodeAnswer {
            node_id: "a".into(),
            answer: "ok".into(),
            latency_ms: 100,
            model: "m".into(),
        };
        let long = NodeAnswer {
            node_id: "b".into(),
            answer: "This is a much longer and more detailed answer that explains things properly. It has multiple sentences. And provides context.".into(),
            latency_ms: 100,
            model: "m".into(),
        };
        let scores = rank_answers(&[short, long]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_elo_updates() {
        let mut nodes = vec![
            SwarmNode {
                id: "a".into(),
                url: "".into(),
                model: "m".into(),
                elo: 1500,
                rounds_played: 0,
                rounds_won: 0,
            },
            SwarmNode {
                id: "b".into(),
                url: "".into(),
                model: "m".into(),
                elo: 1500,
                rounds_played: 0,
                rounds_won: 0,
            },
        ];
        let answers = vec![
            NodeAnswer {
                node_id: "a".into(),
                answer: "ans a".into(),
                latency_ms: 100,
                model: "m".into(),
            },
            NodeAnswer {
                node_id: "b".into(),
                answer: "ans b".into(),
                latency_ms: 100,
                model: "m".into(),
            },
        ];
        let updates = update_elo(&mut nodes, &answers, 0);
        // Winner should gain, loser should lose
        assert_eq!(updates.len(), 2);
        assert!(nodes[0].elo > 1500);
        assert!(nodes[1].elo < 1500);
        assert_eq!(nodes[0].rounds_won, 1);
        assert_eq!(nodes[1].rounds_won, 0);
    }
}
