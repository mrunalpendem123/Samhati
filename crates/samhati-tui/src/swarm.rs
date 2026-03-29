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
    pub rankings: Vec<PairwiseResult>,
    pub difficulty: String,
    pub used_debate: bool,
    pub domain: String,
    pub proof_hashes: Vec<[u8; 32]>,
}

/// Domain demand stats — tracks what queries the network is getting.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DemandStats {
    pub code: u64,
    pub math: u64,
    pub reasoning: u64,
    pub general: u64,
    pub total: u64,
}

impl DemandStats {
    pub fn code_pct(&self) -> f64 { if self.total == 0 { 0.0 } else { self.code as f64 / self.total as f64 * 100.0 } }
    pub fn math_pct(&self) -> f64 { if self.total == 0 { 0.0 } else { self.math as f64 / self.total as f64 * 100.0 } }
    pub fn reasoning_pct(&self) -> f64 { if self.total == 0 { 0.0 } else { self.reasoning as f64 / self.total as f64 * 100.0 } }
    pub fn general_pct(&self) -> f64 { if self.total == 0 { 0.0 } else { self.general as f64 / self.total as f64 * 100.0 } }

    fn record(&mut self, domain: &str) {
        self.total += 1;
        match domain {
            "Code" => self.code += 1,
            "Math" => self.math += 1,
            "Reasoning" => self.reasoning += 1,
            _ => self.general += 1,
        }
        self.save();
    }

    fn save(&self) {
        let path = dirs::home_dir().unwrap_or_default().join(".samhati/demand.json");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            std::fs::write(path, json).ok();
        }
    }

    pub fn load() -> Self {
        let path = dirs::home_dir().unwrap_or_default().join(".samhati/demand.json");
        if let Ok(data) = std::fs::read_to_string(&path) {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeAnswer {
    pub node_id: String,
    pub answer: String,
    pub latency_ms: u64,
    pub model: String,
    /// BLAKE3 hash of the TOPLOC proof (or answer text if no logprobs available).
    pub proof_hash: [u8; 32],
}

/// Result of one pairwise comparison by a ranking node.
#[derive(Debug, Clone)]
pub struct PairwiseResult {
    pub ranker_node_id: String,
    pub answer_a_idx: usize,
    pub answer_b_idx: usize,
    /// 1.0 = A wins, 0.0 = B wins, 0.5 = tie
    pub preference: f64,
    pub reasoning: String,
}

/// The swarm orchestrator — manages nodes and coordinates inference.
///
/// Implements the Fortytwo peer-ranked consensus protocol + debate:
///   Phase 1: Fan out prompt to N nodes, collect answers
///   Phase 1.5: Debate round (hard queries only — agents self-correct)
///   Phase 2: Peer ranking — each node judges others' answers via LLM
///   Phase 3: BradleyTerry aggregation on real pairwise preferences
///   Phase 4: ELO update + persist to disk
pub struct SwarmOrchestrator {
    nodes: Arc<Mutex<Vec<SwarmNode>>>,
    client: reqwest::Client,
    signing_key: [u8; 32],
}

const ELO_FILE: &str = ".samhati/elo.json";

impl SwarmOrchestrator {
    pub fn new(signing_key: [u8; 32]) -> Self {
        Self {
            nodes: Arc::new(Mutex::new(Vec::new())),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .redirect(reqwest::redirect::Policy::none()) // Prevent SSRF redirect chains
                .build()
                .unwrap_or_default(),
            signing_key,
        }
    }

    /// Add a remote node (from gossip or manual 'r' key). Validates URL for SSRF.
    pub fn add_node(&self, id: String, url: String, model: String) {
        // Remote nodes: validate URL to prevent SSRF
        if !is_local_url(&url) && !is_safe_inference_url(&url) {
            eprintln!("[swarm] Rejected unsafe URL: {}", url);
            return;
        }
        // Check if we have a persisted ELO for this node
        let saved_elo = load_elo(&id).unwrap_or(1500);
        if let Ok(mut nodes) = self.nodes.lock() {
            nodes.push(SwarmNode {
                id,
                url,
                model,
                elo: saved_elo,
                rounds_played: 0,
                rounds_won: 0,
            });
        }
    }

    pub fn remove_node(&self, id: &str) {
        if let Ok(mut nodes) = self.nodes.lock() {
            nodes.retain(|n| n.id != id);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.lock().map(|n| n.len()).unwrap_or(0)
    }

    pub fn snapshot_nodes(&self) -> Vec<SwarmNode> {
        self.nodes.lock().map(|n| n.clone()).unwrap_or_default()
    }

    /// Run swarm inference with adaptive complexity routing.
    ///
    /// Easy queries  → 3 nodes + peer rank
    /// Hard queries  → 5 nodes + 1 debate round + peer rank
    ///
    /// Every query always goes through the swarm. The network always gets
    /// used, nodes always earn SMTI, answers are always peer-ranked.
    pub async fn infer(&self, prompt: &str) -> Result<SwarmRoundResult> {
        let node_snapshot = {
            let guard = self.nodes.lock().map_err(|e| anyhow::anyhow!("lock: {}", e))?;
            guard.clone()
        };

        if node_snapshot.is_empty() {
            return Err(anyhow::anyhow!("No nodes registered in swarm"));
        }

        let start = Instant::now();
        let available = node_snapshot.len();

        // ── Complexity classification ────────────────────────────────────
        let difficulty = classify_difficulty(prompt);
        let desired_n = match difficulty {
            Difficulty::Easy => 3,
            Difficulty::Medium => 3,
            Difficulty::Hard => 5,
        };
        let n_to_use = desired_n.min(available);
        let use_debate = difficulty == Difficulty::Hard && n_to_use >= 3;

        // Select nodes: pick top N by ELO (higher ELO = better answers historically)
        let mut selected = node_snapshot.clone();
        selected.sort_by(|a, b| b.elo.cmp(&a.elo));
        selected.truncate(n_to_use);

        // ── Phase 1: Fan out — send prompt to selected nodes in parallel ──
        // Each node returns answer + TOPLOC proof hash
        let mut handles = Vec::new();
        for node in &selected {
            let client = self.client.clone();
            let url = node.url.clone();
            let node_id = node.id.clone();
            let model = node.model.clone();
            let prompt_owned = prompt.to_string();
            let sk = self.signing_key;

            handles.push(tokio::spawn(async move {
                let node_start = Instant::now();
                let result = call_node_with_proof(&client, &url, &prompt_owned, &sk, &model).await;
                let latency = node_start.elapsed().as_millis() as u64;
                (node_id, model, result, latency)
            }));
        }

        let mut answers: Vec<NodeAnswer> = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((node_id, model, Ok((answer, proof_hash)), latency)) => {
                    answers.push(NodeAnswer {
                        node_id,
                        answer,
                        latency_ms: latency,
                        model,
                        proof_hash,
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

        // Single answer — return directly
        let difficulty_str = match difficulty {
            Difficulty::Easy => "Easy",
            Difficulty::Medium => "Medium",
            Difficulty::Hard => "Hard",
        }.to_string();

        // Classify domain and record demand
        let domain = classify_domain(prompt);
        let mut demand = DemandStats::load();
        demand.record(&domain);

        if answers.len() == 1 {
            let winner_id = answers[0].node_id.clone();
            let winning_answer = answers[0].answer.clone();
            let ph = vec![answers[0].proof_hash];
            return Ok(SwarmRoundResult {
                winner_id,
                winning_answer,
                confidence: 1.0,
                all_answers: answers,
                elo_updates: vec![],
                n_nodes: 1,
                total_time_ms: start.elapsed().as_millis() as u64,
                rankings: vec![],
                difficulty: difficulty_str,
                used_debate: false,
                domain: domain.clone(),
                proof_hashes: ph,
            });
        }

        // ── Phase 1.5: Debate round (hard queries only) ─────────────────
        // Each node sees all other answers and rewrites its own.
        // From the Multi-Agent Debate paper (arXiv:2305.14325):
        // agents self-correct by seeing others' reasoning chains.
        if use_debate {
            answers = self
                .debate_round(&selected, &answers, prompt)
                .await;
        }

        // ── Phase 2: Peer ranking — each node judges all pairs using its LLM ──
        let rankings = self
            .peer_rank(&selected, &answers, prompt)
            .await;

        // ── Phase 3: BradleyTerry aggregation on real pairwise preferences ──
        let n = answers.len();
        let (winner_idx, confidence) = if rankings.is_empty() {
            let scores = heuristic_scores(&answers);
            heuristic_bt_select(&scores)
        } else {
            bradley_terry_aggregate(&rankings, n)
        };

        let winner = &answers[winner_idx];

        // ── Phase 4: ELO updates (using real samhati-ranking crate) ──
        // Build BT strengths map for the ranking crate
        let bt_strengths: std::collections::HashMap<String, f64> = answers.iter().enumerate().map(|(i, a)| {
            // Approximate strength from BT confidence — winner gets 1.0, others proportional
            let strength = if i == winner_idx { 1.0 } else { (1.0 - confidence as f64).max(0.01) / (n as f64 - 1.0) };
            (a.node_id.clone(), strength)
        }).collect();

        let elo_updates = {
            let mut guard = self.nodes.lock().map_err(|e| anyhow::anyhow!("lock: {}", e))?;
            let updates = update_elo(&mut guard, &answers, winner_idx, &bt_strengths, Some(&domain));
            save_all_elos(&guard);
            updates
        };

        let final_n = answers.len();
        let winner_id = winner.node_id.clone();
        let winning_answer = winner.answer.clone();
        let phs: Vec<[u8; 32]> = answers.iter().map(|a| a.proof_hash).collect();
        Ok(SwarmRoundResult {
            winner_id,
            winning_answer,
            confidence,
            all_answers: answers,
            elo_updates,
            n_nodes: final_n,
            total_time_ms: start.elapsed().as_millis() as u64,
            rankings,
            difficulty: difficulty_str,
            used_debate: use_debate,
            domain: domain.clone(),
            proof_hashes: phs,
        })
    }

    /// Debate round: each node sees all others' answers and rewrites its own.
    /// From the Multi-Agent Debate paper (arXiv:2305.14325).
    /// Only 1 round — most self-correction happens in the first round.
    async fn debate_round(
        &self,
        nodes: &[SwarmNode],
        answers: &[NodeAnswer],
        original_prompt: &str,
    ) -> Vec<NodeAnswer> {
        let mut handles = Vec::new();

        for (i, node) in nodes.iter().enumerate() {
            // Find this node's original answer
            let my_answer = answers.iter()
                .find(|a| a.node_id == node.id)
                .map(|a| a.answer.clone())
                .unwrap_or_default();

            // Collect other nodes' answers (truncated for speed)
            let others: Vec<String> = answers.iter()
                .filter(|a| a.node_id != node.id)
                .map(|a| {
                    let short: String = a.answer.chars().take(500).collect();
                    short
                })
                .collect();

            if others.is_empty() {
                continue;
            }

            let client = self.client.clone();
            let url = node.url.clone();
            let node_id = node.id.clone();
            let model = node.model.clone();
            let prompt = original_prompt.to_string();

            handles.push(tokio::spawn(async move {
                let node_start = Instant::now();
                let result = ask_node_to_debate(
                    &client, &url, &prompt, &my_answer, &others,
                ).await;
                let latency = node_start.elapsed().as_millis() as u64;
                (node_id, model, result, latency)
            }));
        }

        let mut improved: Vec<NodeAnswer> = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((node_id, model, Ok(answer), latency)) => {
                    let proof_hash = *blake3::hash(answer.as_bytes()).as_bytes();
                    improved.push(NodeAnswer {
                        node_id,
                        answer,
                        latency_ms: latency,
                        model,
                        proof_hash,
                    });
                }
                Ok((node_id, model, Err(e), _)) => {
                    eprintln!("Debate failed for {}: {}", node_id, e);
                    // Fall back to original answer
                    if let Some(orig) = answers.iter().find(|a| a.node_id == node_id) {
                        improved.push(orig.clone());
                    }
                }
                Err(e) => {
                    eprintln!("Debate task error: {}", e);
                }
            }
        }

        // If debate produced fewer answers than we started with, keep originals
        if improved.len() < answers.len() {
            for orig in answers {
                if !improved.iter().any(|a| a.node_id == orig.node_id) {
                    improved.push(orig.clone());
                }
            }
        }

        improved
    }

    /// Phase 2: Send all answers to each node for LLM-based pairwise ranking.
    ///
    /// For each ranking node, generate all C(N,2) pairs where the node is NOT
    /// the author of either answer (preventing self-promotion bias).
    /// Each node's LLM produces a preference + reasoning chain.
    /// Peer ranking with random judge assignment (anti-collusion).
    ///
    /// Instead of every node judging every pair, each pair is assigned
    /// to a random subset of judges. Assignment is derived from a seed
    /// (timestamp-based, or Solana slot hash when available) so it's
    /// unpredictable and changes every round. Colluders can't guarantee
    /// they'll judge each other's answers.
    async fn peer_rank(
        &self,
        nodes: &[SwarmNode],
        answers: &[NodeAnswer],
        original_prompt: &str,
    ) -> Vec<PairwiseResult> {
        let n = answers.len();
        if n < 2 { return vec![]; }

        // Build all possible pairs
        let mut all_pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                all_pairs.push((i, j));
            }
        }

        // Generate a round seed for random assignment.
        // Uses BLAKE3(timestamp + prompt) — unpredictable per round.
        // When Solana slot hash is available, use that instead for
        // on-chain verifiable randomness.
        let seed_input = format!(
            "{}:{}:{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            original_prompt,
            nodes.len(),
        );
        let seed_hash = blake3::hash(seed_input.as_bytes());
        let seed_bytes = seed_hash.as_bytes();

        // Assign each pair to exactly ONE judge (not all judges).
        // The judge is chosen by: seed_bytes[pair_index % 32] % num_eligible_judges
        // A judge can't judge pairs containing their own answer.
        let mut handles = Vec::new();

        for (pair_idx, &(i, j)) in all_pairs.iter().enumerate() {
            // Find eligible judges for this pair (not author of either answer)
            let eligible: Vec<&SwarmNode> = nodes
                .iter()
                .filter(|node| node.id != answers[i].node_id && node.id != answers[j].node_id)
                .collect();

            if eligible.is_empty() {
                continue;
            }

            // Pick one judge using the seed — deterministic but unpredictable
            let judge_idx = seed_bytes[pair_idx % 32] as usize % eligible.len();
            let judge = eligible[judge_idx];

            let client = self.client.clone();
            let ranker_url = judge.url.clone();
            let ranker_id = judge.id.clone();
            let answer_a = answers[i].answer.clone();
            let answer_b = answers[j].answer.clone();
            let prompt = original_prompt.to_string();
            let idx_a = i;
            let idx_b = j;

            handles.push(tokio::spawn(async move {
                let result = ask_node_to_rank(
                    &client, &ranker_url, &prompt, &answer_a, &answer_b,
                )
                .await;
                (ranker_id, idx_a, idx_b, result)
            }));
        }

        let mut rankings = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((ranker_id, idx_a, idx_b, Ok((preference, reasoning)))) => {
                    rankings.push(PairwiseResult {
                        ranker_node_id: ranker_id,
                        answer_a_idx: idx_a,
                        answer_b_idx: idx_b,
                        preference,
                        reasoning,
                    });
                }
                Ok((ranker_id, _, _, Err(e))) => {
                    eprintln!("Ranking by {} failed: {}", ranker_id, e);
                }
                Err(e) => {
                    eprintln!("Ranking task error: {}", e);
                }
            }
        }

        rankings
    }
}

// ── Complexity classifier ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Difficulty {
    Easy,
    Medium,
    Hard,
}

/// Classify query difficulty. Fast heuristic (< 1ms).
/// Every query still goes through the swarm — this only controls HOW MANY
/// nodes and whether debate is used.
/// Validate that an inference URL is safe (not SSRF target).
/// Blocks: private IPs, localhost, non-HTTP schemes, metadata endpoints.
fn is_safe_inference_url(url: &str) -> bool {
    // Must be HTTP or HTTPS
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return false;
    }

    // Extract host from URL
    let host = url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split('/')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("");

    if host.is_empty() {
        return false;
    }

    // Block localhost and loopback
    let blocked_hosts = [
        "localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]",
    ];
    if blocked_hosts.contains(&host) {
        return false;
    }

    // Block private IP ranges
    if let Ok(ip) = host.parse::<std::net::Ipv4Addr>() {
        if ip.is_loopback()           // 127.x.x.x
            || ip.is_private()        // 10.x, 172.16-31.x, 192.168.x
            || ip.is_link_local()     // 169.254.x.x (AWS metadata)
            || ip.is_unspecified()    // 0.0.0.0
        {
            return false;
        }
    }

    true
}

/// Allow localhost URLs explicitly (for local swarm nodes added via 's' key).
/// These are trusted — the user started them on their own machine.
fn is_local_url(url: &str) -> bool {
    url.contains("127.0.0.1") || url.contains("localhost") || url.contains("[::1]")
}

fn classify_difficulty(prompt: &str) -> Difficulty {
    let lower = prompt.to_lowercase();
    let word_count = prompt.split_whitespace().count();
    let mut score: i32 = 0;

    // Hard signals: multi-step reasoning, math, proofs, debugging
    let hard_signals = [
        "prove", "derive", "step by step", "theorem", "induction",
        "debug", "fix this", "find the bug", "what's wrong",
        "implement", "design a", "architect",
        "aime", "olympiad", "competition",
        "∫", "∑", "∂", "lim", "→",
    ];
    for sig in hard_signals {
        if lower.contains(sig) { score += 3; }
    }

    // Medium signals: code, domain-specific, explanation
    let medium_signals = [
        "```", "fn ", "def ", "class ", "import ", "function",
        "explain how", "compare", "difference between",
        "write a program", "write code", "algorithm",
        "analyze", "evaluate", "trade-off",
    ];
    for sig in medium_signals {
        if lower.contains(sig) { score += 2; }
    }

    // Easy signals: simple questions, greetings, factual lookups
    let easy_signals = [
        "what is", "who is", "when was", "where is",
        "hello", "hi", "hey", "thanks",
        "translate", "summarize", "define",
        "list", "name", "how many",
    ];
    for sig in easy_signals {
        if lower.contains(sig) { score -= 1; }
    }

    // Long prompts with multiple questions tend to be harder
    let question_marks = prompt.matches('?').count();
    if question_marks >= 3 { score += 2; }
    if word_count > 100 { score += 2; }
    if word_count < 10 { score -= 1; }

    // Multiple code blocks = harder
    let code_blocks = prompt.matches("```").count() / 2;
    if code_blocks >= 2 { score += 3; }

    match score {
        s if s >= 5 => Difficulty::Hard,
        s if s >= 2 => Difficulty::Medium,
        _ => Difficulty::Easy,
    }
}

/// Classify the domain of a query. Used for demand tracking and
/// routing to specialist models.
fn classify_domain(prompt: &str) -> String {
    let lower = prompt.to_lowercase();

    // Code signals
    let code_signals = [
        "```", "fn ", "def ", "class ", "import ", "function ",
        "compile", "runtime", "error:", "syntax", "variable",
        "rust", "python", "javascript", "typescript", "java ", "golang",
        "react", "api", "endpoint", "database", "sql",
        "git", "docker", "kubernetes", "deploy",
        "write a program", "write code", "code",
        "bug", "debug", "fix this",
    ];
    let code_score: i32 = code_signals.iter()
        .filter(|s| lower.contains(*s))
        .count() as i32;

    // Math signals
    let math_signals = [
        "solve", "equation", "calculate", "compute",
        "prove", "theorem", "formula", "integral",
        "derivative", "matrix", "vector", "algebra",
        "geometry", "probability", "statistics",
        "∫", "∑", "∂", "π", "√",
        "math", "number", "prime",
    ];
    let math_score: i32 = math_signals.iter()
        .filter(|s| lower.contains(*s))
        .count() as i32;

    // Reasoning signals
    let reasoning_signals = [
        "explain why", "reason", "logic", "argument",
        "analyze", "evaluate", "compare", "contrast",
        "what if", "hypothetical", "thought experiment",
        "step by step", "chain of thought",
        "pros and cons", "trade-off",
    ];
    let reasoning_score: i32 = reasoning_signals.iter()
        .filter(|s| lower.contains(*s))
        .count() as i32;

    if code_score >= 2 || (code_score >= 1 && math_score == 0 && reasoning_score == 0) {
        "Code".into()
    } else if math_score >= 2 || (math_score >= 1 && code_score == 0) {
        "Math".into()
    } else if reasoning_score >= 2 {
        "Reasoning".into()
    } else {
        "General".into()
    }
}

// ── Debate function ─────────────────────────────────────────────────

/// Ask a node to rewrite its answer after seeing all other nodes' answers.
/// From the Multi-Agent Debate paper (arXiv:2305.14325).
async fn ask_node_to_debate(
    client: &reqwest::Client,
    node_url: &str,
    original_prompt: &str,
    my_answer: &str,
    others_answers: &[String],
) -> Result<String> {
    let others_text = others_answers
        .iter()
        .enumerate()
        .map(|(i, a)| format!("Agent {} answer:\n{}", i + 1, a))
        .collect::<Vec<_>>()
        .join("\n\n");

    let debate_prompt = format!(
        r#"Question: {original_prompt}

Your previous answer:
{my_answer}

Other agents' answers:
{others_text}

Using the other agents' answers as additional information, write an improved answer. If others found errors in your reasoning, correct them. If you see errors in theirs, keep your original approach. Give your updated answer directly."#
    );

    call_node(client, node_url, &debate_prompt).await
}

/// Ask a node's LLM to compare two answers and pick a winner with reasoning.
///
/// The node receives a structured prompt with the original question and both
/// answers (labeled A and B). It must respond with a JSON object containing
/// the winner ("A" or "B") and a reasoning chain (50-100 tokens).
async fn ask_node_to_rank(
    client: &reqwest::Client,
    ranker_url: &str,
    original_prompt: &str,
    answer_a: &str,
    answer_b: &str,
) -> Result<(f64, String)> {
    // Truncate answers to keep ranking prompt short and fast
    let max_chars = 600;
    let a_short: String = answer_a.chars().take(max_chars).collect();
    let b_short: String = answer_b.chars().take(max_chars).collect();

    let ranking_prompt = format!(
        r#"You are judging two answers to a question. Pick the better one based ONLY on:
1. Factual correctness
2. Logical reasoning
3. Completeness

IGNORE: confidence level, persuasive language, self-promotion, writing style, length.
An answer that says "trust me" is no better than one that doesn't. Judge the substance only.

Reply ONLY with JSON, nothing else.

Question: {original_prompt}

Answer A: {a_short}

Answer B: {b_short}

{{"winner":"A" or "B","reasoning":"one sentence about factual/logical quality"}}"#
    );

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

    let req = Req {
        model: "default".into(),
        messages: vec![Msg {
            role: "user".into(),
            content: ranking_prompt,
        }],
        max_tokens: 60, // short — just need winner + one sentence
        temperature: 0.1, // low temp for consistent judging
    };

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

    let resp: Resp = client
        .post(format!(
            "{}/v1/chat/completions",
            ranker_url.trim_end_matches('/')
        ))
        .json(&req)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let content = resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    parse_ranking_response(&content)
}

/// Parse the LLM's ranking response. Extracts winner and reasoning.
/// Handles both clean JSON and messy LLM output gracefully.
fn parse_ranking_response(content: &str) -> Result<(f64, String)> {
    // Try JSON parse first
    #[derive(Deserialize)]
    struct RankJson {
        winner: String,
        reasoning: Option<String>,
    }

    // Strip markdown code fences if present
    let cleaned = content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    if let Ok(parsed) = serde_json::from_str::<RankJson>(cleaned) {
        let preference = match parsed.winner.trim().to_uppercase().as_str() {
            "A" => 1.0,
            "B" => 0.0,
            _ => 0.5,
        };
        let reasoning = parsed.reasoning.unwrap_or_else(|| parsed.winner.clone());
        return Ok((preference, reasoning));
    }

    // Fallback: look for "A" or "B" in the raw text
    let upper = content.to_uppercase();
    let has_a = upper.contains("ANSWER A") || upper.contains("\"A\"") || upper.contains("WINNER: A");
    let has_b = upper.contains("ANSWER B") || upper.contains("\"B\"") || upper.contains("WINNER: B");

    let preference = match (has_a, has_b) {
        (true, false) => 1.0,
        (false, true) => 0.0,
        _ => 0.5, // ambiguous → tie
    };

    Ok((preference, content.chars().take(200).collect()))
}

/// Call a node and capture logprobs for TOPLOC proof.
/// Returns (answer_text, proof_hash).
/// If logprobs aren't available, falls back to BLAKE3 hash of the answer.
async fn call_node_with_proof(
    client: &reqwest::Client,
    base_url: &str,
    prompt: &str,
    signing_key: &[u8; 32],
    model_name: &str,
) -> Result<(String, [u8; 32])> {
    #[derive(Serialize)]
    struct Req {
        model: String,
        messages: Vec<Msg>,
        max_tokens: u32,
        temperature: f32,
        logprobs: bool,
        top_logprobs: u8,
    }
    #[derive(Serialize)]
    struct Msg {
        role: String,
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
        logprobs: true,
        top_logprobs: 8,
    };

    let resp: serde_json::Value = client
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

    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if content.is_empty() {
        return Err(anyhow::anyhow!("Empty response from node"));
    }

    // Build TOPLOC proof from logprobs returned by llama.cpp
    use samhati_toploc::proof::TokenLogits;
    use samhati_toploc::prover::ToplocProver;

    let mut prover = ToplocProver::new(model_name, *signing_key);

    // Extract logprobs from OpenAI-compatible response (llama.cpp / Ollama / vLLM)
    let logprobs_data = &resp["choices"][0]["logprobs"]["content"];
    if let Some(tokens) = logprobs_data.as_array() {
        for token_entry in tokens {
            // Use actual token ID from server if available, else CRC32 of token string
            let token_id = token_entry["id"].as_u64()
                .map(|id| id as u32)
                .unwrap_or_else(|| {
                    let token_str = token_entry["token"].as_str().unwrap_or("");
                    crc32_hash(token_str.as_bytes())
                });

            let mut top_k: Vec<(u32, f32)> = Vec::new();
            if let Some(top_arr) = token_entry["top_logprobs"].as_array() {
                for lp in top_arr.iter().take(8) {
                    let tok_id = lp["id"].as_u64()
                        .map(|id| id as u32)
                        .unwrap_or_else(|| {
                            let tok = lp["token"].as_str().unwrap_or("");
                            crc32_hash(tok.as_bytes())
                        });
                    let logprob = lp["logprob"].as_f64().unwrap_or(-100.0) as f32;
                    top_k.push((tok_id, logprob));
                }
            }

            if top_k.is_empty() {
                top_k.push((token_id, 0.0));
            }

            prover.record_token(TokenLogits { token_id, top_k });
        }
    }

    let proof_hash = if prover.recorded_count() > 0 {
        // Real TOPLOC proof from logprobs — signed, timestamped, node-bound
        match prover.finalize() {
            Ok(proof) => proof.proof_hash(),
            Err(_) => *blake3::hash(content.as_bytes()).as_bytes(),
        }
    } else {
        // No logprobs in response — last resort fallback
        *blake3::hash(content.as_bytes()).as_bytes()
    };

    Ok((content, proof_hash))
}

/// Simple CRC32 for deterministic token_id from token string.
fn crc32_hash(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Call a single node's OpenAI-compatible chat endpoint for inference.
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

/// BradleyTerry MLE aggregation from real pairwise rankings.
///
/// Uses iterative proportional fitting (MM algorithm):
///   p_i ← wins_i / Σ_j (games_ij / (p_i + p_j))
/// Runs 50 iterations, returns (winner_idx, confidence).
fn bradley_terry_aggregate(rankings: &[PairwiseResult], n: usize) -> (usize, f32) {
    if n == 0 {
        return (0, 0.0);
    }
    if n == 1 {
        return (0, 1.0);
    }

    // Accumulate wins and games from pairwise results
    let mut wins = vec![0.0f64; n];
    let mut games = vec![vec![0.0f64; n]; n];

    for r in rankings {
        let a = r.answer_a_idx;
        let b = r.answer_b_idx;
        if a >= n || b >= n {
            continue;
        }
        wins[a] += r.preference;
        wins[b] += 1.0 - r.preference;
        games[a][b] += 1.0;
        games[b][a] += 1.0;
    }

    // IPF iterations
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
            new_p[i] = if denom > 0.0 { wins[i] / denom } else { p[i] };
        }
        let sum: f64 = new_p.iter().sum();
        if sum > 0.0 {
            for v in &mut new_p {
                *v /= sum;
            }
        }
        p = new_p;
    }

    // Winner = argmax(p)
    let (winner_idx, &max_p) = p
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    (winner_idx, max_p as f32)
}

/// Fallback heuristic scoring (used only when all LLM ranking calls fail).
fn heuristic_scores(answers: &[NodeAnswer]) -> Vec<f64> {
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
            let brevity_penalty = if text.len() < 50 { -10.0 } else { 0.0 };
            let latency_penalty = -(a.latency_ms as f64 / 10000.0);
            len_score + has_code + has_list + has_numbers + sentences + paragraphs
                + brevity_penalty + latency_penalty
        })
        .collect()
}

/// Fallback BradleyTerry from heuristic scores (not pairwise).
fn heuristic_bt_select(scores: &[f64]) -> (usize, f32) {
    if scores.is_empty() {
        return (0, 0.0);
    }
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum: f64 = exp_scores.iter().sum();
    let probs: Vec<f64> = exp_scores.iter().map(|e| e / sum).collect();
    let (idx, &p) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    (idx, p as f32)
}

/// Update ELO scores using the real samhati-ranking crate.
/// Proper K-factor (32 new / 16 established), ELO floor (100),
/// domain-specific tracking, and BradleyTerry strength integration.
fn update_elo(
    nodes: &mut [SwarmNode],
    answers: &[NodeAnswer],
    winner_idx: usize,
    bt_strengths: &std::collections::HashMap<String, f64>,
    domain: Option<&str>,
) -> Vec<(String, i32)> {
    use samhati_ranking::{EloStore, RoundOutcome, NodeId};

    let mut store = EloStore::new();

    // Register all participating nodes with their current ELO
    for answer in answers {
        if let Some(node) = nodes.iter().find(|n| n.id == answer.node_id) {
            let mut node_id: NodeId = [0u8; 32];
            let bytes = answer.node_id.as_bytes();
            let len = bytes.len().min(32);
            node_id[..len].copy_from_slice(&bytes[..len]);
            store.register(node_id);
            // Set existing ELO (override default 1500)
            if let Some(rating) = store.ratings_mut().get_mut(&node_id) {
                rating.score = node.elo;
                rating.total_rounds = node.rounds_played as u64;
                rating.rounds_won = node.rounds_won as u64;
            }
        }
    }

    // Build winner NodeId
    let winner_answer = &answers[winner_idx];
    let mut winner_id: NodeId = [0u8; 32];
    let wb = winner_answer.node_id.as_bytes();
    winner_id[..wb.len().min(32)].copy_from_slice(&wb[..wb.len().min(32)]);

    // Convert BT strengths to NodeId keys
    let mut strengths = std::collections::HashMap::new();
    for (id_str, &strength) in bt_strengths {
        let mut nid: NodeId = [0u8; 32];
        let b = id_str.as_bytes();
        nid[..b.len().min(32)].copy_from_slice(&b[..b.len().min(32)]);
        strengths.insert(nid, strength);
    }

    // Build participants list
    let participants: Vec<NodeId> = answers.iter().map(|a| {
        let mut nid: NodeId = [0u8; 32];
        let b = a.node_id.as_bytes();
        nid[..b.len().min(32)].copy_from_slice(&b[..b.len().min(32)]);
        nid
    }).collect();

    let outcome = RoundOutcome {
        participants,
        winner: winner_id,
        strengths,
        domain: domain.map(|d| d.to_string()),
    };

    let deltas = store.update_round(&outcome);

    // Apply deltas back to SwarmNodes
    let mut updates = Vec::new();
    for (nid, delta) in &deltas {
        // Find the node by matching the NodeId bytes
        for answer in answers.iter() {
            let mut check_id: NodeId = [0u8; 32];
            let b = answer.node_id.as_bytes();
            check_id[..b.len().min(32)].copy_from_slice(&b[..b.len().min(32)]);
            if &check_id == nid {
                if let Some(node) = nodes.iter_mut().find(|n| n.id == answer.node_id) {
                    node.elo = (node.elo + delta).max(100);
                    node.rounds_played += 1;
                    if &check_id == &winner_id {
                        node.rounds_won += 1;
                    }
                    updates.push((node.id.clone(), node.elo));
                }
                break;
            }
        }
    }

    updates
}

// ── ELO persistence ─────────────────────────────────────────────

fn elo_path() -> std::path::PathBuf {
    dirs::home_dir().unwrap_or_default().join(ELO_FILE)
}

/// Load all persisted ELO scores from disk.
fn load_elo_map() -> std::collections::HashMap<String, i32> {
    let path = elo_path();
    if let Ok(data) = std::fs::read_to_string(&path) {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    }
}

/// Load ELO for a specific node.
fn load_elo(node_id: &str) -> Option<i32> {
    load_elo_map().get(node_id).copied()
}

/// Save all node ELO scores to disk.
fn save_all_elos(nodes: &[SwarmNode]) {
    let mut map = load_elo_map();
    for node in nodes {
        map.insert(node.id.clone(), node.elo);
    }
    let path = elo_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(json) = serde_json::to_string_pretty(&map) {
        std::fs::write(&path, json).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bradley_terry_aggregate_single() {
        let (idx, conf) = bradley_terry_aggregate(&[], 1);
        assert_eq!(idx, 0);
        assert!((conf - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bradley_terry_aggregate_clear_winner() {
        // A always beats B in all rankings
        let rankings = vec![
            PairwiseResult {
                ranker_node_id: "r1".into(),
                answer_a_idx: 0,
                answer_b_idx: 1,
                preference: 1.0,
                reasoning: "A is better".into(),
            },
            PairwiseResult {
                ranker_node_id: "r2".into(),
                answer_a_idx: 0,
                answer_b_idx: 1,
                preference: 1.0,
                reasoning: "A is better".into(),
            },
        ];
        let (idx, conf) = bradley_terry_aggregate(&rankings, 2);
        assert_eq!(idx, 0);
        assert!(conf > 0.8);
    }

    #[test]
    fn test_bradley_terry_aggregate_close_race() {
        // Mixed preferences
        let rankings = vec![
            PairwiseResult {
                ranker_node_id: "r1".into(),
                answer_a_idx: 0,
                answer_b_idx: 1,
                preference: 0.6,
                reasoning: "A slightly better".into(),
            },
            PairwiseResult {
                ranker_node_id: "r2".into(),
                answer_a_idx: 0,
                answer_b_idx: 1,
                preference: 0.4,
                reasoning: "B slightly better".into(),
            },
        ];
        let (_, conf) = bradley_terry_aggregate(&rankings, 2);
        // Should be close to 0.5 — neither dominates
        assert!(conf < 0.7);
    }

    #[test]
    fn test_parse_ranking_json() {
        let (pref, _) =
            parse_ranking_response(r#"{"winner": "A", "reasoning": "A is more detailed"}"#)
                .unwrap();
        assert!((pref - 1.0).abs() < 0.001);

        let (pref, _) =
            parse_ranking_response(r#"{"winner": "B", "reasoning": "B is correct"}"#).unwrap();
        assert!(pref.abs() < 0.001);
    }

    #[test]
    fn test_parse_ranking_markdown_fenced() {
        let input = "```json\n{\"winner\": \"A\", \"reasoning\": \"better\"}\n```";
        let (pref, _) = parse_ranking_response(input).unwrap();
        assert!((pref - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_ranking_fallback() {
        let (pref, _) = parse_ranking_response("I think Answer A is clearly better").unwrap();
        assert!((pref - 1.0).abs() < 0.001);

        let (pref, _) = parse_ranking_response("Answer B wins this comparison").unwrap();
        assert!(pref.abs() < 0.001);
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
                proof_hash: [0u8; 32],
            },
            NodeAnswer {
                node_id: "b".into(),
                answer: "ans b".into(),
                latency_ms: 100,
                model: "m".into(),
                proof_hash: [0u8; 32],
            },
        ];
        let bt_strengths: std::collections::HashMap<String, f64> = [("a".into(), 1.0), ("b".into(), 0.2)].into();
        let updates = update_elo(&mut nodes, &answers, 0, &bt_strengths, Some("General"));
        assert_eq!(updates.len(), 2);
        assert!(nodes[0].elo > 1500);
        assert!(nodes[1].elo < 1500);
        assert_eq!(nodes[0].rounds_won, 1);
        assert_eq!(nodes[1].rounds_won, 0);
    }
}
