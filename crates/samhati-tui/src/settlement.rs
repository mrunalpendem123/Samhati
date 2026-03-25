//! Solana settlement — record swarm rounds on-chain after each inference.
//!
//! After each swarm round, persists the round data locally as JSON.
//! When Solana settlement is enabled (coordinator authority matches our key),
//! sends the submit_round transaction.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// A completed round ready for on-chain settlement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundPayload {
    pub round_id: u64,
    pub participants: Vec<String>, // Solana pubkeys (base58)
    pub proof_hashes: Vec<[u8; 32]>,
    pub elo_deltas: Vec<i32>,
    pub winner: String,
    pub smti_emitted: u64,
    pub domain: u64,
    pub timestamp: i64,
}

/// Map domain string to on-chain constant.
pub fn domain_to_u64(domain: &str) -> u64 {
    match domain {
        "Code" => 1,
        "Math" => 2,
        "Reasoning" => 3,
        _ => 0, // General
    }
}

/// Build a RoundPayload from swarm round results.
pub fn build_payload(
    result: &crate::swarm::SwarmRoundResult,
    round_counter: u64,
) -> RoundPayload {
    // For now, node IDs are local identifiers (e.g. "node-8001").
    // In production, these would be Solana pubkeys from the gossip announcements.
    let participants: Vec<String> = result
        .all_answers
        .iter()
        .map(|a| a.node_id.clone())
        .collect();

    // Compute ELO deltas from the updates (current - 1500 baseline)
    let elo_deltas: Vec<i32> = result
        .elo_updates
        .iter()
        .map(|(_, elo)| *elo - 1500) // delta from initial
        .collect();

    RoundPayload {
        round_id: round_counter,
        participants,
        proof_hashes: result.proof_hashes.clone(),
        elo_deltas,
        winner: result.winner_id.clone(),
        smti_emitted: 1000, // base emission per round
        domain: domain_to_u64(&result.domain),
        timestamp: chrono::Utc::now().timestamp(),
    }
}

/// Save round to local pending directory.
/// These can be submitted to Solana later by a coordinator.
pub fn save_pending(payload: &RoundPayload) -> Result<()> {
    let dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".samhati/pending_rounds");
    std::fs::create_dir_all(&dir)?;

    let path = dir.join(format!("round_{}.json", payload.round_id));
    let json = serde_json::to_string_pretty(payload)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Get the next round counter (monotonically increasing).
pub fn next_round_id() -> u64 {
    let path = dirs::home_dir()
        .unwrap_or_default()
        .join(".samhati/round_counter");
    let current = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let next = current + 1;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, next.to_string()).ok();
    next
}
