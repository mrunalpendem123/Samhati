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
/// `our_pubkey` is used as the Solana address for local nodes that don't
/// have their own on-chain identity (nodes started via 's' key on same machine).
pub fn build_payload(
    result: &crate::swarm::SwarmRoundResult,
    round_counter: u64,
    our_pubkey: &str,
) -> RoundPayload {
    // Map node IDs to Solana pubkeys and DEDUPLICATE.
    // Solana can't borrow the same account twice in one instruction.
    // Local nodes all share our pubkey, so we collapse them into one entry.
    use std::collections::HashMap;
    let mut deduped: HashMap<String, ([u8; 32], i32)> = HashMap::new();

    for (i, answer) in result.all_answers.iter().enumerate() {
        let pubkey = if answer.node_id.len() >= 32
            && !answer.node_id.contains('-')
            && !answer.node_id.contains(':')
        {
            answer.node_id.clone()
        } else {
            our_pubkey.to_string()
        };

        let proof = result.proof_hashes.get(i).copied().unwrap_or([0u8; 32]);
        let delta = result.elo_updates.get(i).map(|(_, e)| *e - 1500).unwrap_or(0);

        deduped.entry(pubkey).or_insert((proof, delta));
    }

    let participants: Vec<String> = deduped.keys().cloned().collect();
    let proof_hashes: Vec<[u8; 32]> = participants.iter().map(|p| deduped[p].0).collect();
    let elo_deltas: Vec<i32> = participants.iter().map(|p| deduped[p].1).collect();

    let winner = if participants.contains(&result.winner_id) {
        result.winner_id.clone()
    } else {
        our_pubkey.to_string()
    };

    RoundPayload {
        round_id: round_counter,
        participants,
        proof_hashes,
        elo_deltas,
        winner,
        smti_emitted: 1000,
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
