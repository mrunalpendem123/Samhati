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
    // Map node IDs to Solana pubkeys.
    // Local nodes (started via 's' key) use our pubkey since they share our identity.
    // Remote nodes (from gossip) carry their Solana pubkey from the announcement.
    let participants: Vec<String> = result
        .all_answers
        .iter()
        .map(|a| {
            // If it looks like a base58 Solana pubkey (32+ chars, no dashes), use it directly
            if a.node_id.len() >= 32 && !a.node_id.contains('-') && !a.node_id.contains(':') {
                a.node_id.clone()
            } else {
                // Local node — use our pubkey
                our_pubkey.to_string()
            }
        })
        .collect();

    // Compute reputation deltas from the updates (current - 1500 baseline)
    let elo_deltas: Vec<i32> = result
        .elo_updates
        .iter()
        .map(|(_, elo)| *elo - 1500)
        .collect();

    // Ensure arrays match in length (Solana program requires this)
    let n = participants.len();
    let proof_hashes = if result.proof_hashes.len() == n {
        result.proof_hashes.clone()
    } else {
        // Pad or truncate to match participant count
        let mut ph = result.proof_hashes.clone();
        ph.resize(n, [0u8; 32]);
        ph
    };
    let elo_deltas = if elo_deltas.len() == n {
        elo_deltas
    } else {
        let mut ed = elo_deltas;
        ed.resize(n, 0);
        ed
    };

    let winner = if participants.contains(&result.winner_id) {
        result.winner_id.clone()
    } else {
        // Winner ID is a local name — map to our pubkey
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
