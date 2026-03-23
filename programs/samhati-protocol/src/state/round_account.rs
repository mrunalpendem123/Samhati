use anchor_lang::prelude::*;

/// Record of a single inference round, including proofs and ELO deltas.
///
/// PDA seeds: `[b"round", round_id.to_le_bytes()]`
#[account]
pub struct RoundAccount {
    /// Monotonically increasing round identifier.
    pub round_id: u64,
    /// Coordinator authority that submitted this round.
    pub coordinator: Pubkey,
    /// Participating node operators (max 9).
    pub participants: Vec<Pubkey>,
    /// TOPLOC proof hash per participant (same order).
    pub proof_hashes: Vec<[u8; 32]>,
    /// ELO change per participant (same order, can be negative).
    pub elo_deltas: Vec<i32>,
    /// Winner of this round (receives bonus emission).
    pub winner: Pubkey,
    /// Total SMTI emitted for this round.
    pub smti_emitted: u64,
    /// Domain bitmask for this round.
    pub domain: u64,
    /// Unix timestamp when this round was recorded.
    pub timestamp: i64,
    /// PDA bump seed.
    pub bump: u8,
}

impl RoundAccount {
    pub const MAX_PARTICIPANTS: usize = 9;

    /// Conservative max size:
    /// 8 (disc) + 8 + 32 + (4 + 9*32) + (4 + 9*32) + (4 + 9*4) + 32 + 8 + 8 + 8 + 1
    pub const MAX_SIZE: usize = 8 + 8 + 32 + (4 + 9 * 32) + (4 + 9 * 32) + (4 + 9 * 4) + 32 + 8 + 8 + 8 + 1;
}
