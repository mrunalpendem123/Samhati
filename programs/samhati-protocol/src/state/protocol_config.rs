use anchor_lang::prelude::*;

/// Global protocol configuration singleton.
///
/// PDA seeds: `[b"config"]`
#[account]
pub struct ProtocolConfig {
    /// Coordinator authority (upgrade to multisig later).
    pub authority: Pubkey,
    /// SPL token mint for $SMTI.
    pub smti_mint: Pubkey,
    /// Token account used as the reward vault.
    pub reward_vault: Pubkey,
    /// Total inference rounds recorded on-chain.
    pub total_rounds: u64,
    /// Cumulative SMTI emitted across all rounds.
    pub total_smti_emitted: u64,
    /// Current base emission per round (lamports of SMTI).
    pub base_emission_per_round: u64,
    /// Domain demand counters — how many rounds per domain.
    /// Nodes read these to decide which specialist model to run.
    pub domain_code: u64,
    pub domain_math: u64,
    pub domain_reasoning: u64,
    pub domain_general: u64,
    /// PDA bump seed.
    pub bump: u8,
}

/// Domain bitmask values used in submit_round.
pub const DOMAIN_GENERAL: u64 = 0;
pub const DOMAIN_CODE: u64 = 1;
pub const DOMAIN_MATH: u64 = 2;
pub const DOMAIN_REASONING: u64 = 3;

impl ProtocolConfig {
    /// 8 (disc) + 32*3 + 8*7 + 1
    pub const MAX_SIZE: usize = 8 + 32 + 32 + 32 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 1;
}
