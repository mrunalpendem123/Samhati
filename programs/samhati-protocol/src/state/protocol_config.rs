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
    /// PDA bump seed.
    pub bump: u8,
}

impl ProtocolConfig {
    /// 8 (disc) + 32*3 + 8*3 + 1
    pub const MAX_SIZE: usize = 8 + 32 + 32 + 32 + 8 + 8 + 8 + 1;
}
