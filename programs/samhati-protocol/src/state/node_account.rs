use anchor_lang::prelude::*;

/// On-chain identity for a compute node in the Samhati mesh.
///
/// PDA seeds: `[b"node", operator.key().as_ref()]`
#[account]
pub struct NodeAccount {
    /// Wallet of the node operator (signer for registration).
    pub operator: Pubkey,
    /// ELO reputation score. Starts at 1500, floor 100.
    pub elo_score: i32,
    /// Whether the node has passed TOPLOC calibration.
    pub calibrated: bool,
    /// Human-readable model identifier, e.g. "samhati-hindi-3b-v3" (max 64 UTF-8 bytes).
    pub model_name: String,
    /// Model parameter count in billions (e.g. 3 for a 3B model).
    pub model_size_b: u8,
    /// Bitmask of domain specialities (up to 64 domains).
    pub domain_tags: u64,
    /// Total inference rounds participated in.
    pub total_rounds: u64,
    /// Number of rounds won.
    pub rounds_won: u64,
    /// Number of times this node has been slashed.
    pub slash_count: u8,
    /// Accumulated SMTI lamports pending claim.
    pub pending_rewards: u64,
    /// SMTI lamports currently staked.
    pub staked_amount: u64,
    /// PDA bump seed.
    pub bump: u8,
}

impl NodeAccount {
    /// Anchor discriminator (8) + fields.
    /// Pubkey=32, i32=4, bool=1, String=4+64, u8=1, u64=8 x4, u8 x2, u64=8
    pub const MAX_SIZE: usize = 8 + 32 + 4 + 1 + (4 + 64) + 1 + 8 + 8 + 8 + 1 + 8 + 8 + 1;

    pub const INITIAL_ELO: i32 = 1500;
    pub const ELO_FLOOR: i32 = 100;
    pub const MAX_MODEL_NAME_LEN: usize = 64;

    /// Clamp ELO to the floor after any modification.
    pub fn clamp_elo(&mut self) {
        if self.elo_score < Self::ELO_FLOOR {
            self.elo_score = Self::ELO_FLOOR;
        }
    }
}
