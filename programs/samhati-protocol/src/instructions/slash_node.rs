use anchor_lang::prelude::*;
use crate::errors::SamhatiError;
use crate::state::{NodeAccount, ProtocolConfig};

/// Slash penalty: -200 ELO.
pub const SLASH_ELO_PENALTY: i32 = 200;
/// Burn 10% of staked SMTI (basis points).
pub const SLASH_BURN_BPS: u64 = 1000;

#[derive(Accounts)]
pub struct SlashNode<'info> {
    /// Coordinator authority.
    pub authority: Signer<'info>,

    /// Protocol config — authority check.
    #[account(
        seeds = [b"config"],
        bump = config.bump,
        has_one = authority @ SamhatiError::UnauthorizedCoordinator,
    )]
    pub config: Account<'info, ProtocolConfig>,

    /// The NodeAccount to slash.
    #[account(
        mut,
        seeds = [b"node", node_account.operator.as_ref()],
        bump = node_account.bump,
    )]
    pub node_account: Account<'info, NodeAccount>,
}

pub fn handler(ctx: Context<SlashNode>) -> Result<()> {
    let node = &mut ctx.accounts.node_account;

    // Apply ELO penalty.
    node.elo_score = node
        .elo_score
        .checked_sub(SLASH_ELO_PENALTY)
        .ok_or(SamhatiError::ArithmeticOverflow)?;
    node.clamp_elo();

    // Burn 10% of staked SMTI.
    let burn_amount = node
        .staked_amount
        .checked_mul(SLASH_BURN_BPS)
        .ok_or(SamhatiError::ArithmeticOverflow)?
        / 10_000;

    node.staked_amount = node
        .staked_amount
        .checked_sub(burn_amount)
        .ok_or(SamhatiError::InsufficientStake)?;

    node.slash_count = node.slash_count.saturating_add(1);

    msg!(
        "Node slashed: operator={}, new_elo={}, burned={}, slash_count={}",
        node.operator,
        node.elo_score,
        burn_amount,
        node.slash_count
    );
    Ok(())
}
