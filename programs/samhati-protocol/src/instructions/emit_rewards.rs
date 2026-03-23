use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};
use crate::errors::SamhatiError;
use crate::state::{NodeAccount, ProtocolConfig};

/// Maximum slash count before rewards are blocked.
pub const MAX_SLASH_THRESHOLD: u8 = 3;

#[derive(Accounts)]
pub struct EmitRewards<'info> {
    /// Node operator claiming rewards. Must sign.
    #[account(mut)]
    pub operator: Signer<'info>,

    /// The operator's NodeAccount PDA.
    #[account(
        mut,
        seeds = [b"node", operator.key().as_ref()],
        bump = node_account.bump,
        has_one = operator @ SamhatiError::UnauthorizedOperator,
    )]
    pub node_account: Account<'info, NodeAccount>,

    /// Protocol config (read-only, for vault reference).
    #[account(
        seeds = [b"config"],
        bump = config.bump,
    )]
    pub config: Account<'info, ProtocolConfig>,

    /// Reward vault token account (SMTI). PDA authority = vault PDA.
    #[account(
        mut,
        seeds = [b"vault"],
        bump,
    )]
    pub reward_vault: Account<'info, TokenAccount>,

    /// Operator's SMTI token account to receive rewards.
    #[account(
        mut,
        constraint = operator_token_account.owner == operator.key(),
    )]
    pub operator_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
}

pub fn handler(ctx: Context<EmitRewards>) -> Result<()> {
    let node = &mut ctx.accounts.node_account;

    // Must be calibrated.
    require!(node.calibrated, SamhatiError::NodeNotCalibrated);

    // Must not exceed slash threshold.
    require!(
        node.slash_count < MAX_SLASH_THRESHOLD,
        SamhatiError::SlashThresholdExceeded
    );

    // Must have pending rewards.
    let amount = node.pending_rewards;
    require!(amount > 0, SamhatiError::NoPendingRewards);

    // Transfer SMTI from vault to operator.
    let vault_seeds: &[&[u8]] = &[b"vault", &[ctx.bumps.reward_vault]];
    let signer_seeds = &[vault_seeds];

    let cpi_ctx = CpiContext::new_with_signer(
        ctx.accounts.token_program.to_account_info(),
        Transfer {
            from: ctx.accounts.reward_vault.to_account_info(),
            to: ctx.accounts.operator_token_account.to_account_info(),
            authority: ctx.accounts.reward_vault.to_account_info(),
        },
        signer_seeds,
    );
    token::transfer(cpi_ctx, amount)?;

    // Zero out pending rewards.
    node.pending_rewards = 0;

    msg!("Rewards emitted: operator={}, amount={}", node.operator, amount);
    Ok(())
}
