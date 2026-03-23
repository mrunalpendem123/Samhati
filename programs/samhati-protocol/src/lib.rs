use anchor_lang::prelude::*;

pub mod errors;
pub mod instructions;
pub mod state;

use instructions::*;

declare_id!("SAMHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

#[program]
pub mod samhati_protocol {
    use super::*;

    /// Register a new compute node on the Samhati network.
    /// Creates a NodeAccount PDA with initial ELO of 1500.
    pub fn register_node(ctx: Context<RegisterNode>, args: RegisterNodeArgs) -> Result<()> {
        instructions::register_node::handler(ctx, args)
    }

    /// Mark a node as calibrated after passing TOPLOC verification.
    /// Only callable by the coordinator authority.
    pub fn calibrate_node(ctx: Context<CalibrateNode>) -> Result<()> {
        instructions::calibrate_node::handler(ctx)
    }

    /// Submit an inference round with proofs, ELO deltas, and reward distribution.
    /// Only callable by the coordinator authority.
    /// Participant NodeAccounts must be passed as remaining_accounts in order.
    pub fn submit_round(ctx: Context<SubmitRound>, args: SubmitRoundArgs) -> Result<()> {
        instructions::submit_round::handler(ctx, args)
    }

    /// Slash a node for proof mismatch. Reduces ELO by 200 and burns 10% of stake.
    /// Only callable by the coordinator authority.
    pub fn slash_node(ctx: Context<SlashNode>) -> Result<()> {
        instructions::slash_node::handler(ctx)
    }

    /// Claim accumulated SMTI rewards. Permissionless — any node operator
    /// can call for their own node if calibrated and under slash threshold.
    pub fn emit_rewards(ctx: Context<EmitRewards>) -> Result<()> {
        instructions::emit_rewards::handler(ctx)
    }

    /// Initialize the protocol configuration. Called once at deployment.
    pub fn initialize(
        ctx: Context<Initialize>,
        base_emission_per_round: u64,
    ) -> Result<()> {
        let config = &mut ctx.accounts.config;
        config.authority = ctx.accounts.authority.key();
        config.smti_mint = ctx.accounts.smti_mint.key();
        config.reward_vault = ctx.accounts.reward_vault.key();
        config.total_rounds = 0;
        config.total_smti_emitted = 0;
        config.base_emission_per_round = base_emission_per_round;
        config.bump = ctx.bumps.config;

        msg!("Protocol initialized: authority={}", config.authority);
        Ok(())
    }
}

/// Accounts for the one-time protocol initialization.
#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,

    #[account(
        init,
        payer = authority,
        space = state::ProtocolConfig::MAX_SIZE,
        seeds = [b"config"],
        bump,
    )]
    pub config: Account<'info, state::ProtocolConfig>,

    /// CHECK: SMTI mint address, validated by the caller.
    pub smti_mint: UncheckedAccount<'info>,

    /// CHECK: Reward vault token account, validated by the caller.
    pub reward_vault: UncheckedAccount<'info>,

    pub system_program: Program<'info, System>,
}
