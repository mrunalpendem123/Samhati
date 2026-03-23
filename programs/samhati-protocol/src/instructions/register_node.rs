use anchor_lang::prelude::*;
use crate::errors::SamhatiError;
use crate::state::NodeAccount;

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct RegisterNodeArgs {
    pub model_name: String,
    pub model_size_b: u8,
    pub domain_tags: u64,
}

#[derive(Accounts)]
#[instruction(args: RegisterNodeArgs)]
pub struct RegisterNode<'info> {
    /// The node operator who is registering. Must sign.
    #[account(mut)]
    pub operator: Signer<'info>,

    /// NodeAccount PDA — created here.
    #[account(
        init,
        payer = operator,
        space = NodeAccount::MAX_SIZE,
        seeds = [b"node", operator.key().as_ref()],
        bump,
    )]
    pub node_account: Account<'info, NodeAccount>,

    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<RegisterNode>, args: RegisterNodeArgs) -> Result<()> {
    require!(
        args.model_name.len() <= NodeAccount::MAX_MODEL_NAME_LEN,
        SamhatiError::ModelNameTooLong
    );

    let node = &mut ctx.accounts.node_account;
    node.operator = ctx.accounts.operator.key();
    node.elo_score = NodeAccount::INITIAL_ELO;
    node.calibrated = false;
    node.model_name = args.model_name;
    node.model_size_b = args.model_size_b;
    node.domain_tags = args.domain_tags;
    node.total_rounds = 0;
    node.rounds_won = 0;
    node.slash_count = 0;
    node.pending_rewards = 0;
    node.staked_amount = 0;
    node.bump = ctx.bumps.node_account;

    msg!("Node registered: operator={}", node.operator);
    Ok(())
}
