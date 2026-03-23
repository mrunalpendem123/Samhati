use anchor_lang::prelude::*;
use crate::errors::SamhatiError;
use crate::state::{NodeAccount, ProtocolConfig};

#[derive(Accounts)]
pub struct CalibrateNode<'info> {
    /// Coordinator authority — must match ProtocolConfig.authority.
    pub authority: Signer<'info>,

    /// Protocol config PDA (read-only, for authority check).
    #[account(
        seeds = [b"config"],
        bump = config.bump,
        has_one = authority @ SamhatiError::UnauthorizedCoordinator,
    )]
    pub config: Account<'info, ProtocolConfig>,

    /// The NodeAccount to calibrate.
    #[account(
        mut,
        seeds = [b"node", node_account.operator.as_ref()],
        bump = node_account.bump,
    )]
    pub node_account: Account<'info, NodeAccount>,
}

pub fn handler(ctx: Context<CalibrateNode>) -> Result<()> {
    let node = &mut ctx.accounts.node_account;

    require!(!node.calibrated, SamhatiError::AlreadyCalibrated);

    node.calibrated = true;

    msg!("Node calibrated: operator={}", node.operator);
    Ok(())
}
