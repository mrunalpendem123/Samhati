use anchor_lang::prelude::*;
use crate::errors::SamhatiError;
use crate::state::{NodeAccount, ProtocolConfig, RoundAccount};

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct SubmitRoundArgs {
    pub round_id: u64,
    pub participants: Vec<Pubkey>,
    pub proof_hashes: Vec<[u8; 32]>,
    pub elo_deltas: Vec<i32>,
    pub winner: Pubkey,
    pub smti_emitted: u64,
    pub domain: u64,
}

#[derive(Accounts)]
#[instruction(args: SubmitRoundArgs)]
pub struct SubmitRound<'info> {
    /// Coordinator authority.
    #[account(mut)]
    pub authority: Signer<'info>,

    /// Protocol config — verifies authority and tracks totals.
    #[account(
        mut,
        seeds = [b"config"],
        bump = config.bump,
        has_one = authority @ SamhatiError::UnauthorizedCoordinator,
    )]
    pub config: Account<'info, ProtocolConfig>,

    /// RoundAccount PDA — created here.
    #[account(
        init,
        payer = authority,
        space = RoundAccount::MAX_SIZE,
        seeds = [b"round", args.round_id.to_le_bytes().as_ref()],
        bump,
    )]
    pub round_account: Account<'info, RoundAccount>,

    pub system_program: Program<'info, System>,

    // Remaining accounts: one NodeAccount per participant, in order.
    // Passed via `ctx.remaining_accounts`.
}

pub fn handler(ctx: Context<SubmitRound>, args: SubmitRoundArgs) -> Result<()> {
    let n = args.participants.len();

    // Validate array lengths.
    require!(n <= RoundAccount::MAX_PARTICIPANTS, SamhatiError::TooManyParticipants);
    require!(
        args.proof_hashes.len() == n && args.elo_deltas.len() == n,
        SamhatiError::ArrayLengthMismatch
    );

    // Validate winner is a participant.
    require!(
        args.participants.contains(&args.winner),
        SamhatiError::WinnerNotParticipant
    );

    // Validate remaining accounts match participants.
    require!(
        ctx.remaining_accounts.len() == n,
        SamhatiError::ArrayLengthMismatch
    );

    // Update each participant's NodeAccount.
    let clock = Clock::get()?;
    for (i, account_info) in ctx.remaining_accounts.iter().enumerate() {
        // Deserialize the NodeAccount from remaining_accounts.
        let mut node_data: &[u8] = &account_info.try_borrow_data()?;
        let mut node = NodeAccount::try_deserialize(&mut node_data)
            .map_err(|_| SamhatiError::ArrayLengthMismatch)?;

        // Verify the account key matches expected PDA.
        require_keys_eq!(node.operator, args.participants[i]);

        // Apply ELO delta.
        node.elo_score = node
            .elo_score
            .checked_add(args.elo_deltas[i])
            .ok_or(SamhatiError::ArithmeticOverflow)?;
        node.clamp_elo();

        node.total_rounds = node
            .total_rounds
            .checked_add(1)
            .ok_or(SamhatiError::ArithmeticOverflow)?;

        if args.participants[i] == args.winner {
            node.rounds_won = node
                .rounds_won
                .checked_add(1)
                .ok_or(SamhatiError::ArithmeticOverflow)?;
        }

        // Credit pending rewards proportionally (winner gets extra share).
        // Simple split: winner gets 2x share, rest split evenly.
        let base_share = args.smti_emitted / (n as u64 + 1); // +1 for winner bonus
        let reward = if args.participants[i] == args.winner {
            base_share * 2
        } else {
            base_share
        };
        node.pending_rewards = node
            .pending_rewards
            .checked_add(reward)
            .ok_or(SamhatiError::ArithmeticOverflow)?;

        // Serialize back.
        let mut dst = account_info.try_borrow_mut_data()?;
        let mut cursor: &mut [u8] = &mut dst;
        node.try_serialize(&mut cursor)?;
    }

    // Populate the RoundAccount.
    let round = &mut ctx.accounts.round_account;
    round.round_id = args.round_id;
    round.coordinator = ctx.accounts.authority.key();
    round.participants = args.participants;
    round.proof_hashes = args.proof_hashes;
    round.elo_deltas = args.elo_deltas;
    round.winner = args.winner;
    round.smti_emitted = args.smti_emitted;
    round.domain = args.domain;
    round.timestamp = clock.unix_timestamp;
    round.bump = ctx.bumps.round_account;

    // Update protocol totals.
    let config = &mut ctx.accounts.config;
    config.total_rounds = config
        .total_rounds
        .checked_add(1)
        .ok_or(SamhatiError::ArithmeticOverflow)?;
    config.total_smti_emitted = config
        .total_smti_emitted
        .checked_add(args.smti_emitted)
        .ok_or(SamhatiError::ArithmeticOverflow)?;

    msg!("Round {} submitted with {} participants", args.round_id, n);
    Ok(())
}
