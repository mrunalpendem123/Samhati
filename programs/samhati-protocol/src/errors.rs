use anchor_lang::prelude::*;

#[error_code]
pub enum SamhatiError {
    #[msg("Model name exceeds 64-byte maximum")]
    ModelNameTooLong,

    #[msg("Node has not passed TOPLOC calibration")]
    NodeNotCalibrated,

    #[msg("Node is already calibrated")]
    AlreadyCalibrated,

    #[msg("Participant count exceeds the maximum of 9")]
    TooManyParticipants,

    #[msg("Parallel array lengths do not match (participants, proofs, deltas)")]
    ArrayLengthMismatch,

    #[msg("Proof hash mismatch — evidence of invalid computation")]
    ProofHashMismatch,

    #[msg("Node has been slashed too many times to claim rewards")]
    SlashThresholdExceeded,

    #[msg("No pending rewards to claim")]
    NoPendingRewards,

    #[msg("Unauthorized — signer is not the coordinator authority")]
    UnauthorizedCoordinator,

    #[msg("Unauthorized — signer is not the node operator")]
    UnauthorizedOperator,

    #[msg("Arithmetic overflow during ELO or reward calculation")]
    ArithmeticOverflow,

    #[msg("Winner pubkey is not among the round participants")]
    WinnerNotParticipant,

    #[msg("Insufficient staked balance for slash burn")]
    InsufficientStake,
}
