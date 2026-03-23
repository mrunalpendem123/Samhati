interface RewardCounterProps {
  balance: number;
  pending: number;
  label?: string;
}

export default function RewardCounter({
  balance,
  pending,
  label = "SMTI Balance",
}: RewardCounterProps) {
  return (
    <div className="flex flex-col gap-1">
      <span className="stat-label">{label}</span>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-white font-mono">
          {formatSmti(balance)}
        </span>
        <span className="text-sm text-gray-500">SMTI</span>
      </div>
      {pending > 0 && (
        <span className="text-xs text-green-400">
          +{formatSmti(pending)} pending
        </span>
      )}
    </div>
  );
}

function formatSmti(amount: number): string {
  if (amount >= 1_000_000) return `${(amount / 1_000_000).toFixed(2)}M`;
  if (amount >= 1_000) return `${(amount / 1_000).toFixed(1)}K`;
  return amount.toLocaleString();
}
