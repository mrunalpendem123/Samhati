import { useState } from "react";
import { useWallet } from "../hooks/useWallet";
import RewardCounter from "../components/RewardCounter";

export default function Wallet() {
  const { wallet, loading, error, claim } = useWallet();
  const [copied, setCopied] = useState(false);
  const [claimResult, setClaimResult] = useState<string | null>(null);

  const copyPubkey = async () => {
    try {
      await navigator.clipboard.writeText(wallet.pubkey);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard not available
    }
  };

  const handleClaim = async () => {
    setClaimResult(null);
    const amount = await claim();
    if (amount > 0) {
      setClaimResult(`Claimed ${amount} SMTI successfully!`);
      setTimeout(() => setClaimResult(null), 4000);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white">SMTI Wallet</h2>
        <p className="text-sm text-gray-500">
          Manage your Solana-based SMTI rewards
        </p>
      </div>

      {/* Balance card */}
      <div className="glass-panel p-6">
        <RewardCounter
          balance={wallet.balance}
          pending={wallet.pending_rewards}
          label="SMTI Balance"
        />
      </div>

      {/* Public key */}
      <div className="glass-panel p-5">
        <span className="stat-label">Solana Public Key</span>
        <div className="flex items-center gap-2 mt-2">
          <code className="flex-1 text-sm font-mono text-samhati-300 bg-gray-800 px-3 py-2 rounded-lg truncate">
            {wallet.pubkey}
          </code>
          <button
            onClick={copyPubkey}
            className="btn-secondary text-xs px-3 py-2 flex-shrink-0"
          >
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
      </div>

      {/* Claim rewards */}
      <div className="glass-panel p-5">
        <div className="flex items-center justify-between">
          <div>
            <span className="stat-label">Pending Rewards</span>
            <p className="text-xl font-bold font-mono text-white mt-1">
              {wallet.pending_rewards.toLocaleString()}{" "}
              <span className="text-sm text-gray-500">SMTI</span>
            </p>
          </div>
          <button
            onClick={handleClaim}
            disabled={loading || wallet.pending_rewards === 0}
            className="btn-primary"
          >
            {loading ? "Claiming..." : "Claim Rewards"}
          </button>
        </div>
        {claimResult && (
          <p className="text-sm text-green-400 mt-3">{claimResult}</p>
        )}
        {error && <p className="text-sm text-red-400 mt-3">{error}</p>}
      </div>

      {/* Transaction history placeholder */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">
          Transaction History
        </h3>
        <div className="text-center py-8">
          <p className="text-sm text-gray-600">
            Transaction history will appear here once rewards are claimed.
          </p>
        </div>
      </div>

      {/* Staking info */}
      <div className="glass-panel p-5 border-dashed border-gray-700">
        <div className="flex items-center gap-2 mb-2">
          <h3 className="text-sm font-semibold text-gray-300">Staking</h3>
          <span className="px-2 py-0.5 bg-gray-800 text-gray-500 text-xs rounded-full">
            Coming Soon
          </span>
        </div>
        <p className="text-sm text-gray-600">
          Stake your SMTI tokens to earn additional rewards and increase your
          node priority in the mesh network.
        </p>
      </div>
    </div>
  );
}
