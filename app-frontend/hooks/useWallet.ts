import { useState, useEffect, useCallback } from "react";
import type { WalletInfo } from "../lib/types";
import * as api from "../lib/api";

export function useWallet() {
  const [wallet, setWallet] = useState<WalletInfo>({
    pubkey: "Loading...",
    balance: 0,
    pending_rewards: 0,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const info = await api.getWalletInfo();
      setWallet(info);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  }, [refresh]);

  const claim = useCallback(async () => {
    setLoading(true);
    try {
      const amount = await api.claimRewards();
      await refresh();
      return amount;
    } catch (e) {
      setError(String(e));
      return 0;
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  return { wallet, loading, error, claim, refresh };
}
