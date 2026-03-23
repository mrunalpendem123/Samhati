import { useState, useEffect, useCallback } from "react";
import type { NodeStatus } from "../lib/types";
import * as api from "../lib/api";

export function useNode() {
  const [status, setStatus] = useState<NodeStatus>({
    running: false,
    model: null,
    elo_score: 1200,
    inferences_served: 0,
    uptime_secs: null,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await api.getNodeStatus();
      setStatus(s);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, [refresh]);

  const start = useCallback(async (modelName: string) => {
    setLoading(true);
    try {
      await api.startNode(modelName);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  const stop = useCallback(async () => {
    setLoading(true);
    try {
      await api.stopNode();
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  return { status, loading, error, start, stop, refresh };
}
