import { useState, useEffect } from "react";
import { useNode } from "../hooks/useNode";
import EloDisplay from "../components/EloDisplay";
import RewardCounter from "../components/RewardCounter";
import type { DashboardStats } from "../lib/types";
import * as api from "../lib/api";

export default function Dashboard() {
  const { status, start, stop, loading } = useNode();
  const [stats, setStats] = useState<DashboardStats | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const s = await api.getDashboardStats();
        setStats(s);
      } catch {
        // Stats fetch failed silently
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatUptime = (secs: number | null | undefined): string => {
    if (!secs || secs <= 0) return "0s";
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Dashboard</h2>
          <p className="text-sm text-gray-500">Node performance and earnings</p>
        </div>
        <button
          onClick={() =>
            status.running ? stop() : start("llama-3.2-3b")
          }
          disabled={loading}
          className={status.running ? "btn-danger" : "btn-primary"}
        >
          {loading
            ? "..."
            : status.running
              ? "Stop Node"
              : "Start Node"}
        </button>
      </div>

      {/* ELO + Node status row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="stat-card items-center justify-center py-6">
          <EloDisplay score={status.elo_score} size="lg" />
        </div>

        <div className="stat-card">
          <span className="stat-label">Node Status</span>
          <div className="flex items-center gap-2 mt-2">
            <div
              className={`w-3 h-3 rounded-full ${
                status.running
                  ? "bg-green-400 animate-pulse"
                  : "bg-gray-600"
              }`}
            />
            <span className="text-lg font-semibold">
              {status.running ? "Online" : "Offline"}
            </span>
          </div>
          {status.model && (
            <span className="text-sm text-gray-400 mt-1">{status.model}</span>
          )}
          <span className="text-xs text-gray-500 mt-2">
            Uptime: {formatUptime(status.uptime_secs)}
          </span>
        </div>

        <div className="stat-card">
          <RewardCounter
            balance={stats?.earnings_total ?? 0}
            pending={0}
            label="Total Earnings"
          />
          <div className="grid grid-cols-2 gap-2 mt-3">
            <div>
              <span className="text-xs text-gray-500">Today</span>
              <p className="text-sm font-mono text-white">
                {stats?.earnings_today ?? 0} SMTI
              </p>
            </div>
            <div>
              <span className="text-xs text-gray-500">This Week</span>
              <p className="text-sm font-mono text-white">
                {stats?.earnings_week ?? 0} SMTI
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="stat-card">
          <span className="stat-label">Inferences Served</span>
          <span className="stat-value">
            {(stats?.inferences_served ?? 0).toLocaleString()}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Network Nodes</span>
          <span className="stat-value">
            {(stats?.network_nodes_online ?? 0).toLocaleString()}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Network Inferences / 24h</span>
          <span className="stat-value">
            {(stats?.network_inferences_24h ?? 0).toLocaleString()}
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Uptime</span>
          <span className="stat-value">
            {formatUptime(stats?.uptime_secs)}
          </span>
        </div>
      </div>

      {/* Domain breakdown */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          Domain Breakdown
        </h3>
        <div className="space-y-3">
          {(stats?.domain_breakdown ?? []).map((d) => (
            <div key={d.domain} className="flex items-center gap-3">
              <span className="text-sm text-gray-400 w-20">{d.domain}</span>
              <div className="flex-1 bg-gray-800 rounded-full h-2.5">
                <div
                  className="bg-samhati-500 h-2.5 rounded-full transition-all duration-500"
                  style={{ width: `${d.percentage}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 w-16 text-right">
                {d.count} ({d.percentage}%)
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ELO History placeholder */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          ELO History
        </h3>
        {stats?.elo_history && stats.elo_history.length > 0 ? (
          <div className="space-y-1">
            {stats.elo_history.map((point, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="text-gray-500 w-40">{point.timestamp}</span>
                <span className="text-samhati-300 font-mono">{point.score}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-600">
            ELO history will appear here once the node starts serving inferences.
            Connect a charting library (recharts, chart.js) for visualization.
          </p>
        )}
      </div>
    </div>
  );
}
