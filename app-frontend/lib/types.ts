// ── Node types ─────────────────────────────────────────────────────────────

export interface NodeStatus {
  running: boolean;
  model: string | null;
  elo_score: number;
  inferences_served: number;
  uptime_secs: number | null;
}

// ── Chat types ─────────────────────────────────────────────────────────────

export type ChatMode = "quick" | "best";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  mode?: ChatMode;
  model?: string;
}

// ── Wallet types ───────────────────────────────────────────────────────────

export interface WalletInfo {
  pubkey: string;
  balance: number;
  pending_rewards: number;
}

// ── Model types ────────────────────────────────────────────────────────────

export interface ModelInfo {
  id: string;
  name: string;
  domain: string;
  size_bytes: number;
  size_display: string;
  description: string;
  smti_bonus: string;
  downloaded: boolean;
  download_progress: number | null;
}

// ── Dashboard types ────────────────────────────────────────────────────────

export interface DashboardStats {
  earnings_today: number;
  earnings_week: number;
  earnings_total: number;
  inferences_served: number;
  uptime_secs: number;
  elo_score: number;
  elo_history: EloPoint[];
  domain_breakdown: DomainStat[];
  network_nodes_online: number;
  network_inferences_24h: number;
}

export interface EloPoint {
  timestamp: string;
  score: number;
}

export interface DomainStat {
  domain: string;
  count: number;
  percentage: number;
}

// ── Settings types ─────────────────────────────────────────────────────────

export interface AppSettings {
  modelsDir: string;
  maxVramMb: number;
  maxRamMb: number;
  maxConnections: number;
  relayPreference: "auto" | "direct" | "relay";
  autoUpdateModels: boolean;
  theme: "dark" | "light";
}
