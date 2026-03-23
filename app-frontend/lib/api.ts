import { invoke } from "@tauri-apps/api/core";
import type {
  NodeStatus,
  WalletInfo,
  ModelInfo,
  DashboardStats,
} from "./types";

// ── Node commands ──────────────────────────────────────────────────────────

export async function startNode(modelName: string): Promise<string> {
  return invoke<string>("start_node", { modelName });
}

export async function stopNode(): Promise<void> {
  return invoke<void>("stop_node");
}

export async function getNodeStatus(): Promise<NodeStatus> {
  return invoke<NodeStatus>("get_node_status");
}

// ── Chat commands ──────────────────────────────────────────────────────────

export async function sendChat(
  message: string,
  mode: string
): Promise<string> {
  return invoke<string>("send_chat", { message, mode });
}

// ── Wallet commands ────────────────────────────────────────────────────────

export async function getWalletInfo(): Promise<WalletInfo> {
  return invoke<WalletInfo>("get_wallet_info");
}

export async function claimRewards(): Promise<number> {
  return invoke<number>("claim_rewards");
}

// ── Model commands ─────────────────────────────────────────────────────────

export async function listModels(): Promise<ModelInfo[]> {
  return invoke<ModelInfo[]>("list_models");
}

export async function downloadModel(modelId: string): Promise<void> {
  return invoke<void>("download_model", { modelId });
}

// ── Dashboard commands ─────────────────────────────────────────────────────

export async function getDashboardStats(): Promise<DashboardStats> {
  return invoke<DashboardStats>("get_dashboard_stats");
}
