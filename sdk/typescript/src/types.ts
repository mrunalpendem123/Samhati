/** OpenAI-compatible types with Samhati extensions. */

// ── Request Types ──

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: ToolCall[];
}

export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatParams {
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  /** Swarm routing mode: "quick" (N=3), "best" (N=7), "local" (N=1) */
  mode?: "quick" | "best" | "local";
  /** Domain hint for specialist routing */
  domain?: string;
  /** Include TOPLOC proof in response */
  includeProof?: boolean;
}

// ── Response Types ──

export interface ChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Choice[];
  usage?: Usage;
  // Samhati extensions
  samhati_node_id?: string;
  samhati_confidence?: number;
  samhati_proof?: string;
  samhati_n_nodes?: number;
}

export interface Choice {
  index: number;
  message: ChatMessage;
  finish_reason: string | null;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

// ── Streaming Types ──

export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: StreamChoice[];
  // Samhati extensions (typically on final chunk)
  samhati_node_id?: string;
  samhati_confidence?: number;
  samhati_proof?: string;
  samhati_n_nodes?: number;
}

export interface StreamChoice {
  index: number;
  delta: DeltaMessage;
  finish_reason: string | null;
}

export interface DeltaMessage {
  role?: string;
  content?: string;
  tool_calls?: ToolCall[];
}

// ── Model / Health Types ──

export interface Model {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface ModelList {
  object: string;
  data: Model[];
}

export interface HealthStatus {
  status: string;
  node_id?: string;
  version?: string;
  models_loaded?: string[];
  peers_connected?: number;
}

// ── Client Options ──

export interface SamhatiOptions {
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
}
