/** Samhati SDK — OpenAI-compatible client for the Samhati decentralized AI network. */

export { Samhati } from "./client.js";
export { parseSSEStream } from "./streaming.js";
export {
  SamhatiError,
  APIError,
  AuthenticationError,
  RateLimitError,
  TimeoutError,
  ConnectionError,
} from "./errors.js";
export type {
  ChatMessage,
  ChatParams,
  ChatCompletion,
  ChatCompletionChunk,
  Choice,
  StreamChoice,
  DeltaMessage,
  Usage,
  ToolCall,
  Model,
  ModelList,
  HealthStatus,
  SamhatiOptions,
} from "./types.js";
