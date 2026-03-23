/** Samhati client — OpenAI-compatible TypeScript client. */

import {
  APIError,
  AuthenticationError,
  RateLimitError,
  TimeoutError as SamhatiTimeoutError,
  ConnectionError as SamhatiConnectionError,
} from "./errors.js";
import { parseSSEStream } from "./streaming.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatParams,
  HealthStatus,
  Model,
  ModelList,
  SamhatiOptions,
} from "./types.js";

const DEFAULT_BASE_URL = "http://localhost:8000";
const DEFAULT_TIMEOUT = 60_000; // ms
const DEFAULT_MAX_RETRIES = 2;

export class Samhati {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeout: number;
  private readonly maxRetries: number;

  constructor(options: SamhatiOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  }

  // ── Private helpers ──

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "User-Agent": "samhati-typescript/0.1.0",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private async request<T>(
    method: string,
    path: string,
    options?: { body?: unknown; stream?: boolean }
  ): Promise<Response> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      try {
        const response = await fetch(`${this.baseUrl}${path}`, {
          method,
          headers: this.buildHeaders(),
          body: options?.body ? JSON.stringify(options.body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          // Retry on 429
          if (response.status === 429 && attempt < this.maxRetries) {
            lastError = new RateLimitError();
            continue;
          }
          await this.handleErrorResponse(response);
        }

        return response;
      } catch (error) {
        clearTimeout(timeoutId);
        if (error instanceof APIError) throw error;

        lastError = error as Error;
        if ((error as Error).name === "AbortError") {
          if (attempt >= this.maxRetries) {
            throw new SamhatiTimeoutError("Request timed out");
          }
          continue;
        }
        if (attempt >= this.maxRetries) {
          throw new SamhatiConnectionError((error as Error).message);
        }
      }
    }

    throw new SamhatiConnectionError(lastError?.message ?? "Unknown error");
  }

  private async handleErrorResponse(response: Response): Promise<never> {
    let body: unknown;
    let message: string;
    try {
      body = await response.json();
      const err = (body as Record<string, unknown>)?.error;
      if (err && typeof err === "object" && "message" in (err as Record<string, unknown>)) {
        message = (err as Record<string, string>).message;
      } else {
        message = String(err ?? response.statusText);
      }
    } catch {
      message = response.statusText;
    }

    if (response.status === 401) throw new AuthenticationError(message);
    if (response.status === 429) throw new RateLimitError(message);
    throw new APIError(message, response.status, body);
  }

  // ── Public API ──

  /**
   * Create a chat completion (non-streaming).
   */
  async chat(params: ChatParams): Promise<ChatCompletion> {
    const body = this.buildChatBody({ ...params, stream: false });
    const response = await this.request("POST", "/v1/chat/completions", { body });
    return (await response.json()) as ChatCompletion;
  }

  /**
   * Create a streaming chat completion. Yields chunks via an async generator.
   */
  async *chatStream(params: ChatParams): AsyncGenerator<ChatCompletionChunk> {
    const body = this.buildChatBody({ ...params, stream: true });
    const response = await this.request("POST", "/v1/chat/completions", {
      body,
      stream: true,
    });
    yield* parseSSEStream(response);
  }

  /**
   * List available models.
   */
  async models(): Promise<Model[]> {
    const response = await this.request("GET", "/v1/models");
    const data = (await response.json()) as ModelList;
    return data.data;
  }

  /**
   * Check node health.
   */
  async health(): Promise<HealthStatus> {
    const response = await this.request("GET", "/v1/health");
    return (await response.json()) as HealthStatus;
  }

  // ── Helpers ──

  private buildChatBody(params: ChatParams & { stream?: boolean }): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: params.model ?? "samhati-general-3b",
      messages: params.messages,
      temperature: params.temperature ?? 0.7,
      max_tokens: params.max_tokens ?? 2048,
      stream: params.stream ?? false,
      samhati_mode: params.mode ?? "best",
      samhati_proof: params.includeProof ?? false,
    };
    if (params.domain !== undefined) {
      body.samhati_domain = params.domain;
    }
    return body;
  }
}
