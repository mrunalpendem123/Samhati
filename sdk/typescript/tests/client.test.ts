import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Samhati } from "../src/client.js";
import { AuthenticationError, APIError } from "../src/errors.js";
import type { ChatCompletion, ModelList, HealthStatus } from "../src/types.js";

const BASE_URL = "http://localhost:8000";

// ── Fixtures ──

const CHAT_RESPONSE: ChatCompletion = {
  id: "chatcmpl-abc123",
  object: "chat.completion",
  created: 1700000000,
  model: "samhati-general-3b",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hello!" },
      finish_reason: "stop",
    },
  ],
  usage: {
    prompt_tokens: 10,
    completion_tokens: 5,
    total_tokens: 15,
  },
  samhati_node_id: "node-xyz",
  samhati_confidence: 0.92,
  samhati_n_nodes: 7,
};

const MODELS_RESPONSE: ModelList = {
  object: "list",
  data: [
    { id: "samhati-general-3b", object: "model", created: 1700000000, owned_by: "samhati" },
    { id: "samhati-code-7b", object: "model", created: 1700000000, owned_by: "samhati" },
  ],
};

const HEALTH_RESPONSE: HealthStatus = {
  status: "ok",
  node_id: "node-xyz",
  version: "0.2.0",
  models_loaded: ["samhati-general-3b"],
  peers_connected: 12,
};

// ── Mock fetch ──

const originalFetch = globalThis.fetch;

function mockFetch(handler: (url: string, init?: RequestInit) => Promise<Response>) {
  globalThis.fetch = vi.fn(handler) as unknown as typeof fetch;
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function sseResponse(chunks: unknown[]): Response {
  const body = chunks
    .map((c) => `data: ${JSON.stringify(c)}\n\n`)
    .concat(["data: [DONE]\n\n"])
    .join("");
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ── Tests ──

describe("Samhati", () => {
  it("should create a chat completion", async () => {
    mockFetch(async (url) => {
      expect(url).toBe(`${BASE_URL}/v1/chat/completions`);
      return jsonResponse(CHAT_RESPONSE);
    });

    const client = new Samhati({ baseUrl: BASE_URL });
    const result = await client.chat({
      messages: [{ role: "user", content: "Hi" }],
    });

    expect(result.id).toBe("chatcmpl-abc123");
    expect(result.choices[0].message.content).toBe("Hello!");
    expect(result.samhati_node_id).toBe("node-xyz");
    expect(result.samhati_confidence).toBe(0.92);
    expect(result.samhati_n_nodes).toBe(7);
  });

  it("should send samhati-specific params", async () => {
    let capturedBody: Record<string, unknown> = {};
    mockFetch(async (_url, init) => {
      capturedBody = JSON.parse(init?.body as string);
      return jsonResponse(CHAT_RESPONSE);
    });

    const client = new Samhati({ baseUrl: BASE_URL });
    await client.chat({
      messages: [{ role: "user", content: "Hi" }],
      mode: "quick",
      domain: "medical",
      includeProof: true,
    });

    expect(capturedBody.samhati_mode).toBe("quick");
    expect(capturedBody.samhati_domain).toBe("medical");
    expect(capturedBody.samhati_proof).toBe(true);
  });

  it("should stream chat completions", async () => {
    const streamChunks = [
      {
        id: "chatcmpl-abc123",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "samhati-general-3b",
        choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
      },
      {
        id: "chatcmpl-abc123",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "samhati-general-3b",
        choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: null }],
      },
      {
        id: "chatcmpl-abc123",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "samhati-general-3b",
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        samhati_node_id: "node-xyz",
      },
    ];

    mockFetch(async () => sseResponse(streamChunks));

    const client = new Samhati({ baseUrl: BASE_URL });
    const collected = [];
    for await (const chunk of client.chatStream({
      messages: [{ role: "user", content: "Hi" }],
    })) {
      collected.push(chunk);
    }

    expect(collected.length).toBe(3);
    expect(collected[1].choices[0].delta.content).toBe("Hello");
    expect(collected[2].samhati_node_id).toBe("node-xyz");
  });

  it("should list models", async () => {
    mockFetch(async () => jsonResponse(MODELS_RESPONSE));

    const client = new Samhati({ baseUrl: BASE_URL });
    const models = await client.models();

    expect(models.length).toBe(2);
    expect(models[0].id).toBe("samhati-general-3b");
  });

  it("should check health", async () => {
    mockFetch(async () => jsonResponse(HEALTH_RESPONSE));

    const client = new Samhati({ baseUrl: BASE_URL });
    const health = await client.health();

    expect(health.status).toBe("ok");
    expect(health.peers_connected).toBe(12);
  });

  it("should throw AuthenticationError on 401", async () => {
    mockFetch(async () =>
      jsonResponse({ error: { message: "Invalid API key" } }, 401)
    );

    const client = new Samhati({ baseUrl: BASE_URL });
    await expect(
      client.chat({ messages: [{ role: "user", content: "Hi" }] })
    ).rejects.toThrow(AuthenticationError);
  });

  it("should throw APIError on 500", async () => {
    mockFetch(async () =>
      jsonResponse({ error: { message: "Internal error" } }, 500)
    );

    const client = new Samhati({ baseUrl: BASE_URL });
    await expect(
      client.chat({ messages: [{ role: "user", content: "Hi" }] })
    ).rejects.toThrow(APIError);
  });

  it("should include Authorization header when api key set", async () => {
    let capturedHeaders: Headers | undefined;
    mockFetch(async (_url, init) => {
      capturedHeaders = new Headers(init?.headers as Record<string, string>);
      return jsonResponse(HEALTH_RESPONSE);
    });

    const client = new Samhati({ baseUrl: BASE_URL, apiKey: "sk-test-123" });
    await client.health();

    expect(capturedHeaders?.get("Authorization")).toBe("Bearer sk-test-123");
  });
});
