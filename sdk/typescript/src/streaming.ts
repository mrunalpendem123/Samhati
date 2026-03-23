/** SSE streaming handler for Samhati SDK. */

import type { ChatCompletionChunk } from "./types.js";

/**
 * Parse an SSE stream from a fetch Response into an async generator
 * of ChatCompletionChunk objects.
 */
export async function* parseSSEStream(
  response: Response
): AsyncGenerator<ChatCompletionChunk> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Response body is null");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete lines
      const lines = buffer.split("\n");
      // Keep the last (possibly incomplete) line in the buffer
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(":")) continue;
        if (!trimmed.startsWith("data: ")) continue;

        const payload = trimmed.slice("data: ".length).trim();
        if (payload === "[DONE]") return;

        const chunk: ChatCompletionChunk = JSON.parse(payload);
        yield chunk;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
