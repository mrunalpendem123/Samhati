"""SSE streaming handler for Samhati SDK."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

from samhati.types import ChatCompletionChunk

if TYPE_CHECKING:
    import httpx


def parse_sse_line(line: str) -> dict | None:
    """Parse a single SSE data line into a dict, or None if not a data line."""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data: "):
        return None
    payload = line[len("data: "):]
    if payload.strip() == "[DONE]":
        return None
    return json.loads(payload)


class SyncSSEStream:
    """Synchronous iterator over an SSE response that yields ChatCompletionChunk."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = self._iter_lines()

    def _iter_lines(self) -> Iterator[ChatCompletionChunk]:
        buffer = ""
        for chunk in self._response.iter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                parsed = parse_sse_line(line)
                if parsed is not None:
                    yield ChatCompletionChunk.from_dict(parsed)

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return self._iterator

    def __next__(self) -> ChatCompletionChunk:
        return next(self._iterator)

    def close(self) -> None:
        self._response.close()


class AsyncSSEStream:
    """Async iterator over an SSE response that yields ChatCompletionChunk."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        buffer = ""
        async for chunk in self._response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                parsed = parse_sse_line(line)
                if parsed is not None:
                    yield ChatCompletionChunk.from_dict(parsed)

    async def close(self) -> None:
        await self._response.aclose()
