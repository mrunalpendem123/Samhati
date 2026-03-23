"""Tests for the Samhati Python SDK."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from samhati import (
    Samhati,
    AsyncSamhati,
    ChatCompletion,
    ChatCompletionChunk,
    Model,
    HealthStatus,
)
from samhati.exceptions import APIError, AuthenticationError, RateLimitError


BASE_URL = "http://localhost:8000"

# ── Fixtures ──

CHAT_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "samhati-general-3b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
    "samhati_node_id": "node-xyz",
    "samhati_confidence": 0.92,
    "samhati_n_nodes": 7,
}

MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {"id": "samhati-general-3b", "object": "model", "created": 1700000000, "owned_by": "samhati"},
        {"id": "samhati-code-7b", "object": "model", "created": 1700000000, "owned_by": "samhati"},
    ],
}

HEALTH_RESPONSE = {
    "status": "ok",
    "node_id": "node-xyz",
    "version": "0.2.0",
    "models_loaded": ["samhati-general-3b"],
    "peers_connected": 12,
}

STREAM_CHUNKS = [
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "samhati-general-3b",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "samhati-general-3b",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "samhati-general-3b",
        "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
        "samhati_node_id": "node-xyz",
        "samhati_confidence": 0.92,
        "samhati_n_nodes": 7,
    },
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "samhati-general-3b",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]


def _make_sse_body(chunks: list[dict]) -> str:
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


# ── Sync Client Tests ──


@respx.mock
def test_chat_completion():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    client = Samhati(base_url=BASE_URL)
    result = client.chat(messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(result, ChatCompletion)
    assert result.id == "chatcmpl-abc123"
    assert result.choices[0].message.content == "Hello!"
    assert result.samhati_node_id == "node-xyz"
    assert result.samhati_confidence == 0.92
    assert result.samhati_n_nodes == 7
    assert result.usage is not None
    assert result.usage.total_tokens == 15
    client.close()


@respx.mock
def test_chat_with_mode_and_domain():
    route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    client = Samhati(base_url=BASE_URL)
    client.chat(
        messages=[{"role": "user", "content": "Hi"}],
        mode="quick",
        domain="medical",
        include_proof=True,
    )
    body = json.loads(route.calls[0].request.content)
    assert body["samhati_mode"] == "quick"
    assert body["samhati_domain"] == "medical"
    assert body["samhati_proof"] is True
    client.close()


@respx.mock
def test_chat_streaming():
    sse_body = _make_sse_body(STREAM_CHUNKS)
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
        )
    )
    client = Samhati(base_url=BASE_URL)
    chunks = list(client.chat(messages=[{"role": "user", "content": "Hi"}], stream=True))
    assert len(chunks) == 4
    assert chunks[1].choices[0].delta.content == "Hello"
    assert chunks[2].choices[0].delta.content == " world"
    assert chunks[2].samhati_node_id == "node-xyz"
    assert chunks[3].choices[0].finish_reason == "stop"
    client.close()


@respx.mock
def test_models():
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json=MODELS_RESPONSE)
    )
    client = Samhati(base_url=BASE_URL)
    models = client.models()
    assert len(models) == 2
    assert models[0].id == "samhati-general-3b"
    assert isinstance(models[0], Model)
    client.close()


@respx.mock
def test_health():
    respx.get(f"{BASE_URL}/v1/health").mock(
        return_value=httpx.Response(200, json=HEALTH_RESPONSE)
    )
    client = Samhati(base_url=BASE_URL)
    health = client.health()
    assert health.status == "ok"
    assert health.peers_connected == 12
    assert isinstance(health, HealthStatus)
    client.close()


@respx.mock
def test_authentication_error():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            401, json={"error": {"message": "Invalid API key"}}
        )
    )
    client = Samhati(base_url=BASE_URL)
    with pytest.raises(AuthenticationError):
        client.chat(messages=[{"role": "user", "content": "Hi"}])
    client.close()


@respx.mock
def test_api_error():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "Internal error"}}
        )
    )
    client = Samhati(base_url=BASE_URL)
    with pytest.raises(APIError) as exc_info:
        client.chat(messages=[{"role": "user", "content": "Hi"}])
    assert exc_info.value.status_code == 500
    client.close()


# ── Async Client Tests ──


@respx.mock
@pytest.mark.asyncio
async def test_async_chat():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_RESPONSE)
    )
    async with AsyncSamhati(base_url=BASE_URL) as client:
        result = await client.chat(messages=[{"role": "user", "content": "Hi"}])
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello!"
    assert result.samhati_confidence == 0.92


@respx.mock
@pytest.mark.asyncio
async def test_async_models():
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(200, json=MODELS_RESPONSE)
    )
    async with AsyncSamhati(base_url=BASE_URL) as client:
        models = await client.models()
    assert len(models) == 2


@respx.mock
@pytest.mark.asyncio
async def test_async_health():
    respx.get(f"{BASE_URL}/v1/health").mock(
        return_value=httpx.Response(200, json=HEALTH_RESPONSE)
    )
    async with AsyncSamhati(base_url=BASE_URL) as client:
        health = await client.health()
    assert health.status == "ok"
