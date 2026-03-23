"""Samhati client — OpenAI-compatible sync and async clients."""

from __future__ import annotations

from collections.abc import Iterator
from typing import overload, Literal

import httpx

from samhati.exceptions import (
    APIError,
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    TimeoutError,
)
from samhati.streaming import AsyncSSEStream, SyncSSEStream
from samhati.types import (
    ChatCompletion,
    ChatCompletionChunk,
    HealthStatus,
    Model,
)

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_RETRIES = 2


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "samhati-python/0.1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _build_chat_body(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    mode: str,
    domain: str | None,
    include_proof: bool,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "samhati_mode": mode,
        "samhati_proof": include_proof,
    }
    if domain is not None:
        body["samhati_domain"] = domain
    return body


def _handle_error_response(response: httpx.Response) -> None:
    """Raise the appropriate exception for an error HTTP response."""
    try:
        body = response.json()
    except Exception:
        body = None

    message = ""
    if body and isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            message = err.get("message", response.text)
        else:
            message = str(err) if err else response.text
    else:
        message = response.text

    status = response.status_code
    if status == 401:
        raise AuthenticationError(message)
    if status == 429:
        raise RateLimitError(message)
    raise APIError(message, status_code=status, body=body)


# ── Synchronous Client ──


class Samhati:
    """OpenAI-compatible client for Samhati decentralized AI network."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=_build_headers(api_key),
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Samhati:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Private helpers ──

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                if stream:
                    req = self._client.build_request(method, path, json=json)
                    response = self._client.send(req, stream=True)
                else:
                    response = self._client.request(method, path, json=json)
                if response.status_code >= 400:
                    # Don't retry client errors (except 429)
                    if response.status_code == 429 and attempt < self.max_retries:
                        last_exc = RateLimitError()
                        continue
                    _handle_error_response(response)
                return response
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise ConnectionError(str(exc)) from exc
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise TimeoutError(str(exc)) from exc
        # Should not reach here, but just in case:
        raise ConnectionError(str(last_exc))

    # ── Public API ──

    @overload
    def chat(
        self,
        messages: list[dict],
        model: str = ...,
        temperature: float = ...,
        max_tokens: int = ...,
        *,
        stream: Literal[False] = ...,
        mode: str = ...,
        domain: str | None = ...,
        include_proof: bool = ...,
    ) -> ChatCompletion: ...

    @overload
    def chat(
        self,
        messages: list[dict],
        model: str = ...,
        temperature: float = ...,
        max_tokens: int = ...,
        *,
        stream: Literal[True],
        mode: str = ...,
        domain: str | None = ...,
        include_proof: bool = ...,
    ) -> Iterator[ChatCompletionChunk]: ...

    def chat(
        self,
        messages: list[dict],
        model: str = "samhati-general-3b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        mode: str = "best",
        domain: str | None = None,
        include_proof: bool = False,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion. Set stream=True for SSE streaming."""
        body = _build_chat_body(
            messages, model, temperature, max_tokens, stream, mode, domain, include_proof
        )
        if stream:
            response = self._request("POST", "/v1/chat/completions", json=body, stream=True)
            return SyncSSEStream(response)
        response = self._request("POST", "/v1/chat/completions", json=body)
        return ChatCompletion.from_dict(response.json())

    def models(self) -> list[Model]:
        """List available models."""
        response = self._request("GET", "/v1/models")
        data = response.json()
        return [Model.from_dict(m) for m in data.get("data", [])]

    def health(self) -> HealthStatus:
        """Check node health."""
        response = self._request("GET", "/v1/health")
        return HealthStatus.from_dict(response.json())


# ── Async Client ──


class AsyncSamhati:
    """Async OpenAI-compatible client for Samhati decentralized AI network."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=_build_headers(api_key),
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncSamhati:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ── Private helpers ──

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                if stream:
                    req = self._client.build_request(method, path, json=json)
                    response = await self._client.send(req, stream=True)
                else:
                    response = await self._client.request(method, path, json=json)
                if response.status_code >= 400:
                    if response.status_code == 429 and attempt < self.max_retries:
                        last_exc = RateLimitError()
                        continue
                    _handle_error_response(response)
                return response
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise ConnectionError(str(exc)) from exc
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise TimeoutError(str(exc)) from exc
        raise ConnectionError(str(last_exc))

    # ── Public API ──

    async def chat(
        self,
        messages: list[dict],
        model: str = "samhati-general-3b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        mode: str = "best",
        domain: str | None = None,
        include_proof: bool = False,
    ) -> ChatCompletion | AsyncSSEStream:
        """Create a chat completion. Set stream=True for async SSE streaming."""
        body = _build_chat_body(
            messages, model, temperature, max_tokens, stream, mode, domain, include_proof
        )
        if stream:
            response = await self._request("POST", "/v1/chat/completions", json=body, stream=True)
            return AsyncSSEStream(response)
        response = await self._request("POST", "/v1/chat/completions", json=body)
        return ChatCompletion.from_dict(response.json())

    async def models(self) -> list[Model]:
        """List available models."""
        response = await self._request("GET", "/v1/models")
        data = response.json()
        return [Model.from_dict(m) for m in data.get("data", [])]

    async def health(self) -> HealthStatus:
        """Check node health."""
        response = await self._request("GET", "/v1/health")
        return HealthStatus.from_dict(response.json())
