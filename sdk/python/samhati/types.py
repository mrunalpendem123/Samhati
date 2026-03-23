"""Type definitions for Samhati SDK, OpenAI-compatible with Samhati extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Chat Completion Types ──


@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    type: str
    function: FunctionCall


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ChatMessage:
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"] is not None:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in data["tool_calls"]
            ]
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
        )


@dataclass
class Choice:
    index: int
    message: ChatMessage
    finish_reason: str | None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    """OpenAI-compatible chat completion with Samhati extensions."""

    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    # Samhati extensions
    samhati_node_id: str | None = None
    samhati_confidence: float | None = None
    samhati_proof: bytes | None = None
    samhati_n_nodes: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ChatCompletion:
        choices = [
            Choice(
                index=c["index"],
                message=ChatMessage.from_dict(c["message"]),
                finish_reason=c.get("finish_reason"),
            )
            for c in data["choices"]
        ]
        usage = None
        if "usage" in data and data["usage"] is not None:
            u = data["usage"]
            usage = Usage(
                prompt_tokens=u["prompt_tokens"],
                completion_tokens=u["completion_tokens"],
                total_tokens=u["total_tokens"],
            )
        proof_raw = data.get("samhati_proof")
        proof = None
        if proof_raw is not None:
            if isinstance(proof_raw, bytes):
                proof = proof_raw
            elif isinstance(proof_raw, str):
                import base64
                proof = base64.b64decode(proof_raw)
        return cls(
            id=data["id"],
            object=data["object"],
            created=data["created"],
            model=data["model"],
            choices=choices,
            usage=usage,
            samhati_node_id=data.get("samhati_node_id"),
            samhati_confidence=data.get("samhati_confidence"),
            samhati_proof=proof,
            samhati_n_nodes=data.get("samhati_n_nodes"),
        )


# ── Streaming Types ──


@dataclass
class DeltaMessage:
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> DeltaMessage:
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"] is not None:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", ""),
                    function=FunctionCall(
                        name=tc["function"].get("name", ""),
                        arguments=tc["function"].get("arguments", ""),
                    ),
                )
                for tc in data["tool_calls"]
            ]
        return cls(
            role=data.get("role"),
            content=data.get("content"),
            tool_calls=tool_calls,
        )


@dataclass
class StreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: str | None


@dataclass
class ChatCompletionChunk:
    """OpenAI-compatible streaming chunk with Samhati extensions."""

    id: str
    object: str
    created: int
    model: str
    choices: list[StreamChoice]
    # Samhati extensions (typically on final chunk only)
    samhati_node_id: str | None = None
    samhati_confidence: float | None = None
    samhati_proof: bytes | None = None
    samhati_n_nodes: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ChatCompletionChunk:
        choices = [
            StreamChoice(
                index=c["index"],
                delta=DeltaMessage.from_dict(c["delta"]),
                finish_reason=c.get("finish_reason"),
            )
            for c in data["choices"]
        ]
        proof_raw = data.get("samhati_proof")
        proof = None
        if proof_raw is not None:
            if isinstance(proof_raw, bytes):
                proof = proof_raw
            elif isinstance(proof_raw, str):
                import base64
                proof = base64.b64decode(proof_raw)
        return cls(
            id=data["id"],
            object=data["object"],
            created=data["created"],
            model=data["model"],
            choices=choices,
            samhati_node_id=data.get("samhati_node_id"),
            samhati_confidence=data.get("samhati_confidence"),
            samhati_proof=proof,
            samhati_n_nodes=data.get("samhati_n_nodes"),
        )


# ── Model / Health Types ──


@dataclass
class Model:
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "samhati"

    @classmethod
    def from_dict(cls, data: dict) -> Model:
        return cls(
            id=data["id"],
            object=data.get("object", "model"),
            created=data.get("created", 0),
            owned_by=data.get("owned_by", "samhati"),
        )


@dataclass
class HealthStatus:
    status: str
    node_id: str | None = None
    version: str | None = None
    models_loaded: list[str] = field(default_factory=list)
    peers_connected: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> HealthStatus:
        return cls(
            status=data["status"],
            node_id=data.get("node_id"),
            version=data.get("version"),
            models_loaded=data.get("models_loaded", []),
            peers_connected=data.get("peers_connected", 0),
        )
