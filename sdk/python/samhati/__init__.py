"""Samhati SDK — OpenAI-compatible client for the Samhati decentralized AI network."""

from samhati.client import Samhati, AsyncSamhati
from samhati.types import (
    ChatCompletion,
    ChatCompletionChunk,
    Model,
    HealthStatus,
)
from samhati.exceptions import (
    SamhatiError,
    APIError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ConnectionError,
)

__version__ = "0.1.0"

__all__ = [
    "Samhati",
    "AsyncSamhati",
    "ChatCompletion",
    "ChatCompletionChunk",
    "Model",
    "HealthStatus",
    "SamhatiError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "ConnectionError",
]
