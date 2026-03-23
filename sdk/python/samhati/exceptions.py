"""Custom exceptions for the Samhati SDK."""

from __future__ import annotations


class SamhatiError(Exception):
    """Base exception for all Samhati SDK errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class APIError(SamhatiError):
    """Raised when the API returns an error response."""

    def __init__(
        self,
        message: str,
        status_code: int,
        body: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(message)

    def __str__(self) -> str:
        return f"APIError(status={self.status_code}): {self.message}"


class AuthenticationError(APIError):
    """Raised when the API key is invalid or missing (HTTP 401)."""

    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, status_code=401)


class RateLimitError(APIError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429)


class TimeoutError(SamhatiError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message)


class ConnectionError(SamhatiError):
    """Raised when unable to connect to the Samhati node."""

    def __init__(self, message: str = "Failed to connect to Samhati node") -> None:
        super().__init__(message)
