"""Custom exceptions for SGLangClient."""


class SGLangClientError(Exception):
    """Base exception for all SGLangClient errors."""


class SGLangHTTPError(SGLangClientError):
    """HTTP error from SGLang server."""

    def __init__(self, message: str, *, status: int, body: str = ""):
        """Initialize an `SGLangHTTPError` instance."""
        super().__init__(message)
        self.status = status
        self.body = body


class SGLangContextLengthError(SGLangHTTPError):
    """Prompt/context exceeds the model's maximum length (400 + length keywords)."""


class SGLangThrottledError(SGLangHTTPError):
    """Rate-limited or temporarily unavailable (429, 503)."""


class SGLangConnectionError(SGLangClientError):
    """Connection-level failure (connect, timeout, DNS)."""


class SGLangDecodingError(SGLangClientError):
    """Server returned non-JSON response body."""
