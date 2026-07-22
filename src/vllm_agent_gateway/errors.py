from __future__ import annotations


class GatewayError(ValueError):
    """Expected client-facing gateway failure."""

    def __init__(self, message: str, status_code: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class AuthenticationError(GatewayError):
    def __init__(self, message: str = "Missing or invalid gateway API key."):
        super().__init__(message, status_code=401, code="authentication_error")


class RequestTooLargeError(GatewayError):
    def __init__(self, message: str = "Request body exceeds the configured limit."):
        super().__init__(message, status_code=413, code="request_too_large")


class CapacityError(GatewayError):
    def __init__(self, message: str, retry_after: int = 1):
        super().__init__(message, status_code=429, code="capacity_exceeded")
        self.retry_after = retry_after
