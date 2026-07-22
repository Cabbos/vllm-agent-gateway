from __future__ import annotations

from typing import Any


class DocumentError(ValueError):
    """Base error raised by document loading and conversion code."""

    code = "document_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "status_code": self.status_code,
            "details": dict(self.details),
        }


class DocumentSourceError(DocumentError):
    code = "invalid_document_source"


class DocumentFormatError(DocumentError):
    code = "invalid_document_format"


class DocumentTooLargeError(DocumentError):
    code = "document_too_large"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=413, details=details)


class DocumentLimitError(DocumentError):
    code = "document_limit_exceeded"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=413, details=details)


class RemoteDocumentDeniedError(DocumentError):
    code = "remote_document_denied"


class RemoteDocumentFetchError(DocumentError):
    code = "remote_document_fetch_failed"

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, details=details)
