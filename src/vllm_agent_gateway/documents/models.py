from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentLimits:
    """Resource ceilings applied before and during document conversion."""

    max_raw_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 64
    max_rendered_pages: int = 24
    max_extracted_chars: int = 500_000
    max_page_pixels: int = 16_000_000
    text_page_threshold: int = 40
    render_scale: float = 2.0
    jpeg_quality: int = 82

    def __post_init__(self) -> None:
        positive = {
            "max_raw_bytes": self.max_raw_bytes,
            "max_pdf_pages": self.max_pdf_pages,
            "max_page_pixels": self.max_page_pixels,
            "text_page_threshold": self.text_page_threshold,
            "render_scale": self.render_scale,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Document limits must be positive: {', '.join(invalid)}")
        if self.max_rendered_pages < 0 or self.max_extracted_chars < 0:
            raise ValueError("Rendered-page and extracted-character limits cannot be negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    data: bytes
    media_type: str
    source_kind: str
    title: str

    @property
    def raw_bytes(self) -> int:
        return len(self.data)


@dataclass(slots=True)
class DocumentStats:
    code: str
    source_kind: str
    raw_bytes: int
    pages: int = 0
    searchable_pages: int = 0
    rendered_pages: int = 0
    text_chars: int = 0
    truncated_pages: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
