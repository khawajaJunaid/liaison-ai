"""PDF to positioned text lines.

Digital PDFs already carry their text and coordinates, so they never touch OCR. Only a page
with no usable text layer is rasterised and sent to an OcrEngine. Both paths produce the same
Line objects (text + page + bbox in PDF points), which is what lets every extracted value
point back at a box on the page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pymupdf

from .config import MIN_TEXT_CHARS, OCR_DPI
from .domain import Source


@dataclass
class Line:
    text: str
    page: int
    bbox: tuple[float, float, float, float]


class OcrUnavailable(RuntimeError):
    """Raised by an engine that is not installed or not configured."""


class OcrEngine(Protocol):
    name: str

    def read(self, png: bytes) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Return (text, pixel_bbox) per line, in reading order."""


class NullOcr:
    """Default engine: refuses loudly so a scanned lease is flagged, not silently emptied."""

    name = "none"

    def read(self, png: bytes):
        raise OcrUnavailable("no OCR engine configured (set LEASE_AGENT_OCR=paddle)")


def make_ocr(name: str) -> OcrEngine | None:
    """Build the OCR engine named in settings. An unknown name fails at startup, not on a scan."""
    if name in ("", "none"):
        return None
    if name == "paddle":
        from .ocr_paddle import PaddleOcrEngine  # optional heavy dependency, imported on demand

        return PaddleOcrEngine()
    raise ValueError(f"unknown OCR engine '{name}' (expected none or paddle)")


@dataclass
class Document:
    lines: list[Line]
    page_count: int
    ocr_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        parts, self._spans, pos = [], [], 0
        for ln in self.lines:
            self._spans.append((pos, pos + len(ln.text)))
            parts.append(ln.text)
            pos += len(ln.text) + 1
        self.text = "\n".join(parts)

    def locate(self, start: int, end: int) -> Source | None:
        """Source for a character span of self.text: the lines it touches, on the first page."""
        hit = [ln for ln, (a, b) in zip(self.lines, self._spans) if a < end and b > start]
        if not hit:
            return None
        return self._source(hit)

    def locate_lines(self, first: int, last: int) -> Source:
        return self._source(self.lines[first:last + 1])

    @staticmethod
    def _source(hit: list[Line]) -> Source:
        page = hit[0].page
        same = [ln for ln in hit if ln.page == page]
        return Source(
            page=page,
            bbox=(min(l.bbox[0] for l in same), min(l.bbox[1] for l in same),
                  max(l.bbox[2] for l in same), max(l.bbox[3] for l in same)),
            quote=" ".join(l.text.strip() for l in same)[:300],
        )


def _text_lines(page: pymupdf.Page, number: int) -> list[Line]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            text = "".join(span["text"] for span in ln["spans"]).strip()
            if text:
                lines.append(Line(text, number, tuple(ln["bbox"])))
    return lines


def load_document(path: Path, ocr: OcrEngine | None = None) -> Document:
    pdf = pymupdf.open(path)
    lines: list[Line] = []
    ocr_pages: list[int] = []
    warnings: list[str] = []
    for number, page in enumerate(pdf, start=1):
        page_lines = _text_lines(page, number)
        if sum(len(l.text) for l in page_lines) >= MIN_TEXT_CHARS:
            lines.extend(page_lines)
            continue
        if ocr is None:
            warnings.append(f"Page {number} has no text layer and no OCR engine is configured.")
            continue
        zoom = OCR_DPI / 72
        png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
        try:
            for text, (x0, y0, x1, y1) in ocr.read(png):
                lines.append(Line(text.strip(), number, (x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom)))
            ocr_pages.append(number)
        except OcrUnavailable as exc:
            warnings.append(f"Page {number} is a scan and OCR is unavailable: {exc}.")
    return Document(lines=lines, page_count=len(pdf), ocr_pages=ocr_pages, warnings=warnings)
