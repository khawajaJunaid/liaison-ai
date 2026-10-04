"""PaddleOCR-VL engine for scanned leases (optional install).

    pip install "paddleocr[doc-parser]"
    LEASE_AGENT_OCR=paddle uvicorn app.main:app

PaddleOCR-VL (~0.9B parameters, Apache-2.0) was chosen for its OmniDocBench v1.6 score (96.34)
and its lead on the scanned subset of Real5-OmniDocBench. It runs locally: no API key and no
lease text leaves the machine.

NOT TESTED in this repository's CI: it needs a multi-GB model download. It is written to the
PaddleOCR 3.x `PaddleOCRVL.predict` API, and the unit tests use a fake engine for the OCR
routing. Verify against the PaddleOCR version you install. It returns layout blocks, not
lines, so every line split from a block shares that block's box: provenance for scanned
leases is paragraph-level, coarser than the line-level boxes digital PDFs give.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .ingest import OcrUnavailable


class PaddleOcrEngine:
    name = "paddleocr-vl"

    def __init__(self):
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise OcrUnavailable('paddleocr is not installed (pip install "paddleocr[doc-parser]")') from exc
        self._pipeline = PaddleOCRVL()

    def read(self, png: bytes) -> list[tuple[str, tuple[float, float, float, float]]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            path.write_bytes(png)
            results = self._pipeline.predict(str(path))
            lines: list[tuple[str, tuple[float, float, float, float]]] = []
            for res in results:
                data = res.json if hasattr(res, "json") else res
                data = data.get("res", data)
                for block in data.get("parsing_res_list", []):
                    x0, y0, x1, y1 = block["block_bbox"]
                    for text in str(block.get("block_content", "")).splitlines():
                        if text.strip():
                            lines.append((text.strip(), (float(x0), float(y0), float(x1), float(y1))))
            return lines
