"""PaddleOCR engines for scanned leases (optional install).

    pip install -r requirements-ocr.txt
    LEASE_AGENT_OCR=paddle uvicorn app.main:app

Two engines, because only one of them runs everywhere:

paddle      Classic PP-OCR text detection and recognition. Line-level boxes, runs on CPU, and is
            the engine verified end to end in this repo (scripts/try_ocr.py on the scanned sample,
            on an Intel Mac with no CUDA). Set LEASE_AGENT_OCR_LANG for another language, for
            example `ar` for Arabic.

paddle-vl   PaddleOCR-VL (~0.9B parameters, Apache-2.0). It scores higher on OmniDocBench v1.6
            (96.34) but needs PaddlePaddle >= 3.2, which is not published for Intel macOS, so it
            was NOT run here. It returns layout blocks rather than lines, so its boxes are
            paragraph-level. Use it on Linux or Apple silicon, and verify before relying on it.

Both run locally: no API key, and no lease text leaves the machine. The first run downloads models.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .ingest import OcrUnavailable

Box = tuple[float, float, float, float]


def _reading_order(lines: list[tuple[str, Box]]) -> list[tuple[str, Box]]:
    """Top to bottom, and left to right within a row. Lines whose tops are within half a line height
    of the row's first line are one row, so slightly uneven fragments stay in order."""
    rows: list[list[tuple[str, Box]]] = []
    for line in sorted(lines, key=lambda ln: ln[1][1]):
        height = line[1][3] - line[1][1]
        if rows and abs(line[1][1] - rows[-1][0][1][1]) <= height / 2:
            rows[-1].append(line)
        else:
            rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda ln: ln[1][0])]


class PaddleOcrEngine:
    """Classic PP-OCR. Lightweight mobile models: much faster on CPU than the server models."""

    name = "paddleocr"

    def __init__(self, lang: str | None = None):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OcrUnavailable("paddleocr is not installed (pip install -r requirements-ocr.txt)") from exc
        self.lang = lang or os.environ.get("LEASE_AGENT_OCR_LANG", "en")
        # Orientation and unwarping stages are off: leases are upright pages, and they add CPU time.
        # oneDNN is off because it stalls on some macOS CPUs.
        self._ocr = PaddleOCR(
            lang=self.lang, enable_mkldnn=False, cpu_threads=4,
            text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    def read(self, png: bytes) -> list[tuple[str, Box]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            path.write_bytes(png)
            lines: list[tuple[str, Box]] = []
            for res in self._ocr.predict(str(path)):
                data = res.json if hasattr(res, "json") else res
                data = data.get("res", data)
                for text, box in zip(data.get("rec_texts", []), data.get("rec_boxes", [])):
                    if str(text).strip():
                        x0, y0, x1, y1 = (float(v) for v in box)
                        lines.append((str(text).strip(), (x0, y0, x1, y1)))
        return _reading_order(lines)


class PaddleVLEngine:
    """PaddleOCR-VL. Needs PaddlePaddle >= 3.2. Not run in this repo; see the module docstring."""

    name = "paddleocr-vl"

    def __init__(self):
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise OcrUnavailable('paddleocr is not installed (pip install "paddleocr[doc-parser]")') from exc
        try:
            self._pipeline = PaddleOCRVL()
        except ImportError as exc:  # e.g. fused_rms_norm_ext: the installed PaddlePaddle is too old
            raise OcrUnavailable(f"PaddleOCR-VL cannot load with this PaddlePaddle (it needs >= 3.2): {exc}") from exc

    def read(self, png: bytes) -> list[tuple[str, Box]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            path.write_bytes(png)
            lines: list[tuple[str, Box]] = []
            for res in self._pipeline.predict(str(path)):
                data = res.json if hasattr(res, "json") else res
                data = data.get("res", data)
                for block in data.get("parsing_res_list", []):
                    x0, y0, x1, y1 = block["block_bbox"]
                    for text in str(block.get("block_content", "")).splitlines():
                        if text.strip():
                            lines.append((text.strip(), (float(x0), float(y0), float(x1), float(y1))))
            return lines
