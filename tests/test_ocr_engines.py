"""The Paddle engine adapters, driven by a fake `paddleocr` module (no models, no network).

The real engine is verified separately with `python -m scripts.try_ocr samples/lease_scanned.pdf`.
These tests pin the adapter: which models it asks for, how it reads results, and how it fails.
"""
import sys
import types

import pytest

from app.ingest import OcrUnavailable, make_ocr
from app.ocr_paddle import PaddleOcrEngine, PaddleVLEngine


class _Result:
    def __init__(self, payload):
        self.json = payload


def install_fake_paddleocr(monkeypatch, texts=(), boxes=(), vl_error=None):
    seen = {}

    class PaddleOCR:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def predict(self, path):
            seen["path"] = path
            return [_Result({"res": {"rec_texts": list(texts), "rec_boxes": [list(b) for b in boxes]}})]

    class PaddleOCRVL:
        def __init__(self):
            if vl_error:
                raise vl_error

        def predict(self, path):
            return [_Result({"res": {"parsing_res_list": [
                {"block_bbox": [10, 20, 300, 80], "block_content": "Landlord: A\nTenant: B"}]}})]

    module = types.ModuleType("paddleocr")
    module.PaddleOCR, module.PaddleOCRVL = PaddleOCR, PaddleOCRVL
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    return seen


def test_pp_ocr_adapter_asks_for_the_light_cpu_configuration(monkeypatch):
    seen = install_fake_paddleocr(monkeypatch)
    monkeypatch.delenv("LEASE_AGENT_OCR_LANG", raising=False)
    PaddleOcrEngine()
    kw = seen["kwargs"]
    assert kw["lang"] == "en" and kw["enable_mkldnn"] is False
    assert kw["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert kw["text_recognition_model_name"] == "PP-OCRv5_mobile_rec"
    assert not kw["use_doc_unwarping"] and not kw["use_textline_orientation"]


def test_language_comes_from_the_environment(monkeypatch):
    seen = install_fake_paddleocr(monkeypatch)
    monkeypatch.setenv("LEASE_AGENT_OCR_LANG", "ar")
    PaddleOcrEngine()
    assert seen["kwargs"]["lang"] == "ar"


def test_lines_come_back_in_reading_order_with_pixel_boxes(monkeypatch):
    install_fake_paddleocr(
        monkeypatch,
        texts=["Tenant: B", "Landlord: A", "LEASE", "  "],
        boxes=[(150, 400, 500, 430), (150, 322, 500, 360), (150, 144, 480, 175), (0, 0, 1, 1)])
    lines = PaddleOcrEngine().read(b"png")
    assert [t for t, _ in lines] == ["LEASE", "Landlord: A", "Tenant: B"]  # blank text dropped, sorted by y
    assert lines[0][1] == (150.0, 144.0, 480.0, 175.0)


def test_same_row_fragments_are_ordered_left_to_right(monkeypatch):
    install_fake_paddleocr(monkeypatch, texts=["right", "left"], boxes=[(400, 100, 500, 130), (100, 102, 200, 131)])
    assert [t for t, _ in PaddleOcrEngine().read(b"png")] == ["left", "right"]


def test_a_missing_install_is_a_clear_unavailable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)  # makes `import paddleocr` raise ImportError
    with pytest.raises(OcrUnavailable, match="requirements-ocr.txt"):
        PaddleOcrEngine()
    with pytest.raises(OcrUnavailable, match="not installed"):
        PaddleVLEngine()


def test_vl_engine_explains_when_paddlepaddle_is_too_old(monkeypatch):
    install_fake_paddleocr(monkeypatch, vl_error=ImportError("cannot import name 'fused_rms_norm_ext'"))
    with pytest.raises(OcrUnavailable, match=r"needs >= 3\.2"):
        PaddleVLEngine()


def test_vl_engine_splits_blocks_into_lines_sharing_the_block_box(monkeypatch):
    install_fake_paddleocr(monkeypatch)
    lines = PaddleVLEngine().read(b"png")
    assert lines == [("Landlord: A", (10.0, 20.0, 300.0, 80.0)), ("Tenant: B", (10.0, 20.0, 300.0, 80.0))]


def test_engine_names_map_to_the_right_classes(monkeypatch):
    install_fake_paddleocr(monkeypatch)
    assert make_ocr("paddle").name == "paddleocr"
    assert make_ocr("paddle-vl").name == "paddleocr-vl"
    assert make_ocr("none") is None
    with pytest.raises(ValueError, match="paddle-vl"):
        make_ocr("tesseract")
