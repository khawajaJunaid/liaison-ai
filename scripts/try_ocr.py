"""Run the real OCR engine on a scanned lease, without starting the app.

    python -m scripts.try_ocr samples/lease_scanned.pdf            # PaddleOCR-VL (needs the optional install)
    python -m scripts.try_ocr my_phone_photo_of_a_lease.pdf

Prints what the engine read (page, box, text), then the fields the parser got from it and the result of
each owner rule. Compare against the document: wrong or missing values here are what a person would
have to fix on the review screen. The first run downloads the model, which is slow.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.agents import LeaseAgent
from app.config import SEED_DIR
from app.ingest import OcrUnavailable, load_document, make_ocr
from app.rules import load_ruleset
from app.units import UnitRegistry


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the OCR engine on a scanned lease PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--engine", default="paddle", help="paddle (default) or none")
    args = parser.parse_args(argv)
    if not args.pdf.is_file():
        print(f"No such file: {args.pdf}")
        return 2

    try:
        engine = make_ocr(args.engine)
    except (OcrUnavailable, ValueError) as exc:
        print(f"Cannot start the '{args.engine}' engine: {exc}")
        return 1
    print(f"engine={getattr(engine, 'name', 'none')}  file={args.pdf.name}\n")

    started = time.time()
    doc = load_document(args.pdf, engine)
    print(f"read in {time.time() - started:.1f}s: {len(doc.lines)} lines, OCR used on pages {doc.ocr_pages or 'none'}")
    for w in doc.warnings:
        print(f"  WARNING: {w}")
    print("\n--- what the engine read ---")
    for ln in doc.lines:
        x0, y0, x1, y1 = (round(v) for v in ln.bbox)
        print(f"  p{ln.page} ({x0},{y0})-({x1},{y1})  {ln.text}")

    units = UnitRegistry(SEED_DIR / "units.json")
    lease = LeaseAgent(units, load_ruleset(SEED_DIR / "owner_ruleset.json"), ocr=engine).run(args.pdf, args.pdf.name)
    print("\n--- fields the parser extracted ---")
    for name, f in lease.fields.items():
        shown = "(not found)" if f.value is None else str(f.value)[:60]
        print(f"  {name:20s} {shown}")
    print("\n--- owner rules ---")
    for r in lease.rule_results:
        print(f"  {r.rule_id} {r.outcome:16s} {r.reason}")
    print(f"\nunit: {lease.unit_id} ({lease.unit_match});  {len(lease.flags)} flag(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
