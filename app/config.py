"""Paths and thresholds. Environment overrides keep tests and deployments isolated."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "seed"
DEFAULT_DATA_DIR = Path(os.environ.get("LEASE_AGENT_DATA", ROOT / "data"))

# A PDF page with fewer text characters than this is treated as a scan and sent to OCR.
MIN_TEXT_CHARS = 40
OCR_DPI = 200

# Plausibility bounds for "values that look wrong" flags (QAR, residential).
RENT_RANGE = (1_000, 100_000)
