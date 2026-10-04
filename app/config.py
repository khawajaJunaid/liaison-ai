"""Paths, thresholds and settings. Environment overrides keep tests and deployments isolated."""
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "seed"
DEFAULT_DATA_DIR = Path(os.environ.get("LEASE_AGENT_DATA", ROOT / "data"))

# A PDF page with fewer text characters than this is treated as a scan and sent to OCR.
MIN_TEXT_CHARS = 40
OCR_DPI = 200

# Plausibility bounds for "values that look wrong" flags (QAR, residential).
RENT_RANGE = (1_000, 100_000)


@dataclass(frozen=True)
class Settings:
    """What the agents are built from: which engines, and optionally which steps in which order.

    LEASE_AGENT_OCR           none | paddle
    LEASE_AGENT_VISION        stub | anthropic
    LEASE_AGENT_LEASE_STEPS   comma-separated step names; unset keeps the default pipeline
    LEASE_AGENT_ISSUE_STEPS   comma-separated step names; unset keeps the default pipeline
    """

    ocr: str = "none"
    vision: str = "stub"
    lease_steps: tuple[str, ...] | None = None
    issue_steps: tuple[str, ...] | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        def steps(key: str) -> tuple[str, ...] | None:
            parts = tuple(s.strip() for s in env.get(key, "").split(",") if s.strip())
            return parts or None

        return cls(ocr=env.get("LEASE_AGENT_OCR", "none").strip().lower() or "none",
                   vision=env.get("LEASE_AGENT_VISION", "stub").strip().lower() or "stub",
                   lease_steps=steps("LEASE_AGENT_LEASE_STEPS"),
                   issue_steps=steps("LEASE_AGENT_ISSUE_STEPS"))
