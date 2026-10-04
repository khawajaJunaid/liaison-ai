"""The two agents: thin shells around a configurable pipeline of steps.

LeaseAgent and IssueAgent no longer hard-code their flow. Each is a list of registered step
names (see pipeline.py, lease_steps.py, issue_steps.py). Change the list, or register a step,
to change what an agent does; the default lists below reproduce the original behaviour.

Human decisions stay here: they are not pipeline work. A field override calls revalidate(),
which re-runs only the steps marked `rerun`, so a corrected rent or date re-checks the rules and
re-derives the flags while decisions already made on unchanged flags survive.
"""
from __future__ import annotations

from pathlib import Path

from . import issue_steps, lease_steps  # noqa: F401  (importing registers the steps)
from .domain import ExtractedField, Issue, Lease, Unit, new_id
from .extract import parse_date
from .ingest import OcrEngine
from .issue_steps import IssueCtx
from .lease_steps import LeaseCtx
from .pipeline import Pipeline, Services
from .units import UnitRegistry
from .vision import VisionModel

DEFAULT_LEASE_STEPS = ("read_pdf", "extract_fields", "match_unit", "validate_rules", "self_check")
DEFAULT_ISSUE_STEPS = ("assess_photos", "summarise", "draft_work_order")

FIELD_TYPES = {
    "commencement_date": "date", "expiry_date": "date", "term_months": "int",
    "monthly_rent": "float", "annual_rent": "float", "deposit_amount": "float",
    "landlord_signed": "bool", "tenant_signed": "bool", "escalation_defined": "bool",
}


def coerce(name: str, raw) -> object:
    """Turn a human-typed override into the type the rules expect. Raises ValueError if it cannot."""
    kind = FIELD_TYPES.get(name, "str")
    text = str(raw).strip()
    if kind == "date":
        parsed = parse_date(text)
        if parsed is None:
            raise ValueError(f"'{text}' is not a date (use YYYY-MM-DD)")
        return parsed.isoformat()
    if kind == "int":
        return int(float(text.replace(",", "")))
    if kind == "float":
        return float(text.replace(",", ""))
    if kind == "bool":
        if text.lower() in ("true", "yes", "1", "signed"):
            return True
        if text.lower() in ("false", "no", "0", "unsigned"):
            return False
        raise ValueError(f"'{text}' is not yes/no")
    return text


class LeaseAgent:
    def __init__(self, units: UnitRegistry, ruleset: dict, ocr: OcrEngine | None = None,
                 steps: list[str] | tuple[str, ...] | None = None):
        self.units = units
        self.pipeline = Pipeline("lease", steps or DEFAULT_LEASE_STEPS,
                                 Services(units=units, ruleset=ruleset, ocr=ocr))

    def run(self, path: Path, filename: str, states=None) -> Lease:
        ctx = LeaseCtx(lease=Lease(id=new_id("lease"), filename=filename), path=path, states=states)
        self.pipeline.run(ctx)
        return ctx.lease

    def revalidate(self, lease: Lease, states=None) -> None:
        """Re-run the deterministic steps after a human changed a field."""
        self.pipeline.run(LeaseCtx(lease=lease, states=states), only_rerun=True)

    def apply_field_decision(self, lease: Lease, name: str, action: str, value=None, states=None) -> ExtractedField:
        if name not in lease.fields:
            raise KeyError(name)
        fld = lease.fields[name]
        if action == "accept":
            fld.decision = "accepted"
        elif action == "reject":
            fld.decision = "rejected"
        elif action == "override":
            if value is None or str(value).strip() == "":
                raise ValueError("override needs a value")
            if fld.original_value is None:
                fld.original_value = fld.value
            fld.value, fld.decision, fld.extractor, fld.confidence = coerce(name, value), "overridden", "human", 1.0
        else:
            raise ValueError(f"unknown action '{action}'")
        self.revalidate(lease, states)
        return fld


class IssueAgent:
    def __init__(self, vision: VisionModel, steps: list[str] | tuple[str, ...] | None = None):
        self.vision = vision
        self.pipeline = Pipeline("issue", steps or DEFAULT_ISSUE_STEPS, Services(vision=vision))

    def run(self, unit: Unit, photos: list[tuple[str, bytes]], reporter: str, note: str,
            lease_id: str | None) -> Issue:
        ctx = IssueCtx(unit=unit, photos=photos, reporter=reporter, note=note, lease_id=lease_id)
        self.pipeline.run(ctx)
        return ctx.issue
