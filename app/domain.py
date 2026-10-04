"""Domain models. Everything the UI shows or a human can override lives here."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Decision = Literal["pending", "accepted", "rejected", "overridden"]
Outcome = Literal["PASS", "FAIL", "NOT_DETERMINABLE"]
Severity = Literal["high", "medium", "low"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Source(BaseModel):
    """Where a value came from: page (1-based), box in PDF points, and the text read there."""

    page: int
    bbox: tuple[float, float, float, float]
    quote: str


class ExtractedField(BaseModel):
    name: str
    value: Any = None
    source: Source | None = None
    confidence: float = 0.0
    extractor: str = "rules"
    decision: Decision = "pending"
    original_value: Any = None  # what the agent read, kept after a human override
    note: str | None = None

    @property
    def usable(self) -> bool:
        """A rejected field is treated as missing by validation."""
        return self.value is not None and self.decision != "rejected"


class Flag(BaseModel):
    id: str
    kind: str
    severity: Severity
    message: str
    fields: list[str] = Field(default_factory=list)
    decision: Decision = "pending"  # accepted = confirmed problem, rejected = dismissed


class RuleResult(BaseModel):
    rule_id: str
    description: str
    severity: Severity
    outcome: Outcome
    reason: str
    source: Source | None = None


class Lease(BaseModel):
    id: str
    filename: str
    status: Literal["pending_review", "accepted", "rejected"] = "pending_review"
    created_at: str = Field(default_factory=now)
    page_count: int = 0
    ocr_pages: list[int] = Field(default_factory=list)
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    flags: list[Flag] = Field(default_factory=list)
    rule_results: list[RuleResult] = Field(default_factory=list)
    unit_id: str | None = None
    unit_match: str | None = None  # how the unit was matched, for the reviewer
    warnings: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)  # what the agent did, in order


class Unit(BaseModel):
    unit_id: str
    label: str
    type: str
    area_sqm: float
    parking_bay: str | None = None
    status: str
    building_id: str
    building_name: str
    property_id: str
    property_name: str
    lease_id: str | None = None


# --- Part B -----------------------------------------------------------------

class Damage(BaseModel):
    type: str
    severity: Severity
    description: str
    affects: str | None = None  # equipment the damage concerns, if known


class EquipmentItem(BaseModel):
    name: str
    category: str
    condition: str


class PhotoAssessment(BaseModel):
    filename: str
    condition: Literal["new", "good", "worn", "damaged", "unknown"]
    damages: list[Damage] = Field(default_factory=list)
    equipment: list[EquipmentItem] = Field(default_factory=list)
    confidence: float = 0.0
    model: str = ""
    needs_review: bool = False
    decision: Decision = "pending"


class WorkOrder(BaseModel):
    id: str
    title: str
    description: str
    unit_id: str
    priority: Severity
    equipment: list[str] = Field(default_factory=list)
    status: Literal["draft", "accepted", "rejected"] = "draft"


class Issue(BaseModel):
    id: str
    unit_id: str
    lease_id: str | None = None  # the accepted lease on the unit when it was reported
    reporter: str = ""
    note: str = ""
    created_at: str = Field(default_factory=now)
    photos: list[PhotoAssessment] = Field(default_factory=list)
    summary: str = ""
    work_order: WorkOrder | None = None
