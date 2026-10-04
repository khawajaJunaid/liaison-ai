"""The two agents.

LeaseAgent: read -> extract -> self-check -> validate -> flag -> wait for a human. It records
each step in lease.trace. Human overrides call revalidate(), so a corrected rent or date
re-runs the rules and re-derives the flags; decisions already made on unchanged flags survive.

IssueAgent: assess each photo with the vision model, merge the findings, draft a work order.
It only drafts. Nothing reaches a vendor until a person accepts the order.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import RENT_RANGE
from .domain import Damage, ExtractedField, Flag, Issue, Lease, PhotoAssessment, Severity, Unit, WorkOrder, new_id
from .extract import FIELD_NAMES, extract_fields, parse_date
from .ingest import OcrEngine, load_document
from .rules import Context, evaluate
from .units import UnitRegistry
from .vision import VisionModel

REQUIRED = {  # field -> (label, severity if missing)
    "landlord_name": ("Landlord", "high"), "tenant_name": ("Tenant", "high"),
    "unit_ref": ("Unit", "high"), "commencement_date": ("Commencement date", "high"),
    "expiry_date": ("Expiry date", "high"), "term_months": ("Lease term", "medium"),
    "monthly_rent": ("Monthly rent", "high"), "deposit_amount": ("Security deposit", "medium"),
    "escalation_clause": ("Escalation clause", "medium"),
}
CONTRADICTIONS = {  # internal-consistency rules surfaced as flags as well as rule results
    "R4": ["commencement_date", "expiry_date", "term_months"],
    "R6": ["monthly_rent", "annual_rent"],
}
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


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class LeaseAgent:
    def __init__(self, units: UnitRegistry, ruleset: dict, ocr: OcrEngine | None = None):
        self.units, self.ruleset, self.ocr = units, ruleset, ocr

    def run(self, path: Path, filename: str, states=None) -> Lease:
        lease = Lease(id=new_id("lease"), filename=filename)
        doc = load_document(path, self.ocr)
        lease.page_count, lease.ocr_pages, lease.warnings = doc.page_count, doc.ocr_pages, list(doc.warnings)
        lease.trace.append(f"Read {doc.page_count} page(s), {len(doc.lines)} text lines"
                           + (f"; OCR on page(s) {doc.ocr_pages}" if doc.ocr_pages else "; no OCR needed"))
        lease.fields = extract_fields(doc)
        found = sum(1 for f in lease.fields.values() if f.value is not None)
        lease.trace.append(f"Extracted {found} of {len(FIELD_NAMES)} fields, each with page, box and quote")
        self.revalidate(lease, states)
        return lease

    def revalidate(self, lease: Lease, states=None) -> None:
        ref = lease.fields["unit_ref"]
        unit, how = self.units.resolve(ref.value if ref.usable else None, states)
        lease.unit_id, lease.unit_match = (unit.unit_id if unit else None), how
        readable = any(f.value is not None for f in lease.fields.values())
        lease.rule_results = evaluate(self.ruleset, Context(lease.fields, unit, how, readable))
        previous = {f.id: f.decision for f in lease.flags}
        lease.flags = [f.model_copy(update={"decision": previous.get(f.id, "pending")})
                       for f in self._flags(lease, unit, how)]
        tally = {o: sum(r.outcome == o for r in lease.rule_results) for o in ("PASS", "FAIL", "NOT_DETERMINABLE")}
        lease.trace.append(f"Validated: {tally['PASS']} pass, {tally['FAIL']} fail, "
                           f"{tally['NOT_DETERMINABLE']} not determinable; {len(lease.flags)} flag(s) for review")

    def _flags(self, lease: Lease, unit: Unit | None, unit_note: str) -> list[Flag]:
        out: list[Flag] = []

        def add(kind: str, severity: Severity, message: str, fields: list[str] | None = None, key: str = ""):
            fields = fields or []
            out.append(Flag(id=".".join([kind, *fields] or [kind, key]), kind=kind, severity=severity,
                            message=message, fields=fields))

        f = lease.fields
        for i, warning in enumerate(lease.warnings):
            add("scan", "high", warning, key=str(i))
        for name, (label, severity) in REQUIRED.items():
            if not f[name].usable:
                rejected = f[name].decision == "rejected"
                add("missing", severity, f"{label} {'was rejected by a reviewer' if rejected else 'could not be found'}.", [name])
        for name, fld in f.items():
            if fld.decision != "overridden" and fld.note and fld.note.startswith("CONFLICT"):
                add("conflict", "high", fld.note.removeprefix("CONFLICT: "), [name])
            elif (fld.usable and fld.decision == "pending" and fld.confidence < 0.7
                  and not name.endswith("_signed") and not fld.note):
                add("low_confidence", "low", f"{name.replace('_', ' ')} was read with low confidence ({fld.confidence:.0%}).", [name])
            elif fld.usable and fld.decision == "pending" and name == "deposit_amount" and fld.note:
                add("derived", "low", fld.note, [name])
        for rule in lease.rule_results:
            if rule.rule_id in CONTRADICTIONS and rule.outcome == "FAIL":
                add("contradiction", rule.severity, rule.reason, CONTRADICTIONS[rule.rule_id])

        rent, deposit, term = (f[k].value if f[k].usable else None for k in ("monthly_rent", "deposit_amount", "term_months"))
        if rent is not None and not RENT_RANGE[0] <= rent <= RENT_RANGE[1]:
            add("implausible", "medium", f"Monthly rent QAR {rent:,.0f} is outside the usual range QAR {RENT_RANGE[0]:,}-{RENT_RANGE[1]:,}.", ["monthly_rent"])
        if rent and deposit and deposit > 3 * rent:
            add("implausible", "medium", f"Deposit QAR {deposit:,.0f} is more than three months' rent.", ["deposit_amount"])
        if term is not None and term > 60:
            add("implausible", "medium", f"A {term}-month term is unusually long.", ["term_months"])
        landlord = f["landlord_name"].value if f["landlord_name"].usable else None
        tenant = f["tenant_name"].value if f["tenant_name"].usable else None
        if landlord and _norm(landlord) != _norm(self.units.ownership_entity):
            add("party", "high", f"Landlord '{landlord}' is not the property owner '{self.units.ownership_entity}'.", ["landlord_name"])
        if landlord and tenant and _norm(landlord) == _norm(tenant):
            add("party", "high", "Landlord and tenant are the same party.", ["landlord_name", "tenant_name"])

        if f["unit_ref"].usable:
            if unit is None:
                add("unit", "high", f"No matching unit: {unit_note}.", ["unit_ref"])
            elif unit.status != "available":
                add("unit", "high", f"{unit.unit_id} is '{unit.status}'. It must be available before a new lease is linked.", ["unit_ref"])
        return out

    # --- human decisions ---------------------------------------------------

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


# --- Part B -----------------------------------------------------------------

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
CONDITION_RANK = {"unknown": 0, "new": 1, "good": 2, "worn": 3, "damaged": 4}


class IssueAgent:
    def __init__(self, vision: VisionModel):
        self.vision = vision

    def run(self, unit: Unit, photos: list[tuple[str, bytes]], reporter: str, note: str, lease_id: str | None) -> Issue:
        assessments = [self.vision.assess(data, name, note) for name, data in photos]
        issue = Issue(id=new_id("issue"), unit_id=unit.unit_id, lease_id=lease_id, reporter=reporter,
                      note=note, photos=assessments)
        issue.work_order = self._draft(unit, issue, assessments)
        worst = max((a.condition for a in assessments), key=CONDITION_RANK.get, default="unknown")
        issue.summary = (f"{len(assessments)} photo(s); overall condition {worst}; "
                         f"{sum(len(a.damages) for a in assessments)} damage finding(s)")
        return issue

    @staticmethod
    def _draft(unit: Unit, issue: Issue, assessments: list[PhotoAssessment]) -> WorkOrder:
        findings: list[tuple[Damage, str]] = []
        seen: set[tuple[str, str | None]] = set()
        for a in assessments:  # one line per distinct finding, even if several photos show it
            for d in a.damages:
                if (d.type, d.affects) not in seen:
                    seen.add((d.type, d.affects))
                    findings.append((d, a.filename))
        findings.sort(key=lambda p: -SEVERITY_RANK[p[0].severity])
        equipment = list(dict.fromkeys(e.name for a in assessments for e in a.equipment))
        if findings:
            top: Damage = findings[0][0]
            title = f"{top.type}{': ' + top.affects if top.affects else ''} - {unit.label}"
            priority = top.severity
        else:
            title, priority = f"Inspect reported issue - {unit.label}", "low"
        lines = [f"Unit: {unit.label} ({unit.unit_id}), {unit.building_name}, {unit.property_name}.",
                 f"Reported by: {issue.reporter or 'unknown'}."]
        if issue.note:
            lines.append(f"Reporter note: {issue.note}")
        lines += [f"- {d.type} ({d.severity}){' on ' + d.affects if d.affects else ''} [{fn}]" for d, fn in findings] \
            or ["- No damage identified automatically; an inspector should look at the photos."]
        if equipment:
            lines.append("Equipment in photos: " + ", ".join(equipment) + ".")
        review = sum(a.needs_review for a in assessments)
        if review:
            lines.append(f"{review} of {len(assessments)} photo assessment(s) are low confidence and need a human check.")
        return WorkOrder(id=new_id("wo"), title=title, description="\n".join(lines), unit_id=unit.unit_id,
                         priority=priority, equipment=equipment)
