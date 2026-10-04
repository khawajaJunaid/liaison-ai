"""The lease agent's steps. Order and membership are set in agents.DEFAULT_LEASE_STEPS.

read_pdf -> extract_fields -> match_unit -> validate_rules -> self_check

match_unit, validate_rules and self_check are deterministic functions of the extracted fields,
so they are marked rerun: a human override re-runs just those three.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import RENT_RANGE
from .domain import Flag, Lease, Severity, Unit
from .extract import FIELD_NAMES, extract_fields
from .ingest import Document, load_document
from .pipeline import BaseStep, register
from .rules import Context, evaluate

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


@dataclass
class LeaseCtx:
    lease: Lease
    path: Path | None = None
    states: dict | None = None
    doc: Document | None = None
    unit: Unit | None = None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@register("lease", "read_pdf")
class ReadPdf(BaseStep):
    """Read the PDF into positioned text lines; OCR only pages with no text layer."""

    provides = frozenset({"doc"})

    def run(self, ctx: LeaseCtx) -> None:
        doc = load_document(ctx.path, self.services.ocr)
        ctx.doc = doc
        lease = ctx.lease
        lease.page_count, lease.ocr_pages, lease.warnings = doc.page_count, doc.ocr_pages, list(doc.warnings)
        lease.trace.append(f"Read {doc.page_count} page(s), {len(doc.lines)} text lines"
                           + (f"; OCR on page(s) {doc.ocr_pages}" if doc.ocr_pages else "; no OCR needed"))


@register("lease", "extract_fields")
class ExtractFields(BaseStep):
    """Extract lease fields with page, box and quote for each."""

    needs = frozenset({"doc"})
    provides = frozenset({"fields"})

    def run(self, ctx: LeaseCtx) -> None:
        ctx.lease.fields = extract_fields(ctx.doc)
        found = sum(1 for f in ctx.lease.fields.values() if f.value is not None)
        ctx.lease.trace.append(f"Extracted {found} of {len(FIELD_NAMES)} fields, each with page, box and quote")


@register("lease", "match_unit")
class MatchUnit(BaseStep):
    """Match the lease's unit reference to a unit in the owner's records."""

    needs = frozenset({"fields"})
    provides = frozenset({"unit"})
    rerun = True
    uses = ("units",)

    def run(self, ctx: LeaseCtx) -> None:
        ref = ctx.lease.fields["unit_ref"]
        ctx.unit, how = self.services.units.resolve(ref.value if ref.usable else None, ctx.states)
        ctx.lease.unit_id = ctx.unit.unit_id if ctx.unit else None
        ctx.lease.unit_match = how


@register("lease", "validate_rules")
class ValidateRules(BaseStep):
    """Check the lease against the owner's ruleset: PASS, FAIL or NOT_DETERMINABLE."""

    needs = frozenset({"fields", "unit"})
    provides = frozenset({"rules"})
    rerun = True
    uses = ("ruleset",)

    def run(self, ctx: LeaseCtx) -> None:
        lease = ctx.lease
        readable = any(f.value is not None for f in lease.fields.values())
        lease.rule_results = evaluate(self.services.ruleset,
                                      Context(lease.fields, ctx.unit, lease.unit_match or "", readable))
        tally = {o: sum(r.outcome == o for r in lease.rule_results) for o in ("PASS", "FAIL", "NOT_DETERMINABLE")}
        lease.trace.append(f"Validated: {tally['PASS']} pass, {tally['FAIL']} fail, {tally['NOT_DETERMINABLE']} not determinable")


@register("lease", "self_check")
class SelfCheck(BaseStep):
    """Raise flags for what a human should look at: gaps, conflicts, contradictions, odd values."""

    needs = frozenset({"fields", "rules"})
    provides = frozenset({"flags"})
    rerun = True
    uses = ("units",)

    def run(self, ctx: LeaseCtx) -> None:
        lease = ctx.lease
        previous = {f.id: f.decision for f in lease.flags}  # decisions on unchanged flags survive
        lease.flags = [f.model_copy(update={"decision": previous.get(f.id, "pending")})
                       for f in self.flags(lease, ctx.unit)]
        lease.trace.append(f"Flagged {len(lease.flags)} item(s) for review")

    def flags(self, lease: Lease, unit: Unit | None) -> list[Flag]:
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
        owner = self.services.units.ownership_entity
        landlord = f["landlord_name"].value if f["landlord_name"].usable else None
        tenant = f["tenant_name"].value if f["tenant_name"].usable else None
        if landlord and _norm(landlord) != _norm(owner):
            add("party", "high", f"Landlord '{landlord}' is not the property owner '{owner}'.", ["landlord_name"])
        if landlord and tenant and _norm(landlord) == _norm(tenant):
            add("party", "high", "Landlord and tenant are the same party.", ["landlord_name", "tenant_name"])

        if f["unit_ref"].usable:
            if unit is None:
                add("unit", "high", f"No matching unit: {lease.unit_match}.", ["unit_ref"])
            elif unit.status != "available":
                add("unit", "high", f"{unit.unit_id} is '{unit.status}'. It must be available before a new lease is linked.", ["unit_ref"])
        return out
