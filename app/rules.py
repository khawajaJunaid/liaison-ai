"""Deterministic rule engine for the owner's ruleset.

No model is involved: every rule is arithmetic over extracted values, so the answer is
reproducible and the reason can quote the exact clause. The ruleset file decides which
rules run and their severity; this module supplies one function per rule id. A rule id in
the file with no function here reports NOT_DETERMINABLE rather than silently passing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from .domain import ExtractedField, RuleResult, Source, Unit


@dataclass
class Context:
    fields: dict[str, ExtractedField]
    unit: Unit | None
    unit_note: str = ""  # how the unit match went, used when no unit was found
    readable: bool = True  # False when nothing was read (an unOCR'd scan): absence proves nothing

    def val(self, name: str):
        f = self.fields.get(name)
        return f.value if f and f.usable else None

    def src(self, *names: str) -> Source | None:
        for n in names:
            f = self.fields.get(n)
            if f and f.usable and f.source:
                return f.source
        return None


Verdict = tuple[str, str, Source | None]  # outcome, reason, source


def _fmt(n: float) -> str:
    return f"{n:,.0f}" if float(n).is_integer() else f"{n:,.2f}"


def months_between(start: date, end_exclusive: date) -> int | None:
    """Whole months from start to end, or None when the day-of-month does not line up."""
    if end_exclusive.day != start.day:
        return None
    return (end_exclusive.year - start.year) * 12 + end_exclusive.month - start.month


def term_matches(start: date, expiry: date, term: int) -> tuple[bool, int]:
    """Check the stated term against the dates under either end-date convention.

    Leases write the last day as "28 Feb 2027" (inclusive) or "1 Mar 2027" (exclusive), so a
    12-month lease from 1 Mar 2026 is accepted for either. Returns (match, months_implied).
    """
    candidates = [m for m in (months_between(start, expiry + timedelta(days=1)),
                              months_between(start, expiry)) if m is not None]
    if term in candidates:
        return True, term
    implied = candidates[0] if candidates else round((expiry + timedelta(days=1) - start).days / 30.4375)
    return False, implied


def r1_deposit(c: Context) -> Verdict:
    deposit, rent = c.val("deposit_amount"), c.val("monthly_rent")
    if deposit is None or rent is None:
        missing = " and ".join(n for n, v in (("deposit", deposit), ("monthly rent", rent)) if v is None)
        return "NOT_DETERMINABLE", f"Could not read {missing}.", c.src("deposit_amount", "monthly_rent")
    if deposit >= rent:
        return "PASS", f"Deposit QAR {_fmt(deposit)} covers at least one month's rent (QAR {_fmt(rent)}).", c.src("deposit_amount")
    return "FAIL", f"Deposit QAR {_fmt(deposit)} is below one month's rent (QAR {_fmt(rent)}).", c.src("deposit_amount")


def r2_escalation(c: Context) -> Verdict:
    clause, defined = c.val("escalation_clause"), c.val("escalation_defined")
    if clause is None:
        if not c.readable:
            return "NOT_DETERMINABLE", "The document could not be read, so the clause cannot be checked.", None
        return "FAIL", "No rent escalation clause found in the lease.", None
    if defined:
        return "PASS", "Escalation clause states a mechanism (percentage or index).", c.src("escalation_clause")
    return "FAIL", "Escalation clause exists but sets no percentage or mechanism (e.g. 'as mutually agreed').", c.src("escalation_clause")


def r3_term_cap(c: Context) -> Verdict:
    term = c.val("term_months")
    if term is None:
        return "NOT_DETERMINABLE", "Could not read the lease term.", None
    if term <= 36:
        return "PASS", f"Fixed term of {term} months is within the 36-month limit.", c.src("term_months")
    return "FAIL", f"Fixed term of {term} months exceeds 36 months and needs owner approval.", c.src("term_months")


def r4_dates(c: Context) -> Verdict:
    start, expiry, term = c.val("commencement_date"), c.val("expiry_date"), c.val("term_months")
    src = c.src("commencement_date", "expiry_date")
    if start is None or expiry is None:
        return "NOT_DETERMINABLE", "Commencement or expiry date could not be read.", src
    start_d, expiry_d = date.fromisoformat(start), date.fromisoformat(expiry)
    if expiry_d <= start_d:
        return "FAIL", f"Expiry {expiry} is not after commencement {start}.", src
    if term is None:
        return "NOT_DETERMINABLE", "Dates are in order, but the stated term could not be read to compare.", src
    ok, implied = term_matches(start_d, expiry_d, int(term))
    if ok:
        return "PASS", f"{start} to {expiry} is {term} months, matching the stated term.", src
    return "FAIL", f"Stated term is {term} months but {start} to {expiry} spans about {implied} months.", src


def r5_parties(c: Context) -> Verdict:
    missing = [n for n, k in (("landlord", "landlord_name"), ("tenant", "tenant_name")) if c.val(k) is None]
    if missing:
        if not c.readable:
            return "NOT_DETERMINABLE", "The document could not be read, so the parties cannot be checked.", None
        return "FAIL", f"Not identified: {', '.join(missing)}.", None
    landlord, tenant = c.val("landlord_signed"), c.val("tenant_signed")
    src = c.src("landlord_signed", "tenant_signed")
    if landlord is False or tenant is False:
        who = " and ".join(n for n, v in (("landlord", landlord), ("tenant", tenant)) if v is False)
        return "FAIL", f"Signature block is blank for the {who}.", src
    if landlord is None or tenant is None:
        return "NOT_DETERMINABLE", "Both parties are named, but no signature block could be read.", src
    return "PASS", "Both parties are identified and both signature blocks are filled.", src


def r6_annual(c: Context) -> Verdict:
    monthly, annual = c.val("monthly_rent"), c.val("annual_rent")
    if monthly is None or annual is None:
        return "NOT_DETERMINABLE", "Monthly or annual rent could not be read.", c.src("annual_rent", "monthly_rent")
    if abs(annual - monthly * 12) < 0.01:
        return "PASS", f"Annual rent QAR {_fmt(annual)} equals 12 x QAR {_fmt(monthly)}.", c.src("annual_rent")
    return "FAIL", f"Annual rent QAR {_fmt(annual)} does not equal 12 x QAR {_fmt(monthly)} (= QAR {_fmt(monthly * 12)}).", c.src("annual_rent")


def r7_unit(c: Context) -> Verdict:
    if c.val("unit_ref") is None:
        return "NOT_DETERMINABLE", "The lease does not state which unit it covers.", None
    if c.unit is None:
        return "FAIL", f"Unit not found in owner records: {c.unit_note}.", c.src("unit_ref")
    if c.unit.status != "available":
        return "FAIL", f"{c.unit.unit_id} is marked '{c.unit.status}', not available.", c.src("unit_ref")
    return "PASS", f"{c.unit.unit_id} exists and is available.", c.src("unit_ref")


RULES: dict[str, Callable[[Context], Verdict]] = {
    "R1": r1_deposit, "R2": r2_escalation, "R3": r3_term_cap, "R4": r4_dates,
    "R5": r5_parties, "R6": r6_annual, "R7": r7_unit,
}


def load_ruleset(path: Path) -> dict:
    return json.loads(path.read_text())


def evaluate(ruleset: dict, ctx: Context) -> list[RuleResult]:
    results = []
    for rule in ruleset["rules"]:
        fn = RULES.get(rule["id"])
        if fn is None:
            outcome, reason, source = "NOT_DETERMINABLE", f"No implementation for rule {rule['id']}.", None
        else:
            outcome, reason, source = fn(ctx)
        results.append(RuleResult(rule_id=rule["id"], description=rule["description"],
                                  severity=rule["severity"], outcome=outcome, reason=reason, source=source))
    return results
