"""Rule engine: one table per rule, covering PASS, FAIL and NOT_DETERMINABLE."""
from datetime import date

import pytest

from app.domain import ExtractedField, Unit
from app.rules import Context, evaluate, term_matches

RULESET = {"rules": [
    {"id": f"R{i}", "description": f"rule {i}", "check": "", "severity": "high"} for i in range(1, 8)
]}


def unit(status="available"):
    return Unit(unit_id="MC-B-1204", label="Apartment 1204", type="2BR", area_sqm=118, status=status,
                building_id="MC-B", building_name="Tower B", property_id="PROP-MC", property_name="Marina Crest")


def run(rule_id, unit_=None, readable=True, rejected=(), **values):
    fields = {k: ExtractedField(name=k, value=v, decision="rejected" if k in rejected else "pending")
              for k, v in values.items()}
    ctx = Context(fields, unit_, "test", readable)
    return next(r for r in evaluate(RULESET, ctx) if r.rule_id == rule_id)


@pytest.mark.parametrize("deposit,rent,outcome", [
    (9500, 9500, "PASS"), (12000, 9500, "PASS"), (6000, 12000, "FAIL"),
    (None, 9500, "NOT_DETERMINABLE"), (9500, None, "NOT_DETERMINABLE"),
])
def test_r1_deposit(deposit, rent, outcome):
    assert run("R1", deposit_amount=deposit, monthly_rent=rent).outcome == outcome


@pytest.mark.parametrize("clause,defined,readable,outcome", [
    ("5% a year", True, True, "PASS"),
    ("as mutually agreed", False, True, "FAIL"),
    (None, None, True, "FAIL"),
    (None, None, False, "NOT_DETERMINABLE"),
])
def test_r2_escalation(clause, defined, readable, outcome):
    assert run("R2", readable=readable, escalation_clause=clause, escalation_defined=defined).outcome == outcome


@pytest.mark.parametrize("term,outcome", [(12, "PASS"), (36, "PASS"), (37, "FAIL"), (48, "FAIL"), (None, "NOT_DETERMINABLE")])
def test_r3_term_cap(term, outcome):
    assert run("R3", term_months=term).outcome == outcome


@pytest.mark.parametrize("start,expiry,term,outcome", [
    ("2026-03-01", "2027-02-28", 12, "PASS"),      # inclusive last day
    ("2026-03-01", "2027-03-01", 12, "PASS"),      # exclusive end date
    ("2026-04-01", "2029-03-31", 48, "FAIL"),      # term disagrees with dates
    ("2026-03-01", "2026-03-01", 1, "FAIL"),       # expiry not after start
    ("2026-03-01", "2026-01-01", 12, "FAIL"),
    ("2026-03-01", "2027-02-28", None, "NOT_DETERMINABLE"),
    (None, "2027-02-28", 12, "NOT_DETERMINABLE"),
])
def test_r4_dates(start, expiry, term, outcome):
    assert run("R4", commencement_date=start, expiry_date=expiry, term_months=term).outcome == outcome


def test_r4_reason_names_the_real_span():
    reason = run("R4", commencement_date="2026-04-01", expiry_date="2029-03-31", term_months=48).reason
    assert "48" in reason and "36" in reason


def test_term_matches_handles_awkward_day_of_month():
    ok, implied = term_matches(date(2026, 1, 31), date(2027, 1, 30), 12)  # last day is the 30th
    assert ok and implied == 12
    ok, implied = term_matches(date(2026, 1, 31), date(2026, 12, 30), 12)  # really 11 months
    assert not ok and implied == 11


@pytest.mark.parametrize("kw,outcome", [
    (dict(landlord_name="A", tenant_name="B", landlord_signed=True, tenant_signed=True), "PASS"),
    (dict(landlord_name="A", tenant_name="B", landlord_signed=True, tenant_signed=False), "FAIL"),
    (dict(landlord_name="A", tenant_name=None, landlord_signed=True, tenant_signed=True), "FAIL"),
    (dict(landlord_name="A", tenant_name="B", landlord_signed=None, tenant_signed=None), "NOT_DETERMINABLE"),
])
def test_r5_parties(kw, outcome):
    assert run("R5", **kw).outcome == outcome


def test_r5_unreadable_document_is_not_a_failure():
    assert run("R5", readable=False, landlord_name=None, tenant_name=None).outcome == "NOT_DETERMINABLE"


@pytest.mark.parametrize("monthly,annual,outcome", [
    (9500, 114000, "PASS"), (12000, 140000, "FAIL"), (None, 114000, "NOT_DETERMINABLE"), (9500, None, "NOT_DETERMINABLE"),
])
def test_r6_annual(monthly, annual, outcome):
    assert run("R6", monthly_rent=monthly, annual_rent=annual).outcome == outcome


@pytest.mark.parametrize("unit_,ref,outcome", [
    (unit(), "MC-B-1204", "PASS"),
    (unit("occupied"), "MC-B-1204", "FAIL"),
    (None, "MC-Z-9999", "FAIL"),
    (None, None, "NOT_DETERMINABLE"),
])
def test_r7_unit(unit_, ref, outcome):
    assert run("R7", unit_, unit_ref=ref).outcome == outcome


def test_rejected_field_counts_as_missing():
    assert run("R1", rejected=("deposit_amount",), deposit_amount=9500, monthly_rent=9500).outcome == "NOT_DETERMINABLE"


def test_unknown_rule_id_is_not_silently_passed():
    ctx = Context({}, None, "")
    [result] = evaluate({"rules": [{"id": "R99", "description": "new", "check": "", "severity": "low"}]}, ctx)
    assert result.outcome == "NOT_DETERMINABLE"
