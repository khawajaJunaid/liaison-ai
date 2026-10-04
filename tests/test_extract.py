"""Reading leases: field extraction, provenance, OCR routing, unit matching."""
from pathlib import Path

import pytest

from app.agents import LeaseAgent
from app.extract import extract_fields, parse_date
from app.ingest import load_document
from app.rules import load_ruleset
from app.units import UnitRegistry
from scripts.make_samples import GOOD

SEED = Path(__file__).resolve().parent.parent / "seed"


@pytest.fixture(scope="module")
def registry():
    return UnitRegistry(SEED / "units.json")


def fields_of(samples, key):
    return extract_fields(load_document(samples[key]))


def test_good_lease_fields(samples):
    f = fields_of(samples, "good")
    assert f["landlord_name"].value == "Marina Crest Holdings W.L.L."
    assert f["tenant_name"].value == "Sara Al-Mansoori"
    assert f["unit_ref"].value == "MC-B-1204"
    assert (f["commencement_date"].value, f["expiry_date"].value) == ("2026-03-01", "2027-02-28")
    assert f["term_months"].value == 12
    assert (f["monthly_rent"].value, f["annual_rent"].value, f["deposit_amount"].value) == (9500.0, 114000.0, 9500.0)
    assert f["escalation_defined"].value is True
    assert f["landlord_signed"].value is True and f["tenant_signed"].value is True


def test_every_value_points_back_at_the_page(samples):
    for name, fld in fields_of(samples, "good").items():
        if fld.value is not None:
            assert fld.source is not None, name
            assert fld.source.page == 1 and fld.source.quote, name
            x0, y0, x1, y1 = fld.source.bbox
            assert x1 > x0 and y1 > y0, name


def test_percentage_in_escalation_clause_is_not_read_as_rent(samples):
    f = fields_of(samples, "good")
    assert f["monthly_rent"].value == 9500.0 and not f["monthly_rent"].note


def test_defective_lease_conflicting_dates_are_recorded(samples):
    f = fields_of(samples, "defective")
    assert f["commencement_date"].note.startswith("CONFLICT")
    assert "2026-04-15" in f["commencement_date"].note
    assert f["commencement_date"].confidence < 0.7


def test_defective_lease_values(samples):
    f = fields_of(samples, "defective")
    assert f["term_months"].value == 48
    assert f["tenant_signed"].value is False and f["landlord_signed"].value is True
    assert f["escalation_defined"].value is False


@pytest.mark.parametrize("text,iso", [
    ("1 March 2026", "2026-03-01"), ("1st March 2026", "2026-03-01"), ("March 1, 2026", "2026-03-01"),
    ("2026-03-01", "2026-03-01"), ("01/03/2026", "2026-03-01"), ("28 Feb 2027", "2027-02-28"),
])
def test_parse_date_formats(text, iso):
    assert parse_date(text).isoformat() == iso


@pytest.mark.parametrize("text", ["31 February 2026", "not a date", "13/13/2026"])
def test_parse_date_rejects_nonsense(text):
    assert parse_date(text) is None


def test_scanned_lease_without_ocr_is_flagged_not_guessed(samples, registry):
    agent = LeaseAgent(registry, load_ruleset(SEED / "owner_ruleset.json"))
    lease = agent.run(samples["scanned"], "scan.pdf")
    assert lease.warnings and "no OCR engine" in lease.warnings[0]
    assert any(f.kind == "scan" for f in lease.flags)
    assert all(f.value is None for f in lease.fields.values())
    by_rule = {r.rule_id: r.outcome for r in lease.rule_results}
    assert by_rule["R2"] == by_rule["R5"] == "NOT_DETERMINABLE"  # absence of text proves nothing


class FakeOcr:
    """Stands in for PaddleOCR-VL: returns the good lease's lines as pixel boxes at 200 dpi."""

    name = "fake"

    def read(self, png):
        return [(text, (100, 80 + i * 60, 1400, 120 + i * 60)) for i, (_, text) in enumerate(GOOD)]


def test_scanned_lease_goes_through_ocr_engine(samples, registry):
    agent = LeaseAgent(registry, load_ruleset(SEED / "owner_ruleset.json"), ocr=FakeOcr())
    lease = agent.run(samples["scanned"], "scan.pdf")
    assert lease.ocr_pages == [1] and not lease.warnings
    assert lease.fields["monthly_rent"].value == 9500.0
    assert lease.unit_id == "MC-B-1204"
    assert all(r.outcome == "PASS" for r in lease.rule_results)
    x0, y0, x1, y1 = lease.fields["monthly_rent"].source.bbox  # pixel boxes scaled to PDF points
    assert x1 < 595 and y1 < 842


@pytest.mark.parametrize("ref,unit_id,how", [
    ("MC-B-1204", "MC-B-1204", "exact unit id"),
    ("Apartment 1204, Tower B", "MC-B-1204", "apartment number and tower"),
    ("apartment 0902", "MC-B-0902", "apartment number only (tower not stated)"),
    ("Parking Bay A-12", "MC-A-0301", "parking bay"),
])
def test_unit_matching(registry, ref, unit_id, how):
    unit, method = registry.resolve(ref)
    assert unit.unit_id == unit_id and method == how


@pytest.mark.parametrize("ref", [None, "", "Apartment 9999, Tower B", "MC-Z-0001", "somewhere nice"])
def test_unmatched_units_return_none_not_a_guess(registry, ref):
    unit, why = registry.resolve(ref)
    assert unit is None and why


def test_status_overlay_overrides_seed(registry):
    unit, _ = registry.resolve("MC-B-1204", {"MC-B-1204": ("occupied", "lease_x")})
    assert unit.status == "occupied" and unit.lease_id == "lease_x"
