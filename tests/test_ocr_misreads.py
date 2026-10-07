"""Misreads the real OCR produced on the phone-photo sample, pinned so they stay fixed.

Found by running samples/lease_scanned_photo.pdf through PP-OCR: "9,500" came back as "9.500" (read as 9.5),
"Security" as "Securty" (deposit not found), and "5. RENT" as "5.RENT" (heading text leaked into the clause).
"""
from pathlib import Path

import pytest

from app.agents import LeaseAgent
from app.extract import extract_fields
from app.ingest import Document, Line
from app.rules import load_ruleset
from app.units import UnitRegistry
from scripts.make_samples import GOOD

SEED = Path(__file__).resolve().parent.parent / "seed"


def doc_of(*texts):
    lines = [Line(t, 1, (50.0, 50.0 + i * 20, 400.0, 66.0 + i * 20)) for i, t in enumerate(texts)]
    return Document(lines=lines, page_count=1)


@pytest.mark.parametrize("written,value", [
    ("QAR 9,500", 9500.0), ("QAR 9,500.50", 9500.5), ("QAR 12.50", 12.5), ("QAR 1,234,567", 1234567.0),
    ("QAR 114,000.", 114000.0), ("QAR 9.5", 9.5), ("QAR 12000", 12000.0),
])
def test_ordinary_money_formats_are_unchanged(written, value):
    f = extract_fields(doc_of(f"Monthly Rent: {written}, payable monthly."))["monthly_rent"]
    assert f.value == value and f.note is None and f.confidence == 0.9


def test_a_period_for_a_thousands_comma_is_read_as_thousands_and_flagged_for_a_person():
    f = extract_fields(doc_of("Monthly Rent: QAR 9.500, payable monthly in advance."))["monthly_rent"]
    assert f.value == 9500.0  # not 9.5
    assert f.confidence <= 0.6
    assert f.note.startswith("CHECK") and "9.500" in f.note and "Confirm against the page" in f.note


def test_a_period_thousands_amount_with_decimals():
    assert extract_fields(doc_of("Annual Rent: QAR 114.000,50"))["annual_rent"].value == 114000.5


@pytest.mark.parametrize("line", ["Securty Deposit: QAR 9.500, held for the term.", "Deposit: QAR 9,500.",
                                  "Security Deposit: QAR 9,500, held."])
def test_deposit_label_tolerates_an_ocr_typo_and_a_bare_label(line):
    assert extract_fields(doc_of(line))["deposit_amount"].value == 9500.0


def test_a_sentence_mentioning_deposit_is_not_read_as_the_deposit():
    f = extract_fields(doc_of("The deposit shall be refunded within 30 days; rent is QAR 9,500."))
    assert f["deposit_amount"].value is None


def test_heading_without_a_space_does_not_leak_into_the_clause():
    f = extract_fields(doc_of(
        "5.RENT ESCALATION", "At each anniversary the Monthly Rent increases", "by 5% over the preceding year.",
        "6.RENEWAL", "The Tenant may renew for a further term.", "7.TERMINATION",
        "Either party may terminate early on 60 days written notice.", "8.SIGNATURES"))
    assert f["escalation_clause"].value.startswith("At each anniversary")
    assert "5.RENT" not in f["escalation_clause"].value and "6.RENEWAL" not in f["escalation_clause"].value
    assert f["renewal_terms"].value.startswith("The Tenant may renew")
    assert "8.SIGNATURES" not in f["termination_terms"].value


def test_a_body_line_starting_with_a_number_is_not_a_heading():
    f = extract_fields(doc_of("5. RENT ESCALATION", "12.5% annual increase applies to the rent.", "applied each March."))
    assert f["escalation_clause"].value.startswith("12.5% annual increase")


class NoisyOcr:
    """Returns the good lease as the phone-photo OCR read it: the three misreads above."""

    name = "noisy-fake"

    def read(self, png):
        noisy = {"Monthly Rent: QAR 9,500, payable monthly in advance.": "Monthly Rent: QAR 9.500, payable monthly in advance.",
                 "Security Deposit: QAR 9,500, held for the term of the lease.": "Securty Deposit: QAR 9.500, held for the term of the lease.",
                 "5. RENT ESCALATION": "5.RENT ESCALATION", "6. RENEWAL": "6.RENEWAL", "8. SIGNATURES": "8.SIGNATURES"}
        return [(noisy.get(text, text), (100, 80 + i * 60, 1400, 120 + i * 60)) for i, (_, text) in enumerate(GOOD)]


def test_noisy_ocr_gives_the_right_values_and_asks_a_person_to_confirm_the_risky_ones(samples):
    agent = LeaseAgent(UnitRegistry(SEED / "units.json"), load_ruleset(SEED / "owner_ruleset.json"), ocr=NoisyOcr())
    lease = agent.run(samples["scanned"], "photo.pdf")
    f = lease.fields
    assert (f["monthly_rent"].value, f["deposit_amount"].value, f["annual_rent"].value) == (9500.0, 9500.0, 114000.0)
    assert not f["escalation_clause"].value.startswith("5.")
    assert all(r.outcome == "PASS" for r in lease.rule_results)  # the right values, so the rules pass
    check = {x.id for x in lease.flags if x.kind == "check"}
    assert check == {"check.monthly_rent", "check.deposit_amount"}  # but a person still confirms the guessed separators

    agent.apply_field_decision(lease, "monthly_rent", "accept", states=None)
    assert "check.monthly_rent" not in {x.id for x in lease.flags}  # accepting the field clears its flag
