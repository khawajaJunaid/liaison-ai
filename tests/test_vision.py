"""Vision layer: the stub's contract, and how a real model's replies are parsed and distrusted."""
import json

import pytest

from app.agents import IssueAgent
from app.domain import Unit
from app.vision import StubVision, make_vision, parse_assessment


def unit():
    return Unit(unit_id="MC-B-1204", label="Apartment 1204", type="2BR", area_sqm=118, status="occupied",
                building_id="MC-B", building_name="Tower B", property_id="PROP-MC", property_name="Marina Crest")


@pytest.mark.parametrize("name,condition,damage,equipment", [
    ("ac_unit_water_leak.jpg", "damaged", "Water leak", "Split AC unit"),
    ("water_heater_rust_old.jpg", "damaged", "Corrosion", "Water heater"),
    ("kitchen_sink_new.jpg", "new", None, "Sink and faucet"),
    ("fridge.png", "good", None, "Refrigerator"),
    ("IMG_2231.jpg", "unknown", None, None),
])
def test_stub_reads_filenames(name, condition, damage, equipment):
    result = StubVision().assess(b"", name)
    assert result.condition == condition
    assert (damage in [d.type for d in result.damages]) if damage else not result.damages
    assert (equipment in [e.name for e in result.equipment]) if equipment else not result.equipment
    assert result.needs_review  # a stub result is never presented as verified


def test_stub_uses_the_reporters_note():
    result = StubVision().assess(b"", "IMG_1.jpg", note="the ceiling has a crack and a stain")
    assert {d.type for d in result.damages} == {"Crack", "Stain / mould"}


def test_work_order_takes_priority_from_the_worst_finding():
    agent = IssueAgent(StubVision())
    issue = agent.run(unit(), [("kitchen_sink_new.jpg", b""), ("ac_leak.jpg", b""), ("wall_peeling.jpg", b"")],
                      "Sara", "", None)
    wo = issue.work_order
    assert wo.priority == "high" and wo.title.startswith("Water leak")
    assert "Peeling finish (low)" in wo.description
    assert issue.summary.startswith("3 photo(s); overall condition damaged")


FENCED = """```json
{"condition": "worn", "damages": [{"type": "Stain", "severity": "low", "description": "ceiling stain", "affects": null}],
 "equipment": [{"name": "Split AC", "category": "HVAC", "condition": "worn"}], "confidence": 0.82}
```"""


def test_parse_assessment_accepts_fenced_json():
    result = parse_assessment(FENCED, "a.jpg", "m")
    assert result.condition == "worn" and result.damages[0].type == "Stain"
    assert result.equipment[0].name == "Split AC" and not result.needs_review


def test_low_confidence_model_answer_is_marked_for_review():
    raw = json.dumps({"condition": "good", "damages": [], "equipment": [], "confidence": 0.3})
    assert parse_assessment(raw, "a.jpg", "m").needs_review


@pytest.mark.parametrize("raw", [
    "I cannot assess this image.", "{not json", "[]", json.dumps({"condition": "excellent"}),
    json.dumps({"condition": "good", "damages": [{"type": "x", "severity": "catastrophic", "description": ""}]}),
])
def test_malformed_model_replies_degrade_to_unknown(raw):
    result = parse_assessment(raw, "a.jpg", "m")
    assert result.condition == "unknown" and result.needs_review and result.confidence == 0.0


def test_default_vision_is_the_key_free_stub(monkeypatch):
    monkeypatch.delenv("LEASE_AGENT_VISION", raising=False)
    assert make_vision().name == "stub-v1"


def test_same_finding_in_several_photos_is_listed_once():
    agent = IssueAgent(StubVision())
    issue = agent.run(unit(), [("IMG_1.jpg", b""), ("IMG_2.jpg", b"")], "Sara", "water leaking", None)
    assert issue.work_order.description.count("Water leak (high)") == 1
