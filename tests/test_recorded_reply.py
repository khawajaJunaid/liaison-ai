"""A real model reply, recorded from a live run, turned into a work order.

The photo was a wall-mounted air conditioner leaking down a wall onto a laminate floor. The reply below is
what claude-sonnet-5-5 returned for it (see the README). It pins that the app parses a real reply and that
the draft work order keeps the model's explanation of the likely cause, which a technician needs.
"""
import json

from app.domain import Issue, Unit
from app.issue_steps import DraftWorkOrder
from app.vision import parse_assessment

RECORDED = {
    "condition": "damaged",
    "damages": [
        {"type": "water leak / staining", "severity": "high",
         "description": "Brownish-yellow water stains run vertically down the wall from the underside of the air "
                        "conditioner to the baseboard. The streaks look like repeated or ongoing leakage, probably "
                        "from condensate overflow or a blocked or disconnected drain line.",
         "affects": "wall paint and plaster beneath the AC unit"},
        {"type": "standing water on floor", "severity": "high",
         "description": "A puddle of discolored (brownish) liquid sits on the laminate floor at the base of the wall. "
                        "The leak looks recent or active. It could cause swelling or warping of the laminate and the "
                        "baseboard, and it is a slip hazard.",
         "affects": "laminate flooring and baseboard"},
        {"type": "baseboard water exposure", "severity": "medium",
         "description": "The stain runs down to the baseboard, which is wet or exposed to moisture at the joint.",
         "affects": "baseboard / skirting"},
    ],
    "equipment": [
        {"name": "Wall-mounted split air conditioner (indoor unit)", "category": "HVAC",
         "condition": "Likely faulty. The unit looks intact externally, but water is leaking from it or its drain line."},
        {"name": "Fabric sofa (dark gray)", "category": "furniture", "condition": "good, only partially visible"},
    ],
    "confidence": 0.9,
}


def unit():
    return Unit(unit_id="MC-B-1204", label="Apartment 1204", type="2BR", area_sqm=118, status="occupied",
                building_id="MC-B", building_name="Tower B", property_id="PROP-MC", property_name="Marina Crest")


def work_order():
    assessment = parse_assessment(json.dumps(RECORDED), "photo.jpg", "claude-sonnet-5-5")
    issue = Issue(id="i", unit_id="MC-B-1204", reporter="Sara", photos=[assessment])
    return assessment, DraftWorkOrder.draft(unit(), issue, [assessment])


def test_a_real_reply_parses_and_is_trusted_at_high_confidence():
    assessment, _ = work_order()
    assert assessment.condition == "damaged" and assessment.confidence == 0.9
    assert len(assessment.damages) == 3 and not assessment.needs_review


def test_priority_comes_from_the_worst_finding():
    _, wo = work_order()
    assert wo.priority == "high" and "Apartment 1204" in wo.title


def test_the_work_order_keeps_the_models_explanation_of_the_cause():
    _, wo = work_order()
    assert "condensate overflow or a blocked or disconnected drain line" in wo.description
    assert "slip hazard" in wo.description
    assert wo.description.count("[photo.jpg]") == 3  # one line per finding, each citing its photo
