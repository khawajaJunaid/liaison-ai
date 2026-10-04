"""Vision layer: the stub's contract, and how a real model's replies are parsed and distrusted."""
import json

import pytest

from app.agents import IssueAgent
from app.domain import Unit
from app.vision import (PROMPT, AnthropicVision, OpenAIVision, StubVision, VisionUnavailable, make_vision,
                        parse_assessment)


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


# ---- real providers, driven by fake clients: no network, no keys -------------------------------

GOOD_REPLY = json.dumps({
    "condition": "damaged", "confidence": 0.9,
    "damages": [{"type": "Water leak", "severity": "high", "description": "dripping", "affects": "Split AC unit"}],
    "equipment": [{"name": "Split AC unit", "category": "HVAC", "condition": "damaged"}]})


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeAnthropicClient:
    def __init__(self, text=GOOD_REPLY):
        self.calls, self.text = [], text
        self.messages = _Obj(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return _Obj(content=[_Obj(text=self.text)])


class FakeOpenAIClient:
    def __init__(self, content=GOOD_REPLY):
        self.calls, self.content = [], content
        self.chat = _Obj(completions=_Obj(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return _Obj(choices=[_Obj(message=_Obj(content=self.content))])


def test_anthropic_request_shape_and_parsing(monkeypatch):
    monkeypatch.delenv("VISION_MODEL", raising=False)
    client = FakeAnthropicClient()
    result = AnthropicVision(client=client).assess(b"\x01\x02", "leak.png", "dripping")
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["system"] == PROMPT
    image, text = call["messages"][0]["content"]
    assert image["source"]["type"] == "base64" and image["source"]["media_type"] == "image/png"
    assert image["source"]["data"] == "AQI=" and "dripping" in text["text"]
    assert result.condition == "damaged" and result.damages[0].affects == "Split AC unit"
    assert result.model == "claude-sonnet-5-5" and not result.needs_review


def test_openai_request_shape_and_parsing(monkeypatch):
    monkeypatch.delenv("VISION_MODEL", raising=False)
    client = FakeOpenAIClient()
    result = OpenAIVision(client=client).assess(b"\x01\x02", "leak.jpg", "dripping")
    call = client.calls[0]
    assert call["model"] == "gpt-4o" and call["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in call and "max_completion_tokens" not in call  # name differs by model family
    system, user = call["messages"]
    assert system == {"role": "system", "content": PROMPT}
    text, image = user["content"]
    assert "dripping" in text["text"] and image["image_url"]["url"] == "data:image/jpeg;base64,AQI="
    assert result.condition == "damaged" and result.model == "gpt-4o"


def test_vision_model_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("VISION_MODEL", "my-local-vlm")
    assert OpenAIVision(client=FakeOpenAIClient()).model == "my-local-vlm"
    assert AnthropicVision(client=FakeAnthropicClient()).model == "my-local-vlm"


@pytest.mark.parametrize("empty", [None, "", "not json at all"])
def test_openai_empty_or_garbled_reply_degrades_to_unknown(empty):
    result = OpenAIVision(client=FakeOpenAIClient(content=empty)).assess(b"", "a.jpg")
    assert result.condition == "unknown" and result.needs_review and result.confidence == 0.0


def test_missing_keys_fail_at_startup_with_a_clear_message(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(VisionUnavailable, match="ANTHROPIC_API_KEY"):
        make_vision("anthropic")
    with pytest.raises(VisionUnavailable, match="OPENAI_API_KEY"):
        make_vision("openai")


def test_an_unknown_provider_is_an_error_not_a_silent_stub():
    with pytest.raises(ValueError, match="unknown vision provider 'antropic'"):
        make_vision("antropic")
    assert make_vision("  STUB ").name == "stub-v1"
    assert make_vision("").name == "stub-v1"


class FailingVision:
    name = "failing"

    def assess(self, image, filename, note=""):
        raise RuntimeError("401 invalid api key")


def test_a_provider_failure_is_a_502_naming_the_step(make_client, samples):
    client = make_client(vision=FailingVision())
    files = [("photos", ("a.jpg", open(samples["photos"] / "ac_unit_water_leak.jpg", "rb"), "image/jpeg"))]
    res = client.post("/api/issues", data={"unit_id": "MC-B-1204"}, files=files)
    assert res.status_code == 502
    assert "assess_photos" in res.json()["detail"] and "401" in res.json()["detail"]
    assert client.get("/api/units/MC-B-1204").json()["issues"] == []  # nothing half-saved
