"""The pipeline is editable: reorder, drop, add a step, and bad lists fail at startup."""
from pathlib import Path

import pytest

from app.agents import DEFAULT_ISSUE_STEPS, DEFAULT_LEASE_STEPS, IssueAgent, LeaseAgent
from app.config import Settings
from app.domain import Flag, Unit
from app.ingest import make_ocr
from app.main import create_app
from app.pipeline import _REGISTRY, BaseStep, Pipeline, PipelineError, Services, StepError
from app.rules import load_ruleset
from app.units import UnitRegistry
from app.vision import StubVision

SEED = Path(__file__).resolve().parent.parent / "seed"


@pytest.fixture(scope="module")
def registry():
    return UnitRegistry(SEED / "units.json")


@pytest.fixture(scope="module")
def ruleset():
    return load_ruleset(SEED / "owner_ruleset.json")


def lease_agent(registry, ruleset, steps=None):
    return LeaseAgent(registry, ruleset, steps=steps)


# ---- the default pipelines are listed ----------------------------------------

def test_api_lists_the_active_pipelines_and_available_steps(client):
    body = client.get("/api/pipelines").json()
    assert body["lease"]["steps"] == list(DEFAULT_LEASE_STEPS)
    assert body["issue"]["steps"] == list(DEFAULT_ISSUE_STEPS)
    validate = body["lease"]["available"]["validate_rules"]
    assert validate["needs"] == ["fields", "unit"] and validate["provides"] == ["rules"]
    assert validate["reruns_on_override"] is True
    assert "draft_work_order" in body["issue"]["available"]
    assert body["engines"]["vision"] == "stub-v1"


# ---- a bad list fails when the app is built, not on the first upload ---------

def test_unknown_step_name_lists_what_is_available(registry, ruleset):
    with pytest.raises(PipelineError, match=r"unknown lease step.*'nope'.*available:.*validate_rules"):
        lease_agent(registry, ruleset, steps=["read_pdf", "nope"])


def test_a_step_cannot_run_before_what_it_needs(registry, ruleset):
    with pytest.raises(PipelineError, match=r"'extract_fields' needs \['doc'\]"):
        lease_agent(registry, ruleset, steps=["extract_fields", "read_pdf"])
    with pytest.raises(PipelineError, match=r"'validate_rules' needs"):
        lease_agent(registry, ruleset, steps=["validate_rules"])


def test_a_step_with_a_missing_service_fails_at_startup():
    with pytest.raises(PipelineError, match="needs the 'units' service"):
        Pipeline("lease", ["match_unit"], Services())


def test_startup_rejects_a_bad_configured_pipeline(tmp_path):
    with pytest.raises(PipelineError):
        create_app(data_dir=tmp_path / "d", settings=Settings(lease_steps=("read_pdf", "bogus")))


# ---- dropping and reordering changes what the agent does ---------------------

def test_dropping_a_step_removes_its_output(registry, ruleset, samples):
    steps = ["read_pdf", "extract_fields", "match_unit", "validate_rules"]  # no self_check
    lease = lease_agent(registry, ruleset, steps).run(samples["defective"], "d.pdf")
    assert lease.rule_results and lease.flags == []
    assert any(t.startswith("Validated") for t in lease.trace)
    assert not any(t.startswith("Flagged") for t in lease.trace)


def test_an_extraction_only_agent_is_possible(registry, ruleset, samples):
    lease = lease_agent(registry, ruleset, ["read_pdf", "extract_fields"]).run(samples["good"], "g.pdf")
    assert lease.fields["monthly_rent"].value == 9500.0
    assert lease.rule_results == [] and lease.unit_id is None


def test_override_reruns_only_the_deterministic_steps(registry, ruleset, samples):
    agent = lease_agent(registry, ruleset)
    lease = agent.run(samples["good"], "g.pdf")
    agent.apply_field_decision(lease, "monthly_rent", "override", "9600")
    reads = [t for t in lease.trace if t.startswith("Read ")]
    extracts = [t for t in lease.trace if t.startswith("Extracted")]
    validations = [t for t in lease.trace if t.startswith("Validated")]
    assert len(reads) == len(extracts) == 1  # the PDF is not re-read and re-extracted
    assert len(validations) == 2


# ---- adding a task -----------------------------------------------------------

class FlagBigRent(BaseStep):
    """Example task: flag any rent over QAR 9,000 for the owner's attention."""

    needs = frozenset({"fields", "flags"})
    provides = frozenset({"big_rent_check"})
    rerun = True

    def run(self, ctx):
        rent = ctx.lease.fields["monthly_rent"]
        if rent.usable and rent.value > 9000:
            ctx.lease.flags.append(Flag(id="custom.big_rent", kind="custom", severity="low",
                                        message=f"Rent QAR {rent.value:,.0f} is above QAR 9,000.", fields=["monthly_rent"]))


def test_a_new_registered_step_runs_in_place_and_reruns_on_override(registry, ruleset, samples, monkeypatch):
    monkeypatch.setitem(_REGISTRY["lease"], "flag_big_rent", FlagBigRent)
    FlagBigRent.name = "flag_big_rent"
    agent = lease_agent(registry, ruleset, [*DEFAULT_LEASE_STEPS, "flag_big_rent"])
    lease = agent.run(samples["good"], "g.pdf")
    assert "custom.big_rent" in {f.id for f in lease.flags}

    agent.apply_field_decision(lease, "monthly_rent", "override", "8000")  # rerun drops it again
    assert "custom.big_rent" not in {f.id for f in lease.flags}
    agent.apply_field_decision(lease, "monthly_rent", "override", "9500")
    assert "custom.big_rent" in {f.id for f in lease.flags}


class Boom(BaseStep):
    """Always fails."""

    def run(self, ctx):
        raise RuntimeError("kaboom")


def test_a_failing_step_is_named_in_the_error(registry, ruleset, samples, monkeypatch):
    monkeypatch.setitem(_REGISTRY["lease"], "boom", Boom)
    Boom.name = "boom"
    agent = lease_agent(registry, ruleset, [*DEFAULT_LEASE_STEPS, "boom"])
    with pytest.raises(StepError, match=r"lease step 'boom' failed: kaboom"):
        agent.run(samples["good"], "g.pdf")


def test_a_failing_step_is_a_clear_422_at_the_api(make_client, samples, monkeypatch):
    monkeypatch.setitem(_REGISTRY["lease"], "boom", Boom)
    Boom.name = "boom"
    client = make_client(settings=Settings(lease_steps=(*DEFAULT_LEASE_STEPS, "boom")))
    with open(samples["good"], "rb") as fh:
        res = client.post("/api/leases", files={"file": ("g.pdf", fh, "application/pdf")})
    assert res.status_code == 422 and "boom" in res.json()["detail"]


# ---- the issue agent ---------------------------------------------------------

def unit():
    return Unit(unit_id="MC-B-1204", label="Apartment 1204", type="2BR", area_sqm=118, status="occupied",
                building_id="MC-B", building_name="Tower B", property_id="PROP-MC", property_name="Marina Crest")


def test_assessment_only_issue_agent_drafts_no_work_order():
    agent = IssueAgent(StubVision(), steps=["assess_photos", "summarise"])
    issue = agent.run(unit(), [("ac_leak.jpg", b"")], "Sara", "", None)
    assert issue.work_order is None and issue.photos[0].damages and issue.summary


def test_issue_pipeline_without_a_work_order_works_end_to_end(make_client, samples):
    client = make_client(settings=Settings(issue_steps=("assess_photos", "summarise")))
    files = [("photos", ("ac_unit_water_leak.jpg", open(samples["photos"] / "ac_unit_water_leak.jpg", "rb"), "image/jpeg"))]
    issue = client.post("/api/issues", data={"unit_id": "MC-B-1204"}, files=files).json()
    assert issue["work_order"] is None
    res = client.patch(f"/api/issues/{issue['id']}/work-order", json={"action": "accept"})
    assert res.status_code == 404


# ---- configuration -----------------------------------------------------------

def test_settings_default_to_the_key_free_setup():
    s = Settings.from_env({})
    assert (s.ocr, s.vision, s.lease_steps, s.issue_steps) == ("none", "stub", None, None)


def test_settings_read_engines_and_step_lists_from_the_environment():
    s = Settings.from_env({"LEASE_AGENT_OCR": " Paddle ", "LEASE_AGENT_VISION": "ANTHROPIC",
                           "LEASE_AGENT_LEASE_STEPS": "read_pdf, extract_fields ,,match_unit",
                           "LEASE_AGENT_ISSUE_STEPS": "assess_photos"})
    assert s.ocr == "paddle" and s.vision == "anthropic"
    assert s.lease_steps == ("read_pdf", "extract_fields", "match_unit")
    assert s.issue_steps == ("assess_photos",)


def test_ocr_engine_names():
    assert make_ocr("") is None and make_ocr("none") is None
    with pytest.raises(ValueError, match="unknown OCR engine 'tesseract'"):
        make_ocr("tesseract")


def test_configured_step_list_is_what_the_api_reports(make_client):
    client = make_client(settings=Settings(issue_steps=("assess_photos", "summarise")))
    body = client.get("/api/pipelines").json()
    assert body["issue"]["steps"] == ["assess_photos", "summarise"]
    assert body["issue"]["default"] == list(DEFAULT_ISSUE_STEPS)
