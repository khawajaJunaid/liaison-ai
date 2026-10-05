"""Record real pipeline runs for the animated walkthrough.

    python -m scripts.export_trace         # writes docs/traces.js

docs/how-it-works.html plays these back. Nothing is hand-written: every line a step "produces" comes
from running the actual agents on the sample leases and photos, so the animation cannot drift from
the code. Re-run this after changing a step.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.agents import DEFAULT_ISSUE_STEPS, DEFAULT_LEASE_STEPS, IssueAgent, LeaseAgent
from app.config import SEED_DIR
from app.ingest import load_document
from app.rules import load_ruleset
from app.units import UnitRegistry
from app.vision import StubVision
from scripts.make_samples import build_all

ROOT = Path(__file__).resolve().parent.parent
TONE = {"PASS": "ok", "FAIL": "bad", "NOT_DETERMINABLE": "warn", "high": "bad", "medium": "warn", "low": "info"}


def line(text: str, tone: str = "info") -> dict:
    return {"text": text, "tone": tone}


def money_or_text(value) -> str:
    return f"{value:,.0f}" if isinstance(value, float) and value.is_integer() else str(value)


def lease_gate(lease, units: UnitRegistry) -> dict:
    """The human-gate card, plus the unit's real status so the animation never shows a false one."""
    gate = _lease_gate(lease, units)
    unit = units.get(lease.unit_id) if lease.unit_id else None
    gate["unit"] = {"id": unit.unit_id, "status": unit.status} if unit else None
    return gate


def _lease_gate(lease, units: UnitRegistry) -> dict:
    unit = units.get(lease.unit_id) if lease.unit_id else None
    high = [f for f in lease.flags if f.severity == "high"]
    if unit is None:
        return {"tone": "bad", "title": "Accept is blocked",
                "lines": [line("No unit was matched, so there is nothing to link.", "bad"),
                          line("A person fixes the unit field, the rules re-run, then they can accept.")]}
    if unit.status != "available":
        return {"tone": "bad", "title": "Accept is blocked",
                "lines": [line(f"{unit.unit_id} is already '{unit.status}'.", "bad"),
                          line(f"{len(high)} high-severity flag(s) also need a decision."),
                          line("Override the unit, or reject the lease. Nothing changes until a person decides.")]}
    if high:
        return {"tone": "warn", "title": "Person must decide the flags first",
                "lines": [line(f"{len(high)} high-severity flag(s) are undecided."),
                          line("Confirm or dismiss each one, then accept.")]}
    return {"tone": "ok", "title": "Person accepts the lease",
            "lines": [line(f"{unit.unit_id} becomes occupied and is linked to this lease.", "ok"),
                      line("The decision is written to the audit trail.")]}


def lease_scenario(sid: str, title: str, blurb: str, path: Path, units: UnitRegistry, ruleset: dict) -> dict:
    agent = LeaseAgent(units, ruleset)
    doc = load_document(path)
    lease = agent.run(path, path.name)
    f = lease.fields
    found = [(n, x) for n, x in f.items() if x.value is not None and n != "rent_frequency"]

    if doc.lines:
        read = [line(f"{lease.page_count} page, {len(doc.lines)} text lines, each with a box on the page"),
                line("Text layer found, so no OCR is needed", "ok")]
    else:
        read = [line(f"{lease.page_count} page, but it is an image: no text layer", "warn")]
        read += [line(w, "bad") for w in lease.warnings]
    extract = [line(f"{n.replace('_', ' ')} = {money_or_text(x.value)[:46]}   ({x.confidence:.0%})",
                    "warn" if x.note and x.note.startswith("CONFLICT") else "info") for n, x in found]
    extract += [line("CONFLICT: " + x.note.removeprefix("CONFLICT: ")[:90], "warn")
                for _, x in found if x.note and x.note.startswith("CONFLICT")]
    if not found:
        extract = [line("Nothing to read. Fields stay empty instead of being guessed.", "warn")]
    ref = f["unit_ref"].value
    unit = units.get(lease.unit_id) if lease.unit_id else None
    if unit:
        match = [line(f"'{ref}' matches {lease.unit_id} ({lease.unit_match})", "ok"),
                 line(f"Owner records say {unit.unit_id} is {unit.status}", "ok" if unit.status == "available" else "bad")]
    else:
        match = [line(lease.unit_match or "no unit reference", "warn")]
    rules = [line(f"{r.rule_id}  {r.outcome}: {r.reason}", TONE[r.outcome]) for r in lease.rule_results]
    flags = ([line(f"[{x.severity}] {x.message}", TONE[x.severity]) for x in lease.flags]
             or [line("No flags", "ok")])

    outputs = dict(zip(DEFAULT_LEASE_STEPS, [read, extract, match, rules, flags]))
    return {
        "id": sid, "kind": "lease", "title": title, "blurb": blurb,
        "document": [ln.text for ln in doc.lines],
        "marks": {n: x.source.quote for n, x in found if x.source},
        "steps": [{"name": s, "output": outputs[s]} for s in DEFAULT_LEASE_STEPS],
        "gate": lease_gate(lease, units),
    }


def issue_scenario(sid: str, title: str, blurb: str, photos: list[Path], note: str, units: UnitRegistry) -> dict:
    unit = units.get("MC-B-1204")
    agent = IssueAgent(StubVision())
    issue = agent.run(unit, [(p.name, p.read_bytes()) for p in photos], "Sara", note, None)
    assess = []
    for a in issue.photos:
        assess.append(line(f"{a.filename}: {a.condition}  ({a.confidence:.0%}, {a.model})",
                           "bad" if a.condition == "damaged" else "warn" if a.condition in ("worn", "unknown") else "ok"))
        assess += [line(f"   damage: {d.type} ({d.severity})" + (f" on {d.affects}" if d.affects else ""), TONE[d.severity])
                   for d in a.damages]
        assess += [line(f"   equipment: {e.name}") for e in a.equipment]
        if a.needs_review:
            assess.append(line("   low confidence: needs a human look", "warn"))
    wo = issue.work_order
    order = [line(wo.title, "bad" if wo.priority == "high" else "info"), line(f"priority: {wo.priority}")]
    order += [line(t) for t in wo.description.splitlines()[2:6]]
    outputs = dict(zip(DEFAULT_ISSUE_STEPS, [assess, [line(issue.summary)], order]))
    return {
        "id": sid, "kind": "issue", "title": title, "blurb": blurb,
        "photos": [p.name for p in photos], "note": note,
        "steps": [{"name": s, "output": outputs[s]} for s in DEFAULT_ISSUE_STEPS],
        "gate": {"tone": "ok", "title": "Person accepts, edits or rejects the draft",
                 "lines": [line(f"Shows under {unit.label}, next to its lease.", "ok"),
                           line("Nothing is sent to a vendor until a person accepts.")]},
    }


def main() -> None:
    units = UnitRegistry(SEED_DIR / "units.json")
    ruleset = load_ruleset(SEED_DIR / "owner_ruleset.json")
    with tempfile.TemporaryDirectory() as tmp:
        s = build_all(Path(tmp))
        scenarios = [
            lease_scenario("good", "Clean lease", "Everything checks out. A person still accepts it.",
                           s["good"], units, ruleset),
            lease_scenario("defective", "Lease with problems",
                           "Seven rules fail, dates conflict, and the unit is already occupied.",
                           s["defective"], units, ruleset),
            lease_scenario("scanned", "Scanned lease, no OCR",
                           "An image with no text. It is flagged as unreadable instead of guessed.",
                           s["scanned"], units, ruleset),
            issue_scenario("leak", "Photo of a leaking AC", "Clear damage: a high-priority draft work order.",
                           [s["photos"] / "ac_unit_water_leak.jpg"], "dripping onto the floor", units),
            issue_scenario("unclear", "Photo that is not clear", "The agent says it cannot tell, and a person checks.",
                           [s["photos"] / "unlabelled_photo.jpg"], "", units),
        ]
    out = ROOT / "docs" / "traces.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("// Generated by scripts/export_trace.py from real runs. Do not edit by hand.\n"
                   "window.TRACES = " + json.dumps(scenarios, indent=1) + ";\n")
    print(f"wrote {out} ({len(scenarios)} scenarios)")


if __name__ == "__main__":
    main()
