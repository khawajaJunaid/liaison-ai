"""The issue agent's steps. Order and membership are set in agents.DEFAULT_ISSUE_STEPS.

assess_photos -> summarise -> draft_work_order

Each is optional after the first: drop draft_work_order for an assessment-only agent, or add a
step of your own (triage, vendor routing, an owner-versus-tenant cost check) between them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .domain import Damage, Issue, PhotoAssessment, Unit, WorkOrder, new_id
from .pipeline import BaseStep, register

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
CONDITION_RANK = {"unknown": 0, "new": 1, "good": 2, "worn": 3, "damaged": 4}


@dataclass
class IssueCtx:
    unit: Unit
    photos: list[tuple[str, bytes]]
    reporter: str = ""
    note: str = ""
    lease_id: str | None = None
    issue: Issue | None = None
    assessments: list[PhotoAssessment] = field(default_factory=list)


@register("issue", "assess_photos")
class AssessPhotos(BaseStep):
    """Assess each photo with the vision model: condition, damage, equipment."""

    provides = frozenset({"assessments"})
    uses = ("vision",)

    def run(self, ctx: IssueCtx) -> None:
        ctx.assessments = [self.services.vision.assess(data, name, ctx.note) for name, data in ctx.photos]
        ctx.issue = Issue(id=new_id("issue"), unit_id=ctx.unit.unit_id, lease_id=ctx.lease_id,
                          reporter=ctx.reporter, note=ctx.note, photos=ctx.assessments)


@register("issue", "summarise")
class Summarise(BaseStep):
    """One-line summary: photo count, worst condition, number of findings."""

    needs = frozenset({"assessments"})
    provides = frozenset({"summary"})

    def run(self, ctx: IssueCtx) -> None:
        worst = max((a.condition for a in ctx.assessments), key=CONDITION_RANK.get, default="unknown")
        ctx.issue.summary = (f"{len(ctx.assessments)} photo(s); overall condition {worst}; "
                             f"{sum(len(a.damages) for a in ctx.assessments)} damage finding(s)")


@register("issue", "draft_work_order")
class DraftWorkOrder(BaseStep):
    """Draft a work order from the findings. It only drafts: a person accepts it."""

    needs = frozenset({"assessments"})
    provides = frozenset({"work_order"})

    def run(self, ctx: IssueCtx) -> None:
        ctx.issue.work_order = self.draft(ctx.unit, ctx.issue, ctx.assessments)

    @staticmethod
    def draft(unit: Unit, issue: Issue, assessments: list[PhotoAssessment]) -> WorkOrder:
        findings: list[tuple[Damage, str]] = []
        seen: set[tuple[str, str | None]] = set()
        for a in assessments:  # one line per distinct finding, even if several photos show it
            for d in a.damages:
                if (d.type, d.affects) not in seen:
                    seen.add((d.type, d.affects))
                    findings.append((d, a.filename))
        findings.sort(key=lambda p: -SEVERITY_RANK[p[0].severity])
        equipment = list(dict.fromkeys(e.name for a in assessments for e in a.equipment))
        if findings:
            top: Damage = findings[0][0]
            title = f"{top.type}{': ' + top.affects if top.affects else ''} - {unit.label}"
            priority = top.severity
        else:
            title, priority = f"Inspect reported issue - {unit.label}", "low"
        lines = [f"Unit: {unit.label} ({unit.unit_id}), {unit.building_name}, {unit.property_name}.",
                 f"Reported by: {issue.reporter or 'unknown'}."]
        if issue.note:
            lines.append(f"Reporter note: {issue.note}")
        lines += [f"- {d.type} ({d.severity}){' on ' + d.affects if d.affects else ''} [{fn}]" for d, fn in findings] \
            or ["- No damage identified automatically; an inspector should look at the photos."]
        if equipment:
            lines.append("Equipment in photos: " + ", ".join(equipment) + ".")
        review = sum(a.needs_review for a in assessments)
        if review:
            lines.append(f"{review} of {len(assessments)} photo assessment(s) are low confidence and need a human check.")
        return WorkOrder(id=new_id("wo"), title=title, description="\n".join(lines), unit_id=unit.unit_id,
                         priority=priority, equipment=equipment)
