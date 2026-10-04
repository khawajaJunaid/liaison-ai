"""HTTP API and the single-page UI.

The unit is the join between the two features: GET /api/units/{id} returns the unit's lease,
any leases awaiting review, and every issue (with its draft work order) raised against it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .agents import DEFAULT_ISSUE_STEPS, DEFAULT_LEASE_STEPS, IssueAgent, LeaseAgent
from .config import DEFAULT_DATA_DIR, SEED_DIR, Settings
from .domain import new_id
from .ingest import OcrEngine, make_ocr
from .pipeline import available_steps
from .rules import load_ruleset
from .store import Store
from .units import UnitRegistry
from .vision import VisionModel, make_vision

STATIC = Path(__file__).parent / "static"
MAX_PDF = 20 * 1024 * 1024
MAX_PHOTO = 10 * 1024 * 1024
MAX_PHOTOS = 10
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


class FieldDecision(BaseModel):
    action: Literal["accept", "reject", "override"]
    value: str | float | int | bool | None = None


class FlagDecision(BaseModel):
    action: Literal["accept", "reject"]


class WorkOrderDecision(BaseModel):
    action: Literal["accept", "reject"]
    title: str | None = None
    description: str | None = None


def _safe_name(name: str, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name or "photo").name).strip("._") or "photo"
    return f"{index}_{base}"


def create_app(data_dir: Path | None = None, vision: VisionModel | None = None,
               ocr: OcrEngine | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Engines and step lists come from `settings` (environment by default);
    `vision` and `ocr` override the engines directly, which is how tests inject fakes."""
    settings = settings or Settings.from_env()
    app = FastAPI(title="liAIson")
    store = Store(data_dir or DEFAULT_DATA_DIR)
    units = UnitRegistry(SEED_DIR / "units.json")
    ruleset = load_ruleset(SEED_DIR / "owner_ruleset.json")
    lease_agent = LeaseAgent(units, ruleset, ocr if ocr is not None else make_ocr(settings.ocr),
                             steps=settings.lease_steps)
    issue_agent = IssueAgent(vision or make_vision(settings.vision), steps=settings.issue_steps)
    (store.dir / "leases").mkdir(exist_ok=True)

    def lease_or_404(lease_id: str):
        lease = store.get_lease(lease_id)
        if lease is None:
            raise HTTPException(404, "lease not found")
        return lease

    def issue_or_404(issue_id: str):
        issue = store.get_issue(issue_id)
        if issue is None:
            raise HTTPException(404, "issue not found")
        return issue

    def editable(lease):
        if lease.status != "pending_review":
            raise HTTPException(409, f"lease is already {lease.status}")

    def lease_summary(lease):
        f = lease.fields
        return {"id": lease.id, "filename": lease.filename, "status": lease.status,
                "tenant": f["tenant_name"].value, "monthly_rent": f["monthly_rent"].value,
                "pending_flags": sum(x.decision == "pending" for x in lease.flags)}

    # ---- units ----------------------------------------------------------

    @app.get("/api/units")
    def list_units():
        states = store.unit_states()
        leases = {l.id: l for l in store.list_leases()}
        out = []
        for u in units.all(states):
            lease = leases.get(u.lease_id) if u.lease_id else None
            issues = store.issues_for_unit(u.unit_id)
            open_orders = sum(1 for i in issues if i.work_order and i.work_order.status != "rejected")
            out.append({**u.model_dump(), "tenant": lease.fields["tenant_name"].value if lease else None,
                        "open_issues": open_orders})
        return out

    @app.get("/api/units/{unit_id}")
    def unit_view(unit_id: str):
        states = store.unit_states()
        unit = units.get(unit_id, states)
        if unit is None:
            raise HTTPException(404, "unit not found")
        leases = store.list_leases()
        active = next((l for l in leases if l.id == unit.lease_id), None)
        pending = [lease_summary(l) for l in leases if l.unit_id == unit_id and l.status == "pending_review"]
        issues = store.issues_for_unit(unit_id)
        ids = [i.id for i in issues] + [l.id for l in leases if l.unit_id == unit_id]
        return {"unit": unit, "lease": active, "pending_leases": pending, "issues": issues,
                "audit": store.audit_for(*ids) if ids else []}

    @app.get("/api/rules")
    def rules():
        return ruleset

    @app.get("/api/pipelines")
    def pipelines():
        """The steps each agent runs, in order, and every step that could be added."""
        return {
            "lease": {"steps": lease_agent.pipeline.names, "default": list(DEFAULT_LEASE_STEPS),
                      "available": available_steps("lease")},
            "issue": {"steps": issue_agent.pipeline.names, "default": list(DEFAULT_ISSUE_STEPS),
                      "available": available_steps("issue")},
            "engines": {"ocr": settings.ocr, "vision": issue_agent.vision.name},
        }

    # ---- leases ---------------------------------------------------------

    @app.post("/api/leases")
    def upload_lease(file: UploadFile = File(...)):
        data = file.file.read(MAX_PDF + 1)
        if len(data) > MAX_PDF:
            raise HTTPException(413, "PDF is larger than 20 MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "only PDF leases are supported")
        tmp = store.dir / "leases" / f"{new_id('incoming')}.pdf"
        tmp.write_bytes(data)
        try:
            lease = lease_agent.run(tmp, Path(file.filename or "lease.pdf").name, store.unit_states())
        except Exception as exc:  # corrupt PDF etc: a clear 422, not a stack trace
            tmp.unlink(missing_ok=True)
            raise HTTPException(422, f"could not read this PDF: {exc}") from exc
        tmp.rename(store.dir / "leases" / f"{lease.id}.pdf")
        store.save_lease(lease)
        store.audit("lease", lease.id, "agent.run", filename=lease.filename, trace=lease.trace)
        return lease

    @app.get("/api/leases/{lease_id}")
    def get_lease(lease_id: str):
        return lease_or_404(lease_id)

    @app.get("/api/leases/{lease_id}/page/{number}.png")
    def lease_page(lease_id: str, number: int, field: str | None = Query(None)):
        lease = lease_or_404(lease_id)
        pdf = pymupdf.open(store.dir / "leases" / f"{lease.id}.pdf")
        if not 1 <= number <= len(pdf):
            raise HTTPException(404, "no such page")
        page = pdf[number - 1]
        fld = lease.fields.get(field) if field else None
        if fld and fld.source and fld.source.page == number:
            page.draw_rect(pymupdf.Rect(*fld.source.bbox), color=(0.85, 0.1, 0.1), width=1.5)
        return Response(page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).tobytes("png"), media_type="image/png")

    @app.patch("/api/leases/{lease_id}/fields/{name}")
    def decide_field(lease_id: str, name: str, body: FieldDecision):
        lease = lease_or_404(lease_id)
        editable(lease)
        old = lease.fields[name].value if name in lease.fields else None
        try:
            lease_agent.apply_field_decision(lease, name, body.action, body.value, store.unit_states())
        except KeyError:
            raise HTTPException(404, "no such field") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        store.save_lease(lease)
        store.audit("lease", lease.id, f"field.{body.action}", field=name, old=old, new=lease.fields[name].value)
        return lease

    @app.patch("/api/leases/{lease_id}/flags/{flag_id}")
    def decide_flag(lease_id: str, flag_id: str, body: FlagDecision):
        lease = lease_or_404(lease_id)
        editable(lease)
        flag = next((f for f in lease.flags if f.id == flag_id), None)
        if flag is None:
            raise HTTPException(404, "no such flag")
        flag.decision = "accepted" if body.action == "accept" else "rejected"
        store.save_lease(lease)
        store.audit("lease", lease.id, f"flag.{body.action}", flag=flag_id, message=flag.message)
        return lease

    @app.post("/api/leases/{lease_id}/accept")
    def accept_lease(lease_id: str):
        lease = lease_or_404(lease_id)
        editable(lease)
        states = store.unit_states()
        unit = units.get(lease.unit_id, states) if lease.unit_id else None
        if unit is None:
            raise HTTPException(409, "No unit matched. Override the unit field with a valid unit first.")
        if unit.status != "available":
            raise HTTPException(409, f"{unit.unit_id} is '{unit.status}', not available.")
        pending = [f.id for f in lease.flags if f.severity == "high" and f.decision == "pending"]
        if pending:
            raise HTTPException(409, f"Decide every high-severity flag first: {', '.join(pending)}")
        lease.status = "accepted"
        store.set_unit_state(unit.unit_id, "occupied", lease.id)
        store.save_lease(lease)
        store.audit("lease", lease.id, "lease.accept", unit=unit.unit_id)
        return lease

    @app.post("/api/leases/{lease_id}/reject")
    def reject_lease(lease_id: str):
        lease = lease_or_404(lease_id)
        editable(lease)
        lease.status = "rejected"
        store.save_lease(lease)
        store.audit("lease", lease.id, "lease.reject")
        return lease

    # ---- issues ---------------------------------------------------------

    @app.post("/api/issues")
    def report_issue(unit_id: str = Form(...), reporter: str = Form(""), note: str = Form(""),
                     photos: list[UploadFile] = File(...)):
        states = store.unit_states()
        unit = units.get(unit_id, states)
        if unit is None:
            raise HTTPException(404, "unit not found")
        if not 1 <= len(photos) <= MAX_PHOTOS:
            raise HTTPException(422, f"attach between 1 and {MAX_PHOTOS} photos")
        loaded: list[tuple[str, bytes]] = []
        for i, up in enumerate(photos):
            name = _safe_name(up.filename or "", i)
            if Path(name).suffix.lower() not in PHOTO_TYPES:
                raise HTTPException(415, f"{up.filename}: photos must be jpg, png or webp")
            blob = up.file.read(MAX_PHOTO + 1)
            if len(blob) > MAX_PHOTO:
                raise HTTPException(413, f"{up.filename} is larger than 10 MB")
            loaded.append((name, blob))
        issue = issue_agent.run(unit, loaded, reporter.strip(), note.strip(), unit.lease_id)
        folder = store.dir / "issues" / issue.id
        folder.mkdir(parents=True)
        for name, blob in loaded:
            (folder / name).write_bytes(blob)
        store.save_issue(issue)
        store.audit("issue", issue.id, "agent.run", unit=unit_id, model=issue_agent.vision.name, summary=issue.summary)
        return issue

    @app.patch("/api/issues/{issue_id}/work-order")
    def decide_work_order(issue_id: str, body: WorkOrderDecision):
        issue = issue_or_404(issue_id)
        wo = issue.work_order
        if wo is None:
            raise HTTPException(404, "issue has no work order")
        before = (wo.title, wo.description)
        if body.title is not None:
            wo.title = body.title.strip() or wo.title
        if body.description is not None:
            wo.description = body.description
        wo.status = "accepted" if body.action == "accept" else "rejected"
        store.save_issue(issue)
        store.audit("issue", issue.id, f"work_order.{body.action}", edited=before != (wo.title, wo.description))
        return issue

    @app.patch("/api/issues/{issue_id}/photos/{index}")
    def decide_photo(issue_id: str, index: int, body: FlagDecision):
        issue = issue_or_404(issue_id)
        if not 0 <= index < len(issue.photos):
            raise HTTPException(404, "no such photo")
        issue.photos[index].decision = "accepted" if body.action == "accept" else "rejected"
        store.save_issue(issue)
        store.audit("issue", issue.id, f"assessment.{body.action}", photo=issue.photos[index].filename)
        return issue

    @app.get("/api/issues/{issue_id}/photos/{filename}")
    def photo_file(issue_id: str, filename: str):
        folder = (store.dir / "issues" / issue_id).resolve()
        path = (folder / filename).resolve()
        if folder not in path.parents or not path.is_file():
            raise HTTPException(404, "photo not found")
        return FileResponse(path)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
