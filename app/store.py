"""SQLite persistence. JSON documents keyed by id, plus a unit-status overlay and an audit log.

Documents-as-JSON keeps the schema small enough to read in one sitting; the unit_id column on
issues is the join that puts a unit's lease and its issues on one screen. Swapping this for
Postgres means replacing this one class.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .domain import Issue, Lease, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS issues (id TEXT PRIMARY KEY, unit_id TEXT NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS issues_unit ON issues(unit_id);
CREATE TABLE IF NOT EXISTS unit_state (unit_id TEXT PRIMARY KEY, status TEXT NOT NULL, lease_id TEXT);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, entity TEXT NOT NULL,
    entity_id TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL);
"""


class Store:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.dir = data_dir
        self._db = sqlite3.connect(data_dir / "app.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def _run(self, sql: str, *args):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.fetchall()

    # leases
    def save_lease(self, lease: Lease) -> None:
        self._run("INSERT INTO leases(id, data) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                  lease.id, lease.model_dump_json())

    def get_lease(self, lease_id: str) -> Lease | None:
        rows = self._run("SELECT data FROM leases WHERE id=?", lease_id)
        return Lease.model_validate_json(rows[0]["data"]) if rows else None

    def list_leases(self) -> list[Lease]:
        return [Lease.model_validate_json(r["data"]) for r in self._run("SELECT data FROM leases")]

    # issues
    def save_issue(self, issue: Issue) -> None:
        self._run("INSERT INTO issues(id, unit_id, data) VALUES(?, ?, ?) "
                  "ON CONFLICT(id) DO UPDATE SET data=excluded.data", issue.id, issue.unit_id, issue.model_dump_json())

    def get_issue(self, issue_id: str) -> Issue | None:
        rows = self._run("SELECT data FROM issues WHERE id=?", issue_id)
        return Issue.model_validate_json(rows[0]["data"]) if rows else None

    def issues_for_unit(self, unit_id: str) -> list[Issue]:
        rows = self._run("SELECT data FROM issues WHERE unit_id=? ORDER BY rowid DESC", unit_id)
        return [Issue.model_validate_json(r["data"]) for r in rows]

    # unit status overlay
    def set_unit_state(self, unit_id: str, status: str, lease_id: str | None) -> None:
        self._run("INSERT INTO unit_state(unit_id, status, lease_id) VALUES(?, ?, ?) "
                  "ON CONFLICT(unit_id) DO UPDATE SET status=excluded.status, lease_id=excluded.lease_id",
                  unit_id, status, lease_id)

    def unit_states(self) -> dict[str, tuple[str, str | None]]:
        return {r["unit_id"]: (r["status"], r["lease_id"]) for r in self._run("SELECT * FROM unit_state")}

    # audit trail: every human decision and every agent run
    def audit(self, entity: str, entity_id: str, action: str, **detail) -> None:
        self._run("INSERT INTO audit(ts, entity, entity_id, action, detail) VALUES(?, ?, ?, ?, ?)",
                  now(), entity, entity_id, action, json.dumps(detail, default=str))

    def audit_for(self, *entity_ids: str) -> list[dict]:
        marks = ",".join("?" * len(entity_ids))
        rows = self._run(f"SELECT ts, entity, entity_id, action, detail FROM audit "
                         f"WHERE entity_id IN ({marks}) ORDER BY id DESC", *entity_ids)
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]
