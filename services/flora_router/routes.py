"""Durable backend assignment and capacity-aware admission."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from services.flora_router.capacity import Capacity


@dataclass(frozen=True)
class Assignment:
    backend: str
    remote_id: str | None


class Routes:
    def __init__(self, state: Path):
        self.path = state / "jobs.sqlite3"
        self.lock = threading.Lock()
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS router_assignments ("
                "job_id TEXT PRIMARY KEY, backend TEXT NOT NULL, remote_id TEXT, updated_at REAL NOT NULL)"
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, job_id: str) -> Assignment | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT backend,remote_id FROM router_assignments WHERE job_id=?", (job_id,)
            ).fetchone()
        return Assignment(*row) if row else None

    def set_remote(self, job_id: str, remote_id: str) -> None:
        with self.connect() as db:
            row = db.execute("SELECT remote_id FROM router_assignments WHERE job_id=?", (job_id,)).fetchone()
            if not row or (row[0] is not None and row[0] != remote_id):
                raise ValueError("Remote job identity changed")
            db.execute(
                "UPDATE router_assignments SET remote_id=?,updated_at=? WHERE job_id=?",
                (remote_id, time.time(), job_id),
            )

    def select(self, job_id: str, observations: dict[str, Capacity]) -> Assignment | None:
        """Reserve one observed slot; never reassign a job after selection."""
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT backend,remote_id FROM router_assignments WHERE job_id=?", (job_id,)
            ).fetchone()
            if existing:
                return Assignment(*existing)
            rows = db.execute(
                "SELECT a.backend,a.job_id,a.remote_id FROM router_assignments a "
                "JOIN jobs j ON j.id=a.job_id WHERE j.status IN ('queued','running')"
            ).fetchall()
            choices = []
            for backend, observed in observations.items():
                assigned = [(job, remote) for name, job, remote in rows if name == backend]
                active = set(observed.active_job_ids)
                pending = set(observed.pending_gpu_job_ids)
                unseen = {remote or f"local:{job}" for job, remote in assigned if remote not in active}
                waiting = {
                    remote or f"local:{job}"
                    for job, remote in assigned
                    if remote not in active or remote in pending
                }
                api_slots = observed.concurrency - len(active | unseen)
                gpu_slots = len(observed.free_gpus) - len(pending | waiting)
                available = min(api_slots, gpu_slots)
                if available > 0:
                    choices.append((available, backend))
            if not choices:
                return None
            # Prefer the backend with the most immediately available capacity.
            backend = max(choices)[1]
            db.execute(
                "INSERT INTO router_assignments(job_id,backend,remote_id,updated_at) VALUES(?,?,NULL,?)",
                (job_id, backend, time.time()),
            )
            return Assignment(backend, None)
