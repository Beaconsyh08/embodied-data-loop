"""Durable admission and job state for the standalone service."""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class QueueFullError(Exception):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, idempotency TEXT,
                status TEXT NOT NULL, stage TEXT NOT NULL, error TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                kind TEXT NOT NULL DEFAULT 'video', parent_id TEXT, episode_index INTEGER,
                UNIQUE(owner, idempotency))""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_parent_episode ON jobs(parent_id,episode_index)")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status,created_at)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def get(self, job_id: str, owner: str | None = None):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or (owner is not None and row["owner"] != owner):
            return None
        job = dict(row)
        if job["kind"] == "dataset":
            with self.connect() as db:
                job["counts"] = dict(
                    db.execute(
                        "SELECT status,count(*) FROM jobs WHERE parent_id=? GROUP BY status", (job_id,)
                    ).fetchall()
                )
                if job["status"] == "failed" and job["counts"].get("failed"):
                    first = db.execute(
                        "SELECT episode_index,stage,error FROM jobs WHERE parent_id=? AND status='failed' "
                        "ORDER BY episode_index LIMIT 1",
                        (job_id,),
                    ).fetchone()
                    job["error"] = (
                        f"{job['counts']['failed']} episode(s) failed; "
                        f"episode {first['episode_index']} [{first['stage']}]: "
                        f"{first['error'] or 'No diagnostic available'}"
                    )
        return job

    def reserve(self, owner: str, idempotency: str | None, kind: str = "video"):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idempotency:
                row = db.execute(
                    "SELECT * FROM jobs WHERE owner=? AND idempotency=?", (owner, idempotency)
                ).fetchone()
                if row:
                    return dict(row), False
            rows = db.execute(
                "SELECT owner FROM jobs WHERE parent_id IS NULL AND status IN ('uploading','queued','running')"
            ).fetchall()
            if len(rows) >= 16 or sum(row["owner"] == owner for row in rows) >= 4:
                raise QueueFullError
            job_id, now = uuid.uuid4().hex, time.time()
            db.execute(
                "INSERT INTO jobs (id,owner,idempotency,status,stage,created_at,updated_at,kind) "
                "VALUES (?,?,?,'uploading','uploading',?,?,?)",
                (job_id, owner, idempotency, now, now, kind),
            )
        return self.get(job_id), True

    def update(self, job_id: str, *, status: str, stage: str, error: str | None = None):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,stage=?,error=?,updated_at=? WHERE id=?",
                (status, stage, error, time.time(), job_id),
            )
            parent = db.execute("SELECT parent_id FROM jobs WHERE id=?", (job_id,)).fetchone()
            if parent and parent[0]:
                states = dict(
                    db.execute(
                        "SELECT status,count(*) FROM jobs WHERE parent_id=? GROUP BY status", (parent[0],)
                    ).fetchall()
                )
                if not (states.get("queued") or states.get("running")):
                    final = "failed" if states.get("failed") else "completed"
                    db.execute(
                        "UPDATE jobs SET status=?,stage=?,updated_at=? WHERE id=?",
                        (final, final, time.time(), parent[0]),
                    )

    def touch_upload(self, job_id: str):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET updated_at=? WHERE id=? AND status='uploading'",
                (time.time(), job_id),
            )

    def children(self, parent: str, after: int = -1, limit: int = 100):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM jobs WHERE parent_id=? AND episode_index>? ORDER BY episode_index LIMIT ?",
                    (parent, after, limit),
                )
            ]

    def add_children(self, parent: dict, episodes: list[int]):
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for episode in episodes:
                child = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO jobs (id,owner,status,stage,created_at,updated_at,kind,parent_id,episode_index) "
                    "VALUES (?,?,'queued','queued',?,?,'episode',?,?)",
                    (child, parent["owner"], now, now, parent["id"], episode),
                )
            db.execute(
                "UPDATE jobs SET status='running',stage='dataset queued',updated_at=? WHERE id=?",
                (now, parent["id"]),
            )

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE jobs SET status='failed',stage='upload interrupted',error='Upload timed out' "
                "WHERE status='uploading' AND updated_at<?",
                (time.time() - 7200,),
            )
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' AND kind!='dataset' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE jobs SET status='running',stage='validating',updated_at=? WHERE id=?",
                (time.time(), row["id"]),
            )
            return dict(row)

    def interrupted(self):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute("SELECT * FROM jobs WHERE status='running' AND kind!='dataset'")
            ]
