"""Forward explicitly selected, still-queued legacy dataset episodes to H100."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

import requests

from services.flora_api.store import Store
from services.flora_router.app import KeyCatalog
from services.flora_router.routes import Routes
from services.flora_router.transport import Backend
from services.flora_router.worker import (
    RetryLaterError,
    forward,
    prepare_input,
    recover_completed,
    remote_capacity,
)


class OwnerRevokedError(Exception):
    pass


def initialize(store: Store) -> Routes:
    routes = Routes(store.root)
    with store.connect() as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS legacy_bridge_claims "
            "(job_id TEXT PRIMARY KEY, claimed_at REAL NOT NULL)"
        )
    return routes


def active_claim(store: Store):
    with store.connect() as db:
        rows = db.execute(
            "SELECT j.* FROM jobs j JOIN legacy_bridge_claims b ON b.job_id=j.id "
            "WHERE j.status='running' ORDER BY b.claimed_at"
        ).fetchall()
    if len(rows) > 1:
        raise RuntimeError("More than one legacy bridge claim is active")
    return dict(rows[0]) if rows else None


def claim_queued(store: Store, parents: tuple[str, ...]):
    """Atomically fence the original dispatcher and persist the remote assignment."""
    if not parents:
        raise ValueError("At least one explicitly selected parent is required")
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        active = db.execute(
            "SELECT 1 FROM jobs j JOIN legacy_bridge_claims b ON b.job_id=j.id "
            "WHERE j.status='running' LIMIT 1"
        ).fetchone()
        if active:
            return None
        placeholders = ",".join("?" for _ in parents)
        row = db.execute(
            "SELECT * FROM jobs WHERE status='queued' AND kind='episode' "
            f"AND parent_id IN ({placeholders}) ORDER BY created_at,episode_index LIMIT 1",
            parents,
        ).fetchone()
        if not row:
            return None
        job = dict(row)
        directory = store.root / "jobs" / job["id"]
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError("A queued episode already has files; manual inspection is required")
        now = time.time()
        db.execute("INSERT INTO legacy_bridge_claims(job_id,claimed_at) VALUES (?,?)", (job["id"], now))
        db.execute(
            "INSERT INTO router_assignments(job_id,backend,remote_id,updated_at) VALUES (?,'H100',NULL,?)",
            (job["id"], now),
        )
        changed = db.execute(
            "UPDATE jobs SET status='running',stage='H100 · preparing',updated_at=? "
            "WHERE id=? AND status='queued'",
            (now, job["id"]),
        ).rowcount
        if changed != 1:
            raise RuntimeError("Episode claim changed unexpectedly")
        return job


def queued_remaining(store: Store, parents: tuple[str, ...]) -> bool:
    placeholders = ",".join("?" for _ in parents)
    with store.connect() as db:
        return bool(
            db.execute(
                "SELECT 1 FROM jobs WHERE status='queued' AND kind='episode' "
                f"AND parent_id IN ({placeholders}) LIMIT 1",
                parents,
            ).fetchone()
        )


def gateway_has_h100_reservation(state: Path) -> bool:
    """Leave an already selected H100 slot to the unified gateway."""
    with sqlite3.connect(f"file:{state / 'jobs.sqlite3'}?mode=ro", uri=True, timeout=10) as db:
        return bool(
            db.execute(
                "SELECT 1 FROM router_assignments a JOIN jobs j ON j.id=a.job_id "
                "WHERE a.backend='H100' AND j.status IN ('queued','running') LIMIT 1"
            ).fetchone()
        )


def process_job(store: Store, routes: Routes, job: dict, keys: KeyCatalog, backend: Backend) -> bool:
    if recover_completed(store, job):
        return True
    assignment = routes.get(job["id"])
    if assignment is None or assignment.backend != "H100":
        raise RuntimeError("The original H100 assignment is missing")
    manifest_path = store.root / "jobs" / job["id"] / "forward.json"
    if not manifest_path.exists():
        records = json.loads(keys.read_text())
        if not any(record["id"] == job["owner"] and record.get("enabled", True) for record in records):
            raise OwnerRevokedError("API key revoked before execution")
    manifest = prepare_input(store, job, keys)
    return forward(store, routes, job, manifest, assignment, backend)


def main():
    config = json.loads(Path(os.environ["FLORA_ROUTER_CONFIG"]).read_text())
    parents = tuple(item for item in os.environ["FLORA_LEGACY_PARENTS"].split(",") if item)
    if not parents or any(
        len(value) != 32 or any(c not in "0123456789abcdef" for c in value) for value in parents
    ):
        raise ValueError("Invalid legacy parent IDs")
    store = Store(Path(os.environ["FLORA_API_STATE"]))
    routes = initialize(store)
    keys = KeyCatalog(Path(os.environ["FLORA_API_KEYS"]))
    item = config["backends"]["H100"]
    backend = Backend(
        "H100", item["url"].rstrip("/"), item["hostname"], Path(item["ca"]), Path(item["key_file"])
    )
    gateway = Path(os.environ["FLORA_GATEWAY_STATE"])
    with (store.root / "legacy-bridge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            job = active_claim(store)
            if job:
                try:
                    if process_job(store, routes, job, keys, backend):
                        logging.info("Bridge finished episode %s", job["id"])
                        continue
                except OwnerRevokedError:
                    store.update(
                        job["id"],
                        status="failed",
                        stage="preparing",
                        error="API key revoked before execution",
                    )
                    continue
                except (requests.RequestException, RetryLaterError, OSError):
                    logging.warning("Bridge temporarily unavailable for episode %s", job["id"])
                except Exception:
                    logging.exception(
                        "Bridge needs attention for episode %s; retaining its identity", job["id"]
                    )
                time.sleep(10)
                continue
            if not queued_remaining(store, parents):
                logging.info("No selected queued episodes remain")
                return
            try:
                capacity = remote_capacity(config["probe_command"])
                if (
                    capacity.free_gpus
                    and not capacity.pending_gpu_job_ids
                    and len(capacity.active_job_ids) < capacity.concurrency
                    and not gateway_has_h100_reservation(gateway)
                    and shutil.disk_usage(store.root).free >= 100 * 1024**3
                ):
                    job = claim_queued(store, parents)
                    if job:
                        logging.info("Bridge claimed episode %s", job["id"])
                        continue
            except Exception:
                logging.exception("H100 capacity check or claim failed; no new episode was moved")
            time.sleep(10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
