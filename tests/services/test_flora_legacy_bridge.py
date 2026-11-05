"""Legacy cross-host claims preserve the old API's episode identity and recovery."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import requests

from services.flora_api.store import Store
from services.flora_api.worker import recover
from services.flora_router import legacy_bridge, legacy_cleanup
from services.flora_router.legacy_recovery_guard import RecoveryView


def parent_with_episodes(store, name, episodes):
    parent, _ = store.reserve("owner", name, "dataset")
    store.add_children(parent, episodes)
    return parent, store.children(parent["id"])


def test_only_selected_queued_episodes_are_claimed_and_assignment_is_durable(tmp_path):
    store = Store(tmp_path)
    selected, children = parent_with_episodes(store, "selected", [0, 1, 2])
    other, others = parent_with_episodes(store, "other", [0])
    store.update(children[0]["id"], status="completed", stage="completed")
    store.update(children[1]["id"], status="running", stage="FLORA dense · 4s")
    before = [store.get(child["id"]) for child in children[:2]]
    routes = legacy_bridge.initialize(store)
    job = legacy_bridge.claim_queued(store, (selected["id"],))
    assert job["id"] == children[2]["id"]
    assert store.get(job["id"])["stage"] == "H100 · preparing"
    assert routes.get(job["id"]).backend == "H100" and routes.get(job["id"]).remote_id is None
    assert legacy_bridge.active_claim(store)["id"] == job["id"]
    assert legacy_bridge.claim_queued(store, (other["id"],)) is None
    assert [store.get(child["id"]) for child in children[:2]] == before
    assert store.get(others[0]["id"])["status"] == "queued"
    assert not legacy_bridge.queued_remaining(store, (selected["id"],))
    routes.set_remote(job["id"], "remote-id")
    assert legacy_bridge.initialize(store).get(job["id"]).remote_id == "remote-id"


def test_atomic_claim_never_duplicates_original_dispatcher(tmp_path):
    store = Store(tmp_path)
    selected, children = parent_with_episodes(store, "selected", [0])
    legacy_bridge.initialize(store)
    gate = Barrier(2)

    def original():
        gate.wait()
        return store.claim()

    def bridge():
        gate.wait()
        return legacy_bridge.claim_queued(store, (selected["id"],))

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(original), pool.submit(bridge)
        results = [a.result(), b.result()]
    assert sum(result is not None for result in results) == 1
    assert next(result for result in results if result)["id"] == children[0]["id"]
    assert store.get(children[0]["id"])["status"] == "running"


def test_never_replaces_existing_files_of_a_queued_episode(tmp_path):
    store = Store(tmp_path)
    parent, children = parent_with_episodes(store, "selected", [0])
    routes = legacy_bridge.initialize(store)
    directory = tmp_path / "jobs" / children[0]["id"]
    directory.mkdir(parents=True)
    (directory / "existing").write_text("keep")
    with pytest.raises(RuntimeError, match="already has files"):
        legacy_bridge.claim_queued(store, (parent["id"],))
    assert store.get(children[0]["id"])["status"] == "queued"
    assert routes.get(children[0]["id"]) is None
    assert (directory / "existing").read_text() == "keep"


def test_original_worker_restart_skips_bridge_but_recovers_ordinary_jobs(tmp_path):
    store = Store(tmp_path)
    parent, children = parent_with_episodes(store, "selected", [0, 1, 2])
    store.update(children[0]["id"], status="completed", stage="completed")
    store.update(children[1]["id"], status="running", stage="FLORA dense · 4s")
    before = store.get(children[0]["id"])
    legacy_bridge.initialize(store)
    bridge = legacy_bridge.claim_queued(store, (parent["id"],))
    recover(RecoveryView(store), lambda _: None)
    assert store.get(bridge["id"])["status"] == "running"
    assert store.get(children[1]["id"])["status"] == "failed"
    assert store.get(children[0]["id"]) == before
    assert store.get(parent["id"])["status"] == "running"


def test_guard_preserves_original_behavior_without_bridge_table(tmp_path):
    store = Store(tmp_path)
    parent, children = parent_with_episodes(store, "ordinary", [0])
    store.update(children[0]["id"], status="running", stage="preparing")
    recover(RecoveryView(store), lambda _: None)
    assert store.get(children[0]["id"])["status"] == "failed"
    assert store.get(parent["id"])["status"] == "failed"


def test_gateway_reservation_blocks_new_h100_claim(tmp_path):
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        db.execute("CREATE TABLE jobs(id TEXT, status TEXT)")
        db.execute("CREATE TABLE router_assignments(job_id TEXT, backend TEXT)")
        db.execute("INSERT INTO jobs VALUES ('one', 'running')")
        db.execute("INSERT INTO router_assignments VALUES ('one', 'A100')")
    assert not legacy_bridge.gateway_has_h100_reservation(tmp_path)
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        db.execute("UPDATE router_assignments SET backend='H100'")
    assert legacy_bridge.gateway_has_h100_reservation(tmp_path)
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        db.execute("UPDATE jobs SET status='completed'")
    assert not legacy_bridge.gateway_has_h100_reservation(tmp_path)


def test_retry_retains_original_and_remote_identity(tmp_path, monkeypatch):
    store = Store(tmp_path)
    parent, _ = parent_with_episodes(store, "selected", [0])
    routes = legacy_bridge.initialize(store)
    job = legacy_bridge.claim_queued(store, (parent["id"],))
    (tmp_path / "keys.json").write_text(json.dumps([{"id": "owner"}]))
    keys = SimpleNamespace(read_text=(tmp_path / "keys.json").read_text)
    seen = []

    def prepare(*args):
        directory = tmp_path / "jobs" / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "forward.json").write_text("{}")
        return {}

    def forward(_store, _routes, current, _manifest, assignment, _backend):
        seen.append((current["id"], assignment.backend, assignment.remote_id))
        if assignment.remote_id is None:
            routes.set_remote(current["id"], "existing-remote")
            raise requests.ConnectionError("lost response")
        return True

    monkeypatch.setattr(legacy_bridge, "prepare_input", prepare)
    monkeypatch.setattr(legacy_bridge, "forward", forward)
    with pytest.raises(requests.ConnectionError):
        legacy_bridge.process_job(store, routes, job, keys, None)
    assert legacy_bridge.process_job(store, legacy_bridge.initialize(store), job, keys, None)
    assert seen == [(job["id"], "H100", None), (job["id"], "H100", "existing-remote")]


def test_revoked_owner_never_starts_remote_work(tmp_path):
    store = Store(tmp_path)
    parent, _ = parent_with_episodes(store, "selected", [0])
    routes = legacy_bridge.initialize(store)
    job = legacy_bridge.claim_queued(store, (parent["id"],))
    keys = SimpleNamespace(read_text=lambda: json.dumps([{"id": "owner", "enabled": False}]))
    with pytest.raises(legacy_bridge.OwnerRevokedError):
        legacy_bridge.process_job(store, routes, job, keys, None)
    assert routes.get(job["id"]).remote_id is None
    assert not (tmp_path / "jobs" / job["id"]).exists()


def test_guard_cleanup_waits_for_both_queued_and_bridge_owned_attempts(tmp_path):
    store = Store(tmp_path)
    parent, children = parent_with_episodes(store, "selected", [0, 1])
    legacy_bridge.initialize(store)
    assert legacy_cleanup.outstanding(tmp_path / "jobs.sqlite3", (parent["id"],))
    bridge = legacy_bridge.claim_queued(store, (parent["id"],))
    store.update(children[1]["id"], status="running", stage="FLORA dense")
    assert legacy_cleanup.outstanding(tmp_path / "jobs.sqlite3", (parent["id"],))
    store.update(bridge["id"], status="completed", stage="completed")
    assert not legacy_cleanup.outstanding(tmp_path / "jobs.sqlite3", (parent["id"],))
