"""Synthetic local HTTP acceptance of memory jobs and scoped review. No live accounts or Qwen."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from flask import jsonify

from lerobot.data_platform.datasets.lifecycle.service import LifecycleStore
from lerobot.data_platform.memory.sources import read_preview
from lerobot.data_platform.operators.execution import execute_registered
from lerobot.data_platform.routes.curation import register_curation_routes
from lerobot.data_platform.routes.memory import register_memory_routes
from lerobot.data_platform.routes.processing import register_processing_routes
from tests.datasets.test_control_plane import _app, _bootstrap, _store
from tests.datasets.test_memory_annotation import Teacher, real_source


@pytest.fixture
def memory_remote(tmp_path, monkeypatch):
    source, selections = real_source(tmp_path)
    store = _store(tmp_path)
    app = _app(tmp_path, store)
    client = app.test_client()
    ledger = LifecycleStore(tmp_path / "lifecycle")
    caps = {
        "job_protocol": 2,
        "memory_annotation_protocol": 1,
        "memory_batch_protocol": 1,
        "memory_qwen_configured": True,
        "memory_model": "test-model",
        "data_profile_protocol": 100,
        "operations": ["memory.annotate", "memory.export"],
        "operator_versions": {"memory.annotate": "1.0.0", "memory.export": "1.0.0"},
    }
    token, node = store.enroll_node(
        name="memory-node",
        hostname="localhost",
        allowed_roots=[str(tmp_path)],
        writable_roots=[str(tmp_path)],
        capabilities=caps,
        enrollment_token="test",
        expected_token="test",
    )
    location = store.sync_locations(
        node["node_id"],
        [
            {
                "dataset_key": "fixture",
                "root": str(source),
                "output_dir": str(tmp_path / "cache"),
                "metadata": json.loads((source / "meta/info.json").read_text()),
            }
        ],
    )[0]
    agent = SimpleNamespace(
        allowed_roots=[tmp_path],
        writable_roots=[tmp_path],
        state_path=tmp_path / "agent/state.json",
        name="memory-node",
        state=None,
    )
    agent.state_path.parent.mkdir()

    class Relay:
        def request(self, node_id, read, **kwargs):
            return SimpleNamespace(
                status=200, binary=False, body=json.dumps(read_preview(agent, read)).encode()
            )

    ctx = SimpleNamespace(control_plane_store=store, lifecycle_store=ledger, caption_read_relay=Relay())
    register_processing_routes(app, ctx)
    register_curation_routes(app, ctx)
    register_memory_routes(app, ctx)

    @app.get("/api/control/locations/<location_id>/temporal-caption")
    def caption_catalog(location_id):
        return jsonify(
            runs=[
                {"run": f"flora-{i}", "episode_index": i, "variant": "caption", "scheme": {"id": "flora"}}
                for i in range(2)
            ]
        )

    _bootstrap(client)

    def caption_bytes(loc, run):
        raw = Path(selections[int(run.split("-")[-1])]["annotation"]).read_bytes()
        return {**json.loads(raw), "_memory_raw_bytes": raw}

    app.extensions["data_platform_memory_caption"] = caption_bytes
    teacher = Teacher()
    monkeypatch.setattr("lerobot.data_platform.memory.jobs.QwenClient", lambda **_: teacher)
    monkeypatch.setenv("DATA_PLATFORM_MEMORY_MODEL", "test-model")
    base = f"/api/control/locations/{location['location_id']}/memory"
    return SimpleNamespace(
        client=client,
        app=app,
        store=store,
        node=node,
        token=token,
        location=location,
        ledger=ledger,
        agent=agent,
        teacher=teacher,
        base=base,
        source=source,
    )


def prepare(remote, episode=0):
    response = remote.client.post(
        remote.base + "/prepare",
        json={
            "episode_index": episode,
            "flora_run": f"flora-{episode}",
            "task": "Move block",
            "lineage_group": f"recording-{episode}",
        },
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def execute(remote, request_id, tmp_path):
    with patch.dict(os.environ, {"DATA_PLATFORM_RELEASE": "test-release"}):
        response = remote.client.post(
            remote.base + "/jobs", json={"request_id": request_id}, headers={"Idempotency-Key": "delivery"}
        )
    assert response.status_code == 202, response.get_json()
    snapshot = response.get_json()["job"]["options"]["operator_snapshot"]
    assert snapshot["release"] == "test-release"
    assert snapshot["data_format"] == remote.location["metadata"].get("codebase_version", "unknown")
    job_id = response.get_json()["job"]["job_id"]
    repeated = remote.client.post(
        remote.base + "/jobs", json={"request_id": request_id}, headers={"Idempotency-Key": "delivery"}
    )
    assert repeated.get_json()["job"]["job_id"] == job_id
    job = remote.store.claim_job(remote.node["node_id"], worker_instance_id="test-worker")
    assert job is not None
    headers = {
        "Authorization": f"Bearer {remote.token}",
        "X-Job-Attempt": job["execution"]["attempt_id"],
        "X-Job-Credential": job["execution"]["credential"],
    }
    inputs = tmp_path / ("inputs-" + job_id)
    inputs.mkdir()
    for name, expected in job["options"]["input_files"].items():
        response = remote.client.get(f"/api/agents/jobs/{job_id}/curation-inputs/{name}", headers=headers)
        assert response.status_code == 200, response.get_json()
        assert len(response.data) == expected["size"]
        (inputs / name).write_bytes(response.data)
    job["options"]["input_root"] = str(inputs)

    class Client:
        def event(self, *args):
            pass

        def upload_artifact(self, state, job_id, name, path, **kwargs):
            response = remote.client.put(
                f"/api/agents/jobs/{job_id}/curation-artifacts/{name}",
                headers=headers,
                data=path.read_bytes(),
            )
            assert response.status_code == 200, response.get_json()

    remote.agent.client = Client()
    result = execute_registered(remote.agent, job, remote.source, remote.location)
    bad = remote.client.post(
        f"/api/agents/jobs/{job_id}/complete",
        headers={**headers, "X-Job-Credential": "old"},
        json={"status": "done", "result": result},
    )
    assert bad.status_code == 409
    response = remote.client.post(
        f"/api/agents/jobs/{job_id}/complete", headers=headers, json={"status": "done", "result": result}
    )
    assert response.status_code == 200, response.get_json()
    return job_id, headers


def test_prepare_job_publication_review_guards(memory_remote, tmp_path):
    remote = memory_remote
    preflight = prepare(remote)
    assert preflight["pin"]["expected_requests"] == 4
    assert not remote.teacher.calls
    run, headers = execute(remote, preflight["request_id"], tmp_path)
    assert len(remote.teacher.calls) == 4
    response = remote.client.get(remote.base + f"/reviews/{run}")
    assert response.status_code == 200, response.get_json()
    value = response.get_json()
    assert value["state"]["record"]["review"]["status"] == "draft"
    assert remote.client.get(remote.base + f"/reviews/{run}/frames/1.png?cutoff=0").status_code == 404
    assert remote.client.get(remote.base + f"/reviews/{run}/frames/0.png?cutoff=1").status_code == 200
    assert (
        remote.client.post(
            remote.base + f"/reviews/{run}/needs_review", json={"expected_revision": 0, "frame": 0}
        ).status_code
        == 409
    )
    assert (
        remote.client.post(remote.base + "/prepare", json={"root": "/etc", "episode_index": 0}).status_code
        == 400
    )
    assert (
        remote.client.post(
            f"/api/agents/jobs/{run}/complete", headers=headers, json={"status": "done"}
        ).status_code
        == 200
    )
    remote.store.register_user(username="reader", password="synthetic-password", role="viewer")
    remote.client.post("/api/auth/logout", json={})
    remote.client.post("/api/auth/login", json={"username": "reader", "password": "synthetic-password"})
    assert remote.client.get(remote.base + f"/reviews/{run}").status_code == 200
    assert remote.client.post(remote.base + f"/reviews/{run}/approve", json={}).status_code == 403


def test_scope_rechecked_at_read_submission_and_claim(memory_remote):
    from lerobot.data_platform.control.store import DataMutationScope

    remote = memory_remote
    user = remote.store.register_user(username="scoped", password="synthetic-password", role="operator")
    remote.client.post("/api/auth/logout", json={})
    remote.client.post("/api/auth/login", json={"username": "scoped", "password": "synthetic-password"})
    preflight = prepare(remote)
    response = remote.client.post(
        remote.base + "/jobs",
        json={"request_id": preflight["request_id"]},
        headers={"Idempotency-Key": "scope"},
    )
    assert response.status_code == 202
    with remote.store.sessions.begin() as session:
        session.add(DataMutationScope(user_id=user["user_id"], all_locations=False))
    assert remote.client.get(remote.base).status_code == 403
    assert (
        remote.client.post(
            remote.base + "/jobs",
            json={"request_id": preflight["request_id"]},
            headers={"Idempotency-Key": "scope2"},
        ).status_code
        == 403
    )
    assert remote.store.claim_job(remote.node["node_id"], worker_instance_id="worker") is None


def test_agent_capability_revocation_and_incomplete_audit(memory_remote):
    remote = memory_remote
    preflight = prepare(remote)
    response = remote.client.post(
        remote.base + "/pilot",
        json={"request_ids": [preflight["request_id"]], "seed": 42, "audit_batch": "missing"},
    )
    assert response.status_code == 400
    remote.store.heartbeat(remote.node["node_id"], capabilities={"job_protocol": 2})
    assert (
        remote.client.post(
            remote.base + "/jobs",
            json={"request_id": preflight["request_id"]},
            headers={"Idempotency-Key": "cap"},
        ).status_code
        == 409
    )


def frozen_batch(remote):
    import uuid

    audit = str(uuid.uuid4())
    prepared = []
    for episode in range(2):
        response = remote.client.post(
            remote.base + "/prepare",
            json={
                "episode_index": episode,
                "flora_run": f"flora-{episode}",
                "task": "Move block",
                "lineage_group": f"recording-{episode}",
                "audit_batch": audit,
            },
        )
        assert response.status_code == 200
        prepared.append(response.get_json())
    response = remote.client.post(
        remote.base + "/pilot",
        json={
            "request_ids": [p["request_id"] for p in prepared],
            "seed": 42,
            "audit_batch": audit,
        },
    )
    assert response.status_code == 200, response.get_json()
    pilot = response.get_json()
    return prepared, pilot


def test_dataset_batch_one_job_scoped_reviews_and_regeneration(memory_remote, tmp_path):
    remote = memory_remote
    prepared, pilot = frozen_batch(remote)
    body = {"request_ids": [p["request_id"] for p in prepared], "pilot_id": pilot["digest"]}
    assert (
        remote.client.post(
            remote.base + "/batch", json={**body, "request_ids": body["request_ids"][:1]}
        ).status_code
        == 400
    )
    response = remote.client.post(remote.base + "/batch", json=body)
    assert response.status_code == 200, response.get_json()
    request_id = response.get_json()["request_id"]
    run, _ = execute(remote, request_id, tmp_path)
    assert len(remote.store.list_jobs()) == 1
    result = remote.store.get_job(run)["result"]["summary"]
    assert len(result["episodes"]) == 2 and all(e["has_review"] for e in result["episodes"])
    # A changed delivery token must not accidentally duplicate a dataset batch.
    duplicate = remote.client.post(
        remote.base + "/jobs",
        json={"request_id": request_id},
        headers={"Idempotency-Key": "another-delivery"},
    )
    assert duplicate.get_json()["job"]["job_id"] == run
    assert len(remote.store.list_jobs()) == 1
    for episode in result["episodes"]:
        child = episode["run"]
        value = remote.client.get(remote.base + f"/reviews/{child}").get_json()
        assert value["state"]["pin"]["episode_index"] == episode["episode_index"]
        assert value["state"]["record"]["review"]["status"] == "draft"
        assert remote.client.get(remote.base + f"/reviews/{child}/frames/0.png?cutoff=1").status_code == 200
        assert remote.client.get(remote.base + f"/reviews/{child}/frames/1.png?cutoff=0").status_code == 404
        assert remote.client.get(remote.base + f"/reviews/{child}/flora?cutoff=1").status_code == 200
    child = result["episodes"][0]["run"]
    regenerate = remote.client.post(
        remote.base + f"/reviews/{child}/regenerate", json={"expected_revision": 1}
    )
    assert regenerate.status_code == 200
    regenerated, _ = execute(remote, regenerate.get_json()["request_id"], tmp_path)
    assert len(remote.store.get_job(regenerated)["result"]["summary"]["episodes"]) == 1
    assert remote.client.get(remote.base + f"/reviews/{child}").get_json()["revision"] == 1


def test_batch_requires_new_agent_capability_and_preparing_owner(memory_remote):
    remote = memory_remote
    prepared, pilot = frozen_batch(remote)
    body = {"request_ids": [p["request_id"] for p in prepared], "pilot_id": pilot["digest"]}
    remote.store.register_user(username="other-operator", password="synthetic-password", role="operator")
    remote.client.post("/api/auth/logout", json={})
    remote.client.post(
        "/api/auth/login", json={"username": "other-operator", "password": "synthetic-password"}
    )
    assert remote.client.post(remote.base + "/batch", json=body).status_code == 403
    caps = dict(remote.node["capabilities"])
    caps.pop("memory_batch_protocol")
    remote.store.heartbeat(remote.node["node_id"], capabilities=caps)
    assert remote.client.post(remote.base + "/batch", json=body).status_code == 409
    assert not remote.store.list_jobs()


def test_batch_resume_reuses_responses_and_exports_reviewed_children(memory_remote, tmp_path):
    from dataclasses import replace

    from lerobot.data_platform.memory.jobs import execute as execute_memory
    from lerobot.data_platform.operators import ExecutionContext

    remote = memory_remote
    prepared, pilot = frozen_batch(remote)
    response = remote.client.post(
        remote.base + "/batch",
        json={
            "request_ids": [p["request_id"] for p in prepared],
            "pilot_id": pilot["digest"],
        },
    )
    request_id = response.get_json()["request_id"]
    inputs = remote.ledger.root / "operator-results" / request_id / "static"
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    ctx = ExecutionContext(
        source_root=remote.source,
        dataset_key="fixture",
        output_root=tmp_path / "durable",
        work_dir=first,
        artifacts={p.name: p for p in inputs.iterdir()},
    )
    original = execute_memory(ctx, {"request_id": request_id})
    calls = len(remote.teacher.calls)
    resumed = execute_memory(replace(ctx, work_dir=second), {"request_id": request_id})
    assert len(remote.teacher.calls) == calls == 8
    assert [e["run"] for e in original.summary["episodes"]] == [e["run"] for e in resumed.summary["episodes"]]
    run, _ = execute(remote, request_id, tmp_path)
    children = [e["run"] for e in remote.store.get_job(run)["result"]["summary"]["episodes"]]
    confirmations = {"source_confirmed": True, "goal_confirmed": True, "causal_evidence_confirmed": True}
    for child in children:
        response = remote.client.post(
            remote.base + f"/reviews/{child}/approve", json={"expected_revision": 1, **confirmations}
        )
        assert response.status_code == 200, response.get_json()
    remote.store.register_user(username="batch-independent", password="synthetic-password", role="operator")
    remote.client.post("/api/auth/logout", json={})
    remote.client.post(
        "/api/auth/login", json={"username": "batch-independent", "password": "synthetic-password"}
    )
    for child in children:
        response = remote.client.post(
            remote.base + f"/reviews/{child}/approve",
            json={"expected_revision": 2, **confirmations, "independent": True, "adjudicated": True},
        )
        assert response.status_code == 200, response.get_json()
    response = remote.client.post(
        remote.base + "/export", json={"runs": children, "pilot_id": pilot["digest"]}
    )
    assert response.status_code == 200, response.get_json()
    exported, _ = execute(remote, response.get_json()["request_id"], tmp_path)
    summary = remote.store.get_job(exported)["result"]["summary"]
    assert summary["episodes"] == 2
    package = json.loads((Path(summary["output"]) / "package.json").read_text())
    assert any(path.startswith("audit/") for path in package["files"])
