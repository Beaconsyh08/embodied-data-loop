"""Smart API routing must not allocate occupied GPUs or duplicate backend jobs."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from werkzeug.formparser import parse_form_data
from werkzeug.test import EnvironBuilder

from services.flora_api.store import Store
from services.flora_router.capacity import Capacity, eligible_gpus
from services.flora_router.routes import Routes
from services.flora_router.transport import Multipart


def gpu(index, free=80000, *, processes=""):
    return (
        f"<gpu><minor_number>{index}</minor_number><fb_memory_usage><total>81920 MiB</total>"
        f"<free>{free} MiB</free></fb_memory_usage><processes>{processes}</processes></gpu>"
    )


def test_capacity_excludes_busy_processes_memory_and_shared_locks(tmp_path):
    locked = tmp_path / "gpu-2.lock"
    locked.touch()
    stat = locked.stat()
    import os

    locks = f"1: FLOCK ADVISORY WRITE 42 {os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino} 0 EOF"
    xml = (
        "<nvidia_smi_log>"
        + gpu(0)
        + gpu(1, processes="<process_info/>")
        + gpu(2)
        + gpu(3, 1000)
        + "</nvidia_smi_log>"
    )
    assert eligible_gpus(xml, [0, 1, 2, 3], tmp_path, locks) == (0,)


def test_capacity_fails_closed_if_process_information_missing(tmp_path):
    xml = "<nvidia_smi_log>" + gpu(0).replace("<processes></processes>", "") + "</nvidia_smi_log>"
    with pytest.raises(ValueError, match="occupancy"):
        eligible_gpus(xml, [0], tmp_path, "")


def test_capacity_waits_for_existing_pending_api_job(tmp_path):
    store = Store(tmp_path)
    one, _ = store.reserve("owner", "first")
    store.update(one["id"], status="running", stage="waiting")
    routes = Routes(tmp_path)
    busy = Capacity((0,), ("backend-job",), ("backend-job",), 8)
    assert routes.select(one["id"], {"a100": busy}) is None


def test_concurrent_submissions_reserve_only_one_free_gpu(tmp_path):
    store = Store(tmp_path)
    jobs = [store.reserve("owner", str(i))[0] for i in range(2)]
    for job in jobs:
        store.update(job["id"], status="running", stage="waiting")
    routes = Routes(tmp_path)
    available = Capacity((0,), (), (), 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda j: routes.select(j["id"], {"h100": available}), jobs))
    assert sum(result is not None for result in results) == 1


def test_assigned_job_stays_on_same_backend_after_capacity_changes(tmp_path):
    store = Store(tmp_path)
    job, _ = store.reserve("owner", "same")
    store.update(job["id"], status="running", stage="waiting")
    routes = Routes(tmp_path)
    assigned = routes.select(job["id"], {"h100": Capacity((0,), (), (), 1)})
    routes.set_remote(job["id"], "backend-id")
    assert routes.select(job["id"], {"a100": Capacity((1,), (), (), 8)}).backend == assigned.backend
    with pytest.raises(ValueError, match="identity"):
        routes.set_remote(job["id"], "different-id")


def test_multipart_is_streamed_and_preserves_instruction_and_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"first-frame\x00last-frame")
    with Multipart("放好碗", {"main": video}) as multipart:
        size = len(multipart)
        body = b"".join(iter(lambda: multipart.read(7), b""))
        assert len(body) == size
        builder = EnvironBuilder(
            method="POST", data=body, content_type=f"multipart/form-data; boundary={multipart.boundary}"
        )
        environ = builder.get_environ()
        _, form, files = parse_form_data(environ)
        assert form["instruction"] == "放好碗"
        assert files["main_video"].read() == video.read_bytes()


def test_key_catalog_observes_revocation_and_rejects_duplicates(tmp_path):
    from services.flora_router.app import KeyCatalog

    primary = tmp_path / "primary.json"
    extra = tmp_path / "extra.json"
    primary.write_text(json.dumps([{"id": "user", "sha256": "a", "enabled": True}]))
    extra.write_text(json.dumps([{"id": "router-user", "sha256": "b"}]))
    catalog = KeyCatalog(primary, extra)
    assert len(json.loads(catalog.read_text())) == 2
    primary.write_text(json.dumps([{"id": "user", "sha256": "a", "enabled": False}]))
    assert not json.loads(catalog.read_text())[0]["enabled"]
    extra.write_text(json.dumps([{"id": "router-user", "sha256": "a"}]))
    with pytest.raises(ValueError, match="Duplicate"):
        catalog.read_text()


def test_forward_streams_and_resumes_same_remote_job(tmp_path, monkeypatch):
    import threading

    import requests
    from flask import Flask, jsonify, request
    from werkzeug.serving import make_server

    from services.flora_router import worker

    app = Flask(__name__)
    calls = []
    video = tmp_path / "video.mp4"
    video.write_bytes(b"content" * 40000)

    @app.post("/v1/flora/jobs")
    def submit():
        calls.append(
            (
                request.headers["Idempotency-Key"],
                request.form["instruction"],
                request.files["main_video"].read(),
            )
        )
        return jsonify(id="remote-one"), 202

    @app.get("/v1/flora/jobs/<job_id>")
    def status(job_id):
        return jsonify(id=job_id, status="running", stage="working")

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class LocalBackend:
        name = "H100"
        url = f"http://127.0.0.1:{server.server_port}"

        def session(self):
            return requests.Session()

    try:
        store = Store(tmp_path)
        job, _ = store.reserve("owner", "source")
        store.update(job["id"], status="running", stage="waiting")
        routes = Routes(tmp_path)
        assignment = routes.select(job["id"], {"H100": Capacity((0,), (), (), 1)})
        manifest = {"instruction": "放好碗", "videos": {"main": str(video)}}
        assert worker.forward(store, routes, job, manifest, assignment, LocalBackend()) is False
        assert routes.get(job["id"]).remote_id == "remote-one"
        # A restarted worker must query the recorded ID, never create a second backend job.
        assert (
            worker.forward(store, Routes(tmp_path), job, manifest, routes.get(job["id"]), LocalBackend())
            is False
        )
        assert calls == [(f"router:{job['id']}", "放好碗", video.read_bytes())]
    finally:
        server.shutdown()
        thread.join()


def test_episode_result_preserves_source_identity_and_rejects_changed_source(tmp_path, monkeypatch):
    from lerobot.data_platform.operators.implementations.caption import flora as temporal_caption_flora
    from lerobot.data_platform.operators.implementations.caption import snapshot as temporal_caption_snapshot
    from services.flora_router import worker

    store = Store(tmp_path)
    job, _ = store.reserve("owner", "episode-result")
    store.update(job["id"], status="running", stage="validating")
    source = tmp_path / "source"
    source.mkdir()
    manifest = {
        "source": str(source),
        "fingerprint": "source-v1",
        "meta": {"episode_index": 17, "frame_count": 60, "fps": 10, "source_digest": "source-digest"},
    }
    checked = []
    monkeypatch.setattr(temporal_caption_snapshot, "caption_source_fingerprint", lambda *args: "source-v1")
    monkeypatch.setattr(temporal_caption_flora, "validate_flora", lambda value: checked.append(value.copy()))
    result = {"episode_index": 0, "frame_count": 60, "fps": 10, "source_digest": "backend-digest"}
    worker.finish(store, job, manifest, result)
    saved = json.loads((tmp_path / "jobs" / job["id"] / "output" / "caption.json").read_text())
    assert saved["episode_index"] == 17
    assert saved["source_digest"] == "source-digest"
    assert saved["dataset_key"] == f"api/{job['id']}"
    assert checked and store.get(job["id"])["status"] == "completed"
    monkeypatch.setattr(temporal_caption_snapshot, "caption_source_fingerprint", lambda *args: "source-v2")
    with pytest.raises(ValueError, match="changed"):
        worker.finish(store, job, manifest, result)


def test_selects_only_host_with_available_capacity(tmp_path):
    store = Store(tmp_path)
    job, _ = store.reserve("owner", "route-to-a100")
    store.update(job["id"], status="running", stage="waiting")
    routes = Routes(tmp_path)
    selected = routes.select(
        job["id"],
        {
            "H100": Capacity((), (), (), 8),
            "A100": Capacity((3,), (), (), 8),
        },
    )
    assert selected.backend == "A100"
