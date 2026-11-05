"""Standalone API contracts: isolation, resumability, durable dispatch, and shared GPU locks."""

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from services.flora_api.app import create_app
from services.flora_api.store import QueueFullError, Store
from services.flora_api.worker import recover


@pytest.fixture
def api(tmp_path, monkeypatch):
    from types import SimpleNamespace

    # Admission tests must not depend on the developer machine's free disk space.
    monkeypatch.setattr("services.flora_api.app.shutil.disk_usage", lambda _: SimpleNamespace(free=10**12))
    keys = tmp_path / "keys.json"
    tokens = {name: name * 40 for name in ("a", "b")}
    keys.write_text(
        json.dumps(
            [
                {"id": name, "sha256": hashlib.sha256(token.encode()).hexdigest(), "datasets": []}
                for name, token in tokens.items()
            ]
        )
    )
    monkeypatch.setenv("FLORA_API_KEYS", str(keys))
    app = create_app(tmp_path / "state", keys)
    app.config["TESTING"] = True
    return (
        app.test_client(),
        tmp_path / "state",
        keys,
        {name: {"Authorization": f"Bearer {token}"} for name, token in tokens.items()},
    )


def upload_video(client, headers):
    return client.post(
        "/v1/flora/jobs",
        headers=headers,
        data={
            "instruction": "Move an object",
            "left_video": (io.BytesIO(b"0000ftyp0000test"), "left.mp4"),
            "right_video": (io.BytesIO(b"0000ftyp0000test"), "right.mp4"),
        },
    )


def test_auth_isolation_revocation_and_idempotency(api):
    client, state, keys, headers = api
    assert client.get("/v1/flora/datasets").status_code == 401
    assert (
        client.get("/v1/flora/datasets", headers={"Authorization": "Bearer " + "x" * 40}).status_code == 401
    )
    first = upload_video(client, {**headers["a"], "Idempotency-Key": "same"})
    assert first.status_code == 202
    job = first.json["id"]
    repeated = upload_video(client, {**headers["a"], "Idempotency-Key": "same"})
    assert repeated.status_code == 200 and repeated.json["id"] == job
    assert client.get(f"/v1/flora/jobs/{job}", headers=headers["b"]).status_code == 404
    assert client.get(f"/v1/flora/jobs/{job}/result", headers=headers["a"]).status_code == 409
    assert Store(state).get(job)["status"] == "queued"
    records = json.loads(keys.read_text())
    records[0]["enabled"] = False
    keys.write_text(json.dumps(records))
    assert client.get(f"/v1/flora/jobs/{job}", headers=headers["a"]).status_code == 401


def test_missing_dataset_file_reaches_child_parent_and_owner_only(api, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from services.flora_api.worker import execute

    client, state, _, headers = api
    store = Store(state)
    parent, _ = store.reserve("a", "diagnostic", "dataset")
    store.add_children(parent, [0])
    directory = state / "jobs" / parent["id"]
    directory.mkdir(parents=True)
    (directory / "input.json").write_text(json.dumps({"dataset_id": "test"}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    monkeypatch.setenv("FLORA_API_DATASETS", str(catalog))
    source = tmp_path / "private-upload" / "source"
    monkeypatch.setattr("services.flora_api.worker.shutil.disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr("services.flora_api.datasets.resolve_dataset", lambda *args: source)
    monkeypatch.setattr(
        "lerobot.data_platform.temporal_caption_snapshot.caption_source_fingerprint", lambda *a: "same"
    )

    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file", str(source / "meta/episodes_stats.jsonl"))

    monkeypatch.setattr("lerobot.data_platform.temporal_caption.prepare", missing)
    child = store.claim()
    execute(store, child, lambda *a, **k: pytest.fail("Inference must not run"))
    response = client.get(f"/v1/flora/jobs/{child['id']}", headers=headers["a"])
    assert response.json["stage"] == "preparing"
    assert response.json["error"] == "FileNotFoundError: Missing dataset file: meta/episodes_stats.jsonl"
    assert str(tmp_path) not in response.text
    result = client.get(f"/v1/flora/jobs/{parent['id']}", headers=headers["a"]).json
    assert result["status"] == "failed" and "episode 0 [preparing]" in result["error"]
    assert "meta/episodes_stats.jsonl" in result["error"]
    assert client.get(f"/v1/flora/jobs/{parent['id']}", headers=headers["b"]).status_code == 404


def test_public_errors_preserve_actionable_validation_and_hide_internal_details(tmp_path):
    from services.flora_api.errors import public_error

    expected = "FLORA requires left/right wrist views and an optional scene view"
    assert public_error(ValueError(expected)) == "ValueError: " + expected
    error = ValueError("bad /etc/private/config.json token=private-token Bearer private-bearer\x1b\n")
    result = public_error(error)
    assert all(
        secret not in result for secret in ("/etc/private", "private-token", "private-bearer", "\x1b", "\n")
    )
    missing = FileNotFoundError(2, "missing", "/etc/flora-api/runtime.json")
    assert "/etc" not in public_error(missing, tmp_path)
    assert "service file" in public_error(missing, tmp_path)
    assert "private detail" not in public_error(RuntimeError("private detail"))
    with pytest.raises(json.JSONDecodeError) as malformed:
        json.loads('{"private":')
    assert public_error(malformed.value).startswith("Invalid JSON data at line")


def test_dataset_client_prints_failures_and_keeps_successful_results(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from services.flora_api import dataset_client

    monkeypatch.setenv("FLORA_API_KEY", "test-key")
    monkeypatch.setattr(
        "sys.argv",
        ["client", "--url", "https://example.test", "--server-dataset", "demo", "--output", str(tmp_path)],
    )
    (tmp_path / "submission.json").write_text(
        json.dumps(
            {
                "url": "https://example.test",
                "source": "demo",
                "episodes": "all",
                "job_id": "parent",
            }
        )
    )
    failure = {
        "id": "bad",
        "episode_index": 1,
        "stage": "preparing",
        "status": "failed",
        "error": "Missing dataset file: meta/episodes_stats.jsonl",
    }
    calls = []

    def request(method, url, **kwargs):
        calls.append(method)
        if url.endswith("/parent"):
            value = {"status": "failed", "counts": {"completed": 1, "failed": 1}}
        elif url.endswith("after=-1"):
            value = {
                "jobs": [{"id": "good", "episode_index": 0, "status": "completed"}, failure],
                "next_after": 1,
            }
        elif url.endswith("/good/result"):
            value = {"existing_result": True}
        else:
            value = {"jobs": []}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)

    monkeypatch.setattr(
        dataset_client.requests, "Session", lambda: SimpleNamespace(headers={}, request=request)
    )
    with pytest.raises(SystemExit) as stopped:
        dataset_client.main()
    assert stopped.value.code == 1
    assert "Episode 1 FAILED [preparing] (job bad): Missing dataset file" in capsys.readouterr().out
    assert json.loads((tmp_path / "summary.json").read_text())["failures"] == [failure]
    assert json.loads((tmp_path / "episode_000000.json").read_text()) == {"existing_result": True}
    assert set(calls) == {"GET"}


def test_low_disk_rejects_upload_before_admission(api, monkeypatch):
    from types import SimpleNamespace

    client, state, _, headers = api
    monkeypatch.setattr("services.flora_api.app.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    assert upload_video(client, headers["a"]).status_code == 503
    assert Store(state).claim() is None


def test_bad_upload_releases_admission_and_cleans_files(api):
    client, state, _, headers = api
    for _ in range(5):
        response = client.post("/v1/flora/jobs", headers=headers["a"], data={"instruction": "x"})
        assert response.status_code == 400
        assert not (state / "jobs" / response.json["id"]).exists()
    assert upload_video(client, headers["a"]).status_code == 202


def test_large_multipart_video_streams_to_disk(api):
    client, _, _, headers = api
    response = client.post(
        "/v1/flora/jobs",
        headers=headers["a"],
        data={
            "instruction": "test",
            "left_video": (io.BytesIO(b"0000ftyp" + b"x" * 2_000_000), "left.mp4"),
            "right_video": (io.BytesIO(b"0000ftyp" + b"x" * 2_000_000), "right.mp4"),
        },
    )
    assert response.status_code == 202


def test_admission_and_parallel_claims_are_transactional(tmp_path):
    store = Store(tmp_path)
    for _ in range(4):
        job, _ = store.reserve("a", None)
        store.update(job["id"], status="queued", stage="queued")
    with pytest.raises(QueueFullError):
        store.reserve("a", None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda _: Store(tmp_path).claim(), range(8)))
    assert len({j["id"] for j in jobs if j}) == 4


def test_dataset_upload_hash_paths_and_sealing(api):
    client, _, _, headers = api
    data = json.dumps({"codebase_version": "v2.1", "total_episodes": 8}).encode()
    manifest = [{"path": "meta/info.json", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}]
    invalid = [{**manifest[0], "path": "meta/../../secret"}]
    assert client.post("/v1/flora/datasets", headers=headers["a"], json={"files": invalid}).status_code == 400
    created = client.post("/v1/flora/datasets", headers=headers["a"], json={"files": manifest})
    assert created.status_code == 201
    dataset = created.json["id"]
    path = f"/v1/flora/datasets/{dataset}"
    assert client.get(path, headers=headers["b"]).status_code == 404
    assert client.post(path + "/complete", headers=headers["a"], json={}).status_code == 409
    assert (
        client.put(path + "/files/meta/info.json", headers=headers["a"], data=b"x" * len(data)).status_code
        == 400
    )
    assert client.get(path, headers=headers["a"]).json["missing"] == ["meta/info.json"]
    assert client.put(path + "/files/meta/info.json", headers=headers["a"], data=data).status_code == 200
    assert client.post(path + "/complete", headers=headers["a"], json={}).status_code == 200
    assert client.put(path + "/files/meta/info.json", headers=headers["a"], data=data).status_code == 409
    response = client.post(
        "/v1/flora/dataset-jobs",
        headers={**headers["a"], "Idempotency-Key": "batch"},
        json={"dataset_id": dataset},
    )
    assert response.status_code == 202
    assert response.json["counts"] == {"queued": 8}
    parent = response.json["id"]
    assert client.get(f"/v1/flora/jobs/{parent}/result", headers=headers["a"]).status_code == 409
    children = client.get(f"/v1/flora/jobs/{parent}/episodes", headers=headers["a"]).json["jobs"]
    assert [child["episode_index"] for child in children] == list(range(8))
    assert client.get(f"/v1/flora/jobs/{parent}/episodes", headers=headers["b"]).status_code == 404


def test_server_dataset_scopes(api, tmp_path, monkeypatch):
    client, _, keys, headers = api
    source = tmp_path / "dataset"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0", "total_episodes": 2}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"demo": str(source)}))
    monkeypatch.setenv("FLORA_API_DATASETS", str(catalog))
    body = {"dataset_id": "server:demo"}
    assert client.post("/v1/flora/dataset-jobs", json=body, headers=headers["a"]).status_code == 400
    records = json.loads(keys.read_text())
    records[0]["datasets"] = ["demo"]
    keys.write_text(json.dumps(records))
    assert client.get("/v1/flora/datasets", headers=headers["a"]).json["datasets"][0]["id"] == "server:demo"
    response = client.post(
        "/v1/flora/dataset-jobs", json=body, headers={**headers["a"], "Idempotency-Key": "server-batch"}
    )
    assert response.status_code == 202 and response.json["counts"] == {"queued": 2}


def test_recovery_preserves_complete_outputs_and_does_not_repeat_inference(tmp_path):
    store = Store(tmp_path)
    parent, _ = store.reserve("a", "parent", "dataset")
    store.add_children(parent, [0, 1])
    first, second = store.claim(), store.claim()
    result = tmp_path / "jobs" / first["id"] / "output" / "caption.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"valid": true}')
    (result.parent / "completed.json").write_text(
        json.dumps({"sha256": hashlib.sha256(result.read_bytes()).hexdigest()})
    )
    recover(Store(tmp_path), lambda result: result["valid"])
    assert store.get(first["id"])["status"] == "completed"
    assert store.get(second["id"])["status"] == "failed"
    assert store.get(parent["id"])["counts"] == {"completed": 1, "failed": 1}
    assert store.claim() is None
    assert result.read_text() == '{"valid": true}'


def test_shared_platform_gpu_lock_is_respected(tmp_path, monkeypatch):
    import fcntl
    from types import SimpleNamespace

    from lerobot.data_platform.temporal_caption_flora import FloraRuntime, acquire_gpu

    config = FloraRuntime(
        Path("/python"), Path("/python"), Path("/code"), Path("/weights"), tmp_path, gpus=(0, 1)
    )
    monkeypatch.setattr(
        "lerobot.data_platform.temporal_caption_flora.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="81920,81000"),
    )
    monkeypatch.setattr("lerobot.data_platform.temporal_caption_flora._gpu_has_processes", lambda _: False)
    with (tmp_path / "gpu-0.lock").open("a") as platform_lock:
        fcntl.flock(platform_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with acquire_gpu(config, lambda _: None) as allocated:
            assert allocated.gpu == 1
            with (tmp_path / "gpu-1.lock").open("a") as contender, pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_video_normalization_retains_frame_timing(tmp_path):
    import shutil
    import subprocess

    from services.flora_api.worker import inspect_video, prepare

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for the real video fixture")
    for role in ("left", "right"):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x48:rate=10",
                "-frames:v",
                "10",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                str(tmp_path / f"{role}.mp4"),
            ],
            check=True,
        )
    (tmp_path / "input.json").write_text(json.dumps({"roles": ["left", "right"], "instruction": "test"}))
    bundle = prepare(tmp_path)
    assert inspect_video(bundle / "camera-videos/0.mp4").frames == 10
    metadata = json.loads((bundle / "episode.json").read_text())
    assert metadata["fps"] == 10 and metadata["include_signals"] is False


@pytest.mark.parametrize(
    "roles,accepted",
    [
        (["main"], True),
        (["left", "right"], True),
        (["left", "right", "main"], True),
        (["left"], False),
        (["right", "main"], False),
        ([], False),
    ],
)
def test_upload_camera_layouts(api, roles, accepted):
    client, state, _, headers = api
    response = client.post(
        "/v1/flora/jobs",
        headers=headers["a"],
        data={
            "instruction": "Move",
            **{f"{role}_video": (io.BytesIO(b"0000ftyp0000test"), f"{role}.mp4") for role in roles},
        },
    )
    assert response.status_code == (202 if accepted else 400)
    if accepted:
        settings = json.loads((state / "jobs" / response.json["id"] / "input.json").read_text())
        assert set(settings["roles"]) == set(roles)


def test_single_video_normalization_keeps_one_stream(tmp_path):
    import subprocess

    from services.flora_api.worker import inspect_video, prepare

    source = tmp_path / "main.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x16:r=10",
            "-frames:v",
            "10",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(source),
        ],
        check=True,
    )
    (tmp_path / "input.json").write_text(json.dumps({"instruction": "Move", "roles": ["main"]}))
    original = source.read_bytes()
    bundle = prepare(tmp_path)
    meta = json.loads((bundle / "episode.json").read_text())
    assert meta["cameras"] == ["scene"] and meta["include_signals"] is False
    assert len(list((bundle / "camera-videos").glob("*.mp4"))) == 1
    assert inspect_video(bundle / "camera-videos/0.mp4") == inspect_video(source)
    assert source.read_bytes() == original


def test_large_dataset_manifest_uses_available_disk_not_fixed_size_caps(api, monkeypatch):
    from types import SimpleNamespace

    client, _, _, headers = api
    monkeypatch.setattr(
        "services.flora_api.datasets.shutil.disk_usage", lambda _: SimpleNamespace(free=4 * 1024**4)
    )
    manifest = [
        {"path": "meta/info.json", "size": 1, "sha256": "a" * 64},
        {"path": "videos/large.mp4", "size": 1024**4 + 1, "sha256": "b" * 64},
    ]
    assert (
        client.post("/v1/flora/datasets", headers=headers["a"], json={"files": manifest}).status_code == 201
    )
    monkeypatch.setattr("services.flora_api.datasets.shutil.disk_usage", lambda _: SimpleNamespace(free=100))
    assert (
        client.post("/v1/flora/datasets", headers=headers["a"], json={"files": manifest}).status_code == 507
    )


def test_video_probe_accepts_long_high_resolution_and_still_checks_timestamps(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from services.flora_api.worker import inspect_video

    frames = [{"best_effort_timestamp_time": str(i / 60)} for i in range(7201)]
    payload = {"streams": [{"avg_frame_rate": "60/1", "width": 3840, "height": 2160}], "frames": frames}
    monkeypatch.setattr(
        "services.flora_api.worker.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps(payload)),
    )
    assert inspect_video(tmp_path / "video.mp4").frames == 7201
    frames[100]["best_effort_timestamp_time"] = "0"
    with pytest.raises(ValueError, match="constant-frame-rate"):
        inspect_video(tmp_path / "video.mp4")


def test_streamed_multipart_has_correct_length_and_server_accepts_it(api, tmp_path):
    import requests

    from services.flora_api.client import MultipartUpload

    client, state, _, headers = api
    video = tmp_path / "video.mp4"
    video.write_bytes(b"0000ftyp" + b"x" * 2_000_000)
    body = MultipartUpload("移动物体", [("main", video)])
    prepared = requests.Request("POST", "https://example.test", data=body).prepare()
    assert int(prepared.headers["Content-Length"]) == len(body)
    chunks = list(body)
    assert sum(map(len, chunks)) == len(body)
    assert max(map(len, chunks)) <= 1024 * 1024
    response = client.post(
        "/v1/flora/jobs",
        headers=headers["a"],
        data=b"".join(chunks),
        content_type=f"multipart/form-data; boundary={body.boundary}",
    )
    assert response.status_code == 202
    assert (state / "jobs" / response.json["id"] / "main.mp4").read_bytes() == video.read_bytes()


def test_active_upload_heartbeat_prevents_stale_cleanup(tmp_path, monkeypatch):
    from services.flora_api import store as store_module

    now = [10000.0]
    monkeypatch.setattr(store_module.time, "time", lambda: now[0])
    store = Store(tmp_path)
    job, _ = store.reserve("a", None)
    now[0] += 7100
    store.touch_upload(job["id"])
    now[0] += 200
    assert store.claim() is None
    assert store.get(job["id"])["status"] == "uploading"
    now[0] += 7201
    store.claim()
    assert store.get(job["id"])["status"] == "failed"
    store.touch_upload(job["id"])
    assert store.get(job["id"])["status"] == "failed"


def test_large_declared_requests_are_not_rejected_by_previous_caps(api):
    client, _, _, headers = api
    first = upload_video(client, {**headers["a"], "Idempotency-Key": "large"})
    response = client.post(
        "/v1/flora/jobs",
        headers={**headers["a"], "Idempotency-Key": "large"},
        environ_overrides={"CONTENT_LENGTH": str(512 * 1024**2 + 1)},
    )
    assert response.status_code == 200 and response.json["id"] == first.json["id"]
    response = client.put(
        f"/v1/flora/datasets/{'0' * 32}/files/meta/info.json",
        headers=headers["a"],
        environ_overrides={"CONTENT_LENGTH": str(64 * 1024**3 + 1)},
    )
    assert response.status_code == 400


def test_real_video_longer_than_two_minutes_is_prepared(tmp_path):
    import shutil
    import subprocess

    from services.flora_api.worker import inspect_video, prepare

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "main.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x16:r=1",
            "-frames:v",
            "121",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(source),
        ],
        check=True,
    )
    (tmp_path / "input.json").write_text(json.dumps({"instruction": "Move", "roles": ["main"]}))
    bundle = prepare(tmp_path)
    assert inspect_video(bundle / "camera-videos/0.mp4").frames == 121
    assert json.loads((bundle / "episode.json").read_text())["frame_count"] == 121


def test_streaming_client_uploads_over_http(api, tmp_path):
    import threading

    import requests
    from werkzeug.serving import make_server

    from services.flora_api.client import MultipartUpload

    client, state, _, headers = api
    video = tmp_path / "main.mp4"
    video.write_bytes(b"0000ftyp" + b"x" * 2_000_000)
    body = MultipartUpload("Move", [("main", video)])
    server = make_server("127.0.0.1", 0, client.application, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = requests.post(
            f"http://127.0.0.1:{server.server_port}/v1/flora/jobs",
            headers={**headers["a"], "Content-Type": f"multipart/form-data; boundary={body.boundary}"},
            data=body,
            timeout=10,
        )
        assert response.status_code == 202
        assert (state / "jobs" / response.json()["id"] / "main.mp4").read_bytes() == video.read_bytes()
    finally:
        server.shutdown()
        thread.join(timeout=5)
