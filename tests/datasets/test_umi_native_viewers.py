"""Native result access, session isolation, and one-pass replay capture."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask, g

from lerobot.data_platform.control.store import RemoteJob
from lerobot.data_platform.datasets.lifecycle.service import LifecycleStore
from lerobot.data_platform.operators import get_operator
from lerobot.data_platform.routes.processing import register_processing_routes
from lerobot.data_platform.web.umi_viewers import ViewerSessions
from tests.datasets import test_processing

setup = test_processing.setup


@pytest.fixture
def native_app(setup, tmp_path):
    store, _, actor, node, location = setup
    ledger = LifecycleStore(tmp_path / "ledger")
    path = "runtime/native/replay_0/viewer.json"
    target = ledger.root / "operator-results/native-job/static" / path
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"schema_version": 1, "kind": "replay", "episodes": [0], "status_rows": []}))
    with store.sessions.begin() as db:
        db.add(
            RemoteJob(
                job_id="native-job",
                node_id=node["node_id"],
                location_id=location["location_id"],
                requested_by=actor["user_id"],
                operation="umi.replay",
                options={},
                status="done",
                result={"summary": {}, "artifacts": [{"path": path, "kind": "umi_replay_viewer"}]},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )
    app = Flask(__name__, template_folder=str(Path(__file__).parents[2] / "lerobot/data_platform/templates"))
    current = {"actor": actor}
    app.before_request(lambda: setattr(g, "control_plane_user", current["actor"]))
    register_processing_routes(app, SimpleNamespace(control_plane_store=store, lifecycle_store=ledger))
    yield app, app.test_client(), current, actor, path
    app.extensions["umi_viewer_sessions"].close()


def test_native_result_requires_owner_and_current_location_access(native_app, monkeypatch):
    app, client, current, actor, path = native_app
    assert client.get("/umi/results/native-job").status_code == 200
    current["actor"] = None
    assert client.get("/umi/results/native-job").status_code == 401
    current["actor"] = {**actor, "user_id": "another-user"}
    assert client.get("/umi/results/native-job").status_code == 404
    current["actor"] = actor
    manager = app.extensions["data_platform_processing"]

    def denied(*args, **kwargs):
        raise PermissionError("Data access was revoked")

    monkeypatch.setattr(manager, "_location", denied)
    assert client.get("/umi/results/native-job").status_code == 403
    assert client.get("/api/operator-results/native-job/" + path).status_code == 403


def test_viewer_rejects_arbitrary_paths_and_cross_origin_socket(native_app):
    app, client, current, actor, path = native_app
    for value in ("../../request.json", "/etc/passwd", "runtime/request.json"):
        response = client.post("/api/umi-viewer-sessions", json={"job_id": "native-job", "artifact": value})
        assert response.status_code == 400
    assert (
        client.get(
            "/api/umi-viewer-sessions/unknown/socket", headers={"Origin": "https://other.test"}
        ).status_code
        == 403
    )
    current["actor"] = None
    assert client.get("/api/umi-viewer-sessions/unknown/socket").status_code == 401


def test_empty_replay_has_diagnostics_but_does_not_launch(native_app):
    app, client, _, _, path = native_app
    manager = app.extensions["data_platform_processing"]
    target = manager.lifecycle.root / "operator-results/native-job/static" / path
    target.write_text(json.dumps({"episodes": [], "failures": [{"error": "initial IK"}]}))
    response = client.post("/api/umi-viewer-sessions", json={"job_id": "native-job", "artifact": path})
    assert response.status_code == 400
    assert "No executed frames" in response.json["error"]
    assert not app.extensions["umi_viewer_sessions"].sessions


def test_saved_operator_versions_remain_executable():
    for operation in ("umi.replay", "umi.cosmos_id"):
        assert get_operator(operation).version == "1.1.0"
        assert get_operator(operation).entry_point.endswith("_native")
        assert get_operator(operation, "1.0.0").entry_point != get_operator(operation).entry_point


def test_cpu_sessions_reuse_isolate_and_expire(tmp_path, monkeypatch):
    from lerobot.data_platform.operators.implementations.umi.runtime import LAUNDRY_COMMIT

    config = tmp_path / "runtime.json"
    config.write_text(
        json.dumps(
            {"python": sys.executable, "laundry_root": str(tmp_path), "laundry_commit": LAUNDRY_COMMIT}
        )
    )
    monkeypatch.setenv("DATA_PLATFORM_UMI_VIEWER_RUNTIME", str(config))
    calls = []

    class Process:
        pid = 123
        code = None

        def poll(self):
            return self.code

        def wait(self, timeout):
            self.code = -15

    def start(args, **kwargs):
        calls.append((args, kwargs))
        return Process()

    monkeypatch.setattr("lerobot.data_platform.web.umi_viewers.subprocess.Popen", start)
    monkeypatch.setattr("lerobot.data_platform.web.umi_viewers.os.killpg", lambda *args: None)
    sessions = ViewerSessions(tmp_path / "sessions", idle_seconds=10)
    try:
        first = sessions.start("alice", "job", "variant", tmp_path)
        assert sessions.start("alice", "job", "variant", tmp_path) is first
        second = sessions.start("bob", "job", "variant", tmp_path)
        assert second.session_id != first.session_id
        with pytest.raises(KeyError):
            sessions.get(first.session_id, "bob")
        assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "PYTHONPATH" not in calls[0][1]["env"]
        first.touched = time.monotonic() - 20
        sessions.expire()
        assert first.process.poll() == -15
        assert second.session_id in sessions.sessions
    finally:
        sessions.close()


@pytest.mark.parametrize("failure", [False, True])
def test_capture_persists_the_same_execution_including_partial_failure(tmp_path, monkeypatch, failure):
    import os

    import numpy as np

    root = os.environ.get("DATA_PLATFORM_LAUNDRY_ROOT")
    if root:
        monkeypatch.syspath_prepend(str(Path(root) / "src"))
    pytest.importorskip("laundry.umi_replay.replay")
    from lerobot.data_platform.operators.implementations.umi import native_reports

    calls = []

    def generate(controller, reference, source, args, *, on_frame, diagnostics):
        calls.append(source.episode_index)
        for frame in range(2):
            on_frame(frame, np.array([frame, 0.0]), {"left": np.eye(4)}, {"left": np.eye(4)})
        if failure:
            raise ValueError("stopped at safety boundary")
        return {"left": {"position_m": np.zeros(2)}}

    screen = SimpleNamespace(generate_replay=generate)
    monkeypatch.setattr(native_reports, "copy_evaluation_subset", lambda *args: None)
    finish = native_reports.install_replay_capture(screen, tmp_path)
    try:
        args = (SimpleNamespace(driven_sides=["left"]), {}, SimpleNamespace(episode_index=7), None)
        if failure:
            with pytest.raises(ValueError, match="safety"):
                screen.generate_replay(*args)
        else:
            screen.generate_replay(*args)
    finally:
        finish(tmp_path)
    assert calls == [7]
    assert screen.generate_replay is generate
    with np.load(tmp_path / "episode_000007.npz", allow_pickle=False) as data:
        assert data["joints"].tolist() == [[0, 0], [1, 0]]
    manifest = json.loads((tmp_path / "viewer.json").read_text())
    assert manifest["episodes"] == [7]
    assert bool(manifest["failures"]) == failure


def test_native_cache_binds_inputs_parameters_and_artifact_bytes(tmp_path, monkeypatch):
    from lerobot.data_platform.operators.implementations.umi.diagnostics import file_hash
    from lerobot.data_platform.operators.implementations.umi.native_cache import identity, lookup
    from tests.datasets.test_quality_eef import make_eef

    source = make_eef(tmp_path / "source")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("robot configuration")
    config = {
        "native_result_cache_root": str(tmp_path / "cache"),
        "laundry_commit": "pinned",
        "urdf_path": str(urdf),
    }
    key, bound = identity([source], {}, "umi.replay", config)
    entry = tmp_path / "cache" / key
    evidence = entry / "evidence"
    evidence.mkdir(parents=True)
    result = evidence / "result.json"
    result.write_text('{"status":"reported"}')
    (entry / "cache.json").write_text(
        json.dumps(
            {
                "identity": bound,
                "provenance": {"origin": "external_run"},
                "files": {"result.json": file_hash(result)},
            }
        )
    )
    assert lookup([source], {}, "umi.replay", config)[1]["reused"]
    assert lookup([source], {"episodes": [0]}, "umi.replay", config) is None
    result.write_text('{"status":"changed"}')
    with pytest.raises(ValueError, match="evidence changed"):
        lookup([source], {}, "umi.replay", config)


def test_original_viser_browser_uses_cached_execution(native_app, monkeypatch, tmp_path):
    import os
    import shutil
    import threading

    from werkzeug.serving import make_server

    bundle = os.environ.get("DATA_PLATFORM_UMI_REPLAY_BUNDLE")
    runtime_config = os.environ.get("DATA_PLATFORM_UMI_VIEWER_RUNTIME")
    executable = os.environ.get("DATA_PLATFORM_BROWSER_EXECUTABLE")
    if not all([bundle, runtime_config, executable]):
        pytest.skip("Native Viser acceptance requires a saved bundle, CPU runtime and browser")
    playwright = pytest.importorskip("playwright.sync_api")
    app, _, _, _, path = native_app
    ledger = app.extensions["data_platform_processing"].lifecycle
    target = ledger.root / "operator-results/native-job/static" / path
    target.unlink()
    shutil.copytree(bundle, target.parent, dirs_exist_ok=True)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=executable, args=["--no-sandbox", "--enable-unsafe-swiftshader"]
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/umi/results/native-job")
            page.get_by_role("button", name="Open original replay", exact=True).click()
            page.wait_for_function("!document.getElementById('viewer').hidden", timeout=120000)
            frame = page.frame_locator("#viewer")
            play = frame.get_by_role("button", name="▶ 播放", exact=True)
            play.wait_for(timeout=60000)
            frame.get_by_role("checkbox", name="循环", exact=True).check()
            play.click()
            frame.get_by_role("button", name="⏸ 暂停", exact=True).wait_for()
            page.wait_for_timeout(500)
            frame.get_by_role("button", name="⏸ 暂停", exact=True).click()
            assert frame.locator("canvas").count() > 0
            assert frame.locator("img").count() >= 3
            page.screenshot(path=str(tmp_path / "native-viser.png"), full_page=True)
            page.get_by_role("button", name="Close viewer", exact=True).click()
            page.wait_for_function("document.getElementById('viewer').hidden")
            assert not errors, errors
            browser.close()
    finally:
        app.extensions["umi_viewer_sessions"].close()
        server.shutdown()


def test_cosmos_confirmation_binds_targets_and_same_input_diagnosis(setup):
    from lerobot.data_platform.datasets.lifecycle.service import content_digest
    from lerobot.data_platform.processing.service import location_identity
    from lerobot.data_platform.runtime.jobs import JobConflictError

    _, manager, actor, _, location = setup
    targets = [{"episode_index": 0, "part": "left", "start_frame": 2, "end_frame": 5}]
    step = SimpleNamespace(operator_id="umi.cosmos_id", parameters={"targets": targets})
    evidence = [SimpleNamespace(input_identity=location_identity(location))]
    receipt = manager._review_workspace(step, actor["user_id"], location, evidence, {})
    assert receipt["targets_digest"] == content_digest(targets)
    assert receipt["candidate_quality_approved"] is False
    with pytest.raises(JobConflictError):
        manager._review_workspace(step, actor["user_id"], location, [], {})
    evidence[0].input_identity = {}
    with pytest.raises(JobConflictError):
        manager._review_workspace(step, actor["user_id"], location, evidence, {})
