"""Metric examples, immutable human reviews, permissions and real browser editing."""

import copy
import json
from types import SimpleNamespace

import pytest

from lerobot.data_platform.evaluation.metrics import (
    caption_quality,
    dense_counts,
    dense_metrics,
    evaluate,
    matching,
    segmentation,
)
from lerobot.data_platform.evaluation.review import EvaluationStore, digest, draft
from lerobot.data_platform.job_management import JobConflictError
from lerobot.data_platform.lifecycle import LifecycleStore
from lerobot.data_platform.lifecycle_repository import SQLiteLifecycleRepository
from lerobot.data_platform.temporal_caption import artifact_key
from tests.datasets.test_temporal_caption_flora import flora_artifact


def segment(a, b, caption="Move the block"):
    return {"start_s": a, "end_s": b, "caption": caption}


def test_matching_one_to_one_thresholds_and_joint_is_not_product():
    pred = [segment(3, 7), segment(3, 7)]
    gold = [segment(2, 6)]
    assert segmentation(pred, gold, 0.5)["f1"] == pytest.approx(2 / 3)
    assert segmentation(pred, gold, 0.75)["f1"] == 0
    assert len(matching(pred, gold)) == 1
    # Maximum-weight assignment may differ from a greedy or caption-blind match.
    pred = [segment(0, 4), segment(0, 4)]
    gold = [segment(0, 4), segment(0, 4)]
    assert matching(pred, gold, scores={"0:0": 0.9, "0:1": 0.8, "1:0": 0.8, "1:1": 0.1}) == [(0, 1), (1, 0)]


def test_caption_facts_group_normalization_error_family_and_paraphrase():
    ref = {
        "facts": [
            {"group": "action", "text": "Lift"},
            {"group": "action", "text": "Place"},
            {"group": "identity", "text": "Red block"},
        ]
    }
    assert caption_quality(ref, {"covered": ["0", "1", "2"], "errors": []}) == 1
    assert caption_quality(ref, {"covered": ["0"], "errors": []}) == 0.5
    error = {"group": "identity", "family": "wrong object", "evidence": "Blue cube visible"}
    a = {"covered": ["0", "1"], "errors": [error]}
    assert caption_quality(ref, a) == pytest.approx(1 / (1.25 + 0.3125))
    assert caption_quality(ref, {**a, "errors": [error, error]}) == caption_quality(ref, a)


def test_dense_native_fps_severity_missing_and_absent_failures():
    ref = [{"start_s": 0, "end_s": 2, "label": 3}, {"start_s": 0.5, "end_s": 1.5, "label": 1}]
    ten, missing = dense_counts([3] * 10 + [1] * 10, ref, 2, 10)
    thirty, _ = dense_counts([3] * 30 + [1] * 30, ref, 2, 30)
    assert ten == thirty and missing == 0
    result = dense_metrics(ten)
    assert result["samples"] == 2 and result["failure_recall"] == 1
    matrix, _ = dense_counts([], [{"start_s": 0, "end_s": 2, "label": 3}], 2, 10)
    metric = dense_metrics(matrix)
    assert metric["prediction_coverage"] == 0 and metric["macro_f1"] == 0 and metric["failure_recall"] is None


def seed():
    value = draft(flora_artifact(), {"task_hint": ["Move block"], "cameras": ["scene"]})
    value.update(source_run="run", cameras=["0"], camera_names=["scene"])
    for q in value["reference"]["quality"]:
        q["reason"] = "Grasp missed; visible in video"
    return value


def test_review_revisions_invalidation_and_no_unreviewed_scores(tmp_path):
    store = EvaluationStore(SQLiteLifecycleRepository(tmp_path))
    store.create("loc", [seed()], "operator")
    value = store.get("loc", 0)
    assert evaluate([value])["approved"] == 0
    with pytest.raises(ValueError, match="confirm"):
        store.change("loc", 0, 1, "human", "approve", {"reference": value["reference"]})
    value = store.change(
        "loc",
        0,
        1,
        "human",
        "approve",
        {"reference": value["reference"], "confirmed": ["segments", "captions", "quality", "outcome"]},
    )
    assert value["approved_by"] == "human"
    metrics = evaluate([value])
    assert metrics["segmentation_f1_50"]["value"] == 1
    assert metrics["outcome_mae"]["value"] == 0
    assert metrics["joint_f1"]["value"] is None and metrics["oracle_caption_quality"]["value"] is None
    with pytest.raises(JobConflictError):
        store.change("loc", 0, 1, "other", "save", {"reference": value["reference"]})
    value = store.change(
        "loc",
        0,
        2,
        "human",
        "assess",
        {"pair": "0:0", "assessment": {"covered": ["0"], "errors": [], "reason": "Correct visible action"}},
    )
    changed = copy.deepcopy(value["reference"])
    changed["outcome"] = 2
    value = store.change("loc", 0, 3, "human", "save", {"reference": changed})
    assert value["status"] == "draft" and value["approved_by"] is None and value["assessments"] == {}
    assert store.get("loc", 0, 2)["status"] == "approved"
    assert store.repository.get("evaluation_seeds", "loc:0")["reference"]["outcome"] == 4


def test_reference_reject_nan_gap_duplicate_facts_and_oracle_mismatch(tmp_path):
    store = EvaluationStore(SQLiteLifecycleRepository(tmp_path))
    store.create("loc", [seed()], "op")
    value = store.get("loc", 0)
    ref = copy.deepcopy(value["reference"])
    ref["segments"][0]["end_s"] = float("nan")
    with pytest.raises(ValueError):
        store.change("loc", 0, 1, "human", "save", {"reference": ref})
    ref = copy.deepcopy(value["reference"])
    ref["quality"][0]["start_s"] = 0.1
    with pytest.raises(ValueError, match="gap"):
        store.change("loc", 0, 1, "human", "approve", {"reference": ref})
    value = store.change(
        "loc",
        0,
        1,
        "human",
        "approve",
        {"reference": value["reference"], "confirmed": ["segments", "captions", "quality", "outcome"]},
    )
    with pytest.raises(ValueError):
        store.change(
            "loc",
            0,
            2,
            "human",
            "assess",
            {"pair": "0:0", "assessment": {"covered": ["0", "0"], "errors": [], "reason": "x"}},
        )
    oracle = {
        "reference_sha256": "old",
        "model": "FLORA",
        "provenance": "Native reference-window inference",
        "segments": value["prediction"]["segments"],
    }
    with pytest.raises(JobConflictError):
        store.change("loc", 0, 2, "human", "oracle-import", {"oracle": oracle})
    oracle["reference_sha256"] = digest(value["reference"])
    value = store.change("loc", 0, 2, "human", "oracle-import", {"oracle": oracle})
    for i in range(3):
        value = store.change(
            "loc",
            0,
            value["revision"],
            "human",
            "oracle-assess",
            {
                "pair": str(i),
                "assessment": {"covered": ["0"], "errors": [], "reason": "Verified semantic equivalence"},
            },
        )
    assert evaluate([value])["oracle_caption_quality"]["value"] == 1


@pytest.fixture
def evaluation_remote(tmp_path, monkeypatch):
    from lerobot.data_platform.routes.temporal_caption import register_temporal_caption_routes
    from tests.datasets.test_control_plane import _app, _bootstrap, _store

    store = _store(tmp_path)
    app = _app(tmp_path, store)
    ledger = LifecycleStore(tmp_path / "ledger")
    root = tmp_path / "captions"
    monkeypatch.setenv("DATA_PLATFORM_TEMPORAL_CAPTION_ROOT", str(root))
    register_temporal_caption_routes(app, SimpleNamespace(control_plane_store=store, lifecycle_store=ledger))
    client = app.test_client()
    _bootstrap(client)
    _, node = store.enroll_node(
        name="eval",
        hostname="localhost",
        allowed_roots=["/data"],
        writable_roots=["/data"],
        capabilities={"job_protocol": 2},
        enrollment_token="test",
        expected_token="test",
    )
    location = store.sync_locations(
        node["node_id"],
        [
            {
                "dataset_key": "node/source",
                "root": "/data/source",
                "output_dir": "/data/cache",
                "metadata": {"total_episodes": 1},
            }
        ],
    )[0]
    folder = root / artifact_key("node/source") / "run"
    folder.mkdir(parents=True)
    (folder / "caption.json").write_text(json.dumps(flora_artifact()))
    (folder / "episode.json").write_text(
        json.dumps({"task_hint": ["Move <script>bad()</script> block"], "cameras": ["scene"]})
    )
    (folder / "camera-videos").mkdir()
    (folder / "camera-videos/0.mp4").write_bytes(b"video")
    return SimpleNamespace(
        app=app,
        client=client,
        store=store,
        ledger=ledger,
        location=location,
        folder=folder,
        base=f"/api/control/locations/{location['location_id']}/evaluation",
    )


def test_api_pilot_scope_immutable_selection_and_viewer_denial(evaluation_remote):
    r = evaluation_remote
    c = r.client
    assert c.get(r.base).json["pilot"] is None
    result = c.post(r.base + "/pilot", json={"episodes": [0]})
    assert result.status_code == 201, result.json
    assert c.post(r.base + "/pilot", json={"episodes": [0]}).status_code == 409
    assert c.get(r.base + "/metrics").json["approved"] == 0
    assert c.get(r.base + "/episodes/0/video/0", headers={"Range": "bytes=0-1"}).status_code == 206
    assert c.get(r.base + "/episodes/0/video/99").status_code == 404
    r.store.register_user(username="evalviewer", password="test-password-123", role="viewer")
    viewer = r.app.test_client()
    assert (
        viewer.post(
            "/api/auth/login", json={"username": "evalviewer", "password": "test-password-123"}
        ).status_code
        == 200
    )
    assert viewer.get(r.base).status_code == 200
    assert viewer.post(r.base + "/episodes/0", json={}).status_code == 403
    assert r.app.test_client().get(r.base).status_code == 401


def test_evaluation_browser_edit_approve_score_and_mobile(evaluation_remote, tmp_path):
    import os
    import shutil
    import subprocess
    import threading

    from playwright.sync_api import expect, sync_playwright
    from werkzeug.serving import make_server

    r = evaluation_remote
    executable = os.getenv("DATA_PLATFORM_BROWSER_EXECUTABLE") or shutil.which("google-chrome")
    if not executable:
        pytest.skip("Chrome required")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:r=30",
            "-t",
            "0.4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(r.folder / "camera-videos/0.mp4"),
        ],
        check=True,
    )
    server = make_server("127.0.0.1", 0, r.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=executable, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1366, "height": 1000})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            assert page.request.post(
                origin + "/api/auth/login",
                data={"username": "platform-admin", "password": "strong-admin-password"},
            ).ok
            page.goto(origin + f"/remote/{r.location['location_id']}/temporal-caption/evaluation")
            page.locator("#create").click()
            expect(page.locator("#workbench")).to_be_visible()
            expect(page.locator("#instruction")).to_have_text("Move <script>bad()</script> block")
            page.wait_for_function("document.querySelector('video').readyState>=2")
            page.locator("#seg-caption").fill("Reach for the block, corrected by human")
            page.locator("#save").click()
            expect(page.locator("#status")).to_have_text("草稿已保存。")
            page.reload()
            expect(page.locator("#seg-caption")).to_have_value("Reach for the block, corrected by human")
            page.locator("[data-field=reason]").first.fill("The first grasp misses the object")
            for box in page.locator(".confirm").all():
                box.check()
            page.locator("#approve").click()
            expect(page.locator("#record-state")).to_have_text("参考已确认")
            page.locator("[data-tab=assessment]").click()
            expect(page.locator(".pair").first).to_be_visible()
            pair = page.locator(".pair").first
            pair.locator(".covered").check()
            pair.locator("textarea").fill("The action matches the visible reach")
            pair.locator(".save-assessment").click()
            expect(page.locator("#status")).to_contain_text("描述核对已保存")
            page.locator("[data-tab=metrics]").click()
            expect(page.locator("#metric-coverage")).to_contain_text("1 / 1")
            expect(page.locator("#metric-scores")).to_contain_text("100.0%")
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth+1")
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        thread.join()
