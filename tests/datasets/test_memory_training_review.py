"""Frozen training labels remain source-bound, scoped, and visibly incomplete."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import threading

import pytest
from werkzeug.serving import make_server

from scripts.build_memory_training_review import build_snapshot, canonical
from tests.datasets.test_memory_routes import memory_remote  # noqa: F401


def sample():
    record = {
        "episode_id": "episode-000000",
        "task": "Move",
        "facts": [],
        "checkpoints": [{"frame": 0, "memory": "block visible", "subtask": "approach"}],
        "review": {"status": "draft"},
    }
    selected = {
        "episode_index": 0,
        "frame_count": 30,
        "fps": 10,
        "run": "caption-0",
        "annotation_sha256": hashlib.sha256(b"caption").hexdigest(),
        "lineage_group": "recording",
        "split": "train",
        "task": "Move",
    }
    media = {
        "id": "episode-000000",
        "dataset_key": "fixture",
        "dataset_version_id": "source",
        "annotation_sha256": selected["annotation_sha256"],
        "episode_index": 0,
        "frame_count": 30,
        "fps": 10,
        "lineage_group": "recording",
        "cameras": ["image", "left_wrist_image", "right_wrist_image"],
    }
    payload = canonical({k: v for k, v in record.items() if k != "review"})
    evidence = {
        "record_sha256": payload,
        "checked_frames": [0],
        "receipts_sha256": "receipts",
        "student_input_protocol": 2,
        "complete": False,
    }
    stopped = {
        "episode-000000": {
            "accepted_checkpoints": 1,
            "planned_checkpoints": 3,
            "stopped_at": 10,
            "issues": {"frame": 10, "state": "needs_review", "issues": ["uncertain"]},
        }
    }
    ledger = {
        "schema": "mem_ledger/v1",
        "bundle_sha256": "bundle",
        "episodes": [record],
        "model_checks": {
            "episode-000000": {
                "status": "model_checked",
                "payload_sha256": payload,
                "checked_frames": [0],
                "receipts_sha256": "receipts",
                "bundle_sha256": "bundle",
            }
        },
    }
    proof = {
        "schema": "ee3_model_trial_verification/v1",
        "actual_model_requests": 0,
        "human_approvals": 0,
        "source_fingerprint": "source",
        "binding": {"manifest_sha256": "selection-hash"},
        "episodes": {"episode-000000": evidence},
        "stopped_episodes": stopped,
    }
    coverage = {
        "quality": "model_only_experiment",
        "human_approvals": 0,
        "requested_episodes": 1,
        "excluded_episodes": [],
        "stopped_episodes": copy.deepcopy(stopped),
        "included_checkpoints": 1,
        "complete_episodes": 0,
        "verification_sha256": "proof-hash",
    }
    selection = {"source_fingerprint": "source", "episodes": [selected]}
    bundle_manifest = {
        "schema": "mem_dataset_export/v1",
        "episode_count": 1,
        "files": {"episodes.jsonl": "episodes-hash"},
    }
    hashes = {
        "selection": "selection-hash",
        "verification": "proof-hash",
        "bundle_manifest": "bundle",
        "episodes": "episodes-hash",
    }
    return ledger, coverage, proof, selection, [media], bundle_manifest, hashes


def test_snapshot_rejects_changed_training_record_and_source():
    args = sample()
    snapshot = build_snapshot(*args)
    assert snapshot["episodes"][0]["stopped_at"] == 10
    assert snapshot["episodes"][0]["issues"]["issues"] == ["uncertain"]
    changed = copy.deepcopy(args)
    changed[0]["episodes"][0]["checkpoints"][0]["memory"] = "tampered"
    with pytest.raises(ValueError, match="mismatch"):
        build_snapshot(*changed)
    changed = copy.deepcopy(args)
    changed[4][0]["annotation_sha256"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        build_snapshot(*changed)


def test_review_camera_labels_match_video_layout():
    from lerobot.data_platform.datasets.camera_layout import review_camera_order

    assert review_camera_order(["image", "left_wrist_image", "right_wrist_image"]) == [
        "left_wrist_image",
        "image",
        "right_wrist_image",
    ]
    assert review_camera_order([]) == []


def install_fixture(remote):
    snapshot = build_snapshot(*sample())
    folder = remote.ledger.root / "memory-training-reviews" / remote.location["location_id"]
    folder.mkdir(parents=True)
    (folder / "snapshot.json").write_text(json.dumps(snapshot))
    remote.app.extensions["data_platform_memory_caption"] = lambda location, run: {
        "_memory_raw_bytes": b"caption",
        "source_fingerprint": "source",
        "episode_index": 0,
        "frame_count": 30,
        "fps": 10,
    }
    remote.app.extensions["data_platform_memory_video_cameras"] = lambda location, run: [
        "left_wrist_image",
        "image",
        "right_wrist_image",
    ]


def test_training_review_enforces_scope_and_source(memory_remote):  # noqa: F811
    from lerobot.data_platform.control.store import DataMutationScope

    remote = memory_remote
    install_fixture(remote)
    assert remote.client.get(remote.base + "/training-review").status_code == 200
    assert remote.client.get(remote.base + "/training-review/0").get_json()["stopped_at"] == 10
    assert remote.client.get(remote.base + "/training-review/1").status_code == 404
    remote.app.extensions["data_platform_memory_caption"] = lambda location, run: {
        "_memory_raw_bytes": b"wrong"
    }
    assert remote.client.get(remote.base + "/training-review/0").status_code == 409
    user = remote.store.register_user(username="scoped", password="synthetic-password", role="operator")
    with remote.store.sessions.begin() as session:
        session.add(DataMutationScope(user_id=user["user_id"], all_locations=False))
    remote.client.post("/api/auth/logout", json={})
    remote.client.post("/api/auth/login", json={"username": "scoped", "password": "synthetic-password"})
    assert remote.client.get(remote.base + "/training-review").status_code == 403
    assert remote.client.get(remote.base + "/training-review/0").status_code == 403


def test_training_review_browser_shows_stop(memory_remote, tmp_path):  # noqa: F811
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = (
        os.getenv("DATA_PLATFORM_BROWSER_EXECUTABLE")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
    )
    ffmpeg = shutil.which("ffmpeg")
    if not chrome or not ffmpeg:
        pytest.skip("Chrome and ffmpeg are required")
    remote = memory_remote
    install_fixture(remote)
    snapshot_file = (
        remote.ledger.root / "memory-training-reviews" / remote.location["location_id"] / "snapshot.json"
    )
    snapshot = json.loads(snapshot_file.read_text())
    second = copy.deepcopy(snapshot["episodes"][0])
    second.update(episode_index=1, caption_run="caption-1")
    snapshot["episodes"].append(second)
    snapshot_file.write_text(json.dumps(snapshot))
    remote.app.extensions["data_platform_memory_caption"] = lambda location, run: {
        "_memory_raw_bytes": b"caption",
        "source_fingerprint": "source",
        "episode_index": 1 if run == "caption-1" else 0,
        "frame_count": 30,
        "fps": 10,
    }
    video = tmp_path / "review.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=96x32:r=10:d=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    server = make_server("127.0.0.1", 0, remote.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = browser.new_page()
            page.route(
                "**/caption/video", lambda route: route.fulfill(path=str(video), content_type="video/mp4")
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            assert (
                page.request.post(
                    base + "/api/auth/login",
                    data={"username": "platform-admin", "password": "strong-admin-password"},
                ).status
                == 200
            )
            page.goto(base + f"/remote/{remote.location['location_id']}/memory/training-review?episode=1")
            playwright.expect(page.locator("#episodes")).to_have_value("1")
            playwright.expect(page.locator("#memory")).to_have_text("block visible")
            playwright.expect(page.locator("#coverage")).to_contain_text("1/3")
            playwright.expect(page.locator("#cameras")).to_have_text("Left wrist · Main · Right wrist")
            page.wait_for_function("document.querySelector('#video').readyState >= 2")
            page.locator("#video").evaluate("video => video.play()")
            page.wait_for_function("document.querySelector('#video').currentTime > 0.1")
            page.locator("#points button.pending").click()
            playwright.expect(page.locator("#memory")).to_contain_text("No accepted memory")
            playwright.expect(page.locator("#issues")).to_contain_text("uncertain")
            page.locator("#filter").select_option("complete")
            playwright.expect(page.locator("#status")).to_contain_text("No episodes")
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join()
