"""FLORA evidence integrity, executor selection, and attempt-scoped publication."""

import copy
import json

import pytest

from lerobot.data_platform.annotations.jobs import publish_run, validate_options, validate_source
from lerobot.data_platform.operators.implementations.caption.core import prepare
from lerobot.data_platform.operators.implementations.caption.flora import (
    build_artifact,
    camera_roles,
    disagreement_intervals,
    review_partition,
    validate_flora,
)
from lerobot.data_platform.routes.temporal_caption_jobs import caption_scheme_status
from tests.datasets import test_temporal_caption_jobs as job_fixtures


@pytest.fixture
def remote(tmp_path, monkeypatch):
    value = job_fixtures.remote.__wrapped__(tmp_path, monkeypatch)
    client, store, node, token, location, root = value
    location = store.sync_locations(
        node["node_id"],
        [
            {
                **location,
                "metadata": {
                    **location["metadata"],
                    "total_episodes": 1,
                    "features": {name: {"dtype": "image"} for name in ["left_wrist", "scene", "right_wrist"]},
                },
            }
        ],
    )[0]
    return client, store, node, token, location, root


def flora_artifact(key="node/source"):
    native = {
        "segments": [
            {"start_s": 0, "end_s": 0.1, "caption": "Reach", "status": "complete"},
            {"start_s": 0.15, "end_s": 0.3, "caption": "Grasp", "status": "complete"},
            {"start_s": 0.25, "end_s": 0.4, "caption": "Lift", "status": "complete"},
        ]
    }
    windows = [
        {"scores": [1] * 8 + [3] * 4, "output": "Window one: failure then normal"},
        {"scores": [0] * 3 + [2] * 5 + [3] * 4, "output": "Window two: suboptimal then normal"},
    ]
    reward = {
        "terminal": {"score": 4, "rationale": "Completed after retry"},
        "windows": windows,
        "merged_scores": [1] * 8 + [3] * 4,
        "merged_source_window": [0] * 12,
        "disagreement_intervals": disagreement_intervals(windows, 12),
        "human_reviewed": False,
        "alignment": "Test alignment",
    }
    return build_artifact({"episode_index": 0, "frame_count": 12, "fps": 30}, key, native, reward, [])


def write_run(path, key="node/source"):
    path.mkdir()
    (path / "caption.json").write_text(json.dumps(flora_artifact(key)))
    (path / "episode.json").write_text(
        json.dumps(
            {"episode_index": 0, "frame_count": 12, "fps": 30, "include_signals": False, "signals": []}
        )
    )
    (path / "video.mp4").write_bytes(b"review-video")
    return path


def test_native_gap_overlap_and_ceil_are_preserved():
    artifact = flora_artifact()
    segments = artifact["result"]["segments"]
    assert [(s["start_frame"], s["end_frame"]) for s in segments] == [(0, 3), (3, 5), (5, 8), (8, 9), (9, 12)]
    assert segments[1]["caption"] == "Unlabeled interval"
    assert segments[3]["native_segment_indices"] == [1, 2]
    assert artifact["native_segments"][1]["start_s"] == 0.15
    assert all(s["confidence"] == "not_reported" for s in segments)
    assert artifact["reward"]["disagreement_intervals"] == [{"start_frame": 3, "end_frame": 8}]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["reward"]["merged_scores"].pop(),
        lambda a: a["reward"]["windows"][0]["scores"].__setitem__(0, True),
        lambda a: a["reward"]["disagreement_intervals"].clear(),
        lambda a: a["reward"]["merged_source_window"].__setitem__(0, 20),
        lambda a: a["reward"]["terminal"].update(score=6),
        lambda a: a["reward"].update(human_reviewed=True),
        lambda a: a.update(include_signals=True),
        lambda a: a["native_segments"][0].update(end_s=float("nan")),
        lambda a: a["result"]["segments"][0].update(caption="Invented action"),
    ],
)
def test_reject_inconsistent_rewards_or_native_captions(mutation):
    artifact = flora_artifact()
    mutation(artifact)
    with pytest.raises(ValueError):
        validate_flora(artifact)


def test_reject_incomplete_native_results():
    native = copy.deepcopy(flora_artifact()["native_segments"])
    native[0]["status"] = "failed"
    with pytest.raises(ValueError, match="incomplete"):
        review_partition(native, 12, 30)


def test_visual_only_options_and_camera_identity():
    assert validate_options({"episode_index": 0, "scheme": "flora"})["include_signals"] is False
    with pytest.raises(ValueError, match="visual only"):
        validate_options({"episode_index": 0, "scheme": "flora", "include_signals": True})
    assert camera_roles(["image_right", "scene", "image_left"])["main"] == "scene"
    assert camera_roles(["right_wrist", "left_wrist"]) == {"left": "left_wrist", "right": "right_wrist"}
    for camera in ("image", "observation.images.main", "left_wrist"):
        assert camera_roles([camera]) == {"main": camera}
    for names in ([], ["a", "b", "c"], ["left", "scene"], ["left_right", "scene", "other"]):
        with pytest.raises(ValueError):
            camera_roles(names)


def test_flora_readiness_does_not_require_dashscope():
    capabilities = {
        "caption_flora_protocol": 1,
        "caption_flora_configured": True,
        "caption_configured": False,
        "caption_schemes": ["flora", "video_events"],
    }
    status = {s["id"]: s for s in caption_scheme_status(capabilities, None, {"role": "operator"})}
    assert status["flora"]["can_submit"]
    assert not status["video_events"]["can_submit"]
    assert not any(s["can_submit"] for s in caption_scheme_status(capabilities, None, {"role": "viewer"}))
    capabilities["caption_flora_protocol"] = 0
    assert not next(
        s for s in caption_scheme_status(capabilities, None, {"role": "operator"}) if s["id"] == "flora"
    )["can_submit"]


def test_mixed_flora_readiness_requires_upgraded_executor_and_disables_other_methods():
    capabilities = {
        "caption_flora_protocol": 1,
        "caption_flora_configured": True,
        "caption_configured": True,
        "caption_schemes": ["flora", "video_events"],
    }
    metadata = {
        "data_platform_mixed_episode_viewer": 1,
        "features": {name: {"dtype": "video"} for name in ["scene", "left_wrist", "right_wrist"]},
    }

    def status():
        return {s["id"]: s for s in caption_scheme_status(capabilities, None, {"role": "operator"}, metadata)}

    assert not status()["flora"]["can_submit"]
    capabilities["mixed_episode_flora_protocol"] = 1
    assert status()["flora"]["can_submit"]
    assert not status()["video_events"]["can_submit"]


def test_mixed_flora_queue_and_claim_require_compatible_agent(remote):
    client, store, node, _, location, _ = remote
    location = store.sync_locations(
        node["node_id"],
        [
            {
                **location,
                "metadata": {**location["metadata"], "data_platform_mixed_episode_viewer": 1},
            }
        ],
    )[0]
    current = next(n for n in store.list_nodes() if n["node_id"] == node["node_id"])
    caps = {
        **current["capabilities"],
        "caption_flora_protocol": 1,
        "caption_flora_configured": True,
        "caption_signal_options": True,
        "caption_schemes": ["flora"],
    }
    store.heartbeat(node["node_id"], capabilities=caps)
    api = f"/api/control/locations/{location['location_id']}/temporal-caption/jobs"
    body = {"episode_index": 0, "scheme": "flora"}
    headers = {"Idempotency-Key": "mixed-flora"}
    assert client.post(api, json=body, headers=headers).status_code == 409
    store.heartbeat(node["node_id"], capabilities={**caps, "mixed_episode_flora_protocol": 1})
    assert client.post(api, json=body, headers=headers).status_code == 202
    store.heartbeat(node["node_id"], capabilities=caps)
    assert store.claim_job(node["node_id"], worker_instance_id="old-worker") is None
    store.heartbeat(node["node_id"], capabilities={**caps, "mixed_episode_flora_protocol": 1})
    assert store.claim_job(node["node_id"], worker_instance_id="new-worker") is not None


@pytest.mark.parametrize(
    "cameras", [["image"], ["left_wrist", "right_wrist"], ["left_wrist", "scene", "right_wrist"]]
)
def test_flora_queue_publish_read_and_tamper_rejection(remote, tmp_path, cameras):
    client, store, node, token, location, root = remote
    location = store.sync_locations(
        node["node_id"],
        [
            {
                **location,
                "metadata": {
                    **location["metadata"],
                    "features": {name: {"dtype": "image"} for name in cameras},
                },
            }
        ],
    )[0]
    current = next(n for n in store.list_nodes() if n["node_id"] == node["node_id"])
    store.heartbeat(
        node["node_id"],
        capabilities={
            **current["capabilities"],
            "caption_configured": False,
            "caption_flora_protocol": 1,
            "caption_flora_configured": True,
            "caption_schemes": ["flora"],
            "caption_signal_options": True,
        },
    )
    api = f"/api/control/locations/{location['location_id']}/temporal-caption"
    response = client.post(
        api + "/jobs", json={"episode_index": 0, "scheme": "flora"}, headers={"Idempotency-Key": "flora"}
    )
    assert response.status_code == 202, response.get_json()
    job = response.get_json()["job"]
    assert job["options"]["include_signals"] is False
    claim = store.claim_job(node["node_id"], worker_instance_id="flora-worker")
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Job-Attempt": claim["execution"]["attempt_id"],
        "X-Job-Credential": claim["execution"]["credential"],
    }
    run = write_run(tmp_path / "run")
    endpoint = f"/api/agents/jobs/{job['job_id']}"
    for path in run.iterdir():
        assert (
            client.put(
                endpoint + f"/caption-artifacts/{path.name}", headers=headers, data=path.read_bytes()
            ).status_code
            == 200
        )
    completed = client.post(endpoint + "/complete", headers=headers, json={"status": "done"})
    assert completed.status_code == 200, completed.get_json()
    response = client.get(api + f"/{job['job_id']}/caption")
    assert response.status_code == 200
    assert response.get_json()["reward"]["disagreement_intervals"] == [{"start_frame": 3, "end_frame": 8}]
    saved = next(root.glob("*/*/caption.json"))
    value = json.loads(saved.read_text())
    value["reward"]["disagreement_intervals"] = []
    saved.write_text(json.dumps(value))
    assert client.get(api + f"/{job['job_id']}/caption").status_code == 422


def test_invalid_flora_never_publishes(tmp_path):
    run = write_run(tmp_path / "run")
    value = flora_artifact()
    value["reward"]["merged_scores"] = []
    (run / "caption.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        publish_run(run, tmp_path / "published", "node/source", "run")
    assert not (tmp_path / "published").exists()


def test_export_native_camera_video_keeps_resolution_and_source(tmp_path):
    import av

    from tests.datasets.test_temporal_caption import (
        test_export_reads_only_selected_episode_from_shared_v3_shard,
    )

    test_export_reads_only_selected_episode_from_shared_v3_shard(tmp_path)
    source = tmp_path / "source"
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "flora-export"
    prepare(source, 1, output, export_camera_videos=True)
    assert not (output / "frames").exists()
    assert not list(output.rglob("*.png"))
    with av.open(str(output / "camera-videos/0.mp4")) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == 2
    assert (frames[0].width, frames[0].height) == (8, 8)
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("local_page", [False, True], ids=["remote", "local"])
def test_browser_flora_review_seek_switch_mobile_and_readonly(remote, tmp_path, monkeypatch, local_page):
    """Exercise real HTML/JS and authenticated routes, not template substring checks."""
    import shutil
    import subprocess
    import threading
    from types import SimpleNamespace

    from werkzeug.serving import make_server

    from tests.datasets.test_temporal_caption_jobs import make_hybrid_run, make_run

    playwright = pytest.importorskip("playwright.sync_api")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    client, store, node, _, location, root = remote
    from lerobot.data_platform.ai_tools.service import AssistantService
    from lerobot.data_platform.routes.ai_assistant import register_ai_assistant_routes
    from lerobot.data_platform.routes.processing import register_processing_routes
    from lerobot.data_platform.routes.temporal_caption import register_temporal_caption_routes
    from tests.datasets.test_ai_assistant import FakeQwen
    from tests.datasets.test_ai_assistant_context import text_response
    from tests.datasets.test_control_plane import _app

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-key")
    ctx = SimpleNamespace(
        control_plane_store=store,
        lifecycle_store=None,
        repo_key=lambda key: key,
        static_dir_for_key=lambda key: tmp_path if key == location["dataset_key"] else None,
    )
    if local_page:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta/info.json").write_text(json.dumps(location["metadata"]))
        ctx.ensure_dataset_loaded = lambda _: (
            SimpleNamespace(root=tmp_path, meta=SimpleNamespace(total_episodes=2)),
            None,
        )
        monkeypatch.setattr(
            "lerobot.data_platform.runtime.local.agent_capabilities_for_local",
            lambda: {
                "caption_schemes": ["flora"],
                "caption_flora_protocol": 1,
                "caption_flora_configured": True,
            },
        )
    client = _app(tmp_path, store).test_client()
    register_temporal_caption_routes(client.application, ctx)
    register_processing_routes(client.application, ctx)
    register_ai_assistant_routes(client.application, ctx)
    fake = FakeQwen([text_response("Review conversation saved")])
    client.application.extensions["data_platform_assistant"] = AssistantService(
        client.application.extensions["data_platform_processing"], client_factory=lambda **_: fake
    )
    run = write_run(tmp_path / "browser-run")
    instruction = '把杯子放到右边。\nKeep <img src=x onerror="window.instructionInjected=true"> as text.'
    second_instruction = "Open the drawer."
    metadata = json.loads((run / "episode.json").read_text())
    metadata["task_hint"] = [instruction]
    (run / "episode.json").write_text(json.dumps(metadata))
    repaired = json.loads((run / "caption.json").read_text())
    repaired["coverage_repair"] = {
        "status": "draft",
        "original_run": "original-preserved",
        "human_reviewed": False,
        "intervals": [
            {
                "start_frame": 0,
                "end_frame": 3,
                "reuse": False,
                "needs_review": True,
                "uncertainty": ["Gripper contact is unclear"],
            }
        ],
    }
    (run / "caption.json").write_text(json.dumps(repaired))
    (run / "video.mp4").unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=192x64:r=30",
            "-frames:v",
            "12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(run / "video.mp4"),
        ],
        check=True,
    )
    publish_run(run, root, "node/source", "2f26744f-5a02-4f8e-8777-a106e8079fa7")
    second = tmp_path / "second-episode"
    shutil.copytree(run, second)
    for name in ("caption.json", "episode.json"):
        data = json.loads((second / name).read_text())
        data["episode_index"] = 1
        if name == "episode.json":
            data["task_hint"] = [second_instruction]
        (second / name).write_text(json.dumps(data))
    publish_run(second, root, "node/source", "second-episode")
    publish_run(make_hybrid_run(tmp_path / "legacy", key="node/source"), root, "node/source", "legacy")
    alternate = store.sync_locations(
        node["node_id"],
        [{**location, "root": str(tmp_path / "alternate"), "dataset_key": "node/alternate"}],
    )[0]
    publish_run(
        make_run(tmp_path / "alternate-run", key="node/alternate"),
        root,
        "node/alternate",
        "alternate-result",
    )
    store.register_user(username="flora-reader", password="reader-password", display_name="Reader")
    current = next(n for n in store.list_nodes() if n["node_id"] == node["node_id"])
    store.heartbeat(
        node["node_id"],
        capabilities={
            **current["capabilities"],
            "caption_flora_protocol": 1,
            "caption_flora_configured": True,
            "caption_schemes": ["flora"],
            "caption_signal_options": True,
        },
    )
    server = make_server("127.0.0.1", 0, client.application, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with playwright.sync_playwright() as driver:
            browser = driver.chromium.launch(
                headless=True, executable_path=shutil.which("google-chrome") or None
            )
            context = browser.new_context(viewport={"width": 1360, "height": 1000})
            assert context.request.post(
                origin + "/api/auth/login", data={"username": "flora-reader", "password": "reader-password"}
            ).ok
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            if not local_page:

                def memory_response(route):
                    if f"/locations/{location['location_id']}/" not in route.request.url:
                        route.fulfill(status=404, json={})
                        return
                    episode = int(route.request.url.rsplit("/", 1)[-1])
                    route.fulfill(
                        json={
                            "episode_index": episode,
                            "frame_count": 12,
                            "fps": 30,
                            "caption_run": "2f26744f-5a02-4f8e-8777-a106e8079fa7"
                            if episode == 0
                            else "second-episode",
                            "source_fingerprint": "source",
                            "stopped_at": 9,
                            "checkpoints": [
                                {"frame": 0, "memory": f"Episode {episode} first", "subtask": "Approach"},
                                {"frame": 5, "memory": f"Episode {episode} second", "subtask": "Grasp"},
                            ],
                            "issues": {"revisions": [{"revision": 0, "issues": ["Uncertain contact"]}]},
                        }
                    )

                page.route("**/memory/training-review/*", memory_response)
            route = (
                "/node/source/temporal-caption"
                if local_page
                else f"/remote/{location['location_id']}/temporal-caption"
            )
            page.goto(origin + route + "/review?scheme=flora")
            page.locator("#reward-review").wait_for(state="visible")
            page.wait_for_function("document.querySelector('#video').readyState >= 1")
            playwright.expect(page.locator("#original-instruction")).to_be_visible()
            playwright.expect(page.locator("#task")).to_have_text(instruction)
            assert not page.locator("#task img").count()
            assert page.evaluate("window.instructionInjected !== true")
            assert page.locator("#reward-tracks canvas").count() == 3
            assert page.locator("#reward-overall-score").inner_text() == "4"
            assert page.locator("#reward-values .reward-window-score").count() == 1
            assert page.locator("#reward-values strong").inner_text() == "1"
            assert page.locator("#reward-merged-score").inner_text() == "1 / 3"
            assert page.locator("#reward-score-conflict").is_hidden()
            assert "no human approval" in page.locator("#coverage-repair-status").inner_text()
            assert "original-preserved" in page.locator("#coverage-repair-status").inner_text()
            assert "Gripper contact is unclear" in page.locator("#segments").inner_text()
            expect = playwright.expect
            if not local_page:
                expect(page.locator("#training-memory-panel")).to_be_visible()
                expect(page.locator("#training-memory-value")).to_have_text("Episode 0 first")
                assert (
                    page.locator("#training-memory-panel").bounding_box()["x"]
                    > page.locator("#caption-panel").bounding_box()["x"]
                )
                page.locator("#video").evaluate("v => {v.currentTime = 6 / 30;}")
                expect(page.locator("#training-memory-value")).to_have_text("Episode 0 second")
                page.locator("#training-memory-jump").click()
                expect(page.locator("#training-memory-value")).to_contain_text("No accepted memory")
                expect(page.locator("#training-memory-issues")).to_contain_text("Uncertain contact")
                assert "episode=0" in page.locator("#training-memory-full").get_attribute("href")
                # A delayed response for episode 0 must not replace episode 1's memory.
                page.evaluate("""() => {
                  const original = window.fetch;
                  window.fetch = (...args) => String(args[0]).endsWith('/memory/training-review/0')
                    ? new Promise(resolve => {window.releaseDelayedMemory = () => {
                        window.fetch = original; original(...args).then(resolve);
                      };}) : original(...args);
                  loadTrainingMemory(generation, artifact);
                }""")
                page.wait_for_function("typeof window.releaseDelayedMemory === 'function'")
                page.get_by_role("button", name="Next episode", exact=True).click()
                expect(page.locator("#training-memory-value")).to_have_text("Episode 1 first")
                page.evaluate("window.releaseDelayedMemory()")
                expect(page.locator("#training-memory-value")).to_have_text("Episode 1 first")
                page.get_by_role("button", name="Previous episode", exact=True).click()
                expect(page.locator("#training-memory-value")).to_have_text("Episode 0 first")
                page.route(
                    "**/memory/training-review/0",
                    lambda route: route.fulfill(
                        json={
                            "episode_index": 0,
                            "frame_count": 12,
                            "fps": 30,
                            "caption_run": "different-run",
                            "source_fingerprint": "different-source",
                        }
                    ),
                )
                page.evaluate("() => {loadTrainingMemory(generation, artifact);}")
                page.wait_for_timeout(200)
                expect(page.locator("#training-memory-panel")).to_be_hidden()
                page.unroute("**/memory/training-review/0")
                page.evaluate("() => {loadTrainingMemory(generation, artifact);}")
                expect(page.locator("#training-memory-value")).to_have_text("Episode 0 first")
            page.locator("#caption-datasets > summary").click()
            expect(page.locator("#caption-datasets-list a")).to_have_count(2)
            page.route("**/api/temporal-caption/datasets", lambda r: r.fulfill(status=503, json={}))
            page.locator("#caption-datasets-refresh").click()
            expect(page.locator("#caption-datasets-status")).to_contain_text("503")
            expect(page.locator("#caption-datasets-list a")).to_have_count(2)
            page.unroute("**/api/temporal-caption/datasets")
            page.locator("#caption-datasets-refresh").click()
            expect(page.locator("#caption-datasets-status")).to_contain_text("2 / 2")
            page.locator("#caption-datasets-search").fill("no matching dataset")
            expect(page.locator("#caption-datasets-status")).to_contain_text("No matching")
            page.locator("#caption-datasets-search").fill("alternate")
            page.locator("#caption-datasets-search").press("ArrowDown")
            assert page.locator("#episode").input_value() == "0"
            page.locator("#caption-datasets-list a").click()
            page.wait_for_url(f"**/remote/{alternate['location_id']}/temporal-caption/review**")
            expect(page.locator("#scheme")).to_have_value("multiview_semantics")
            expect(page.locator("#review")).to_be_visible()
            expect(page.locator("#task")).to_have_text("Not supplied")
            expect(page.locator("#training-memory-panel")).to_be_hidden()
            assert "run=alternate-result" in page.url and "scheme=multiview_semantics" in page.url
            assert "episode=25" in page.url
            page.select_option("#variant", "alternate-result/coarse")
            expect(page.locator("#review")).to_be_visible()
            page.reload()
            expect(page.locator("#variant")).to_have_value("alternate-result/coarse")
            expect(page.locator("#review")).to_be_visible()
            page.locator("#caption-datasets > summary").click()
            page.locator("#caption-datasets-search").fill("source")
            page.locator("#caption-datasets-list a").click()
            page.locator("#review").wait_for(state="visible")
            page.select_option("#scheme", "flora")
            page.locator("#reward-review").wait_for(state="visible")
            expect(page.locator("#ai-message")).to_be_enabled()
            page.get_by_role("button", name="Next episode", exact=True).click()
            expect(page.locator("#episode")).to_have_value("1")
            expect(page.locator("#task")).to_have_text(second_instruction)
            if not local_page:
                expect(page.locator("#training-memory-value")).to_have_text("Episode 1 first")
            expect(page.locator("#episode-position")).to_have_text("· 2 / 2")
            expect(page.locator("#episode-next")).to_be_disabled()
            # Hold restoration of the unpersisted Episode 0 conversation. A late 404
            # must not replace a newly submitted conversation or clear its response.
            page.evaluate("""() => {
              const original = window.fetch;
              window.fetch = (...args) => {
                if (String(args[0]).includes('/api/ai-assistant/conversations/')) {
                  return new Promise(resolve => {window.finishRestore = () => {
                    window.fetch = original;
                    resolve(new Response(JSON.stringify({error:'Conversation not found'}),
                      {status:404, headers:{'Content-Type':'application/json'}}));
                  };});
                }
                return original(...args);
              };
            }""")
            page.get_by_role("button", name="Previous episode", exact=True).click()
            expect(page.locator("#episode")).to_have_value("0")
            expect(page.locator("#task")).to_have_text(instruction)
            expect(page.locator("#episode-previous")).to_be_disabled()
            page.wait_for_function("document.querySelector('#video').readyState >= 1")
            page.keyboard.press("Control+Shift+Space")
            page.wait_for_function("typeof window.finishRestore === 'function'")
            assert page.locator("#ai-message").is_disabled()
            assert page.locator("#ai-send").is_disabled()
            page.evaluate("window.finishRestore()")
            expect(page.locator("#ai-message")).to_be_enabled()
            expect(page.locator("#assistant-target")).to_contain_text("Episode 0")
            page.locator("#ai-message").fill("Discuss this episode")
            page.locator("#ai-message").press("Space")
            page.locator("#ai-message").press("ArrowDown")
            page.locator("#ai-message").press("ArrowRight")
            assert page.locator("#episode").input_value() == "0"
            assert page.locator("#video").evaluate("v=>v.paused && v.playbackRate===1")
            page.locator("#ai-message").press("Enter")
            expect(page.locator("#ai-history")).to_contain_text("Review conversation saved")
            assert {tool["function"]["name"] for tool in fake.requests[0]["tools"]} == {
                "read_context",
                "describe_operator",
            }
            assert "temporal_caption_review" in json.dumps(fake.requests)
            assert "Current caption text, FLORA scores" in json.dumps(fake.requests)
            page.select_option("#episode", "1")
            expect(page.locator("#assistant-target")).to_contain_text("Episode 1")
            expect(page.locator("#ai-history")).not_to_contain_text("Review conversation saved")
            page.select_option("#episode", "0")
            expect(page.locator("#ai-history")).to_contain_text("Review conversation saved")
            page.locator("#assistant-follow").uncheck()
            page.select_option("#episode", "1")
            expect(page.locator("#assistant-target")).to_contain_text("Episode 0")
            page.locator("#assistant-follow").check()
            expect(page.locator("#assistant-target")).to_contain_text("Episode 1")
            page.select_option("#episode", "0")
            page.reload()
            expect(page.locator("#ai-history")).to_contain_text("Review conversation saved")
            page.get_by_role("button", name="Collapse AI assistant", exact=True).click()
            page.locator("#reward-review").wait_for(state="visible")
            assert not store.list_jobs()
            score_box = page.locator("#reward-summary").bounding_box()
            video_box = page.locator("#video").bounding_box()
            assert abs(score_box["y"] - video_box["y"]) < 2
            assert score_box["x"] > video_box["x"] + video_box["width"]
            page.locator(".reward-overall summary").click()
            assert page.locator("#reward-terminal").is_visible()
            assert page.locator("#reward-terminal").inner_text() == "Completed after retry"
            page.locator(".reward-overall summary").click()
            assert page.locator('[data-window="0"] .timeline-label small').inner_text() == "0.00–0.40 s"
            assert page.locator('[data-window="1"] .timeline-label small').inner_text() == "0.10–0.40 s"

            def check_alignment():
                bounds = page.locator("#timeline").bounding_box()
                assert bounds
                for selector in ["#ticks", ".timeline-overlay", "#reward-tracks canvas"]:
                    for element in page.locator(selector).all():
                        box = element.bounding_box()
                        assert box and abs(box["x"] - bounds["x"]) < 1
                        assert abs(box["width"] - bounds["width"]) < 1
                band = page.locator(".disagreement-band").bounding_box()
                assert abs(band["x"] - bounds["x"] - bounds["width"] * 3 / 12) < 1
                assert abs(band["width"] - bounds["width"] * 5 / 12) < 1
                bars = page.locator("#timeline button")
                for index, segment in enumerate(flora_artifact()["result"]["segments"]):
                    box = bars.nth(index).bounding_box()
                    assert abs(box["x"] - bounds["x"] - bounds["width"] * segment["start_frame"] / 12) < 1
                    assert (
                        abs(
                            box["width"]
                            - bounds["width"] * (segment["end_frame"] - segment["start_frame"]) / 12
                        )
                        < 1
                    )

            # Playback shortcuts match Viewer, without stealing form/slider keyboard input.
            page.locator("#video").focus()
            for _ in range(4):
                page.keyboard.press("ArrowLeft")
            assert page.locator("#rate").input_value() == "0.25"
            assert page.locator("#video").evaluate("v=>v.playbackRate") == 0.25
            page.locator("#video").evaluate("v=>{v.currentTime=0;v.loop=true}")
            page.keyboard.press("Space")
            page.wait_for_function("!document.querySelector('#video').paused")
            page.keyboard.press("Space")
            assert page.locator("#video").evaluate("v=>v.paused")
            page.keyboard.down("Space")
            page.wait_for_function("!document.querySelector('#video').paused")
            page.keyboard.down("Space")
            assert not page.locator("#video").evaluate("v=>v.paused")
            page.keyboard.up("Space")
            page.keyboard.press("Space")
            for expected in ["0.5", "1", "2", "3", "5", "5"]:
                page.keyboard.press("ArrowRight")
                assert page.locator("#rate").input_value() == expected
            assert page.locator("#video").evaluate("v=>v.paused")
            page.keyboard.press("Control+ArrowLeft")
            assert page.locator("#rate").input_value() == "5"
            page.locator("#rate").focus()
            assert page.locator("#rate").evaluate("""e=>{
                const event=new KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true,cancelable:true});
                e.dispatchEvent(event);return !event.defaultPrevented;
            }""")
            assert page.locator("#video").evaluate("v=>v.playbackRate") == 5
            page.evaluate("""() => {
                const input=document.createElement('textarea');input.id='shortcut-input';
                document.body.append(input);input.focus();
            }""")
            page.keyboard.press("Space")
            page.keyboard.press("ArrowLeft")
            assert page.locator("#shortcut-input").input_value() == " "
            assert page.locator("#video").evaluate("v=>v.paused && v.playbackRate===5")
            page.evaluate("document.querySelector('#shortcut-input').remove()")
            page.select_option("#rate", "1")
            page.locator("#video").evaluate("v=>{v.loop=false;v.currentTime=0}")

            # Follow long captions within their pane, never move the video or the document.
            page.evaluate("""() => {
                window.originalCaption=structuredClone(artifact);
                const data=structuredClone(artifact);
                data.result.segments.forEach(s=>{s.evidence='Visible evidence. '.repeat(60)});
                render(data);
            }""")
            page.wait_for_function("document.querySelector('#video').readyState>=1")
            page.locator("#video").evaluate("v=>v.currentTime=.3")
            page.wait_for_function("active===artifact.result.segments.length-1")
            page.wait_for_function("document.querySelector('#caption-panel').scrollTop>100")
            assert page.evaluate("window.scrollY") == 0
            bounds = page.locator("#caption-panel").bounding_box()
            active_card = page.locator("#segments .active").bounding_box()
            assert bounds["y"] <= active_card["y"] < bounds["y"] + bounds["height"]
            assert page.locator("#follow-caption").is_visible()
            page.locator("#follow-caption").uncheck()
            scroll_before = page.locator("#caption-panel").evaluate("e=>e.scrollTop")
            page.locator("#video").evaluate("v=>v.currentTime=0")
            page.wait_for_function("active===0")
            page.wait_for_timeout(100)
            assert page.locator("#caption-panel").evaluate("e=>e.scrollTop") == scroll_before
            page.locator("#follow-caption").check()
            page.wait_for_function("""() => {
                const panel=document.querySelector('#caption-panel').getBoundingClientRect();
                const card=document.querySelector('#segments .active').getBoundingClientRect();
                return card.top>=panel.top && card.top<panel.bottom;
            }""")
            page.evaluate("render(window.originalCaption);delete window.originalCaption")
            page.wait_for_function("document.querySelector('#video').readyState>=1")
            check_alignment()
            page.evaluate("""() => {
                renderReward({...artifact,reward:{...artifact.reward,
                    windows:Array.from({length:6},(_,i)=>artifact.reward.windows[i%2])}});
                updateReward(0);
            }""")
            for width, height in [(1366, 768), (1280, 720), (1920, 1080), (1024, 768)]:
                page.set_viewport_size({"width": width, "height": height})
                assert page.evaluate("document.documentElement.scrollHeight <= innerHeight")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                for selector in [
                    "#original-instruction",
                    "#video",
                    "#reward-summary",
                    "#current-caption",
                    ".track-box",
                ]:
                    box = page.locator(selector).bounding_box()
                    assert box and box["y"] >= 0 and box["y"] + box["height"] <= height
                assert page.locator("#video").bounding_box()["height"] >= 160
                assert page.locator(".review-panel-content").bounding_box()["height"] >= 120
                assert page.locator("#reward-summary").bounding_box()["height"] < height / 2
                check_alignment()
                assert page.locator(".track-box").evaluate("e=>e.scrollHeight<=e.clientHeight+1")
            page.evaluate("renderReward(artifact);updateReward(0)")
            page.set_viewport_size({"width": 1360, "height": 1000})
            page.locator("#loop").check()
            page.locator("#reward-conflicts button").click()
            page.wait_for_function(
                "document.querySelector('#reward-overlay').textContent.includes('Windows disagree')"
            )
            assert abs(page.locator("#video").evaluate("v=>v.currentTime") - 0.1) < 0.04
            assert not page.locator("#loop").is_checked()
            assert page.locator("#reward-overall-score").inner_text() == "4"
            assert page.locator("#reward-values strong").all_text_contents() == ["1", "2"]
            assert "Window 1" in page.locator(".reward-window-score.is-source").inner_text()
            assert page.locator("#reward-score-conflict").is_visible()
            page.wait_for_function("document.querySelector('#playhead').style.left === '25%'")
            assert page.locator('[data-window="0"]').get_attribute("class").endswith("reward-source")
            page.locator(".reward-windows > summary").click()
            assert not page.locator('[data-window="1"] canvas').is_visible()
            page.locator(".reward-windows > summary").click()
            page.locator("#reward-tracks canvas").first.focus()
            page.keyboard.press("ArrowRight")
            page.wait_for_function("document.querySelector('#video').currentTime > .12")
            page.locator("#reward-tracks canvas").first.click(position={"x": 1, "y": 10})
            page.wait_for_function("document.querySelector('#video').currentTime === 0")
            page.locator("#timeline button").last.click()
            page.wait_for_function("document.querySelector('#video').currentTime >= .3")
            page.locator("#video").evaluate("v=>v.pause()")
            page.wait_for_function("document.querySelector('#reward-merged-score').textContent === '3 / 3'")
            assert page.locator("#reward-overall-score").inner_text() == "4"
            assert page.locator("#reward-values strong").all_text_contents() == ["3", "3"]
            assert page.locator("#reward-score-conflict").is_hidden()
            page.select_option("#scheme", "hybrid_windows")
            page.locator("#reward-review").wait_for(state="hidden")
            assert page.locator("#reward-summary").is_hidden()
            assert page.locator("#reward-details").is_hidden()
            assert page.locator("#reward-overall-score").inner_text() == ""
            assert page.locator(".disagreement-band").count() == 0
            assert page.locator("#reward-tracks canvas").count() == 0
            page.select_option("#scheme", "flora")
            page.locator("#reward-review").wait_for(state="visible")
            page.route(
                "**/legacy/caption",
                lambda route: route.fulfill(status=503, json={"error": "Temporarily unavailable"}),
            )
            page.select_option("#scheme", "hybrid_windows")
            page.wait_for_function("document.querySelector('#status').textContent.includes('503')")
            assert page.locator("#reward-review").is_hidden()
            assert page.locator("#original-instruction").is_hidden()
            assert page.locator("#reward-tracks canvas").count() == 0
            page.unroute("**/legacy/caption")
            page.goto(origin + route + "/review?scheme=video_events")
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('No saved results')"
            )
            assert page.locator("#review").is_hidden()
            assert page.locator("#original-instruction").is_hidden()
            page.goto(origin + route + "/review?scheme=flora")
            page.locator("#reward-review").wait_for(state="visible")
            page.set_viewport_size({"width": 390, "height": 844})
            expect(page.locator("#original-instruction")).to_be_visible()
            expect(page.locator("#task")).to_have_text(instruction)
            if not local_page:
                expect(page.locator("#training-memory-panel")).to_be_visible()
            page.locator("#caption-datasets > summary").click()
            expect(page.locator("#caption-datasets-list a")).to_have_count(2)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator("#caption-datasets-search").press("Escape")
            assert not page.locator("#caption-datasets").get_attribute("open")
            check_alignment()
            page.evaluate("window.scrollTo(0,200)")
            mobile_scroll = page.evaluate("window.scrollY")
            page.locator("#video").evaluate("v=>v.currentTime=.3")
            page.wait_for_function("active===artifact.result.segments.length-1")
            page.wait_for_timeout(100)
            assert page.evaluate("window.scrollY") == mobile_scroll
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.get_by_role("button", name="Open AI assistant", exact=True).click()
            expect(page.locator("#ai-message")).to_be_enabled()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.get_by_role("button", name="Collapse AI assistant", exact=True).click()
            page.evaluate("""() => {
                const reward=structuredClone(artifact.reward);
                reward.windows=Array.from({length:20},()=>({scores:Array(12).fill(3),output:'Normal'}));
                reward.disagreement_intervals=[];
                renderReward({...artifact,reward});
            }""")
            assert not page.locator(".reward-windows").get_attribute("open")
            assert page.locator("#reward-tracks canvas").count() == 21
            assert page.locator(".disagreement-band").count() == 0
            assert "No overlapping-window" in page.locator("#reward-conflicts").inner_text()
            page.locator(".reward-windows > summary").click()
            assert page.locator("#reward-tracks canvas").last.is_visible()
            page.evaluate("renderReward(artifact)")
            page.goto(origin + route)
            page.locator('[data-caption-id="job-scheme"] option[value="flora"]').wait_for(state="attached")
            page.select_option('[data-caption-id="job-scheme"]', "flora")
            assert page.locator('[data-caption-id="include-signals"]').is_disabled()
            assert not page.locator('[data-caption-id="include-signals"]').is_checked()
            assert page.locator('[data-caption-id="job-submit"]').is_disabled()
            assert "Read-only" in page.locator('[data-caption-id="job-readiness"]').inner_text()
            page.get_by_role("button", name="Open AI assistant", exact=True).click()
            expect(page.locator("#ai-message")).to_be_enabled()
            expect(page.locator("#assistant-target")).to_contain_text("Annotation")
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_parent_death_stops_worker_and_its_children(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    pid_file = tmp_path / "pids.json"
    worker = (
        "import json,subprocess,sys,time;from pathlib import Path;"
        "from lerobot.data_platform.operators.implementations.caption.flora_worker import guard_parent;"
        "guard_parent();"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"Path({str(pid_file)!r}).write_text(json.dumps([__import__('os').getpid(),child.pid]));"
        "time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,os,time;"
        f"subprocess.Popen([sys.executable,'-c',{worker!r}],"
        "env={**os.environ,'FLORA_PARENT_PID':str(os.getpid())},start_new_session=True);time.sleep(60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code])
    pids = []

    def alive(pid):
        path = Path(f"/proc/{pid}/stat")
        return path.exists() and path.read_text().split(")", 1)[1].split()[0] != "Z"

    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        pids = json.loads(pid_file.read_text())
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(alive(pid) for pid in pids)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if pids and any(alive(pid) for pid in pids):
            os.killpg(pids[0], signal.SIGKILL)


@pytest.mark.parametrize("video_backed", [False, True])
def test_executor_exports_visual_only_and_publishes_without_api_key(tmp_path, monkeypatch, video_backed):
    import shutil
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot.data_platform.ai_tools import qwen
    from lerobot.data_platform.annotations.jobs import execute_caption_job, validate_run
    from lerobot.data_platform.operators.implementations.caption import flora as temporal_caption_flora
    from tests.datasets.test_temporal_caption import (
        test_export_reads_only_selected_episode_from_shared_v3_shard,
    )

    test_export_reads_only_selected_episode_from_shared_v3_shard(tmp_path)
    source = tmp_path / "source"
    info = json.loads((source / "meta/info.json").read_text())
    for name in ("left_wrist", "right_wrist"):
        info["features"][name] = {"dtype": "image"}
    (source / "meta/info.json").write_text(json.dumps(info))
    parquet = source / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(parquet).to_pylist()
    for row in rows:
        row.update(left_wrist=row["head_image"], right_wrist=row["head_image"])
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    if video_backed:
        import subprocess

        info["video_path"] = "videos/{video_key}.mp4"
        episodes_path = source / "meta/episodes/file-000.parquet"
        episodes = pq.read_table(episodes_path).to_pylist()
        (source / "videos").mkdir()
        cameras = ["head_image", "left_wrist", "right_wrist"]
        for camera in cameras:
            info["features"][camera]["dtype"] = "video"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=16x16:r=30",
                    "-frames:v",
                    "4",
                    "-c:v",
                    "libx264",
                    str(source / "videos" / f"{camera}.mp4"),
                ],
                check=True,
            )
            for ep in episodes:
                ep.update(
                    {
                        f"videos/{camera}/chunk_index": 0,
                        f"videos/{camera}/file_index": 0,
                        f"videos/{camera}/from_timestamp": ep["episode_index"] * 2 / 30,
                        f"videos/{camera}/to_timestamp": (ep["episode_index"] + 1) * 2 / 30,
                    }
                )
        (source / "meta/info.json").write_text(json.dumps(info))
        pq.write_table(pa.Table.from_pylist(episodes), episodes_path)
        pq.write_table(pq.read_table(parquet).drop(cameras), parquet)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(
        qwen, "QwenClient", lambda **kwargs: pytest.fail("FLORA must not construct an API client")
    )
    monkeypatch.setattr(temporal_caption_flora.FloraRuntime, "load", lambda: None)

    def infer(bundle, output, key, model, **kwargs):
        meta = json.loads((bundle / "episode.json").read_text())
        assert meta["include_signals"] is False
        assert all(set(row) == {"frame"} for row in meta["signals"])
        assert len(list((bundle / "camera-videos").glob("*.mp4"))) == 3
        count = meta["frame_count"]
        windows = [{"scores": [3] * count, "output": "All normal"}]
        value = build_artifact(
            meta,
            key,
            {
                "segments": [
                    {"start_s": 0, "end_s": count / meta["fps"], "caption": "Move", "status": "complete"}
                ]
            },
            {
                "terminal": {"score": 4, "rationale": "Completed"},
                "windows": windows,
                "merged_scores": [3] * count,
                "merged_source_window": [0] * count,
                "disagreement_intervals": [],
                "human_reviewed": False,
            },
            [],
        )
        output.mkdir()
        (output / "caption.json").write_text(json.dumps(value))
        shutil.copyfile(bundle / "episode.json", output / "episode.json")
        shutil.copyfile(bundle / "video.mp4", output / "video.mp4")

    monkeypatch.setattr(temporal_caption_flora, "infer_flora", infer)
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    client = SimpleNamespace(
        event=lambda *args: None,
        upload_artifact=lambda state, job, name, path, **kwargs: shutil.copyfile(path, uploaded / name),
    )
    agent = SimpleNamespace(client=client, state=None, state_path=tmp_path / "state.json")
    options = {"episode_index": 1, "scheme": "flora", "include_signals": False}
    result = execute_caption_job(
        agent, {"job_id": "flora", "options": options}, source, {"dataset_key": "node/source"}
    )
    assert result["scheme"] == "flora"
    records = validate_run(uploaded, key="node/source", options=options)
    assert len(records) == 1 and records[0]["source_fingerprint"]


def test_video_shard_export_uses_selected_episode_range(tmp_path):
    import av
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.datasets.test_temporal_caption import (
        test_export_reads_only_selected_episode_from_shared_v3_shard,
    )

    test_export_reads_only_selected_episode_from_shared_v3_shard(tmp_path)
    root = tmp_path / "source"
    info = json.loads((root / "meta/info.json").read_text())
    info["features"]["head_image"]["dtype"] = "video"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    (root / "meta/info.json").write_text(json.dumps(info))
    path = root / "videos/head_image/chunk-000/file-000.mp4"
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": "0"}
        for color in [(255, 0, 0)] * 2 + [(0, 255, 0)] * 2:
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), color, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    ep_path = root / "meta/episodes/file-000.parquet"
    episodes = pq.read_table(ep_path).to_pylist()
    for ep in episodes:
        ep.update(
            {
                "videos/head_image/chunk_index": 0,
                "videos/head_image/file_index": 0,
                "videos/head_image/from_timestamp": ep["episode_index"] * 2 / 30,
                "videos/head_image/to_timestamp": (ep["episode_index"] + 1) * 2 / 30,
            }
        )
    pq.write_table(pa.Table.from_pylist(episodes), ep_path)
    data = root / "data/chunk-000/file-000.parquet"
    pq.write_table(pq.read_table(data).drop(["head_image"]), data)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "video-export"
    prepare(root, 1, output, export_camera_videos=True)
    with av.open(str(output / "camera-videos/0.mp4")) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == 2
    assert all(f.to_ndarray(format="rgb24")[0, 0, 1] > 240 for f in frames)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    episodes[1]["videos/head_image/to_timestamp"] = 3 / 30
    pq.write_table(pa.Table.from_pylist(episodes), ep_path)
    with pytest.raises(ValueError, match="range"):
        prepare(root, 1, tmp_path / "invalid-export", export_camera_videos=True)
    assert not (tmp_path / "invalid-export").exists()


def test_two_wrist_long_episode_chunks_keep_frame_count(tmp_path):
    import subprocess

    import av

    from lerobot.data_platform.operators.implementations.caption.flora_worker import caption_inputs

    videos = {}
    for role, color in [("left", "red"), ("right", "blue")]:
        path = tmp_path / f"{role}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=16x16:r=4",
                "-frames:v",
                "485",
                "-c:v",
                "libx264",
                str(path),
            ],
            check=True,
        )
        videos[role] = str(path)
    inputs, instruction = caption_inputs(tmp_path, {"videos": videos, "frame_count": 485, "fps": 4})
    assert "left wrist" in instruction and "No scene camera" in instruction
    assert [offset for offset, _ in inputs] == [0, 242 / 4]
    counts = []
    for _, paths in inputs:
        assert list(paths) == ["main"]
        with av.open(str(paths["main"])) as video:
            frames = list(video.decode(video=0))
            counts.append(len(frames))
            assert frames[0].width == 32 and frames[0].height == 16
            image = frames[0].to_ndarray(format="rgb24")
            assert image[8, 8, 0] > 240 and image[8, 24, 2] > 240
    assert counts == [242, 243]


def test_long_native_partition_can_publish_more_than_forty_actions():
    segments = [{"start_s": i, "end_s": i + 1, "caption": "Move", "status": "complete"} for i in range(100)]
    assert len(review_partition(segments, 3000, 30)["segments"]) == 100


def test_video_dataset_passes_submission_and_export_validation():
    info = {
        "codebase_version": "v3.0",
        "features": {name: {"dtype": "video"} for name in ["left_wrist", "right_wrist"]},
    }
    validate_source(info)
    caps = {"caption_schemes": ["flora"], "caption_flora_protocol": 1, "caption_flora_configured": True}
    status = caption_scheme_status(caps, None, {"role": "operator"}, metadata=info)
    assert next(s for s in status if s["id"] == "flora")["can_submit"]


@pytest.mark.parametrize("stats", [None, "not training statistics"])
def test_single_view_v21_inference_export_does_not_need_training_stats(tmp_path, stats):
    import av

    from tests.datasets.test_temporal_caption import test_export_v21_embedded_images_and_source_preservation

    test_export_v21_embedded_images_and_source_preservation(tmp_path)
    root = tmp_path / "source"
    path = root / "meta/episodes_stats.jsonl"
    if stats is None:
        path.unlink()
    else:
        path.write_text(stats)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    bundle = tmp_path / "native-single"
    prepare(root, 1, bundle, export_camera_videos=True)
    metadata = json.loads((bundle / "episode.json").read_text())
    assert camera_roles(metadata["cameras"]) == {"main": "image"}
    assert metadata["frame_count"] == 2
    with av.open(str(bundle / "camera-videos/0.mp4")) as video:
        assert len(list(video.decode(video=0))) == 2
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "roles,expected", [(["main"], "native_single_view"), (["left", "main", "right"], "native_three_view")]
)
def test_native_caption_calls_preserve_original_views(tmp_path, monkeypatch, roles, expected):
    import sys
    from types import ModuleType, SimpleNamespace

    from lerobot.data_platform.operators.implementations.caption.flora_worker import caption_stage

    calls = []
    paths = {role: str(tmp_path / f"{role}.mp4") for role in roles}
    segments = [{"start_s": 0.0, "end_s": 10.0, "caption": "Move", "status": "complete"}]

    def inputs(main, profile, left, right, yam32, instruction):
        assert str(main) == paths["main"] and profile == "yam" and not yam32 and not instruction
        assert (str(left) if left else None) == paths.get("left")
        assert (str(right) if right else None) == paths.get("right")
        return dict(paths), profile

    def invoke(kind):
        def run(videos, *args, **kwargs):
            assert videos == paths
            calls.append(kind)
            return segments, {}

        return run

    models = SimpleNamespace(Session=lambda: SimpleNamespace(close=lambda: None), MODELS={})
    pipeline = SimpleNamespace(
        segment=invoke("segment"), caption_single=invoke("single"), caption_three=invoke("three")
    )
    package = ModuleType("batch_utils")
    package.models, package.pipeline = models, pipeline
    api = ModuleType("batch_utils.api")
    api.DataEnginePipeline = SimpleNamespace(_inputs=inputs)
    monkeypatch.setitem(sys.modules, "batch_utils", package)
    monkeypatch.setitem(sys.modules, "batch_utils.api", api)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kwargs: None)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    request = {
        "runtime": {"caption_code": str(tmp_path), "weights": str(tmp_path)},
        "videos": paths,
        "frame_count": 100,
        "fps": 10,
    }
    caption_stage("segment", tmp_path, request)
    caption_stage("caption", tmp_path, request)
    assert calls == ["segment", "single" if len(roles) == 1 else "three"]
    assert (
        json.loads((tmp_path / "caption.json").read_text())["input_adaptation"]["camera_layout"] == expected
    )
    assert not (tmp_path / "wrists.mp4").exists()


def test_flora_long_episode_output_and_many_segments_remain_valid():
    count = 36001
    native = [{"start_s": 0, "end_s": count / 30, "caption": "Move", "status": "complete"}]
    windows = [{"scores": [1] * count, "output": "Normal"}]
    reward = {
        "terminal": {"score": 4, "rationale": "Completed"},
        "windows": windows,
        "merged_scores": [1] * count,
        "merged_source_window": [0] * count,
        "disagreement_intervals": [],
        "human_reviewed": False,
        "alignment": "Test",
    }
    artifact = build_artifact(
        {"episode_index": 0, "frame_count": count, "fps": 30}, "api/test", {"segments": native}, reward, []
    )
    validate_flora(artifact)
    many = [{"start_s": i, "end_s": i + 1, "caption": "Move", "status": "complete"} for i in range(1001)]
    assert len(review_partition(many, 1001, 1)["segments"]) == 1001


def test_flora_long_episode_reaches_gpu_allocation(tmp_path, monkeypatch):
    from lerobot.data_platform.operators.implementations.caption import flora as temporal_caption_flora

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "episode.json").write_text(
        json.dumps(
            {
                "include_signals": False,
                "frame_count": 36001,
                "fps": 30,
                "cameras": ["scene"],
                "task_hint": ["Move"],
            }
        )
    )
    config = type("Config", (), {"lock_dir": tmp_path / "locks"})()
    monkeypatch.setattr(temporal_caption_flora.FloraRuntime, "load", lambda: config)
    monkeypatch.setattr(
        temporal_caption_flora, "acquire_gpu", lambda *a: (_ for _ in ()).throw(RuntimeError("gpu reached"))
    )
    with pytest.raises(RuntimeError, match="gpu reached"):
        temporal_caption_flora.infer_flora(bundle, tmp_path / "output", "api/test")


def test_long_flora_input_uses_bounded_chunks_without_losing_frames(tmp_path, monkeypatch):
    from lerobot.data_platform.operators.implementations.caption import (
        flora_worker as temporal_caption_flora_worker,
    )

    commands = []
    monkeypatch.setattr(
        temporal_caption_flora_worker.subprocess, "run", lambda args, **kw: commands.append(args)
    )
    request = {"videos": {"main": str(tmp_path / "input.mp4")}, "frame_count": 36001, "fps": 30}
    chunks, _ = temporal_caption_flora_worker.caption_inputs(tmp_path, request)
    assert len(chunks) == 11
    assert chunks[0][0] == 0 and chunks[-1][0] == (10 * 36001 // 11) / 30
    ranges = [command[command.index("-vf") + 1] for command in commands]
    assert ranges[0].startswith("trim=start_frame=0:end_frame=3272,")
    assert ranges[-1].startswith("trim=start_frame=32728:end_frame=36001,")
