"""Mixed evaluation episodes keep camera availability and source signal semantics."""

import csv
import json
import shutil
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from flask import Flask

from lerobot.data_platform.cli import run_precompute
from lerobot.data_platform.operators.implementations.data_profile import require_dataset_operation
from lerobot.data_platform.operators.implementations.mixed_episode_viewer import episode_viewer_spec
from lerobot.data_platform.operators.implementations.preprocess.dataset_version import validate_v3_dataset
from lerobot.data_platform.operators.implementations.viewer_signals import viewer_signal_presentation
from lerobot.data_platform.viewer import _columns_from_csv_header, _serve_csv_stripped


@pytest.fixture
def mixed_root(tmp_path):
    root = tmp_path / "mixed"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    cameras = ["observation.images.left_wrist", "observation.images.scene", "observation.images.right_wrist"]
    h10 = {
        "state": {"dtype": "float32", "shape": [2], "names": ["joint_a", "joint_b"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["command_a", "command_b"]},
    }
    umi = {"state": {"dtype": "float32", "shape": [3], "names": ["left_x", "right_qz", "head_qw"]}}
    eef = {"left_arm_quaternion_pose": {"dtype": "float32", "shape": [2], "names": ["x", "qw"]}}
    specs = [
        (cameras, h10, "h10_w"),
        ([cameras[0], cameras[2]], umi, "UMI-GripperBody-Head"),
        ([cameras[1]], {}, "droid"),
        ([cameras[0], cameras[2]], eef, "genrobot_das"),
    ]
    global_features = {
        "timestamp": {"dtype": "float64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "state": {"dtype": "float32", "shape": [3]},
        "action": {"dtype": "float32", "shape": [2]},
        **eef,
    }
    for key in cameras:
        global_features[key] = {"dtype": "video", "shape": [16, 16, 3]}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "mixed_episode_evaluation_v1",
        "data_platform_mixed_episode_viewer": 1,
        "fps": 30,
        "total_episodes": 4,
        "total_frames": 12,
        "total_tasks": 1,
        "chunks_size": 1000,
        "features": global_features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text("{}")
    pq.write_table(pa.table({"task_index": [0], "task": ["test"]}), root / "meta/tasks.parquet")
    episodes = []
    for i, (present, signals, robot) in enumerate(specs):
        table = {
            "timestamp": pa.array([0.0, 1 / 30, 2 / 30]),
            "episode_index": pa.array([i] * 3),
            "frame_index": pa.array([0, 1, 2]),
            "index": pa.array([i * 3 + j for j in range(3)]),
            "task_index": pa.array([0] * 3),
        }
        vals = {
            "state": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            if i == 0
            else [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            "action": [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            "left_arm_quaternion_pose": [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
        }
        for key in ["state", "action", "left_arm_quaternion_pose"]:
            table[key] = pa.array(vals[key] if key in signals else [None] * 3, type=pa.list_(pa.float32()))
        pq.write_table(pa.table(table), root / f"data/chunk-000/file-{i:03d}.parquet")
        ep = {
            "episode_index": i,
            "tasks": ["test"],
            "length": 3,
            "dataset_from_index": i * 3,
            "dataset_to_index": i * 3 + 3,
            "data/chunk_index": 0,
            "data/file_index": i,
            "viewer/cameras_json": json.dumps(present),
            "viewer/signals_json": json.dumps(signals),
            "viewer/robot_type": robot,
        }
        for key in cameras:
            for name, value in [
                ("chunk_index", 0),
                ("file_index", i),
                ("from_timestamp", 0.0),
                ("to_timestamp", 0.1),
            ]:
                ep[f"videos/{key}/{name}"] = value if key in present else None
            if key in present:
                video = root / f"videos/{key}/chunk-000/file-{i:03d}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=red:s=16x16:r=30",
                        "-frames:v",
                        "3",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(video),
                    ],
                    check=True,
                )
        episodes.append(ep)
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


def test_mixed_precompute_preserves_views_and_raw_signals(mixed_root, tmp_path):
    assert validate_v3_dataset(mixed_root)["total_episodes"] == 4
    cache = tmp_path / "cache"
    run_precompute(
        mixed_root, repo_id="remote/mixed", output_dir=cache, prepare_workers=1, show_progress=False
    )
    static = cache / "static"
    manifest = json.loads((static / "viewer_manifest.json").read_text())
    assert manifest["mixed_episode_viewer"]
    assert [len(e["viewer"]["camera_keys"]) for e in manifest["episodes"]] == [3, 2, 1, 2]
    for i, count in enumerate([3, 2, 1, 2]):
        assert len(list((static / "videos").glob(f"*/episode_{i:06d}_h264.mp4"))) == count
    with (static / "csv/episode_000000_ds1.csv").open() as f:
        h10 = list(csv.DictReader(f))
    assert [float(h10[0][k]) for k in ["joint_a", "joint_b", "command_a", "command_b"]] == [1, 2, 10, 20]
    with (static / "csv/episode_000001_ds1.csv").open() as f:
        umi = list(csv.DictReader(f))
    assert float(umi[0]["state.left_x"]) == pytest.approx(0.1)
    assert "joint_a" not in umi[0]
    with (static / "csv/episode_000002_ds1.csv").open() as f:
        assert next(csv.reader(f)) == ["timestamp"]
    with (static / "csv/episode_000003_ds1.csv").open() as f:
        eef = list(csv.DictReader(f))
    assert float(eef[0]["x"]) == pytest.approx(0.1)
    columns = _columns_from_csv_header(
        static / "csv/episode_000001_ds1.csv", manifest["episodes"][1]["viewer"]["signal_features"]
    )
    assert (
        viewer_signal_presentation(
            manifest["episodes"][1]["viewer"]["signal_features"], columns, "UMI-GripperBody-Head"
        )["mode"]
        == "end_effector"
    )
    with Flask(__name__).test_request_context():
        response = _serve_csv_stripped(
            static / "csv/episode_000001_ds1.csv",
            None,
            manifest["episodes"][1]["viewer"]["signal_features"],
            normalize_grippers=False,
        )
    assert b"state.left_x" in response.data
    assert (
        run_precompute(
            mixed_root, repo_id="remote/mixed", output_dir=cache, prepare_workers=1, show_progress=False
        ).prepared
        is False
    )
    with pytest.raises(ValueError, match="read-only"):
        require_dataset_operation(mixed_root, "preprocess.merge")
    with pytest.raises(ValueError, match="read-only"):
        require_dataset_operation(mixed_root, "analysis")
    require_dataset_operation(mixed_root, "caption.annotate")
    require_dataset_operation(mixed_root, "curation.snapshot")
    require_dataset_operation(mixed_root, "snapshot")


def test_mixed_snapshot_keeps_source_read_only(mixed_root):
    from lerobot.data_platform.datasets.curation import scan_snapshot
    from lerobot.data_platform.datasets.lifecycle.service import dataset_snapshot

    before = dataset_snapshot(mixed_root)
    snapshot = scan_snapshot(mixed_root, "node/mixed", "mixed-test")
    assert snapshot.version["fingerprint"] == before[0]
    assert dataset_snapshot(mixed_root) == before


def test_mixed_snapshot_rejects_wrong_episode_signal_width(mixed_root):
    from lerobot.data_platform.datasets.curation import scan_snapshot

    path = mixed_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    column = table.schema.get_field_index("state")
    table = table.set_column(column, "state", pa.array([[1.0], [2.0], [3.0]], type=pa.list_(pa.float32())))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="state width"):
        scan_snapshot(mixed_root, "node/mixed", "mixed-test")


def test_mixed_rejects_unregistered_camera_and_missing_video(mixed_root):
    info = json.loads((mixed_root / "meta/info.json").read_text())
    with pytest.raises(ValueError, match="cameras"):
        episode_viewer_spec(
            info,
            {"viewer/cameras_json": '["unknown"]', "viewer/signals_json": "{}", "viewer/robot_type": "x"},
        )
    with pytest.raises(ValueError, match="signal"):
        episode_viewer_spec(
            info,
            {
                "viewer/cameras_json": '["observation.images.scene"]',
                "viewer/signals_json": json.dumps({"state": {"dtype": "float32", "shape": [4]}}),
                "viewer/robot_type": "x",
            },
        )
    (mixed_root / "videos/observation.images.left_wrist/chunk-000/file-001.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        validate_v3_dataset(mixed_root)


def test_flora_mixed_uses_native_views_fps_and_checks_source(mixed_root, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lerobot.data_platform.annotations.jobs import execute_caption_job, validate_run
    from lerobot.data_platform.operators.implementations.caption import flora as temporal_caption_flora
    from lerobot.data_platform.operators.implementations.caption.mixed import prepare_flora_mixed
    from lerobot.data_platform.operators.implementations.caption.snapshot import caption_source_fingerprint

    episodes = pq.read_table(mixed_root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    records = []
    for episode in episodes:
        index = episode["episode_index"]
        cameras = json.loads(episode["viewer/cameras_json"])
        source_names = [key.rsplit(".", 1)[-1] for key in cameras]
        native = mixed_root / "native" / f"episode_{index:06d}"
        (native / "camera-videos").mkdir(parents=True)
        frames, fps = (1, 10) if index == 2 else (3, 30)
        for j, key in enumerate(cameras):
            source = mixed_root / f"videos/{key}/chunk-000/file-{index:03d}.mp4"
            target = native / "camera-videos" / f"{j}.mp4"
            if index == 2:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(source), "-frames:v", "1", "-r", "10", str(target)],
                    check=True,
                )
            else:
                shutil.copyfile(source, target)
        shutil.copyfile(native / "camera-videos/0.mp4", native / "video.mp4")
        sample = f"sample-{index}"
        (native / "episode.json").write_text(
            json.dumps(
                {
                    "episode_index": 100 + index,
                    "frame_count": frames,
                    "fps": fps,
                    "duration_s": frames / fps,
                    "cameras": source_names,
                    "task_hint": ["test"],
                    "sample_id": sample,
                    "source_digest": "original",
                    "signals": [{"state": [1]}],
                }
            )
        )
        records.append(
            {
                "episode_index": index,
                "source_episode_index": 100 + index,
                "sample_id": sample,
                "native_relative": native.relative_to(mixed_root).as_posix(),
                "camera_keys": cameras,
                "source_frames": frames,
                "source_fps": fps,
                "viewer_repeat_factor": 3 if index == 2 else 1,
                "videos": {
                    key: {
                        "source_camera": name,
                        "source_video": (native / "camera-videos" / f"{j}.mp4")
                        .relative_to(mixed_root)
                        .as_posix(),
                    }
                    for j, (key, name) in enumerate(zip(cameras, source_names, strict=True))
                },
            }
        )
    (mixed_root / "provenance").mkdir()
    provenance = mixed_root / "provenance/episodes.json"
    provenance.write_text(json.dumps(records))

    fingerprint = caption_source_fingerprint(mixed_root, tmp_path / "hashes")
    for index, expected_count, expected_fps in [(0, 3, 30), (2, 1, 10), (3, 2, 30)]:
        target = tmp_path / f"bundle-{index}"
        prepare_flora_mixed(mixed_root, index, target)
        result = json.loads((target / "episode.json").read_text())
        assert result["episode_index"] == index
        assert result["source_episode_index"] == 100 + index
        assert result["fps"] == expected_fps
        assert len(result["cameras"]) == expected_count
        assert result["signals"] == []
        assert len(list((target / "camera-videos").glob("*.mp4"))) == expected_count

    monkeypatch.setattr(temporal_caption_flora.FloraRuntime, "load", lambda: None)

    def infer(bundle, output, key, model, **kwargs):
        meta = json.loads((bundle / "episode.json").read_text())
        assert meta["fps"] == 10 and meta["frame_count"] == 1 and meta["episode_index"] == 2
        assert meta["signals"] == [] and len(meta["cameras"]) == 1
        reward = {
            "terminal": {"score": 4, "rationale": "Done"},
            "windows": [{"scores": [3], "output": "Normal"}],
            "merged_scores": [3],
            "merged_source_window": [0],
            "disagreement_intervals": [],
            "human_reviewed": False,
        }
        artifact = temporal_caption_flora.build_artifact(
            meta,
            key,
            {"segments": [{"start_s": 0, "end_s": 0.1, "caption": "Move", "status": "complete"}]},
            reward,
            [],
        )
        output.mkdir()
        (output / "caption.json").write_text(json.dumps(artifact))
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
    options = {"episode_index": 2, "scheme": "flora", "include_signals": False}
    execute_caption_job(
        agent, {"job_id": "flora", "options": options}, mixed_root, {"dataset_key": "node/mixed"}
    )
    assert validate_run(uploaded, key="node/mixed", options=options)[0]["fps"] == 10

    video = mixed_root / "native/episode_000002/camera-videos/0.mp4"
    with video.open("ab") as stream:
        stream.write(b"changed")
    assert caption_source_fingerprint(mixed_root, tmp_path / "hashes") != fingerprint
    records[2]["source_fps"] = 30
    provenance.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="provenance"):
        prepare_flora_mixed(mixed_root, 2, tmp_path / "invalid")


def test_cached_viewer_switches_episode_media_and_signal_metadata(mixed_root, tmp_path, monkeypatch):
    from pathlib import Path

    from lerobot.data_platform import viewer
    from tests.datasets.test_control_plane import _bootstrap, _store

    store = _store(tmp_path)
    _, node = store.enroll_node(
        name="test",
        hostname="test",
        allowed_roots=["/datasets"],
        writable_roots=["/datasets"],
        capabilities={},
        enrollment_token="test",
        expected_token="test",
    )
    location = store.sync_locations(
        node["node_id"],
        [{"dataset_key": "remote/mixed", "root": "/datasets/mixed", "output_dir": "/datasets/cache"}],
    )[0]
    cache = tmp_path / "cache"
    run_precompute(
        mixed_root, repo_id="remote/mixed", output_dir=cache, prepare_workers=1, show_progress=False
    )
    store.mark_viewer_ready(
        location["location_id"], viewer_url="/remote/mixed/episode_0", cache_root=str(cache)
    )
    monkeypatch.setenv("DATA_PLATFORM_BOOTSTRAP_TOKEN", "bootstrap-secret")
    console = tmp_path / "console/static"
    console.mkdir(parents=True)
    app = viewer.run_server(
        dataset=None,
        episodes=None,
        max_frames=None,
        prepare_videos=False,
        downsample=None,
        precompute_csv=False,
        precomputed_only=True,
        host="127.0.0.1",
        port=0,
        static_folder=console,
        template_folder=Path(viewer.__file__).parent / "templates",
        database_url=f"sqlite:///{tmp_path / 'control-plane.db'}",
        remote_cache_root=tmp_path,
        start_server=False,
    )
    client = app.test_client()
    assert _bootstrap(client).status_code == 200
    for episode, names in [
        (0, ["left_wrist", "scene", "right_wrist"]),
        (1, ["left_wrist", "right_wrist"]),
        (2, ["scene"]),
        (3, ["left_wrist", "right_wrist"]),
    ]:
        response = client.get(f"/remote/mixed/episode_{episode}?direct=1")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert html.count("data-video-label=") == len(names)
        for name in names:
            assert f'data-video-label="observation.images.{name}"' in html
        if episode in [1, 3]:
            assert 'data-video-label="observation.images.scene"' not in html
        csv_response = client.get(f"/remote/mixed/episode_{episode}/data.csv")
        assert csv_response.status_code == 200
        text = csv_response.get_data(as_text=True)
        if episode == 0:
            assert "joint_a" in text and "command_a" in text and "state.left_x" not in text
        if episode == 1:
            assert "state.left_x" in text and "joint_a" not in text
        if episode == 2:
            assert text.splitlines()[0] == "timestamp"
        if episode == 3:
            assert "qw" in text and "joint_a" not in text


def test_mixed_agent_capability_checked_at_submission_and_claim(mixed_root, tmp_path):
    from lerobot.data_platform.runtime.agent import _dataset_payload
    from tests.datasets.test_control_plane import _app, _bootstrap, _store
    from tests.datasets.test_data_platform_agent import _agent

    store = _store(tmp_path)
    client = _app(tmp_path, store).test_client()
    assert _bootstrap(client).status_code == 200
    agent = _agent(tmp_path, tmp_path, tmp_path)
    supported = {**agent.capabilities(), "job_protocol": 1}
    old = {key: value for key, value in supported.items() if key != "mixed_episode_viewer_protocol"}
    token, node = store.enroll_node(
        name="mixed-test",
        hostname="mixed-test",
        allowed_roots=[str(tmp_path)],
        writable_roots=[str(tmp_path)],
        capabilities=old,
        enrollment_token="enroll",
        expected_token="enroll",
    )
    location = store.sync_locations(node["node_id"], [_dataset_payload(mixed_root, node_name="mixed-test")])[
        0
    ]
    url = f"/api/control/locations/{location['location_id']}/viewer-jobs"
    assert client.post(url, json={}).status_code == 409
    headers = {"Authorization": f"Bearer {token}"}
    assert (
        client.post("/api/agents/heartbeat", headers=headers, json={"capabilities": supported}).status_code
        == 200
    )
    assert client.post(url, json={"force_recompute_stage": True}).status_code == 400
    assert client.post(url, json={}).status_code == 202
    assert (
        client.post("/api/agents/heartbeat", headers=headers, json={"capabilities": old}).status_code == 200
    )
    assert client.post("/api/agents/jobs/claim", headers=headers, json={}).get_json()["job"] is None
    assert (
        client.post("/api/agents/heartbeat", headers=headers, json={"capabilities": supported}).status_code
        == 200
    )
    assert client.post("/api/agents/jobs/claim", headers=headers, json={}).get_json()["job"] is not None
