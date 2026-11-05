"""UMI reports preserve source bytes and distinguish acquisition flags from evidence."""

import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from lerobot.data_platform.operators import ExecutionContext, get_operator
from lerobot.data_platform.operators.execution import execute_registered
from lerobot.data_platform.operators.schema import validate_parameters
from lerobot.data_platform.precompute.umi.comparison import run_comparison
from lerobot.data_platform.precompute.umi.diagnostics import diagnose, diagnose_part, episode_table
from lerobot.data_platform.precompute.umi.runtime import runtime_unavailable
from lerobot.data_platform.precompute.umi.worker import apply_frozen_alignment
from lerobot.data_platform.processing.planner import PipelineSpec
from tests.datasets.test_lifecycle import _snapshot
from tests.datasets.test_quality_eef import make_eef


def native(root):
    make_eef(root, "v3.0", "wxyz")
    path = root / "data/chunk-000/file-000.parquet"
    values = pq.read_table(path).to_pydict()
    info = json.loads((root / "meta/info.json").read_text())
    positions = np.asarray(values["left_arm_quaternion_pose"])[:, :3]
    r = Rotation.from_euler("xyz", np.tile([0.8, 0.4, -0.3], (600, 1)))
    flags = np.arange(600) % 100 < 50
    legacy = np.where(flags[:, None], r.as_rotvec(), r.as_euler("xyz"))
    for part in ("left_arm", "right_arm", "head"):
        values[part + "_quaternion_pose"] = np.c_[positions, r.as_quat()[:, [3, 0, 1, 2]]].tolist()
        values[part + "_pose"] = np.c_[positions, legacy].tolist()
        values[part + "_pose_is_valid"] = flags[:, None].tolist()
        info["features"][part + "_quaternion_pose"] = dict(info["features"]["left_arm_quaternion_pose"])
    pq.write_table(pa.table(values), path)
    (root / "meta/info.json").write_text(json.dumps(info))
    return root


def test_mixed_encoding_flags_are_reported_independently(tmp_path):
    source = native(tmp_path / "raw")
    before = _snapshot(source)
    report = diagnose(source, tmp_path / "report")
    assert _snapshot(source) == before
    assert report["summary"]["frames"] == 600
    assert report["summary"]["recorded_flag_semantics"] == "unconfirmed"
    p = report["episodes"][0]["parts"]["left_arm"]
    assert p["encoding_by_recorded_flag"]["true"]["rotvec"] == 150
    assert p["encoding_by_recorded_flag"]["false"]["euler_xyz"] == 150
    # The spike falls on a true flag: neither flag value is treated as a repair label.
    assert p["edge_statistics"]["recorded_true"]["position_jumps"] == 2
    assert p["recorded_false_frames"] == 150
    assert p["position_jump_frames"] == [100, 101]
    assert report["episodes"][1]["parts"]["left_arm"]["position_jump_frames"] == []
    json.dumps(report, allow_nan=False)


def test_split_v3_episode_preserves_identity_and_complete_rows(tmp_path):
    source = native(tmp_path / "raw")
    path = source / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 120), path)
    pq.write_table(table.slice(120), path.with_name("file-001.parquet"))
    report = diagnose(source, tmp_path / "report", episodes=[0])
    assert report["episodes"][0]["frames"] == 300
    assert report["episodes"][0]["parts"]["head"]["position_jump_frames"] == [100, 101]


def test_invalid_quaternion_and_timestamp_do_not_make_fake_rotation(tmp_path):
    pose = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (6, 1))
    pose[1, 3:] *= -1
    pose[2, 3:] = 0
    pose[3, 0] = float("nan")
    result = diagnose_part(pose, np.array([0, 1, 2, 2, 4, 5], dtype=float))
    assert result["rotation_jump_frames"] == []
    assert result["bad_timestamp_edges"] == [3]
    assert result["nonfinite_ranges"] == [[3, 4]]
    assert result["invalid_quaternion_ranges"] == [[2, 3]]
    json.dumps(result, allow_nan=False)


def test_diagnosis_rejects_output_overlap_and_unnamed_pose(tmp_path):
    source = native(tmp_path / "raw")
    with pytest.raises(ValueError, match="outside"):
        diagnose(source, source / "report")
    info = json.loads((source / "meta/info.json").read_text())
    info["features"]["head_quaternion_pose"].pop("names")
    (source / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="explicit"):
        diagnose(source, tmp_path / "report")


def test_episode_reader_rejects_duplicate_frames(tmp_path):
    source = native(tmp_path / "raw")
    path = source / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pydict()
    rows["frame_index"][1] = 0
    pq.write_table(pa.table(rows), path)
    with pytest.raises(ValueError, match="unique and contiguous"):
        episode_table(source, {"episode_index": 0, "length": 300}, ["timestamp"])


def test_episode_reader_checks_other_shards_for_duplicate_rows(tmp_path):
    source = native(tmp_path / "raw")
    path = source / "data/chunk-000/file-000.parquet"
    pq.write_table(pq.read_table(path).slice(0, 1), path.with_name("file-001.parquet"))
    with pytest.raises(ValueError, match="row count"):
        episode_table(source, {"episode_index": 0, "length": 300}, ["timestamp"])


def test_comparison_requires_matching_registered_sources(tmp_path):
    source = native(tmp_path / "raw")
    with pytest.raises(ValueError, match="registered"):
        run_comparison(ExecutionContext(source, work_dir=tmp_path / "out"), {})
    other = tmp_path / "other"
    shutil.copytree(source, other)
    meta = other / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(meta).to_pylist()
    rows[0]["length"] = 299
    pq.write_table(pa.Table.from_pylist(rows), meta)
    with pytest.raises(ValueError, match="frame counts"):
        run_comparison(ExecutionContext(source, source_roots=(source, other), work_dir=tmp_path / "out"), {})


def test_pipeline_has_review_and_evidence_gate():
    targets = [{"episode_index": 0, "part": "left", "start_frame": 10, "end_frame": 20}]
    with pytest.raises(ValueError, match="Missing required evidence"):
        PipelineSpec.from_dict(
            {
                "name": "bad",
                "steps": [
                    {"step_id": "id", "operator_id": "umi.cosmos_id", "parameters": {"targets": targets}}
                ],
            }
        )
    spec = PipelineSpec.from_dict(
        {
            "name": "review",
            "steps": [
                {"step_id": "diagnose", "operator_id": "umi.diagnose"},
                {
                    "step_id": "id",
                    "operator_id": "umi.cosmos_id",
                    "parameters": {"targets": targets},
                    "artifacts": ["diagnose"],
                },
            ],
        }
    )
    assert spec.steps[1].review_before
    assert get_operator("umi.cosmos_id").output_type == "artifact"
    with pytest.raises(ValueError, match="Unsupported parameters"):
        validate_parameters(get_operator("umi.cosmos_id"), {"targets": targets, "model_python": "/untrusted"})
    compare = PipelineSpec.from_dict(
        {
            "name": "compare",
            "steps": [
                {
                    "step_id": "compare",
                    "operator_id": "umi.compare",
                    "options": {"source_location_ids": ["other"]},
                }
            ],
        }
    )
    assert compare.steps[0].options["source_location_ids"] == ["other"]


def test_worker_does_not_accept_foreign_secondary_node(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    agent = SimpleNamespace(
        state_path=tmp_path / "state.json", allowed_roots=[tmp_path], state=SimpleNamespace(node_id="node")
    )
    job = {
        "job_id": "test",
        "operation": "umi.compare",
        "location_id": "primary",
        "execution": {"attempt_id": "a"},
        "options": {
            "parameters": {},
            "source_location_ids": ["primary", "other"],
            "_source_locations": [
                {"location_id": "primary", "node_id": "node", "root": str(source)},
                {"location_id": "other", "node_id": "foreign", "root": str(tmp_path / "other")},
            ],
        },
    }
    with pytest.raises(ValueError, match="registered node inputs"):
        execute_registered(agent, job, source, {"dataset_key": "x"})


def test_model_runtime_unavailable_without_node_configuration(monkeypatch):
    monkeypatch.delenv("DATA_PLATFORM_UMI_RUNTIME", raising=False)
    assert "node-owned" in runtime_unavailable("umi.cosmos_id")


def test_unknown_acquisition_flag_is_not_counted_false():
    pose = np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1))
    result = diagnose_part(pose, np.arange(3), [True, None, False])
    assert result["recorded_false_frames"] == 1
    assert result["recorded_unknown_frames"] == 1
    assert result["recorded_false_ranges"] == [[2, 3]]
    assert result["edge_statistics"]["flag_transition"]["edges"] == 0


def test_replay_freezes_world_hand_head_mapping_and_initial_height():
    @dataclass
    class Episode:
        transforms: np.ndarray
        metadata: dict
        end_effector_names: tuple = ("leftHandEE", "rightHandEE", "headEE")

    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("z", 0.8).as_matrix()
    world[:3, 3] = [1, 2, 3]
    hand = world.copy()
    hand[:3, 3] += [0.1, 0.2, 0.3]
    offset = Rotation.from_euler("x", 0.2).as_matrix()
    setup = {
        "world_transform": world.tolist(),
        "hand_world_transform": hand.tolist(),
        "head_orientation_offset": offset.tolist(),
        "torso_initial_q": -0.4,
    }
    source = Episode(np.tile(np.eye(4), (4, 3, 1, 1)), {})
    source.transforms[2, :, :3, 3] = [4, 5, 6]
    source.transforms[3, 2, :3, :3] = Rotation.from_euler("y", 0.7).as_matrix()
    pivot = np.array([0.1, 0.2, 0.3])
    result = apply_frozen_alignment(source, Episode(source.transforms, setup), pivot)
    assert result.metadata["torso_initial_q"] == -0.4
    assert np.allclose(result.transforms[:, 0], hand @ source.transforms[:, 0])
    assert np.allclose(
        result.transforms[:, 2, :3, :3], world[:3, :3] @ source.transforms[:, 2, :3, :3] @ offset
    )
    expected_pivot = (source.transforms[:, 2, :3, 3] + source.transforms[:, 2, :3, :3] @ pivot) @ world[
        :3, :3
    ].T + world[:3, 3]
    actual_pivot = result.transforms[:, 2, :3, 3] + result.transforms[:, 2, :3, :3] @ pivot
    assert np.allclose(actual_pivot, expected_pivot)


def test_gpu_worker_publishes_evidence_without_private_conversion(tmp_path, monkeypatch):
    from lerobot.data_platform import temporal_caption_flora
    from lerobot.data_platform.precompute.umi import runtime

    source = native(tmp_path / "source")
    work = tmp_path / "attempt/bundle/static"
    config = {
        "host_python": "/usr/bin/python3",
        "laundry_root": str(tmp_path),
        "lock_dir": str(tmp_path / "locks"),
        "gpus": [7],
        "timeout_seconds": 60,
    }
    monkeypatch.setattr(runtime, "load_runtime", lambda _: config)

    @contextmanager
    def reserve(*args):
        yield SimpleNamespace(gpu=7)

    monkeypatch.setattr(temporal_caption_flora, "acquire_gpu", reserve)

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            request = json.loads(Path(command[-1]).read_text())
            output = Path(request["output"])
            assert not output.is_relative_to(work.parent)
            (output / "result.json").write_text('{"status":"reported"}')
            (output / "eval_0/data").mkdir(parents=True)
            (output / "eval_0/data/large-private.parquet").write_bytes(b"private")
            (output / "report.html").write_text("<p>report</p>")

        def poll(self):
            return 0

    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    result = runtime.run_replay(ExecutionContext(source, work_dir=work), {})
    assert result.summary == {"status": "reported"}
    assert {p["path"] for p in result.artifacts} == {
        "runtime/result.json",
        "runtime/report.html",
        "runtime/worker.log",
    }
    assert not list(work.rglob("*.parquet"))
    assert not (work.parent.parent / "umi-runtime-private").exists()


def test_umi_diagnose_job_publishes_verified_artifacts(tmp_path):
    from lerobot.data_platform.curation import validate_bundle
    from tests.datasets.test_data_platform_agent import _agent

    source = native(tmp_path / "source")
    before = _snapshot(source)
    agent = _agent(tmp_path, source, tmp_path)
    target = {"location_id": "location", "dataset_key": "umi/source", "dataset_version_id": None}
    job = {
        "job_id": "umi-job",
        "operation": "umi.diagnose",
        "execution": {"attempt_id": "attempt"},
        "location": {"root": str(source), "dataset_key": "umi/source"},
        "options": {
            "parameters": {},
            "target": target,
            "operator_snapshot": {"operator_id": "umi.diagnose", "version": "1.0.0"},
        },
    }
    staged = tmp_path / "published"

    def upload(_state, _job, name, path, **kwargs):
        assert kwargs["curation"]
        destination = staged / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    agent.client.upload_artifact = upload
    result = agent.execute_job(job)
    assert result["summary"]["episodes"] == 2
    validate_bundle(staged, "umi.diagnose", target)
    assert (staged / "static/umi_diagnostics.html").is_file()
    assert _snapshot(source) == before


def test_umi_pipeline_browser_selects_registered_comparison(tmp_path):
    import os
    import threading

    from werkzeug.serving import make_server

    from lerobot.data_platform.operators.registry import operation_ids, worker_manifest
    from lerobot.data_platform.routes.processing import register_processing_routes
    from tests.datasets.test_control_plane import _app, _bootstrap, _store

    executable = os.environ.get("DATA_PLATFORM_BROWSER_EXECUTABLE")
    if not executable:
        pytest.skip("Set DATA_PLATFORM_BROWSER_EXECUTABLE for real-browser acceptance")
    playwright = pytest.importorskip("playwright.sync_api")
    control = _store(tmp_path)
    app = _app(tmp_path, control)
    register_processing_routes(app, SimpleNamespace(control_plane_store=control, lifecycle_store=None))
    assert _bootstrap(app.test_client()).status_code == 200
    root = native(tmp_path / "source")
    info = json.loads((root / "meta/info.json").read_text())
    # Catalog validation uses declared schema; video rendering is verified separately.
    for part in ("left_arm", "right_arm", "head"):
        info["features"][part + "_image"] = {"dtype": "image", "shape": [224, 224, 3]}
    _, node = control.enroll_node(
        name="umi-node",
        hostname="localhost",
        allowed_roots=[str(tmp_path)],
        writable_roots=[str(tmp_path)],
        enrollment_token="test",
        expected_token="test",
        capabilities={
            **worker_manifest(),
            "operations": sorted(operation_ids(remote=True)),
            "job_protocol": 2,
            "data_profile_protocol": 100,
        },
    )
    locations = control.sync_locations(
        node["node_id"],
        [
            {
                "dataset_key": "umi/" + name,
                "root": str(root if name == "raw" else tmp_path / "candidate"),
                "output_dir": str(tmp_path / (name + "-cache")),
                "metadata": info,
            }
            for name in ("raw", "candidate")
        ],
    )
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(
                executable_path=executable, args=["--no-sandbox", "--disable-gpu"]
            )
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            assert (
                page.request.post(
                    base + "/api/auth/login",
                    data={"username": "platform-admin", "password": "strong-admin-password"},
                ).status
                == 200
            )
            page.goto(base + "/pipelines?location_id=" + locations[0]["location_id"])
            page.get_by_role("button", name="Add step", exact=True).click()
            page.locator("#step-editor").get_by_label("Operator", exact=True).select_option("umi.compare")
            page.get_by_label("Comparison dataset", exact=True).select_option(locations[1]["location_id"])
            page.get_by_label("Name", exact=True).fill("UMI comparison")
            page.get_by_role("button", name="Save version", exact=True).click()
            page.wait_for_function("!document.getElementById('run').disabled")
            definition = page.request.get(base + "/api/pipelines").json()["pipelines"][0]
            saved = page.request.get(base + "/api/pipelines/" + definition["pipeline_id"]).json()
            assert saved["definition"]["steps"][0]["options"]["source_location_ids"] == [
                locations[1]["location_id"]
            ]
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.mark.parametrize("operation", ["umi.replay", "umi.cosmos_id"])
def test_foreign_python_workers_do_not_inherit_host_site_packages(tmp_path, monkeypatch, operation):
    import os
    import subprocess
    import sys

    from lerobot.data_platform.precompute.umi import diagnostics, worker

    source = native(tmp_path / "source")
    out = tmp_path / "output"
    out.mkdir()
    foreign = tmp_path / "host-site-packages"
    foreign.mkdir()
    (foreign / "umi_host_only.py").write_text('raise RuntimeError("host module leaked")\n')
    laundry = tmp_path / "laundry"
    (laundry / "src").mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", str(foreign))
    monkeypatch.setenv("PYTHONHOME", "/wrong-host-python")
    monkeypatch.setattr(worker, "prepare_evaluation", lambda source, *args: source)
    monkeypatch.setattr(diagnostics, "source_fingerprint", lambda root: {"meta/info.json": "same"})
    actual_run = subprocess.run
    checked = []

    def probe(*args, **kwargs):
        env = kwargs.get("env", os.environ.copy())
        result = actual_run(
            [
                sys.executable,
                "-c",
                'import importlib.util; assert importlib.util.find_spec("umi_host_only") is None',
            ],
            env=env,
            check=True,
        )
        assert "PYTHONHOME" not in env
        assert env["PYTHONPATH"] == str(laundry / "src")
        checked.append(True)
        (out / "result.json").write_text('{"status":"reported"}')
        return result

    if operation == "umi.cosmos_id":

        def fake_cosmos(*args):
            probe()
            return {"status": "needs_review"}

        monkeypatch.setattr(worker, "cosmos", fake_cosmos)
    else:
        monkeypatch.setattr(worker.subprocess, "run", probe)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": operation,
                "output": str(out),
                "runtime": {"laundry_root": str(laundry), "cumotion_python": sys.executable},
                "source_roots": [str(source)],
                "parameters": {},
            }
        )
    )
    worker.run_request(request)
    assert checked == [True]
