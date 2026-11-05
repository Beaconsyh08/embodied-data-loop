"""Synthetic-only memory safety, revision and compatibility regressions. No model requests."""

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from lerobot.data_platform.execution import StopRequestedError, atomic_json
from lerobot.data_platform.lifecycle_repository import SQLiteLifecycleRepository
from lerobot.data_platform.memory.contracts import AnnotationConfig, checkpoints, history_frames
from lerobot.data_platform.memory.generation import call_model, generate
from lerobot.data_platform.memory.review import MemoryReviewStore
from lerobot.data_platform.memory.sources import select_pilot
from lerobot.data_platform.qwen import QwenRequestError


def sample(tmp_path, count=5):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    frames = []
    for frame in range(count):
        images = []
        for camera in ("main", "left", "right"):
            path = f"{frame}-{camera}.png"
            Image.new("RGB", (20, 10), (frame * 20, 0, 0)).save(bundle / path)
            images.append({"camera": camera, "path": path, "decode_error_s": 0})
        frames.append({"frame_index": frame, "timestamp": frame / 4, "images": images})
    (bundle / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    (bundle / "episodes.jsonl").write_text(
        json.dumps({"id": "episode-000000", "frames_file": "frames.jsonl"}) + "\n"
    )
    pin = {
        "task": "Move the block",
        "cameras": ["main", "left", "right"],
        "config": AnnotationConfig("test-model").to_dict(),
        "timestamps": [f["timestamp"] for f in frames],
        "checkpoints": list(range(count)),
        "frame_count": count,
        "segment_starts": [0],
        "episode_index": 0,
        "source_fingerprint": "source",
    }
    return bundle, pin


def draft(frame):
    return {
        "facts": [],
        "checkpoint": {
            "frame": frame,
            "memory": "No events have been confirmed yet.",
            "subtask": "Observe block",
            "memory_fact_ids": [],
            "required_fact_ids": [],
            "allowed_fact_ids": [],
            "acceptable_actions": [{"action": "observe", "object": "block", "target": "table"}],
            "decision": "usable",
            "decision_evidence_frames": [frame],
        },
    }


class Teacher:
    def __init__(self, *, unsupported=None, invalid=None, uncertain=None):
        self.calls = []
        self.unsupported = unsupported
        self.invalid = invalid
        self.uncertain = uncertain

    def post_chat_completion(self, payload):
        self.calls.append(copy.deepcopy(payload))
        text = payload["messages"][0]["content"][0]["text"]
        context = json.loads(text.split("\n")[-1])
        if "current_frame" in context:
            self.frame = context["current_frame"]
            value = draft(self.frame)
            if self.frame == self.invalid:
                value["checkpoint"]["decision_evidence_frames"] = [self.frame + 1]
            if self.frame == self.uncertain:
                value["facts"] = [
                    {
                        "id": "unclear",
                        "text": "Possibly a second block",
                        "confirmed_at_frame": self.frame,
                        "status": "uncertain",
                        "evidence": [{"frame": self.frame, "camera": "main"}],
                    }
                ]
        else:
            value = {
                "supported": self.frame != self.unsupported,
                "issues": ["Count only visible in earlier video"] if self.frame == self.unsupported else [],
            }
        return {"choices": [{"message": {"content": json.dumps(value)}}], "usage": {"total_tokens": 10}}


def test_schedule_history():
    timestamps = [i / 10 for i in range(31)]
    assert checkpoints(timestamps, [0, 7]) == [0, 7, 10, 20, 30]
    config = AnnotationConfig("test")
    assert history_frames(timestamps, 0, config) == []
    assert history_frames(timestamps, 1, config) == []
    assert len(history_frames(timestamps, 30, config)) == 8
    assert all(timestamps[i] <= 3 and timestamps[i] >= 1 for i in history_frames(timestamps, 30, config))
    with pytest.raises(ValueError):
        checkpoints([0, 0], [0])


def test_two_isolated_requests_stop_and_resume(tmp_path):
    bundle, pin = sample(tmp_path)
    # These source fields must never reach either model request.
    pin.update(reward="FUTURE_REWARD", caption="FUTURE_CAPTION", state="SECRET_STATE", action="SECRET_ACTION")
    teacher = Teacher(unsupported=2)
    state = generate(bundle, pin, tmp_path / "run", teacher)
    assert len(teacher.calls) == 6
    assert state["entries"][-1]["state"] == "needs_review"
    assert state["record"]["checkpoints"][-1]["decision"] == "needs_review"
    for index, payload in enumerate(teacher.calls):
        text = json.dumps(payload)
        assert all(
            v not in text for v in ("FUTURE_REWARD", "FUTURE_CAPTION", "SECRET_STATE", "SECRET_ACTION")
        )
        if index % 2:
            context = json.loads(payload["messages"][0]["content"][0]["text"].split("\n")[-1])
            assert set(context) == {"task", "previous_memory", "proposed_output"}
            assert len(payload["messages"][0]["content"]) == 2
    resumed = Teacher()
    again = generate(bundle, pin, tmp_path / "run", resumed, previous=state)
    assert not resumed.calls  # identical blocked responses are durable and remain pending review
    assert again["entries"][-1]["state"] == "needs_review"


def test_invalid_and_uncertain_outputs_stop(tmp_path):
    bundle, pin = sample(tmp_path)
    for field, status in (("invalid", "invalid_response"), ("uncertain", "needs_review")):
        teacher = Teacher(**{field: 1})
        state = generate(bundle, pin, tmp_path / field, teacher)
        assert state["entries"][-1]["state"] == status
        assert state["entries"][-1]["frame"] == 1
        assert len(teacher.calls) == (3 if field == "invalid" else 4)


def test_transport_retry_cancel_and_schema_not_repaired(tmp_path):
    class Failing:
        def __init__(self):
            self.calls = 0

        def post_chat_completion(self, payload):
            self.calls += 1
            raise QwenRequestError("Rate limited", retryable=True, status_code=429)

    client = Failing()
    with pytest.raises(QwenRequestError):
        call_model(client, {}, tmp_path / "receipt.json", lambda: None, sleep=lambda _: None)
    assert client.calls == 3
    assert len(json.loads((tmp_path / "receipt.json").read_text())["attempts"]) == 3

    def stop():
        raise StopRequestedError()

    with pytest.raises(StopRequestedError):
        call_model(client, {}, tmp_path / "cancel.json", stop)
    assert client.calls == 3


def test_edit_invalidates_suffix_and_approval_and_conflicts(tmp_path):
    from lerobot.data_platform.job_management import JobConflictError

    bundle, pin = sample(tmp_path)
    state = generate(bundle, pin, tmp_path / "run", Teacher())
    store = MemoryReviewStore(SQLiteLifecycleRepository(tmp_path / "db"))
    store.create("run", "location", state)
    confirmations = {"source_confirmed": True, "goal_confirmed": True, "causal_evidence_confirmed": True}
    approved = store.change("run", 1, "alice", "approve", confirmations)
    assert approved["state"]["record"]["review"]["reviewer"] == "alice"
    with pytest.raises(ValueError, match="distinct"):
        store.change(
            "run", 2, "alice", "approve", {**confirmations, "independent": True, "adjudicated": True}
        )
    store.change("run", 2, "bob", "approve", {**confirmations, "independent": True, "adjudicated": True})
    point = draft(1)["checkpoint"]
    point["subtask"] = "Inspect block"
    changed = store.change(
        "run",
        3,
        "alice",
        "edit",
        {"frame": 1, "facts": [], "checkpoint": point, "reason": "Correct next task"},
    )
    assert changed["state"]["record"]["review"] == {"status": "draft"}
    assert [e["state"] for e in changed["state"]["entries"]] == [
        "complete",
        "complete",
        "stale",
        "stale",
        "stale",
    ]
    assert store.get("run", 2)["state"]["record"]["review"]["status"] == "approved"
    with pytest.raises(JobConflictError):
        store.change("run", 3, "bob", "needs_review", {"frame": 1})
    teacher = Teacher()
    new = generate(bundle, pin, tmp_path / "new", teacher, previous=changed["state"])
    assert len(teacher.calls) == 6
    assert new["record"]["checkpoints"][1]["subtask"] == "Inspect block"


def test_pilot_whole_groups_and_shortfall():
    rows = [
        {"episode_index": i, "duration_s": i + 1, "identity": {"lineage_group": str(i)}} for i in range(14)
    ]
    pilot = select_pilot(rows)
    assert pilot == select_pilot(rows)
    assert pilot["counts"] == {"train": 6, "calibration": 1, "validation": 2, "test": 1}
    assert pilot["shortfall"] == 0
    assert select_pilot(rows[:4])["shortfall"] == 6
    for row in rows:
        row["identity"]["lineage_group"] = "one recording"
    assert len(set(select_pilot(rows)["assignments"].values())) == 1


def test_causal_video_decoder_never_chooses_future(tmp_path):
    from lerobot.data_platform.mem_export import _video_images
    from tests.datasets.test_mem_export import fixture

    root, _, _ = fixture(tmp_path, version="v2.1", video=True)
    path = root / "videos/head/000000.mp4"
    nearest = list(_video_images(path, [0.26], 0.26))
    causal = list(_video_images(path, [0.26], 0.51, causal=True))
    assert nearest[0][1] == 0.5
    assert causal[0][1] == 0


def real_source(tmp_path, count=2):
    """Tiny image-backed LeRobot source; all pixels and captions are synthetic."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot.data_platform.temporal_caption_flora import build_artifact
    from lerobot.data_platform.temporal_caption_snapshot import caption_source_fingerprint

    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data").mkdir()
    cameras = ["main", "left_wrist", "right_wrist"]
    info = {
        "codebase_version": "v2.1",
        "fps": 2,
        "total_episodes": count,
        "total_frames": 2 * count,
        "data_path": "data/{episode_index:06d}.parquet",
        "features": {c: {"dtype": "image"} for c in cameras},
    }
    atomic_json(source / "meta/info.json", info)
    (source / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": ep, "length": 2, "tasks": ["Move block"]}) + "\n"
            for ep in range(count)
        )
    )
    for ep in range(count):
        image = io.BytesIO()
        Image.new("RGB", (16, 16), (ep * 20, 0, 0)).save(image, format="PNG")
        rows = [
            {
                "episode_index": ep,
                "frame_index": f,
                "timestamp": f / 2,
                **{c: {"bytes": image.getvalue(), "path": None} for c in cameras},
            }
            for f in range(2)
        ]
        pq.write_table(pa.Table.from_pylist(rows), source / f"data/{ep:06d}.parquet")
    fingerprint = caption_source_fingerprint(source, tmp_path / "hashes")
    selections = []
    for ep in range(count):
        artifact = build_artifact(
            {"episode_index": ep, "frame_count": 2, "fps": 2},
            "fixture",
            {"segments": [{"start_s": 0, "end_s": 1, "caption": "Move block", "status": "complete"}]},
            {
                "terminal": {"score": 3, "rationale": "Synthetic only"},
                "windows": [{"scores": [3, 3], "output": "Synthetic reward"}],
                "merged_scores": [3, 3],
                "merged_source_window": [0, 0],
                "disagreement_intervals": [],
                "human_reviewed": False,
            },
            [],
        )
        artifact["source_fingerprint"] = fingerprint
        path = tmp_path / f"flora-{ep}.json"
        atomic_json(path, artifact)
        selections.append(
            {
                "dataset_key": "fixture",
                "episode_index": ep,
                "flora_run": f"flora-{ep}",
                "annotation": str(path),
                "cameras": cameras,
                "task": "Move block",
                "lineage_group": f"recording-{ep}",
            }
        )
    return source, selections


def test_source_binding_export_and_agent_adapter(tmp_path, monkeypatch):
    from lerobot.data_platform.memory.jobs import execute
    from lerobot.data_platform.memory.sources import preview
    from lerobot.data_platform.operators import ExecutionContext

    source, selections = real_source(tmp_path)
    config = AnnotationConfig("test-model")
    monkeypatch.setenv("DATA_PLATFORM_MEMORY_MODEL", "test-model")
    pin = preview(source, selections[0], config, tmp_path / "hashes")
    assert pin["checkpoints"] == [0, 1]
    static = tmp_path / "published"
    static.mkdir()
    input_path = tmp_path / "input.json"
    atomic_json(
        input_path,
        {"request_id": "test", "operation": "memory.annotate", "dataset_key": "fixture", "pin": pin},
    )
    teacher = Teacher()
    monkeypatch.setattr("lerobot.data_platform.memory.jobs.QwenClient", lambda **_: teacher)
    context = ExecutionContext(
        source_root=source,
        output_root=tmp_path / "data-disk",
        dataset_key="fixture",
        work_dir=static,
        artifacts={"input.json": input_path, "flora.json": Path(selections[0]["annotation"])},
    )
    result = execute(context, {"request_id": "test"})
    assert len(teacher.calls) == 4
    assert result.summary["status"] == "draft"
    assert (static / "review.json").exists()
    assert (static / "frames/1.png").exists()
    execute(context, {"request_id": "test"})
    assert len(teacher.calls) == 4
    with pytest.raises(ValueError, match="fingerprint"):
        artifact = json.loads(Path(selections[1]["annotation"]).read_text())
        artifact["source_fingerprint"] = "bad"
        atomic_json(Path(selections[1]["annotation"]), artifact)
        preview(source, selections[1], config, tmp_path / "hashes")


def test_changed_sufficiency_prompt_invalidates_prefix(tmp_path, monkeypatch):
    import lerobot.data_platform.memory.generation as generation

    bundle, pin = sample(tmp_path, count=1)
    state = generate(bundle, pin, tmp_path / "run", Teacher())
    monkeypatch.setattr(generation, "CHECK_PROMPT", generation.CHECK_PROMPT + "Check identity again.\n")
    changed = Teacher()
    generate(bundle, pin, tmp_path / "run", changed, previous=state)
    assert len(changed.calls) == 2


def test_empty_provider_choices_are_visible_invalid_response(tmp_path):
    class Empty:
        def post_chat_completion(self, payload):
            return {"choices": []}

    bundle, pin = sample(tmp_path)
    state = generate(bundle, pin, tmp_path / "run", Empty())
    assert len(state["entries"]) == 1
    assert state["entries"][0]["state"] == "invalid_response"
    assert not state["record"]["checkpoints"]
