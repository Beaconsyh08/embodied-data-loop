import json

import pytest

from lerobot.data_platform.memory import model_trial


def point(frame, *, memory="No events have been confirmed yet."):
    return {
        "facts": [],
        "checkpoint": {
            "frame": frame,
            "memory": memory,
            "subtask": "Approach the visible garment in the drum.",
            "memory_fact_ids": [],
            "required_fact_ids": [],
            "allowed_fact_ids": [],
            "acceptable_actions": [{"action": "approach", "object": "garment", "target": "drum"}],
            "decision": "usable",
            "decision_evidence_frames": [frame],
        },
    }


class Client:
    def __init__(self, values):
        self.values = iter(values)
        self.requests = []

    def post_chat_completion(self, payload):
        self.requests.append(payload)
        return {"choices": [{"message": {"content": json.dumps(next(self.values))}}]}


def setup(monkeypatch):
    monkeypatch.setattr(model_trial, "image_at", lambda source, rows, frame, output: str(frame).encode())
    rows = [{"timestamp": i / 30, "state": "FORBIDDEN", "action": "FORBIDDEN"} for i in range(61)]
    pin = {"episode_index": 0, "task": "Transfer garments", "model": "test", "checkpoints": [0, 30]}
    return rows, pin


def test_revision_is_checked_and_never_human_approval(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    client = Client(
        [
            point(0, memory="Unsupported"),
            {"supported": False, "issues": ["Unsupported claim"]},
            point(0),
            {"supported": True, "issues": []},
            point(30),
            {"supported": True, "issues": []},
        ]
    )
    state = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert state["complete"] and state["human_approvals"] == 0
    assert state["record"]["review"] == {"status": "draft"}
    assert len(state["entries"][0]["revisions"]) == 2
    assert "FORBIDDEN" not in json.dumps(client.requests)
    # Student check never receives teacher facts, history or revision comments.
    for request in client.requests[1::2]:
        text = request["messages"][0]["content"][0]["text"]
        context = json.loads(text[len(model_trial.CHECK) + 1 :])
        assert set(context) == {"task", "previous_memory", "proposed_output"}
        assert len([c for c in request["messages"][0]["content"] if c["type"] == "image_url"]) == 1
    text = client.requests[4]["messages"][0]["content"][0]["text"]
    context = json.loads(text[len(model_trial.DRAFT) + 1 :])
    assert context["image_frames"] == [0, 7, 15, 22, 30]


def test_persistent_failure_stops_chain_and_resume_uses_receipts(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    rejected = [point(0), {"supported": False, "issues": ["Ambiguous target"]}] * 3
    client = Client(rejected)
    state = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert not state["complete"] and not state["record"]["checkpoints"]
    assert len(state["entries"]) == 1
    replay = Client([])
    same = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", replay)
    assert same == state and not replay.requests


def test_future_evidence_cannot_enter_chain(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    bad = point(0)
    bad["checkpoint"]["decision_evidence_frames"] = [30]
    client = Client([bad] * 3)
    state = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert not state["record"]["checkpoints"]
    assert all("Unseen or future" in r["issues"][0] for r in state["entries"][0]["revisions"])


def test_explicit_extra_revision_still_requires_student_check(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    pin["max_label_revisions"] = 6
    client = Client(
        [point(0), {"supported": False, "issues": ["Unsupported hand identity"]}] * 3
        + [point(0), {"supported": True, "issues": []}, point(30), {"supported": True, "issues": []}]
    )
    state = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert state["complete"] and len(state["entries"][0]["revisions"]) == 4
    request = client.requests[6]["messages"][0]["content"][0]["text"]
    context = json.loads(request[len(model_trial.DRAFT) + 1 :])
    assert "Conflicting checker comments are not evidence" in context["revision_request"]["instruction"]
    assert state["record"]["review"]["status"] == "draft"


def test_final_repair_uses_student_inputs_without_clip_or_hidden_facts(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    pin["max_label_revisions"] = 9
    client = Client(
        [point(0), {"supported": True, "issues": []}]
        + [point(30), {"supported": False, "issues": ["Unsupported location"]}] * 6
        + [point(30), {"supported": True, "issues": []}]
    )
    state = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert state["complete"] and len(state["entries"][1]["revisions"]) == 7
    content = client.requests[14]["messages"][0]["content"]
    context = json.loads(content[0]["text"][len(model_trial.DRAFT) + 1 :])
    assert context["image_frames"] == [30]
    assert not {"observed_facts", "previous_subtask", "history_frames"} & set(context)
    assert set(context["revision_request"]["proposed_output"]) == {"memory", "subtask"}
    assert len([p for p in content if p["type"] == "image_url"]) == 1


def test_balanced_repair_preserves_prefix_and_checks_only_actual_claims(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    replies = [point(0), {"supported": True, "issues": []}, point(30), {"supported": True, "issues": []}]
    original = Client(replies)
    model_trial.annotate(tmp_path, rows, pin, tmp_path / "original", original)
    pin["student_visible_from_frame"] = 30
    repaired = Client(replies)
    state = model_trial.annotate(tmp_path, rows, pin, tmp_path / "repaired", repaired)
    assert state["complete"] and repaired.requests[:2] == original.requests[:2]
    draft = repaired.requests[2]["messages"][0]["content"]
    context = json.loads(draft[0]["text"][len(model_trial.DRAFT) + 1 :])
    assert context["image_frames"] == [30] and "observed_facts" not in context
    check = repaired.requests[3]["messages"][0]["content"][0]["text"]
    assert check.startswith(model_trial.BALANCED_CHECK)
    assert "does not assert who put it there" in check


def test_protocol_two_checker_has_no_frame_metadata_anywhere(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    pin["student_input_protocol"] = 2
    client = Client(
        [point(0), {"supported": True, "issues": []}, point(30), {"supported": True, "issues": []}]
    )
    state = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert state["complete"]
    for request in client.requests[1::2]:
        content = request["messages"][0]["content"]
        assert [c["type"] for c in content] == ["text", "image_url"]
        prefix = model_trial.BALANCED_CHECK + model_trial.TEXT_TARGET_RULES + "\n"
        context = json.loads(content[0]["text"][len(prefix) :])
        assert set(context) == {"task", "previous_memory", "proposed_output"}
        assert "ORIGINAL EPISODE FRAME ID" not in json.dumps(request)


@pytest.mark.parametrize(
    "memory",
    [
        "At frame 30 a garment is visible.",
        "A garment is visible (f12).",
        "A garment is visible (frame30_revision1_garment).",
        "word " * 51,
    ],
)
def test_protocol_two_rejects_bookkeeping_and_unbounded_memory(tmp_path, monkeypatch, memory):
    rows, pin = setup(monkeypatch)
    pin.update(student_input_protocol=2, checkpoints=[0])
    bad = point(0, memory=memory)
    if "frame30_revision1" in memory:
        bad["facts"] = [
            {
                "id": "frame30_revision1_garment",
                "text": "A garment is visible.",
                "status": "confirmed",
                "confirmed_at_frame": 0,
                "evidence": [{"frame": 0, "camera": "image"}],
            }
        ]
    client = Client([bad] * 3)
    result = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert not result["record"]["checkpoints"] and len(client.requests) == 3
    assert all(r["issues"] for r in result["entries"][0]["revisions"])


def test_protocol_two_repairs_invalid_text_from_student_visible_evidence(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    pin.update(student_input_protocol=2, checkpoints=[0])
    client = Client(
        [point(0, memory="At frame 0 no events are confirmed."), point(0), {"supported": True, "issues": []}]
    )
    result = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert result["complete"]
    prompt = model_trial.DRAFT + model_trial.TEXT_TARGET_RULES + "\n"
    context = json.loads(client.requests[1]["messages"][0]["content"][0]["text"][len(prompt) :])
    assert "observed_facts" not in context
    assert context["image_frames"] == [0]


def test_physical_door_frame_is_not_frame_metadata():
    draft = point(0)
    draft["checkpoint"]["subtask"] = "Move the held cloth toward the basket, clear of the door frame."
    model_trial.validate_target_text(draft, [])
