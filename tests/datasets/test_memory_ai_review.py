import io
import json
import shutil

import pytest
from PIL import Image, ImageChops

from lerobot.data_platform.memory import ai_review, model_trial
from tests.datasets.test_memory_model_trial import Client, point, setup


def verdict(supported=True):
    return {
        "observations": [{"camera": c, "visible": "A garment is visible."} for c in ai_review.CAMERAS],
        "assessments": [
            {
                "field": f,
                "claim": "A visible garment.",
                "supported": supported,
                "basis": "current_image",
                "cameras": ["image"],
                "evidence": "Visible in main view.",
            }
            for f in ("memory", "subtask")
        ],
        "supported": supported,
        "issues": [] if supported else ["Contact is occluded."],
    }


def test_panels_preserve_exact_student_pixels():
    source = Image.new("RGB", (672, 224))
    for i, color in enumerate(("red", "green", "blue")):
        source.paste(color, (224 * i, 0, 224 * (i + 1), 224))
    buf = io.BytesIO()
    source.save(buf, format="PNG")
    panels = ai_review.camera_panels(buf.getvalue())
    for i, panel in enumerate(panels):
        actual = Image.open(io.BytesIO(panel))
        assert actual.size == (224, 224)
        assert ImageChops.difference(actual, source.crop((i * 224, 0, (i + 1) * 224, 224))).getbbox() is None
    bad = io.BytesIO()
    Image.new("RGB", (224, 224)).save(bad, format="PNG")
    with pytest.raises(ValueError, match="fixed"):
        ai_review.camera_panels(bad.getvalue())


@pytest.mark.parametrize(
    "mutation", ["missing_view", "missing_subtask", "false_positive", "unsupported_empty", "missing_evidence"]
)
def test_ai_review_cannot_silently_approve_inconsistent_assessments(mutation):
    review = verdict()
    if mutation == "missing_view":
        review["observations"].pop()
    elif mutation == "missing_subtask":
        review["assessments"].pop()
    elif mutation == "false_positive":
        review["assessments"][0]["supported"] = False
    elif mutation == "unsupported_empty":
        review = verdict(False)
        review["issues"] = []
    else:
        review["assessments"][0]["evidence"] = ""
    with pytest.raises(ValueError):
        ai_review.validate_review(review)


def test_revision_reuses_prefix_and_new_check_has_only_student_inputs(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    pin["student_input_protocol"] = 2
    monkeypatch.setattr(
        ai_review, "camera_panels", lambda raw: [raw + b"main", raw + b"left", raw + b"right"]
    )
    original_client = Client(
        [point(0), {"supported": True, "issues": []}, point(30), {"supported": True, "issues": []}]
    )
    original = model_trial.annotate(tmp_path, rows, pin, tmp_path / "old", original_client)
    (tmp_path / "new").mkdir()
    shutil.copytree(tmp_path / "old/requests", tmp_path / "new/requests")
    pin.update(ai_review_protocol=1, ai_review_from_frame=30)
    client = Client([point(30), verdict()])
    result = model_trial.annotate(tmp_path, rows, pin, tmp_path / "new", client)
    assert result["complete"] and result["human_approvals"] == 0
    assert result["record"]["review"] == {"status": "draft"}
    assert len(client.requests) == 2
    assert result["entries"][0] == original["entries"][0]
    assert result["entries"][1]["revisions"][0]["ai_review"] == verdict()
    check = client.requests[-1]
    content = check["messages"][0]["content"]
    context = json.loads(content[0]["text"][len(ai_review.CHECK_PROMPT) + 1 :])
    assert set(context) == {"task", "previous_memory", "proposed_output"}
    assert [c["type"] for c in content] == ["text", "image_url", "image_url", "image_url"]
    assert "FORBIDDEN" not in json.dumps(client.requests)
    replay = Client([])
    assert model_trial.annotate(tmp_path, rows, pin, tmp_path / "new", replay) == result
    assert not replay.requests
    pin["ai_review_from_frame"] = 0
    with pytest.raises(ValueError, match="Inputs changed"):
        model_trial.annotate(tmp_path, rows, pin, tmp_path / "new", replay)


def test_ai_review_rejection_stops_following_memory(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    monkeypatch.setattr(ai_review, "camera_panels", lambda raw: [raw] * 3)
    pin.update(student_input_protocol=2, ai_review_protocol=1, ai_review_from_frame=0)
    client = Client([point(0), verdict(False)] * 3)
    result = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert not result["complete"] and not result["record"]["checkpoints"]
    assert len(result["entries"]) == 1
    assert all(a["student_check"]["supported"] is False for a in result["entries"][0]["revisions"])


def test_agent_correction_is_teacher_feedback_not_student_evidence(tmp_path, monkeypatch):
    rows, pin = setup(monkeypatch)
    monkeypatch.setattr(ai_review, "camera_panels", lambda raw: [raw] * 3)
    pin.update(
        student_input_protocol=2,
        ai_review_protocol=1,
        ai_review_from_frame=0,
        agent_reviews={"30": "REVIEWER_NOTE: avoid unsupported empty-drum claim"},
    )
    client = Client([point(0), verdict(), point(30), verdict()])
    result = model_trial.annotate(tmp_path, rows, pin, tmp_path / "out", client)
    assert result["complete"]
    assert "REVIEWER_NOTE" not in json.dumps(client.requests[:2])
    assert "REVIEWER_NOTE" in json.dumps(client.requests[2])
    assert "REVIEWER_NOTE" not in json.dumps(client.requests[3])
    assert result["human_approvals"] == 0
