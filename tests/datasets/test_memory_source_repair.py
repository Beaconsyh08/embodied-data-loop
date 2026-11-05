"""Repair geometry is not a model prediction; new captions require isolated visual evidence."""

import copy
import io
import json

import pytest
from PIL import Image

from lerobot.data_platform.memory.source_repair import CAMERAS, repair, repair_plan, validate_caption
from tests.datasets.test_temporal_caption_flora import flora_artifact


def test_repair_preserves_exact_intervals_and_regenerates_clipped_gap_overlap(tmp_path):
    original = flora_artifact()
    untouched = copy.deepcopy(original)
    plan = repair_plan(original)
    assert [(r["start_frame"], r["end_frame"], r["reuse"]) for r in plan] == [
        (0, 3, True),
        (3, 5, False),
        (5, 8, False),
        (8, 9, False),
        (9, 12, False),
    ]
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, format="PNG")
    rows = [
        {**{c: {"bytes": buffer.getvalue()} for c in CAMERAS}, "action": "FORBIDDEN_ACTION"}
        for _ in range(12)
    ]

    class Teacher:
        def __init__(self):
            self.calls = []

        def post_chat_completion(self, payload):
            self.calls.append(payload)
            context = json.loads(payload["messages"][0]["content"][0]["text"].split("\n")[-1])
            content = payload["messages"][0]["content"]
            assert payload["response_format"] == {"type": "json_object"}
            assert [part["text"] for part in content[1:] if part["type"] == "text"] == [
                f"ORIGINAL EPISODE FRAME ID: {frame}" for frame in context["image_frames"]
            ]
            assert all(
                context["target_start_frame"] <= f < context["target_end_frame_exclusive"]
                for f in context["image_frames"]
            )
            assert not any(
                word in json.dumps(payload)
                for word in ("FORBIDDEN_ACTION", "Completed after retry", '"Reach"', '"Grasp"', '"Lift"')
            )
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "caption": "The gripper is visible.",
                                    "uncertainty": [],
                                    "evidence": [{"frame": context["image_frames"][0], "camera": "image"}],
                                }
                            )
                        }
                    }
                ]
            }

    teacher = Teacher()
    output = tmp_path / "output"
    result = repair(original, "original-sha", tmp_path, rows, "Task", output, teacher, "test", "provider")
    assert original == untouched and len(teacher.calls) == 4
    assert result["native_segments"][0]["caption"] == "Reach"
    assert all(len(r["native_segment_indices"]) == 1 for r in result["result"]["segments"])
    assert result["reward"] == original["reward"]
    audit = result["coverage_repair"]
    assert audit["status"] == "draft" and not audit["human_reviewed"]
    assert audit["intervals"][3]["needs_review"]  # one frame cannot establish motion
    repair(original, "original-sha", tmp_path, rows, "Task", output, teacher, "test", "provider")
    assert len(teacher.calls) == 4  # exact input receipts are reused
    repair(original, "original-sha", tmp_path, rows, "Task", output, teacher, "test", "changed-provider")
    assert len(teacher.calls) == 8


@pytest.mark.parametrize(
    "change",
    [
        {"caption": ""},
        {"evidence": [{"frame": 8, "camera": "image"}]},
        {"evidence": [{"frame": 3, "camera": "state"}]},
        {"uncertainty": "maybe"},
    ],
)
def test_invalid_repair_response_never_silently_fixed(change):
    value = {"caption": "Visible gripper", "evidence": [{"frame": 3, "camera": "image"}], "uncertainty": []}
    with pytest.raises(ValueError):
        validate_caption({**value, **change}, [3, 4])
