"""Pinned inputs and strict causal contracts for memory annotation protocol 1."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass

EMPTY_MEMORY = "No events have been confirmed yet."
PROTOCOL = 1


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class AnnotationConfig:
    model: str
    history_seconds: float = 2.0
    history_hz: float = 4.0
    temperature: float = 0.0
    prompt_version: str = "memory-1"
    panel_size: int = 224

    def __post_init__(self):
        if not self.model or not isinstance(self.model, str) or len(self.model) > 128:
            raise ValueError("A configured model is required")
        if (
            self.history_seconds,
            self.history_hz,
            self.temperature,
            self.prompt_version,
            self.panel_size,
        ) != (2.0, 4.0, 0.0, "memory-1", 224):
            raise ValueError("Unsupported protocol 1 sampling or prompt configuration")

    def to_dict(self):
        return asdict(self)


def checkpoints(timestamps: list[float], starts: list[int], extra=()) -> list[int]:
    if (
        not timestamps
        or any(not math.isfinite(t) for t in timestamps)
        or any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False))
    ):
        raise ValueError("Timestamps must be finite and strictly increasing")
    selected = {0, len(timestamps) - 1, *starts, *extra}
    if any(type(t) is not int or not 0 <= t < len(timestamps) for t in selected):
        raise ValueError("Checkpoint outside episode")
    next_second = timestamps[0]
    for index, timestamp in enumerate(timestamps):
        if timestamp >= next_second:
            selected.add(index)
            next_second = timestamp + 1.0
    return sorted(selected)


def history_frames(timestamps: list[float], cutoff: int, config: AnnotationConfig) -> list[int]:
    """Past-only samples in [t-2,t), with current t supplied separately. No future rounding."""
    end = timestamps[cutoff]
    selected = set()
    for step in range(1, int(config.history_seconds * config.history_hz) + 1):
        target = end - step / config.history_hz
        index = bisect_right(timestamps, target, hi=cutoff) - 1
        if index >= 0 and timestamps[index] >= end - config.history_seconds:
            selected.add(index)
    return sorted(selected)


def validate_point(value: dict, facts: list[dict], *, frame: int, cameras: list[str], visible: set[int]):
    if not isinstance(value, dict) or set(value) != {"facts", "checkpoint"}:
        raise ValueError("Expected exactly facts and checkpoint")
    known = {f["id"]: f for f in facts}
    if not isinstance(value["facts"], list):
        raise ValueError("facts must be a list")
    for fact in value["facts"]:
        if not isinstance(fact, dict) or set(fact) != {
            "id",
            "text",
            "status",
            "confirmed_at_frame",
            "evidence",
        }:
            raise ValueError("Invalid fact schema")
        if not isinstance(fact["id"], str) or not fact["id"] or fact["id"] in known:
            raise ValueError("Duplicate or invalid fact ID")
        if not isinstance(fact["text"], str) or not fact["text"].strip():
            raise ValueError("Fact text is required")
        if type(fact["confirmed_at_frame"]) is not int or fact["confirmed_at_frame"] != frame:
            raise ValueError("Invalid fact confirmation time")
        if fact["status"] not in {"confirmed", "uncertain"} or not fact["evidence"]:
            raise ValueError("Facts require certainty and evidence")
        for ev in fact["evidence"]:
            if (
                set(ev) != {"frame", "camera"}
                or type(ev["frame"]) is not int
                or ev["frame"] not in visible
                or ev["camera"] not in cameras
            ):
                raise ValueError("Unseen or future fact evidence")
        known[fact["id"]] = fact
    point = value["checkpoint"]
    fields = {
        "frame",
        "memory",
        "subtask",
        "memory_fact_ids",
        "required_fact_ids",
        "allowed_fact_ids",
        "acceptable_actions",
        "decision",
        "decision_evidence_frames",
    }
    if (
        not isinstance(point, dict)
        or set(point) != fields
        or type(point["frame"]) is not int
        or point["frame"] != frame
    ):
        raise ValueError("Invalid checkpoint schema or frame")
    for name in ("memory", "subtask"):
        if not isinstance(point[name], str) or not point[name].strip():
            raise ValueError("Memory and subtask must be nonempty strings")
    if point["decision"] not in {"usable", "needs_review"}:
        raise ValueError("Invalid decision")
    for name in ("memory_fact_ids", "required_fact_ids", "allowed_fact_ids"):
        ids = point[name]
        if (
            not isinstance(ids, list)
            or any(not isinstance(fid, str) for fid in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("Duplicate or invalid fact references")
        if any(
            fid not in known
            or known[fid]["status"] != "confirmed"
            or known[fid]["confirmed_at_frame"] > frame
            for fid in ids
        ):
            raise ValueError("Unknown, uncertain or future memory fact")
    if not set(point["required_fact_ids"]) <= set(point["memory_fact_ids"]) <= set(point["allowed_fact_ids"]):
        raise ValueError("Required/memory/allowed facts are inconsistent")
    if not isinstance(point["acceptable_actions"], list) or (
        point["decision"] == "usable" and not point["acceptable_actions"]
    ):
        raise ValueError("Usable checkpoints require acceptable actions")
    for action in point["acceptable_actions"]:
        if (
            not isinstance(action, dict)
            or set(action) != {"action", "object", "target"}
            or any(not isinstance(v, str) for v in action.values())
        ):
            raise ValueError("Invalid acceptable action")
    if (
        not isinstance(point["decision_evidence_frames"], list)
        or not point["decision_evidence_frames"]
        or any(type(f) is not int or f not in visible for f in point["decision_evidence_frames"])
    ):
        raise ValueError("Unseen or future decision evidence")
    return value


def validate_check(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"supported", "issues"}
        or type(value["supported"]) is not bool
    ):
        raise ValueError("Invalid student input check")
    if not isinstance(value["issues"], list) or any(
        not isinstance(s, str) or not s.strip() for s in value["issues"]
    ):
        raise ValueError("Invalid student input issues")
    if not value["supported"] and not value["issues"]:
        raise ValueError("Unsupported labels require an explanation")
    return value


def input_binding(pin):
    """The schedule may gain a human cutoff without invalidating earlier identical inputs."""
    return digest({k: v for k, v in pin.items() if k not in {"checkpoints", "expected_requests"}})


def provider_config_digest(model):
    import os

    from lerobot.data_platform.ai_tools.qwen import DEFAULT_DASHSCOPE_BASE_URL

    return digest(
        {
            "endpoint": DEFAULT_DASHSCOPE_BASE_URL,
            "model": model,
            "revision": os.getenv("DATA_PLATFORM_MEMORY_MODEL_REVISION", ""),
        }
    )
