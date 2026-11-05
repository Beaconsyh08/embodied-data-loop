"""Closed input and model-output vocabulary; no inferred execution parameters."""

import uuid
from dataclasses import dataclass

EVENTS = {
    "search",
    "operator_selected",
    "form_submitted",
    "preview_blocked",
    "draft_saved",
    "draft_modified",
    "confirmed",
    "cancelled",
    "result_opened",
    "feedback",
}
CONTROLS = {"assistant", "pipeline", "jobs", "operators", "search", "results", "feedback"}
REASONS = {
    "none",
    "wrong_operator",
    "wrong_order",
    "wrong_input",
    "wrong_parameters",
    "missing_review",
    "unclear",
    "execution_failure",
    "infrastructure",
    "other",
}
VERDICTS = {"resolved", "unresolved", "modify"}
CATEGORIES = {"intent", "clarification", "correction", "preference", "workflow", "unreliable"}
LABELS = {
    "split",
    "convert",
    "quality",
    "viewer",
    "unknown_goal",
    "missing_information",
    "wrong_operator",
    "wrong_order",
    "wrong_input",
    "wrong_parameters",
    "missing_review",
    "language_zh",
    "language_en",
    "concise",
    "detailed",
    "unreliable",
}
PREFERENCES = {
    "language_zh": "请用中文回答。",
    "language_en": "Please respond in English.",
    "concise": "请使用简洁的回答。",
    "detailed": "请在回答中提供详细解释。",
}


def exact(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Unexpected learning fields")


def identifier(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("A canonical UUID is required")
    return value


def revision(value, expected):
    from lerobot.data_platform.job_management import JobConflictError

    if type(value) is not int or value != expected:
        raise JobConflictError("Learning record changed; reload before editing")


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    source_id: str
    revision: int
    digest: str

    @classmethod
    def parse(cls, value):
        exact(value, {"kind", "source_id", "revision", "digest"})
        if value["kind"] != "learning_episode" or type(value["revision"]) is not int:
            raise ValueError("Unsupported learning evidence")
        if any(not isinstance(value[k], str) or len(value[k]) != 64 for k in ("source_id", "digest")):
            raise ValueError("Invalid evidence identity")
        return cls(**value)


def extraction(value, sources):
    exact(value, {"items"})
    if not isinstance(value["items"], list) or len(value["items"]) > 8:
        raise ValueError("At most eight bounded learning items")
    result = []
    for item in value["items"]:
        exact(item, {"kind", "label", "source", "quote"})
        if item["kind"] not in CATEGORIES or item["label"] not in LABELS:
            raise ValueError("Unknown learning category")
        if item["kind"] == "preference" and item["label"] not in PREFERENCES:
            raise ValueError("Unsupported preference")
        text = sources.get(item["source"])
        quote = item["quote"]
        if not text or not isinstance(quote, str) or not 1 <= len(quote) <= 240 or quote not in text:
            raise ValueError("Learning items require an exact authorized source excerpt")
        result.append(dict(item))
    return {"items": result}
