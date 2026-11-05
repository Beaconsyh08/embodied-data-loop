"""Immutable review revisions and optimistic concurrency on the lifecycle repository."""

from __future__ import annotations

import copy

from lerobot.data_platform.job_management import JobConflictError
from lerobot.data_platform.memory.contracts import checkpoints, digest, validate_point


def review_payload(record):
    return {k: v for k, v in record.items() if k != "review"}


class MemoryReviewStore:
    def __init__(self, repository):
        self.repository = repository

    def get(self, run, revision=None, *, connection=None):
        kind, key = ("memory_heads", run) if revision is None else ("memory_revisions", f"{run}:{revision}")
        value = self.repository.get(kind, key, connection=connection)
        if value is None:
            raise KeyError(run)
        return value

    def create(self, run, location, state):
        with self.repository.transaction() as connection:
            old = self.repository.get("memory_heads", run, connection=connection)
            if old:
                return old
            value = {
                "run": run,
                "location_id": location,
                "revision": 1 + max((p["revision"] for p in state.get("parents", [])), default=0),
                "state": copy.deepcopy(state),
                "actor": None,
            }
            self._save(value, connection)
            return value

    def _save(self, value, connection):
        self.repository.put(
            "memory_revisions",
            f"{value['run']}:{value['revision']}",
            value,
            immutable=True,
            connection=connection,
        )
        self.repository.put("memory_heads", value["run"], value, connection=connection)

    def change(self, run, expected, actor, action, body):
        with self.repository.transaction() as connection:
            current = self.get(run, connection=connection)
            if type(expected) is not int or expected != current["revision"]:
                raise JobConflictError("Review revision changed; reload before editing")
            value = copy.deepcopy(current)
            state = value["state"]
            record, entries, pin = state["record"], state["entries"], state["pin"]
            if action == "approve":
                if (
                    [p["frame"] for p in record["checkpoints"]] != pin["checkpoints"]
                    or any(e["state"] != "complete" for e in entries)
                    or any(p["decision"] != "usable" for p in record["checkpoints"])
                ):
                    raise ValueError("Resolve every scheduled checkpoint before approval")
                if not all(
                    body.get(k) is True
                    for k in ("source_confirmed", "goal_confirmed", "causal_evidence_confirmed")
                ):
                    raise ValueError("Source, goal and causal evidence must be explicitly confirmed")
                review = record["review"]
                payload_hash = digest(review_payload(record))
                if body.get("independent"):
                    if review.get("status") != "approved" or review.get("payload_sha256") != payload_hash:
                        raise ValueError("Primary approval is required before independent review")
                    if actor == review["reviewer"]:
                        raise ValueError("Test review requires a distinct logged-in reviewer")
                    if body.get("adjudicated") is not True:
                        raise ValueError("Independent review requires explicit adjudication")
                    review.update(second_reviewer=actor, adjudicated=True)
                else:
                    record["review"] = {
                        "status": "approved",
                        "reviewer": actor,
                        "source_confirmed": True,
                        "goal_confirmed": True,
                        "causal_evidence_confirmed": True,
                        "payload_sha256": payload_hash,
                    }
            elif action in {"edit", "checkpoint", "needs_review"}:
                frame = body.get("frame")
                if type(frame) is not int or not 0 <= frame < pin["frame_count"]:
                    raise ValueError("Frame outside episode")
                if action == "checkpoint":
                    pin["checkpoints"] = checkpoints(
                        pin["timestamps"], pin["segment_starts"], [*pin["checkpoints"], frame]
                    )
                    pin["expected_requests"] = len(pin["checkpoints"]) * 2
                    state["pin_digest"] = digest(pin)
                else:
                    entry = next((e for e in entries if e["frame"] == frame), None)
                    if entry is None:
                        raise ValueError("Checkpoint has no generated record")
                    if action == "edit":
                        earlier = [f for f in record["facts"] if f["confirmed_at_frame"] < frame]
                        candidate = {"facts": body.get("facts"), "checkpoint": body.get("checkpoint")}
                        validate_point(
                            candidate,
                            earlier,
                            frame=frame,
                            cameras=pin["cameras"],
                            visible={frame, *entry["history_frames"]},
                        )
                        record["facts"] = (
                            earlier
                            + candidate["facts"]
                            + [f for f in record["facts"] if f["confirmed_at_frame"] > frame]
                        )
                        record["checkpoints"] = sorted(
                            [p for p in record["checkpoints"] if p["frame"] != frame]
                            + [candidate["checkpoint"]],
                            key=lambda p: p["frame"],
                        )
                        entry["state"] = (
                            "complete" if candidate["checkpoint"]["decision"] == "usable" else "needs_review"
                        )
                        entry["human_resolution"] = {
                            "reviewer": actor,
                            "reason": str(body.get("reason") or "").strip(),
                        }
                        if not entry["human_resolution"]["reason"]:
                            raise ValueError("A human resolution reason is required")
                    else:
                        entry["state"] = "needs_review"
                        for point in record["checkpoints"]:
                            if point["frame"] == frame:
                                point["decision"] = "needs_review"
                for entry in entries:
                    if entry["frame"] > frame or (action == "checkpoint" and entry["frame"] == frame):
                        entry["state"] = "stale"
                record["review"] = {"status": "draft"}
            else:
                raise ValueError("Unknown review action")
            value.update(revision=expected + 1, actor=actor)
            self._save(value, connection)
            return value
