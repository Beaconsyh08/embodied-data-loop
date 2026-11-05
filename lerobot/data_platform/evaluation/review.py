"""Immutable pilot selection, model snapshots and optimistic human review revisions."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time

from lerobot.data_platform.evaluation.metrics import WEIGHTS, edges
from lerobot.data_platform.job_management import JobConflictError


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def text(value, name, required=False):
    if not isinstance(value, str) or len(value) > 12000 or (required and not value.strip()):
        raise ValueError(f"{name}: enter a nonempty text of at most 12000 characters")


def interval(value, duration):
    if not isinstance(value, dict) or not all(number(value.get(k)) for k in ("start_s", "end_s")):
        raise ValueError("Segment boundaries must be finite source seconds")
    if not 0 <= value["start_s"] < value["end_s"] <= duration + 1e-6:
        raise ValueError("Segment boundaries are outside the source video")


def validate_reference(ref, duration, approve=False):
    if not isinstance(ref, dict) or set(ref) != {"segments", "quality", "outcome", "outcome_reason", "notes"}:
        raise ValueError("Invalid reference fields")
    for name in ("segments", "quality"):
        if not isinstance(ref[name], list) or len(ref[name]) > 1000:
            raise ValueError("Too many annotation intervals")
        for item in ref[name]:
            interval(item, duration)
    for seg in ref["segments"]:
        if set(seg) != {"start_s", "end_s", "caption", "facts"}:
            raise ValueError("Invalid segment fields")
        text(seg["caption"], "Caption", approve)
        if not isinstance(seg["facts"], list) or len(seg["facts"]) > 100:
            raise ValueError("Invalid reference facts")
        for fact in seg["facts"]:
            if not isinstance(fact, dict) or set(fact) != {"group", "text"} or fact["group"] not in WEIGHTS:
                raise ValueError("Invalid fact group")
            text(fact["text"], "Reference fact", True)
        if approve and not any(f["group"] == "action" for f in seg["facts"]):
            raise ValueError("Each approved segment needs at least one supported action fact")
    for item in ref["quality"]:
        if (
            set(item) != {"start_s", "end_s", "label", "reason"}
            or type(item["label"]) is not int
            or item["label"] not in (1, 2, 3)
        ):
            raise ValueError("Quality labels must be 1 failure, 2 suboptimal, or 3 normal")
        text(item["reason"], "Quality reason", approve and item["label"] != 3)
    if ref["outcome"] is not None and (type(ref["outcome"]) is not int or not 1 <= ref["outcome"] <= 5):
        raise ValueError("Outcome must be an integer from 1 to 5")
    text(ref["outcome_reason"], "Outcome reason", approve)
    text(ref["notes"], "Notes")
    if approve:
        if ref["outcome"] is None or not ref["segments"]:
            raise ValueError("Review outcome and at least one action segment before approval")
        end = 0
        for q in sorted(ref["quality"], key=lambda q: q["start_s"]):
            if q["start_s"] > end + 1e-6:
                raise ValueError("Quality labels must cover the full video; fill the unlabeled gap")
            end = max(end, q["end_s"])
        if end < duration - 1e-6:
            raise ValueError("Quality labels must cover the full video")


def validate_assessment(value, segment):
    if not isinstance(value, dict) or set(value) != {"covered", "errors", "reason"}:
        raise ValueError("Invalid semantic assessment")
    covered = value["covered"]
    if (
        not isinstance(covered, list)
        or len(covered) != len(set(covered))
        or any(
            str(i) not in {str(j) for j in range(len(segment["facts"]))} or not isinstance(i, str)
            for i in covered
        )
    ):
        raise ValueError("Covered facts must refer to distinct reference facts")
    if not isinstance(value["errors"], list) or len(value["errors"]) > 100:
        raise ValueError("Invalid factual errors")
    for error in value["errors"]:
        if (
            not isinstance(error, dict)
            or set(error) != {"group", "family", "evidence"}
            or error["group"] not in WEIGHTS
        ):
            raise ValueError("Invalid factual error family")
        text(error["family"], "Error family", True)
        text(error["evidence"], "Error evidence", True)
    text(value["reason"], "Assessment explanation", True)


def draft(artifact, metadata):
    segments = [{k: s[k] for k in ("start_s", "end_s", "caption")} for s in artifact["native_segments"]]
    duration, fps = artifact["duration_s"], artifact["fps"]
    dense = artifact["reward"]["merged_scores"]
    quality = []
    for frame, label in enumerate(dense):
        if label not in (1, 2, 3):
            continue  # Unknown regions stay missing; never turn them into normal labels.
        start, end = frame / fps, min((frame + 1) / fps, duration)
        if quality and quality[-1]["label"] == label and abs(quality[-1]["end_s"] - start) < 1e-6:
            quality[-1]["end_s"] = end
        elif start < end:
            quality.append({"start_s": start, "end_s": end, "label": label, "reason": ""})
    return {
        "schema_version": 1,
        "episode_index": artifact["episode_index"],
        "duration_s": duration,
        "fps": fps,
        "instruction": "\n".join(metadata.get("task_hint", [])),
        "prediction": {
            "segments": segments,
            "outcome": artifact["reward"]["terminal"].get("score"),
            "dense": dense,
        },
        "source_sha256": digest(artifact),
        "source_digest": artifact.get("source_digest"),
        "model": artifact.get("model"),
        "model_revisions": artifact.get("model_revisions"),
        "annotation_origin": "flora_assisted_human_review",
        "reference": {
            "segments": [{**s, "facts": [{"group": "action", "text": s["caption"]}]} for s in segments],
            "quality": quality,
            "outcome": artifact["reward"]["terminal"].get("score"),
            "outcome_reason": artifact["reward"]["terminal"].get("rationale", ""),
            "notes": "",
        },
        "status": "draft",
        "assessments": {},
        "oracle": None,
        "oracle_assessments": {},
    }


class EvaluationStore:
    def __init__(self, repository):
        self.repository = repository

    def pilot(self, location, *, connection=None):
        return self.repository.get("evaluation_pilots", location, connection=connection)

    def get(self, location, episode, revision=None, *, connection=None):
        key = f"{location}:{episode}"
        value = self.repository.get(
            "evaluation_heads" if revision is None else "evaluation_revisions",
            key if revision is None else f"{key}:{revision}",
            connection=connection,
        )
        if value is None:
            raise KeyError("Evaluation episode not found")
        return value

    def _save(self, value, connection):
        key = f"{value['location_id']}:{value['episode_index']}"
        self.repository.put(
            "evaluation_revisions", f"{key}:{value['revision']}", value, immutable=True, connection=connection
        )
        self.repository.put("evaluation_heads", key, value, connection=connection)

    def create(self, location, seeds, actor):
        if not 1 <= len(seeds) <= 10 or len({s["episode_index"] for s in seeds}) != len(seeds):
            raise ValueError("A pilot needs 1–10 distinct episodes")
        with self.repository.transaction() as conn:
            if self.pilot(location, connection=conn):
                raise JobConflictError("Pilot already exists; its selection is frozen")
            pilot = {
                "location_id": location,
                "episodes": [s["episode_index"] for s in seeds],
                "created_by": actor,
                "created_at": time.time(),
                "selection": "Task-diverse pilot, not a random estimate of all 100 episodes",
            }
            for seed in seeds:
                value = {
                    **copy.deepcopy(seed),
                    "location_id": location,
                    "revision": 1,
                    "actor": actor,
                    "updated_at": time.time(),
                    "action": "seed",
                    "approved_by": None,
                }
                self.repository.put(
                    "evaluation_seeds",
                    f"{location}:{seed['episode_index']}",
                    value,
                    immutable=True,
                    connection=conn,
                )
                self._save(value, conn)
            self.repository.put("evaluation_pilots", location, pilot, immutable=True, connection=conn)
        return pilot

    def change(self, location, episode, expected, actor, action, body):
        with self.repository.transaction() as conn:
            current = self.get(location, episode, connection=conn)
            if type(expected) is not int or current["revision"] != expected:
                raise JobConflictError(
                    "Another reviewer saved changes. Reload before saving; your local edits are retained."
                )
            value = copy.deepcopy(current)
            if action in ("save", "approve"):
                ref = body.get("reference")
                validate_reference(ref, value["duration_s"], approve=action == "approve")
                if action == "approve" and body.get("confirmed") != [
                    "segments",
                    "captions",
                    "quality",
                    "outcome",
                ]:
                    raise ValueError("Explicitly confirm all four annotation sections")
                if ref != value["reference"]:
                    value.update(assessments={}, oracle=None, oracle_assessments={})
                value.update(
                    reference=ref,
                    status="approved" if action == "approve" else "draft",
                    approved_by=actor if action == "approve" else None,
                )
            elif action in ("assess", "oracle-assess"):
                if value["status"] != "approved":
                    raise ValueError("Freeze the human reference before assessing captions")
                index = body.get("pair")
                if action == "assess":
                    possible = {
                        f"{i}:{j}": j
                        for i, j in edges(value["prediction"]["segments"], value["reference"]["segments"])
                    }
                    target = "assessments"
                else:
                    if not value.get("oracle"):
                        raise ValueError("No oracle-window predictions have been imported")
                    possible = {str(i): i for i in range(len(value["reference"]["segments"]))}
                    target = "oracle_assessments"
                if index not in possible:
                    raise ValueError("Caption pair is not eligible under this reference")
                assessment = body.get("assessment")
                validate_assessment(assessment, value["reference"]["segments"][possible[index]])
                value[target][index] = assessment
            elif action == "oracle-import":
                if value["status"] != "approved":
                    raise ValueError("Freeze the human reference before oracle inference")
                oracle = body.get("oracle")
                if not isinstance(oracle, dict) or set(oracle) != {
                    "reference_sha256",
                    "model",
                    "segments",
                    "provenance",
                }:
                    raise ValueError("Invalid oracle prediction fields")
                if oracle["reference_sha256"] != digest(value["reference"]):
                    raise JobConflictError(
                        "Oracle prediction was generated for a different reference revision"
                    )
                text(oracle["model"], "Oracle model", True)
                text(oracle["provenance"], "Oracle inference provenance", True)
                if not isinstance(oracle["segments"], list) or len(oracle["segments"]) != len(
                    value["reference"]["segments"]
                ):
                    raise ValueError("Oracle predictions must cover every reference window")
                for p, g in zip(oracle["segments"], value["reference"]["segments"], strict=False):
                    if set(p) != {"start_s", "end_s", "caption"} or (p["start_s"], p["end_s"]) != (
                        g["start_s"],
                        g["end_s"],
                    ):
                        raise ValueError("Oracle windows must exactly match frozen reference intervals")
                    text(p["caption"], "Oracle caption")
                value.update(oracle=oracle, oracle_assessments={})
            else:
                raise ValueError("Unknown review action")
            value.update(revision=expected + 1, actor=actor, action=action, updated_at=time.time())
            self._save(value, conn)
            return value
