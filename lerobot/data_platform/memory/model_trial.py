"""Explicit model-only annotation experiment; never grants human review or gold status."""

from __future__ import annotations

import json
import re
from pathlib import Path

from lerobot.data_platform.memory import ai_review
from lerobot.data_platform.memory.contracts import (
    EMPTY_MEMORY,
    AnnotationConfig,
    digest,
    history_frames,
    validate_check,
    validate_point,
)
from lerobot.data_platform.memory.generation import call_model, request_payload, response_json
from lerobot.data_platform.memory.source_repair import CAMERAS, source_mosaic
from lerobot.data_platform.runtime.supervisor import atomic_json

DRAFT = """Produce a causal robot high-level subtask and compact textual memory, in English.
Images are synchronized mosaics: LEFT PANEL = image (main), CENTER = left_wrist_image,
RIGHT = right_wrist_image. Each image has its original episode frame ID immediately before it.
The task is an instruction, never evidence of execution. Use only these current/past images
and the provided prior facts/memory. No future information is available.
Memory retains decision-relevant observed facts, especially completed transfers, failures,
and last confirmed locations. Do not recount every visible background detail. No numeric count
unless individual completed transfers are actually supported. No invented object identities,
hidden contacts, success, or claims that nothing ever happened before the first visible frame.
At the first frame an empty memory is valid. Unclear background details need not be claimed.
Describe observed state conservatively. A subtask is a feasible NEXT instruction, not a claim
that it already occurred; it can include approaching before grasping. Do not claim success from
gripper pose alone. Avoid handedness unless needed and supported. Do not output generic waiting
or inspection solely to pass validation when a grounded task-directed action is visible.
If prior memory is stale, update it using observed evidence. Retain prior out-of-view facts
only when still relevant. Never put fact IDs in the natural-language memory or subtask.
Use needs_review when ambiguity prevents a meaningful grounded target; do not force usable.
Return ONLY a JSON object with exactly facts and checkpoint, using this schema:
{"facts":[{"id":"unique_id","text":"observed fact","status":"confirmed",
"confirmed_at_frame":CURRENT_FRAME,"evidence":[{"frame":SUPPLIED_FRAME,"camera":"image"}]}],
"checkpoint":{"frame":CURRENT_FRAME,"memory":"concise relevant memory",
"subtask":"next feasible task-directed instruction","memory_fact_ids":["unique_id"],
"required_fact_ids":["unique_id"],"allowed_fact_ids":["unique_id"],
"acceptable_actions":[{"action":"verb","object":"object","target":"destination"}],
"decision":"usable","decision_evidence_frames":[CURRENT_FRAME]}}.
facts contains ONLY new facts; existing IDs cannot be repeated. Facts can be uncertain, but
never reference an uncertain fact in memory_fact_ids, required_fact_ids or allowed_fact_ids.
Required IDs must be a subset of memory IDs, themselves a subset of allowed confirmed IDs.
Populate memory_fact_ids with the IDs for ALL facts summarized in memory; it is NOT normally empty.
For example, if memory mentions f1 and f4, memory_fact_ids=["f1","f4"], and required_fact_ids
may be ["f4"] but never contain an ID absent from memory_fact_ids. Replace example IDs.
Every NEW fact.confirmed_at_frame must equal current_frame EXACTLY, even when its evidence
was observed in an earlier supplied frame. Evidence time and the current confirmation time differ.
All new assertions in memory need matching facts. With no confirmed relevant events use exactly:
No events have been confirmed yet. All outputs are model drafts, never human approval.
"""
CHECK = """Independently verify the proposed robot memory and next subtask against ONLY the
student-visible task, current image, and previous memory. The proposed output is NOT evidence.
The synchronized image panels are LEFT = main, CENTER = left wrist, RIGHT = right wrist.
Reject unsupported new facts, counts, success, hidden history, mistaken identities or changed
out-of-view facts. Prior memory can support previously established events without seeing them
again. A next instruction is prospective, not proof of execution: approaching/grasping a visible
target does not require that contact already occurred. Assess task feasibility and visible target,
not whether motion already happened. Empty initial memory asserts no confirmed events, not proof
of zero prior events. Do not demand invisible details that the proposed output does not assert.
Return JSON only: {"supported":true|false,"issues":["specific unsupported claim"]}.
If all claims and the instruction are supported, issues must be empty. This is MODEL review.
"""
BALANCED_CHECK = """Check whether the EXACT proposed memory and prospective subtask are supported
by the current image, task, and previous memory. Judge only claims actually made. The proposal
is not evidence. Image panels LEFT=main, CENTER=left wrist, RIGHT=right wrist, all current.
Distinguish observable STATE from an EVENT or IDENTITY: 'a garment is visible in the basket'
does not assert who put it there, that it was just released, its origin, or its identity. Such
a visible state needs no release video. Conversely, 'the garment was successfully released',
an exact transferred count, or 'the same garment' DOES require supporting memory/current evidence.
An object may visibly change position or a gripper may visibly open since previous memory;
the old state must not override current pixels. Prior memory supports retained past events,
but cannot justify a claim that an old transient position remains true when current views disagree.
A prospective next instruction need not have happened yet. Check its grounded target and
feasibility. Do not invent an unstated claim, demand proof of it, then reject the actual label.
Reject real unsupported assertions or instructions; accept fully supported ones. Return only
JSON {"supported":true|false,"issues":["quote the unsupported assertion and concise reason"]}.
For supported=true issues must be []. For false list only actual errors, at most three concise
items. No speculation about what a benchmark expects, no self-dialogue or alternative verdicts.
This is model verification, never human approval.
"""

TEXT_TARGET_RULES = """The student never receives frame indices, timestamps, or ledger fact IDs.
They are bookkeeping for evidence only. NEVER put them into memory or subtask, even if the
previous draft did so. Memory must be at most 50 words: retain relevant established events and
the latest decision-relevant state, not a chronological narration. Subtask must be at most
35 words and describe one feasible next step. At the initial image, absence of prior memory
does not prove no earlier transfers occurred; do not assert an unseen negative history.
"""


def validate_target_text(draft, existing_facts):
    """Reject bookkeeping in student targets; never silently rewrite a model response."""
    point = draft["checkpoint"]
    for key, limit in (("memory", 50), ("subtask", 35)):
        value = point[key]
        if len(value.split()) > limit:
            raise ValueError(f"{key} must be at most {limit} words; summarize relevant facts")
        if re.search(
            r"\bframes?\s*[:#]?\s*\d+|\b(?:checkpoints?|timestamps?)\b|\bf(?:act)?[_-]?\d+\b", value, re.I
        ):
            raise ValueError(f"{key} contains frame or ledger bookkeeping absent from student input")
        for fact in [*existing_facts, *draft["facts"]]:
            fid = fact["id"]
            if re.search(r"[\d_]", fid) and re.search(r"(?<!\w)" + re.escape(fid) + r"(?!\w)", value):
                raise ValueError(f"{key} must express facts naturally, without ledger IDs")


def payload(config, prompt, context, images, frame_ids):
    result = request_payload(config, prompt, context, images)
    content = result["messages"][0]["content"]
    labelled = content[:1]
    for frame, item in zip(frame_ids, content[1:], strict=True):
        labelled.extend([{"type": "text", "text": f"ORIGINAL EPISODE FRAME ID: {frame}"}, item])
    result["messages"][0]["content"] = labelled
    result["response_format"] = {"type": "json_object"}
    return result


def image_at(source, rows, frame, output):
    target = output / "images" / f"{frame:06d}.png"
    if not target.exists():
        # Same 224-pixel panels as the training contract, with no text overlays.
        target.write_bytes(source_mosaic(source, rows[frame], panel_size=224))
    return target.read_bytes()


def annotate(source, rows, pin, output: Path, client, stop=lambda: None, progress=lambda _: None):
    """Sequential checkpoints; bounded, explicit revisions at rejected cutoffs; no chain jumps."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    strict_student = pin.get("student_input_protocol") == 2
    draft_prompt = DRAFT + TEXT_TARGET_RULES if strict_student else DRAFT
    prompts = (
        [draft_prompt, CHECK, BALANCED_CHECK]
        if strict_student or "student_visible_from_frame" in pin
        else [DRAFT, CHECK]
    )
    if pin.get("ai_review_protocol") == 1:
        if not strict_student or pin.get("ai_review_from_frame") not in pin["checkpoints"]:
            raise ValueError("AI review requires protocol 2 and a scheduled revision cutoff")
        prompts += [ai_review.DRAFT_GUIDANCE, ai_review.CHECK_PROMPT]
    binding = {
        "pin": pin,
        "prompts": digest(prompts),
        "mode": "model_only_experiment/v2" if strict_student else "model_only_experiment/v1",
    }
    path = output / "binding.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("Inputs changed; create a new annotation revision")
    atomic_json(path, binding)
    record = {
        "episode_id": f"episode-{pin['episode_index']:06d}",
        "task": pin["task"],
        "facts": [],
        "checkpoints": [],
        "challenges": [],
        "review": {"status": "draft"},
    }
    entries = []
    config = AnnotationConfig(model=pin["model"])
    timestamps = [r["timestamp"] for r in rows]
    for frame in pin["checkpoints"]:
        stop()
        reviewing = pin.get("ai_review_protocol") == 1 and frame >= pin["ai_review_from_frame"]
        past = history_frames(timestamps, frame, config)
        ids = [*past, frame]
        images = [image_at(source, rows, i, output) for i in ids]
        previous = record["checkpoints"][-1] if record["checkpoints"] else {}
        context = {
            "task": pin["task"],
            "current_frame": frame,
            "camera_order": CAMERAS,
            "image_frames": ids,
            "observed_facts": record["facts"],
            "previous_memory": previous.get("memory", EMPTY_MEMORY),
            "previous_subtask": previous.get("subtask", ""),
        }
        agent_notes = pin.get("agent_reviews", {}).get(str(frame))
        if agent_notes:
            context["agent_review_notes"] = agent_notes
        entry = {"frame": frame, "state": "needs_review", "revisions": []}
        student_visible = frame >= pin.get("student_visible_from_frame", float("inf"))
        for revision in range(pin.get("max_label_revisions", 3)):
            stop()
            draft_context, draft_images, draft_ids = context, images, ids
            if revision >= (1 if strict_student else 6) or student_visible:
                # Final bounded repair uses exactly the student's semantic information.
                # Opaque fact IDs support ledger references, not additional historical facts.
                feedback = context.get("revision_request", {})
                rejected_draft = feedback.get("rejected_draft")
                rejected = rejected_draft.get("checkpoint", {}) if isinstance(rejected_draft, dict) else {}
                if not isinstance(rejected, dict):
                    rejected = {}
                draft_context = {
                    "task": pin["task"],
                    "current_frame": frame,
                    "camera_order": CAMERAS,
                    "image_frames": [frame],
                    "previous_memory": context["previous_memory"],
                    "previous_memory_fact_ids": previous.get("memory_fact_ids", []),
                    "new_fact_id_prefix": f"frame{frame}_revision{revision}_",
                    "revision_request": {
                        "revision": revision,
                        "proposed_output": {k: rejected.get(k) for k in ("memory", "subtask")},
                        "issues": feedback.get("issues", [feedback.get("schema_error", "Recheck grounding")]),
                    },
                    "instruction": "Relabel from ONLY the current image and previous_memory. "
                    "No clip or extra historical facts are supplied. Keep memory to one or two concise "
                    "sentences of relevant supported facts, with no frame numbers or camera narration. "
                    "Refer to previous_memory_fact_ids for retained prior claims and create uniquely "
                    "prefixed new fact IDs only for new current-image observations. A rejected draft "
                    "and checker comments are not evidence. Produce a task-directed next subtask if "
                    "supported; otherwise retain needs_review. Do not invent events to complete the episode.",
                }
                draft_images, draft_ids = images[-1:], [frame]
                if agent_notes:
                    draft_context["agent_review_notes"] = agent_notes
            active_prompt = draft_prompt
            if reviewing:
                active_prompt += ai_review.DRAFT_GUIDANCE
                draft_images = [*draft_images[:-1], *ai_review.camera_panels(draft_images[-1])]
                draft_ids = [*draft_ids[:-1], frame, frame, frame]
            draft_request = payload(config, active_prompt, draft_context, draft_images, draft_ids)
            request_hash = digest(draft_request)
            folder = output / "requests" / request_hash
            folder.mkdir(parents=True, exist_ok=True)
            attempt = {"revision": revision, "draft_sha256": request_hash}
            entry["revisions"].append(attempt)
            draft = None
            try:
                draft = response_json(call_model(client, draft_request, folder / "draft.json", stop))
                validate_point(draft, record["facts"], frame=frame, cameras=CAMERAS, visible=set(draft_ids))
                if strict_student:
                    validate_target_text(draft, record["facts"])
                check_context = {
                    "task": pin["task"],
                    "previous_memory": context["previous_memory"],
                    "proposed_output": {k: draft["checkpoint"][k] for k in ("memory", "subtask")},
                }
                if reviewing:
                    check_request = request_payload(
                        config, ai_review.CHECK_PROMPT, check_context, ai_review.camera_panels(images[-1])
                    )
                    check_request["response_format"] = {"type": "json_object"}
                elif strict_student:
                    # Exactly the student's semantic inputs: no labelled frame IDs, clip,
                    # hidden facts, or revision comments anywhere in the request content.
                    check_request = request_payload(
                        config, BALANCED_CHECK + TEXT_TARGET_RULES, check_context, images[-1:]
                    )
                    check_request["response_format"] = {"type": "json_object"}
                else:
                    check_request = payload(
                        config,
                        BALANCED_CHECK if student_visible else CHECK,
                        check_context,
                        images[-1:],
                        [frame],
                    )
                attempt["check_sha256"] = digest(check_request)
                review_response = response_json(
                    call_model(client, check_request, folder / "check.json", stop)
                )
                check = (
                    ai_review.validate_review(review_response)
                    if reviewing
                    else validate_check(review_response)
                )
                if reviewing:
                    attempt["ai_review"] = review_response
                attempt["student_check"] = check
                accepted = (
                    check["supported"]
                    and not check["issues"]
                    and draft["checkpoint"]["decision"] == "usable"
                    and all(f["status"] == "confirmed" for f in draft["facts"])
                )
                if accepted:
                    record["facts"].extend(draft["facts"])
                    record["checkpoints"].append(draft["checkpoint"])
                    entry["state"] = "model_checked"
                    break
                attempt["issues"] = check["issues"] or ["Teacher reported unresolved ambiguity"]
                context["revision_request"] = {
                    "revision": revision + 1,
                    "rejected_draft": draft,
                    "issues": attempt["issues"],
                    "instruction": "Re-examine the same evidence. Correct unsupported claims; do not invent a replacement.",
                }
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                attempt["issues"] = [str(exc)]
                context["revision_request"] = {"schema_error": str(exc), "revision": revision + 1}
                context["revision_request"].update(
                    rejected_draft=draft,
                    exact_confirmation_frame=frame,
                    instruction="Correct the invalid fields in this draft. Every new confirmed_at_frame must equal "
                    f"{frame}. All required IDs must be in memory_fact_ids; all memory IDs in allowed_fact_ids. "
                    "Include all referenced memory facts. Do not change visual claims merely to satisfy schema.",
                )
            if revision >= 2:
                context["revision_request"]["instruction"] += (
                    " Keep the memory concise and decision-relevant. Do not narrate frame numbers or camera "
                    "visibility in the memory. Avoid unnecessary hand identity, counts, exact contact, or "
                    "inside/outside claims when the evidence is ambiguous. Preserve supported relevant facts "
                    "and give a meaningful task-directed next instruction; if this is impossible, needs_review. "
                    "Conflicting checker comments are not evidence: resolve them from the supplied images."
                )
            atomic_json(output / "pending.json", entry)
        entries.append(entry)
        state = {
            "schema": "memory_model_trial/v1",
            "pin": pin,
            "record": record,
            "entries": entries,
            "human_approvals": 0,
            "training_quality": "model_only",
            "complete": False,
        }
        atomic_json(output / "review.json", state)
        progress({"current": len(entries), "total": len(pin["checkpoints"]), "state": entry["state"]})
        if entry["state"] != "model_checked":
            break
    state["complete"] = len(record["checkpoints"]) == len(pin["checkpoints"])
    atomic_json(output / "review.json", state)
    return state
