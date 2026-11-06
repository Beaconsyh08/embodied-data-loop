"""Two isolated Qwen requests per cutoff, durable receipts and prefix-only resume."""

from __future__ import annotations

import base64
import copy
import io
import json
import time
from pathlib import Path

from PIL import Image

from lerobot.data_platform.ai_tools.qwen import QwenRequestError
from lerobot.data_platform.memory.contracts import (
    EMPTY_MEMORY,
    AnnotationConfig,
    digest,
    history_frames,
    input_binding,
    validate_check,
    validate_point,
)
from lerobot.data_platform.runtime.supervisor import atomic_json

DRAFT_PROMPT = """Label robot high-level textual memory using only the supplied current and past views.
Task and previous instructions are goals, never proof of execution or success. Do not infer contact
from pose, double count repeated objects, or invent identities. Uncertainty, count conflicts and
ambiguous identity require decision needs_review and uncertain facts. Return strict JSON only:
{facts:[{id,text,status:confirmed|uncertain,confirmed_at_frame:CURRENT_FRAME,
evidence:[{frame:SUPPLIED_FRAME,camera:CAMERA_NAME}]}],checkpoint:{frame:CURRENT_FRAME,
memory:concise English summary,subtask:next feasible English instruction,memory_fact_ids:[],
required_fact_ids:[],allowed_fact_ids:[],acceptable_actions:[{action,object,target}],
decision:usable|needs_review,decision_evidence_frames:[SUPPLIED_FRAME]}}.
Only NEW facts with unique IDs. Required subset memory subset allowed; reference only confirmed
facts. Empty memory: No events have been confirmed yet. usable is a draft, never human approval.
"""
CHECK_PROMPT = """Independently check a proposed robot subtask and memory label. You receive only the
student's task, current three-view image and previous memory, plus the proposed output. Decide
whether EACH claim and instruction is inferable from those student inputs. The proposal is not
evidence. Reject hidden-history dependencies, uncertain identities, unsupported counts or success.
Return strict JSON {supported:boolean,issues:[specific unsupported claim or ambiguity]}. This is
an input-sufficiency check, not human approval. No earlier clip or fact ledger is available to you.
"""


def mosaic(bundle: Path, images: list[dict]) -> bytes:
    errors = [image.get("decode_error_s", 0) for image in images]
    if max(errors, default=0) - min(errors, default=0) > 0.05:
        raise ValueError("Camera synchronization exceeds 50 milliseconds")
    canvas = Image.new("RGB", (672, 224))
    if len(images) != 3:
        raise ValueError("Exactly main, left and right views are required")
    for index, image in enumerate(images):
        path = (bundle / image["path"]).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError("Image path escapes bundle")
        if image.get("decode_error_s", 0) > 1e-9:
            raise ValueError("Decoded image is after the cutoff")
        with Image.open(path) as source:
            view = source.convert("RGB")
        view.thumbnail((224, 224), Image.Resampling.LANCZOS)
        canvas.paste(view, (index * 224 + (224 - view.width) // 2, (224 - view.height) // 2))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def request_payload(config, prompt, context, images):
    content = [{"type": "text", "text": prompt + "\n" + json.dumps(context, ensure_ascii=False)}]
    content.extend(
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(im).decode()}}
        for im in images
    )
    return {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": content}],
    }


def call_model(client, payload, receipt: Path, stop, *, sleep=time.sleep):
    fingerprint = digest(payload)
    attempts = []
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["input_sha256"] == fingerprint and saved.get("response") is not None:
            return saved["response"]
        if saved["input_sha256"] == fingerprint:
            attempts = saved.get("attempts", [])
    if len(attempts) >= 3 or (attempts and not attempts[-1].get("retryable", True)):
        raise QwenRequestError(
            "Stored request exhausted its retries; explicitly regenerate to retry", retryable=False
        )
    for number in range(len(attempts) + 1, 4):
        stop()
        started = time.monotonic()
        try:
            response = client.post_chat_completion(payload)
        except QwenRequestError as exc:
            attempts.append(
                {
                    "attempt": number,
                    "elapsed_s": time.monotonic() - started,
                    "error": str(exc),
                    "retryable": exc.retryable,
                }
            )
            atomic_json(receipt, {"input_sha256": fingerprint, "attempts": attempts, "response": None})
            if not exc.retryable or number == 3:
                raise
            for _ in range(2 ** (number - 1) * 10):
                stop()
                sleep(0.1)
        else:
            attempts.append({"attempt": number, "elapsed_s": time.monotonic() - started})
            atomic_json(receipt, {"input_sha256": fingerprint, "attempts": attempts, "response": response})
            stop()
            return response


def response_json(response):
    # Invalid JSON/fences/schema are errors; never coerce or silently repair labels.
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Expected a text JSON response")
    return json.loads(content)


def generate(
    bundle: Path,
    pin: dict,
    output: Path,
    client,
    *,
    previous=None,
    stop=lambda: None,
    progress=lambda _: None,
):
    config = AnnotationConfig(**pin["config"])
    episode = json.loads((bundle / "episodes.jsonl").read_text().splitlines()[0])
    frames = [json.loads(row) for row in (bundle / episode["frames_file"]).read_text().splitlines()]
    timestamps = [row["timestamp"] for row in frames]
    record = {
        "episode_id": episode["id"],
        "task": pin["task"],
        "facts": [],
        "checkpoints": [],
        "challenges": [],
        "review": {"status": "draft"},
    }
    entries = []
    output.mkdir(parents=True, exist_ok=True)
    prior_points = {p["frame"]: p for p in (previous or {}).get("record", {}).get("checkpoints", [])}
    prior_entries = {p["frame"]: p for p in (previous or {}).get("entries", [])}
    prefix_valid = bool(previous and input_binding(previous["pin"]) == input_binding(pin))
    for frame in pin["checkpoints"]:
        stop()
        history = history_frames(timestamps, frame, config)
        visible = {frame, *history}
        earlier = record["checkpoints"][-1] if record["checkpoints"] else {}
        context = {
            "task": pin["task"],
            "current_frame": frame,
            "camera_order": pin["cameras"],
            "history_frames": history,
            "image_frames": [frame, *history],
            "observed_facts": record["facts"],
            "previous_memory": earlier.get("memory", EMPTY_MEMORY),
            "previous_subtask": earlier.get("subtask", ""),
        }
        images = [mosaic(bundle, frames[i]["images"]) for i in [frame, *history]]
        payload = request_payload(config, DRAFT_PROMPT, context, images)
        input_hash = digest({"pin": input_binding(pin), "payload": payload, "check_prompt": CHECK_PROMPT})
        old = prior_entries.get(frame)
        if prefix_valid and old and old.get("input_sha256") == input_hash and old.get("state") == "complete":
            value = {
                "facts": [f for f in previous["record"]["facts"] if f["confirmed_at_frame"] == frame],
                "checkpoint": prior_points[frame],
            }
            validate_point(value, record["facts"], frame=frame, cameras=pin["cameras"], visible=visible)
            record["facts"].extend(copy.deepcopy(value["facts"]))
            record["checkpoints"].append(copy.deepcopy(value["checkpoint"]))
            entries.append(copy.deepcopy(old))
            continue
        prefix_valid = False
        folder = output / "requests" / input_hash
        folder.mkdir(parents=True, exist_ok=True)
        entry = {
            "frame": frame,
            "input_sha256": input_hash,
            "history_frames": history,
            "state": "generating",
            "issues": [],
            "prompt_sha256": digest({"draft": DRAFT_PROMPT, "check": CHECK_PROMPT}),
        }
        entries.append(entry)
        state = {
            "schema": "memory_review/v1",
            "pin": pin,
            "pin_digest": digest(pin),
            "record": record,
            "entries": entries,
        }
        try:
            draft = response_json(call_model(client, payload, folder / "draft.json", stop))
            validate_point(draft, record["facts"], frame=frame, cameras=pin["cameras"], visible=visible)
            check_context = {
                "task": pin["task"],
                "previous_memory": context["previous_memory"],
                "proposed_output": {k: draft["checkpoint"][k] for k in ("memory", "subtask")},
            }
            check = validate_check(
                response_json(
                    call_model(
                        client,
                        request_payload(config, CHECK_PROMPT, check_context, images[:1]),
                        folder / "check.json",
                        stop,
                    )
                )
            )
            entry["student_check"] = check
            entry["issues"] = check["issues"]
            needs_review = (
                not check["supported"]
                or bool(check["issues"])
                or draft["checkpoint"]["decision"] == "needs_review"
                or any(f["status"] == "uncertain" for f in draft["facts"])
            )
            if needs_review:
                # This is an explicit gate, with the original model response retained in the receipt.
                draft["checkpoint"]["decision"] = "needs_review"
                entry["state"] = "needs_review"
                if not entry["issues"]:
                    entry["issues"] = ["Teacher reported uncertainty; human review required"]
            else:
                entry["state"] = "complete"
            record["facts"].extend(draft["facts"])
            record["checkpoints"].append(draft["checkpoint"])
        except (ValueError, KeyError, IndexError, TypeError, QwenRequestError) as exc:
            entry["state"] = "request_error" if isinstance(exc, QwenRequestError) else "invalid_response"
            entry["issues"] = [
                str(exc)
                if isinstance(exc, QwenRequestError)
                else "Response failed strict JSON/schema/causal validation"
            ]
        atomic_json(output / "review.json", state)
        progress(
            {
                "phase": "memory_annotation",
                "current": len(entries),
                "total": len(pin["checkpoints"]),
                "message": f"Frame {frame}: {entry['state']}",
            }
        )
        if entry["state"] != "complete":
            break
    state = {
        "schema": "memory_review/v1",
        "pin": pin,
        "pin_digest": digest(pin),
        "record": record,
        "entries": entries,
    }
    atomic_json(output / "review.json", state)
    return state
