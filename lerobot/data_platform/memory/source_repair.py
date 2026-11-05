"""Audited coverage repair; original FLORA observations are never overwritten."""

from __future__ import annotations

import copy
import io
import math
import time
from pathlib import Path

from PIL import Image

from lerobot.data_platform.memory.contracts import AnnotationConfig, digest
from lerobot.data_platform.memory.generation import call_model, request_payload, response_json
from lerobot.data_platform.temporal_caption_flora import review_partition, validate_flora

PROMPT = """Describe only visually supported activity or state in the TARGET interval.
Images are chronological synchronized mosaics, left to right: main, left wrist, right wrist.
The task is a goal, not evidence of completed actions. No original captions are provided.
Do not invent motion from a single frame, hidden objects, counts, contact or success.
For an unclear action, describe the visible state and explain the uncertainty. A still robot
or an unidentifiable object can be described without inventing an action.
Return strict JSON only: {"caption":"concise English description", "evidence":[
{"frame":SUPPLIED_FRAME,"camera":"image|left_wrist_image|right_wrist_image"}],
"uncertainty":["specific ambiguity if any"]}. This is an unapproved source caption,
not a memory/subtask training label. Only TARGET frames are shown; do not change boundaries.
For evidence.frame use the exact ORIGINAL EPISODE FRAME ID written immediately before each
image. The ordinal image position (0 through 7) is NOT a frame ID. Do not use ordinal indices.
"""
CAMERAS = ["image", "left_wrist_image", "right_wrist_image"]


def repair_payload(config, context, images):
    payload = request_payload(config, PROMPT, context, images)
    content = payload["messages"][0]["content"]
    labelled = content[:1]
    for frame, item in zip(context["image_frames"], content[1:], strict=True):
        labelled.extend([{"type": "text", "text": f"ORIGINAL EPISODE FRAME ID: {frame}"}, item])
    payload["messages"][0]["content"] = labelled
    payload["max_tokens"] = 1024
    payload["response_format"] = {"type": "json_object"}
    return payload


def repair_plan(original):
    validate_flora(original)
    count, fps = original["frame_count"], original["fps"]
    spans = [
        (min(count, math.ceil(s["start_s"] * fps - 1e-7)), min(count, math.ceil(s["end_s"] * fps - 1e-7)))
        for s in original["native_segments"]
    ]
    planned = []
    for cell in original["result"]["segments"]:
        start, end = cell["start_frame"], cell["end_frame"]
        ids = cell["native_segment_indices"]
        unchanged = len(ids) == 1 and spans[ids[0]] == (start, end)
        # Include both ends without duplicating a short interval's frames.
        frames = sorted({start + i * (end - start - 1) // 7 for i in range(8)})
        planned.append(
            {
                "start_frame": start,
                "end_frame": end,
                "original_segment_indices": ids,
                "reuse": unchanged,
                "image_frames": [] if unchanged else frames,
            }
        )
    return planned


def source_mosaic(root: Path, row, *, panel_size=448):
    canvas = Image.new("RGB", (3 * panel_size, panel_size))
    for index, camera in enumerate(CAMERAS):
        value = row[camera]
        if value.get("bytes"):
            source = io.BytesIO(value["bytes"])
        else:
            source = (root / value["path"]).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError("Image escapes source root")
        with Image.open(source) as image:
            view = image.convert("RGB")
            view.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
            canvas.paste(
                view, (index * panel_size + (panel_size - view.width) // 2, (panel_size - view.height) // 2)
            )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def validate_caption(value, frames):
    if not isinstance(value, dict) or set(value) != {"caption", "evidence", "uncertainty"}:
        raise ValueError("Invalid coverage-repair response fields")
    if not isinstance(value["caption"], str) or not value["caption"].strip():
        raise ValueError("Missing repaired caption")
    if not isinstance(value["uncertainty"], list) or any(
        not isinstance(s, str) or not s.strip() for s in value["uncertainty"]
    ):
        raise ValueError("Invalid uncertainty list")
    if not isinstance(value["evidence"], list) or not value["evidence"]:
        raise ValueError("Missing visual evidence")
    for evidence in value["evidence"]:
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {"frame", "camera"}
            or type(evidence["frame"]) is not int
            or evidence["frame"] not in frames
            or evidence["camera"] not in CAMERAS
        ):
            raise ValueError("Unseen source-caption evidence")
    return value


def repair(
    original,
    original_sha256,
    root,
    rows,
    task,
    output,
    client,
    model,
    provider_digest,
    stop=lambda: None,
    progress=lambda value: None,
):
    """Receipts resume by exact model payload; invalid responses are never silently repaired."""
    from lerobot.data_platform.execution import atomic_json

    planned = repair_plan(original)
    output.mkdir(parents=True, exist_ok=True)
    segments, records = [], []
    config = AnnotationConfig(model)
    for index, cell in enumerate(planned):
        stop()
        start, end = cell["start_frame"], cell["end_frame"]
        record = dict(cell)
        if cell["reuse"]:
            caption = original["native_segments"][cell["original_segment_indices"][0]]["caption"]
            record.update(generator="original_flora", uncertainty=[], evidence=[])
        else:
            context = {
                "task": task,
                "episode_index": original["episode_index"],
                "target_start_frame": start,
                "target_end_frame_exclusive": end,
                "image_frames": cell["image_frames"],
                "camera_order": CAMERAS,
            }
            images = []
            for frame in cell["image_frames"]:
                stop()
                image = source_mosaic(root, rows[frame])
                path = output / "frames" / f"{frame}.png"
                path.parent.mkdir(exist_ok=True)
                if path.exists() and path.read_bytes() != image:
                    raise ValueError("Source image changed during repair")
                path.write_bytes(image)
                images.append(image)
            payload = repair_payload(config, context, images)
            payload_hash = digest(payload)
            binding = digest(
                {"payload": payload_hash, "provider": provider_digest, "source_annotation": original_sha256}
            )
            receipt = output / "requests" / f"{binding}.json"
            response = call_model(client, payload, receipt, stop)
            value = validate_caption(response_json(response), cell["image_frames"])
            caption = value["caption"]
            record.update(
                generator=model,
                receipt=receipt.relative_to(output).as_posix(),
                payload_sha256=payload_hash,
                evidence=value["evidence"],
                uncertainty=value["uncertainty"],
            )
        record["needs_review"] = bool(record["uncertainty"]) or (not cell["reuse"] and end - start == 1)
        segments.append(
            {
                "start_s": start / original["fps"],
                "end_s": end / original["fps"],
                "caption": caption,
                "status": "complete",
            }
        )
        records.append(record)
        atomic_json(
            output / "progress.json", {"completed": index + 1, "total": len(planned), "records": records}
        )
        progress({"completed": index + 1, "total": len(planned)})
    result = copy.deepcopy(original)
    result.update(
        created_at=time.time(),
        model=f"FLORA + {model} coverage repair",
        native_segments=segments,
        result=review_partition(segments, original["frame_count"], original["fps"]),
    )
    # Legacy flora-1 containers remain readable. Explicit provenance distinguishes the
    # new program-partitioned intervals from the retained original TempSeg output.
    result["coverage_repair"] = {
        "schema": "flora_qwen_coverage_repair/v1",
        "status": "draft",
        "human_reviewed": False,
        "original_annotation_sha256": original_sha256,
        "model": model,
        "provider_config_sha256": provider_digest,
        "prompt_sha256": digest(PROMPT),
        "boundary_policy": "frame-aligned exclusive partition of original boundaries",
        "native_segments_semantics": "program-partitioned captions; not native TempSeg predictions",
        "reward_provenance": "unchanged original FLORA reward; never passed to Qwen",
        "needs_review": any(r["needs_review"] for r in records),
        "intervals": records,
    }
    for field in ("memory_source_registration", "stage_metrics"):
        if field in result:
            result["coverage_repair"]["original_" + field] = result.pop(field)
    validate_flora(result)
    assert all(len(c["native_segment_indices"]) == 1 for c in result["result"]["segments"])
    return result
