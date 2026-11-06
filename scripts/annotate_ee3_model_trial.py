#!/usr/bin/env python3
"""User-authorized full EE3 model-only memory experiment; preserves production review gates."""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.data_platform.ai_tools.qwen import QwenClient
from lerobot.data_platform.artifacts.mem_export import digest as file_digest
from lerobot.data_platform.memory import ai_review
from lerobot.data_platform.memory.contracts import checkpoints, digest, provider_config_digest
from lerobot.data_platform.memory.model_trial import BALANCED_CHECK, CHECK, DRAFT, TEXT_TARGET_RULES, annotate
from lerobot.data_platform.memory.source_repair import CAMERAS
from lerobot.data_platform.operators.implementations.caption.snapshot import caption_source_fingerprint
from lerobot.data_platform.runtime.supervisor import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--max-checkpoints", type=int, help="Explicit probe only, never a full episode")
    parser.add_argument("--max-label-revisions", type=int, choices=(3, 6, 9), default=3)
    parser.add_argument(
        "--repair-plan", type=Path, help="Pinned prior stopped cutoffs for student-visible repair"
    )
    parser.add_argument("--execute-model-only", action="store_true")
    parser.add_argument(
        "--ai-review-plan",
        type=Path,
        help="Pinned prior reviews; retain prefix and record AI adjudication from cutoff",
    )
    args = parser.parse_args()
    if args.ai_review_plan and args.repair_plan:
        parser.error("Choose either AI review or the legacy repair plan")
    if not 1 <= args.workers <= 64:
        parser.error("Use 1-64 bounded workers")
    source = args.source.resolve()
    if source.name != "ee3_20260929_2_grasp_from_wm" or args.output.resolve().is_relative_to(source):
        parser.error("EE3 only; output must be outside the immutable source")
    manifest = json.loads(args.manifest.read_text())
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or any(info["features"][c]["dtype"] != "image" for c in CAMERAS):
        raise ValueError("Expected audited image-backed EE3")
    selected = [
        r for r in manifest["episodes"] if args.episodes is None or r["episode_index"] in args.episodes
    ]
    if args.episodes and {r["episode_index"] for r in selected} != set(args.episodes):
        raise ValueError("Unknown episode")
    args.output.mkdir(parents=True, exist_ok=True)
    binding = {
        "manifest_sha256": file_digest(args.manifest),
        "model": "qwen3-vl-plus",
        "provider_sha256": provider_config_digest("qwen3-vl-plus"),
        "prompts_sha256": digest([DRAFT, CHECK]),
        "probe_limit": args.max_checkpoints,
        "max_label_revisions": args.max_label_revisions,
        "student_input_protocol": 2,
    }
    binding["prompts_sha256"] = digest([DRAFT + TEXT_TARGET_RULES, CHECK, BALANCED_CHECK])
    ai_plan = json.loads(args.ai_review_plan.read_text()) if args.ai_review_plan else None
    if ai_plan is not None:
        binding.update(
            ai_review_protocol=1,
            ai_review_plan_sha256=file_digest(args.ai_review_plan),
            prompts_sha256=digest(
                [
                    DRAFT + TEXT_TARGET_RULES,
                    CHECK,
                    BALANCED_CHECK,
                    ai_review.DRAFT_GUIDANCE,
                    ai_review.CHECK_PROMPT,
                ]
            ),
        )
    repair_plan = json.loads(args.repair_plan.read_text()) if args.repair_plan else None
    if repair_plan is not None:
        binding["repair_plan_sha256"] = file_digest(args.repair_plan)
    binding_file = args.output / "binding.json"
    if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
        raise ValueError("Inputs changed; use a new output revision")
    atomic_json(binding_file, binding)
    if caption_source_fingerprint(source, args.output / "hashes") != manifest["source_fingerprint"]:
        raise ValueError("Source fingerprint changed")
    print(
        json.dumps({"episodes": len(selected), "execute": args.execute_model_only, "binding": binding}),
        flush=True,
    )
    if not args.execute_model_only:
        return
    states, lock = {}, threading.Lock()

    def stop():
        if (args.output / "CANCEL").exists():
            raise InterruptedError("Annotation cancelled")

    def progress(index, value):
        with lock:
            states[str(index)] = value
            atomic_json(args.output / "progress.json", {"episodes": states})

    def run(row):
        index = row["episode_index"]
        folder = args.output / f"episode-{index:06d}"
        try:
            stop()
            annotation = (args.manifest.parent / row["annotation_file"]).resolve()
            if not annotation.is_relative_to(args.manifest.parent.resolve()):
                raise ValueError("Annotation escaped input root")
            if file_digest(annotation) != row["annotation_sha256"]:
                raise ValueError("Annotation source changed")
            path = (
                source
                / info["data_path"].format(
                    episode_index=index, episode_chunk=index // int(info.get("chunks_size", 1000))
                )
            ).resolve()
            if not path.is_relative_to(source):
                raise ValueError("Episode escaped source")
            data_hash = file_digest(path)
            # Explicit allowlist excludes all state/action and annotation text from teacher data.
            rows = (
                pq.read_table(
                    path, columns=["episode_index", "frame_index", "timestamp", "task_index", *CAMERAS]
                )
                .sort_by("frame_index")
                .to_pylist()
            )
            tasks = {
                t["task_index"]: t["task"]
                for t in map(json.loads, (source / "meta/tasks.jsonl").read_text().splitlines())
            }
            if len(rows) != row["frame_count"] or any(
                r["episode_index"] != index
                or r["frame_index"] != i
                or abs(r["timestamp"] - rows[0]["timestamp"] - i / row["fps"]) > 0.002
                or tasks[r["task_index"]] != row["task"]
                for i, r in enumerate(rows)
            ):
                raise ValueError("Episode identity, timing, task or count changed")
            schedule = checkpoints([r["timestamp"] for r in rows], row["segment_starts"])
            pin = {
                **row,
                **binding,
                "source_fingerprint": manifest["source_fingerprint"],
                "data_sha256": data_hash,
                "checkpoints": schedule[: args.max_checkpoints] if args.max_checkpoints else schedule,
                "full_checkpoint_count": len(schedule),
                "camera_order": CAMERAS,
            }
            if repair_plan is not None:
                repair = repair_plan[str(index)]
                prior_path = Path(repair["review_path"])
                if file_digest(prior_path) != repair["review_sha256"]:
                    raise ValueError("Prior stopped revision changed")
                prior = json.loads(prior_path.read_text())
                cutoff = repair["frame"]
                if (
                    prior["pin"]["data_sha256"] != data_hash
                    or prior["pin"]["annotation_sha256"] != row["annotation_sha256"]
                    or prior["entries"][-1]["state"] != "needs_review"
                    or prior["entries"][-1]["frame"] != cutoff
                    or cutoff not in schedule
                ):
                    raise ValueError("Repair must start at the pinned rejected cutoff")
                pin["student_visible_from_frame"] = cutoff
            if ai_plan is not None:
                revision = ai_plan[str(index)]
                prior_path = Path(revision["review_path"])
                if file_digest(prior_path) != revision["review_sha256"]:
                    raise ValueError("Prior AI review source changed")
                prior = json.loads(prior_path.read_text())
                cutoff = revision["frame"]
                if (
                    prior["pin"]["data_sha256"] != data_hash
                    or prior["pin"]["annotation_sha256"] != row["annotation_sha256"]
                    or prior["pin"]["student_input_protocol"] != 2
                    or prior["pin"]["provider_sha256"] != binding["provider_sha256"]
                    or prior["pin"]["model"] != binding["model"]
                    or prior["pin"]["source_fingerprint"] != manifest["source_fingerprint"]
                    or prior["pin"]["task"] != row["task"]
                    or prior["pin"]["checkpoints"] != schedule
                    or prior["pin"]["split"] != "train"
                    or row["split"] != "train"
                    or cutoff not in schedule
                ):
                    raise ValueError("AI review requires unchanged frozen TRAIN source and schedule")
                pin["ai_review_from_frame"] = prior["pin"].get("ai_review_from_frame", cutoff)
                pin["prior_review_sha256"] = revision["review_sha256"]
                if prior["pin"].get("agent_reviews"):
                    pin["agent_reviews"] = dict(prior["pin"]["agent_reviews"])
                if revision.get("agent_review_notes"):
                    if not isinstance(revision["agent_review_notes"], str):
                        raise ValueError("Agent review notes must be explicit text, not an approval")
                    pin.setdefault("agent_reviews", {})[str(cutoff)] = revision["agent_review_notes"]
            state = annotate(
                source,
                rows,
                pin,
                folder,
                QwenClient(max_response_bytes=1024 * 1024),
                stop,
                lambda value: progress(index, {"status": "running", **value}),
            )
            if file_digest(path) != data_hash:
                raise ValueError("Source changed during annotation")
            progress(
                index,
                {
                    "status": "model_draft_complete" if state["complete"] else "needs_review",
                    "accepted": len(state["record"]["checkpoints"]),
                    "total": len(pin["checkpoints"]),
                    "probe": bool(args.max_checkpoints),
                },
            )
        except Exception as exc:
            progress(index, {"status": "error", "type": type(exc).__name__, "message": str(exc)})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, row) for row in selected]):
            future.result()
    unchanged = caption_source_fingerprint(source, args.output / "hashes") == manifest["source_fingerprint"]
    atomic_json(
        args.output / "execution.json",
        {
            "binding": binding,
            "episodes": states,
            "source_unchanged": unchanged,
            "human_approvals": 0,
            "probe": bool(args.max_checkpoints),
        },
    )
    if not unchanged:
        raise ValueError("Source changed; do not export")


if __name__ == "__main__":
    main()
