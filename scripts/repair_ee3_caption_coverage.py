#!/usr/bin/env python3
"""Explicit operational CLI for the approved EE3 source repair, not memory generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.data_platform.execution import atomic_json
from lerobot.data_platform.mem_export import digest as file_digest
from lerobot.data_platform.memory.contracts import digest, provider_config_digest
from lerobot.data_platform.memory.source_repair import CAMERAS, PROMPT, repair, repair_plan
from lerobot.data_platform.qwen import QwenClient
from lerobot.data_platform.temporal_caption_snapshot import caption_source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error("Use 1–10 bounded workers")
    source = args.source.resolve()
    if source.name != "ee3_20260929_2_grasp_from_wm" or args.output.resolve().is_relative_to(source):
        parser.error("This repair is scoped to EE3 with output outside the source")
    manifest = json.loads(args.manifest.read_text())
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or any(info["features"][c]["dtype"] != "image" for c in CAMERAS):
        raise ValueError("Expected audited EE3 image-backed v2.1 source")
    model = os.environ.get("DATA_PLATFORM_MEMORY_MODEL", "qwen3-vl-plus")
    provider = provider_config_digest(model)
    selected = [
        r for r in manifest["episodes"] if args.episodes is None or r["episode_index"] in args.episodes
    ]
    if args.episodes and {r["episode_index"] for r in selected} != set(args.episodes):
        raise ValueError("Unknown or already eligible episode")
    plans = []
    for row in selected:
        path = args.manifest.parent / row["original_file"]
        if not path.resolve().is_relative_to(args.manifest.parent.resolve()):
            raise ValueError("Original caption path escaped manifest directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["original_sha256"]:
            raise ValueError("Original caption changed")
        original = json.loads(raw)
        if (
            original["episode_index"] != row["episode_index"]
            or original["source_fingerprint"] != manifest["source_fingerprint"]
            or original["dataset_key"] != manifest["dataset_key"]
        ):
            raise ValueError("Original caption source identity changed")
        plans.append((row, original, repair_plan(original)))
    print(
        json.dumps(
            {
                "episodes": len(plans),
                "qwen_intervals": sum(not c["reuse"] for _, _, p in plans for c in p),
                "preserved_intervals": sum(c["reuse"] for _, _, p in plans for c in p),
                "execute": args.execute,
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    binding = {
        "source_manifest_sha256": file_digest(args.manifest),
        "source": str(source),
        "model": model,
        "provider_config_sha256": provider,
        "prompt_sha256": digest(PROMPT),
    }
    binding_file = args.output / "binding.json"
    if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
        raise ValueError("Repair binding changed; use a new output revision")
    atomic_json(binding_file, binding)
    if caption_source_fingerprint(source, args.output / "hashes") != manifest["source_fingerprint"]:
        raise ValueError("Source fingerprint changed before repair")
    states, lock = {}, threading.Lock()

    def stop():
        if (args.output / "CANCEL").exists():
            raise InterruptedError("Source repair cancelled")

    def progress(index, value):
        with lock:
            states[str(index)] = value
            atomic_json(args.output / "batch-progress.json", {"episodes": states})

    def run(item):
        row, original, planned = item
        index = row["episode_index"]
        folder = args.output / f"episode-{index:06d}"
        stop()
        try:
            path = (
                source
                / info["data_path"].format(
                    episode_index=index, episode_chunk=index // int(info.get("chunks_size", 1000))
                )
            ).resolve()
            if not path.is_relative_to(source):
                raise ValueError("Episode data escapes source")
            source_sha = file_digest(path)
            seal = folder / "completed.json"
            if seal.exists():
                saved = json.loads(seal.read_text())
                if saved["data_sha256"] != source_sha or any(
                    file_digest(folder / p) != h for p, h in saved["files"].items()
                ):
                    raise ValueError("Completed repair package changed")
                progress(index, {"status": "draft", "resumed": True, **saved["summary"]})
                return
            table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", *CAMERAS])
            rows = table.sort_by("frame_index").to_pylist()
            if len(rows) != original["frame_count"] or any(
                r["episode_index"] != index
                or r["frame_index"] != i
                or abs(r["timestamp"] - rows[0]["timestamp"] - i / original["fps"]) > 0.002
                for i, r in enumerate(rows)
            ):
                raise ValueError("Episode timing or identity changed")
            result = repair(
                original,
                row["original_sha256"],
                source,
                rows,
                row["task"],
                folder,
                QwenClient(max_response_bytes=1024 * 1024),
                model,
                provider,
                stop,
                lambda value: progress(index, {"status": "running", **value}),
            )
            stop()
            if file_digest(path) != source_sha:
                raise ValueError("Episode source changed during repair")
            result["coverage_repair"].update(
                original_run=row["original_run"], data_sha256=source_sha, source_binding=binding
            )
            atomic_json(folder / "caption.json", result)
            summary = {
                "intervals": len(planned),
                "qwen_intervals": sum(not c["reuse"] for c in planned),
                "needs_review": result["coverage_repair"]["needs_review"],
            }
            atomic_json(
                seal,
                {
                    "data_sha256": source_sha,
                    "summary": summary,
                    "files": {
                        p.relative_to(folder).as_posix(): file_digest(p)
                        for p in sorted(folder.rglob("*"))
                        if p.is_file()
                    },
                },
            )
            progress(index, {"status": "draft", **summary})
        except Exception as exc:
            progress(index, {"status": "error", "type": type(exc).__name__})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, item) for item in plans]):
            future.result()
    stop()
    if caption_source_fingerprint(source, args.output / "hashes") != manifest["source_fingerprint"]:
        raise ValueError("Source changed during batch; do not publish")
    report = {
        "schema": "ee3_qwen_source_repair_execution/v1",
        "binding": binding,
        "source_reverified": True,
        "episodes": states,
        "human_approvals": 0,
        "training_started": False,
        "memory_annotation_started": False,
    }
    atomic_json(args.output / "execution.json", report)
    print(
        json.dumps(
            {
                "completed": sum(v["status"] == "draft" for v in states.values()),
                "errors": sum(v["status"] == "error" for v in states.values()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
