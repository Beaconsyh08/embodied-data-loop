#!/usr/bin/env python3
"""Rebuild actual teacher/checker inputs from raw images and replay saved responses offline."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.data_platform.artifacts.mem_export import digest as file_digest
from lerobot.data_platform.memory.contracts import digest
from lerobot.data_platform.memory.model_trial import annotate
from lerobot.data_platform.memory.source_repair import CAMERAS
from lerobot.data_platform.operators.implementations.caption.snapshot import caption_source_fingerprint
from lerobot.data_platform.runtime.supervisor import atomic_json


class OfflineOnly:
    def post_chat_completion(self, _payload):
        raise RuntimeError("Reconstructed request has no matching saved response; network is disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads((args.run / "execution.json").read_text())
    if not execution["source_unchanged"] or execution.get("probe"):
        raise ValueError("Require a finished full run with unchanged source")
    info = json.loads((args.source / "meta/info.json").read_text())
    verified, excluded = {}, {}
    for key, status in execution["episodes"].items():
        if status["status"] not in {"model_draft_complete", "needs_review"}:
            raise ValueError(f"Resolve execution error in episode {key} before verification")
        index = int(key)
        folder = args.run / f"episode-{index:06d}"
        stored = json.loads((folder / "review.json").read_text())
        pin = stored["pin"]
        path = (
            args.source
            / info["data_path"].format(
                episode_index=index, episode_chunk=index // int(info.get("chunks_size", 1000))
            )
        ).resolve()
        if not path.is_relative_to(args.source.resolve()) or file_digest(path) != pin["data_sha256"]:
            raise ValueError("Raw source data changed")
        if pin.get("probe_limit") is not None:
            raise ValueError("Probe output cannot be final training data")
        rows = (
            pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", *CAMERAS])
            .sort_by("frame_index")
            .to_pylist()
        )
        with tempfile.TemporaryDirectory(
            prefix="memory-offline-verification-", dir=args.output.parent
        ) as temp:
            scratch = Path(temp)
            # Only response receipts are reused. Every PNG is reconstructed from the immutable raw source.
            shutil.copytree(folder / "requests", scratch / "requests")
            rebuilt = annotate(args.source, rows, pin, scratch, OfflineOnly())
            if rebuilt != stored:
                raise ValueError(f"Replayed annotation changed for episode {index}")
            image_hashes = {p.name: file_digest(p) for p in (scratch / "images").iterdir()}
            if any(file_digest(folder / "images" / name) != sha for name, sha in image_hashes.items()):
                raise ValueError("Saved evidence image changed")
        receipts = {
            str(p.relative_to(folder)): file_digest(p) for p in sorted((folder / "requests").rglob("*.json"))
        }
        record = stored["record"]
        if record["checkpoints"]:
            verified[record["episode_id"]] = {
                "record_sha256": digest({k: v for k, v in record.items() if k != "review"}),
                "review_sha256": file_digest(folder / "review.json"),
                "checked_frames": [p["frame"] for p in record["checkpoints"]],
                "receipts_sha256": digest(receipts),
                "receipt_files": receipts,
                "image_files": image_hashes,
                "data_sha256": pin["data_sha256"],
                "annotation_sha256": pin["annotation_sha256"],
                "student_input_protocol": pin.get("student_input_protocol", 1),
                "complete": stored["complete"],
            }
        if not stored["complete"]:
            excluded[record["episode_id"]] = {
                "stopped_at": stored["entries"][-1]["frame"],
                "accepted_checkpoints": len(record["checkpoints"]),
                "planned_checkpoints": len(pin["checkpoints"]),
                "issues": stored["entries"][-1],
            }
        print(json.dumps({"episode": index, "verified_points": len(record["checkpoints"])}), flush=True)
    expected = pin["source_fingerprint"]
    if caption_source_fingerprint(args.source, args.run / "hashes") != expected:
        raise ValueError("Source fingerprint changed")
    atomic_json(
        args.output,
        {
            "schema": "ee3_model_trial_verification/v1",
            "run": str(args.run.resolve()),
            "binding": execution["binding"],
            "source_fingerprint": expected,
            "episodes": verified,
            "stopped_episodes": excluded,
            "actual_model_requests": 0,
            "human_approvals": 0,
            "method": "Exact saved-response replay with fresh raw-image payload reconstruction",
        },
    )


if __name__ == "__main__":
    main()
