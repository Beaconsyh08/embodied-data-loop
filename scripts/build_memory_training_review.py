#!/usr/bin/env python3
"""Build a read-only review snapshot from an already verified model-only training ledger."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def build_snapshot(ledger, coverage, proof, selection, exported, bundle_manifest, source_hashes):
    if ledger.get("schema") != "mem_ledger/v1" or proof.get("schema") != "ee3_model_trial_verification/v1":
        raise ValueError("Invalid ledger or verification schema")
    if (
        proof.get("actual_model_requests") != 0
        or proof.get("human_approvals") != 0
        or coverage.get("quality") != "model_only_experiment"
        or coverage.get("human_approvals") != 0
    ):
        raise ValueError("Expected offline-verified model-only annotations")
    if (
        proof["source_fingerprint"] != selection["source_fingerprint"]
        or proof["binding"]["manifest_sha256"] != source_hashes["selection"]
    ):
        raise ValueError("Frozen source manifest does not match verification")
    if (
        source_hashes["verification"] != coverage["verification_sha256"]
        or source_hashes["bundle_manifest"] != ledger["bundle_sha256"]
        or bundle_manifest.get("schema") != "mem_dataset_export/v1"
        or bundle_manifest["files"].get("episodes.jsonl") != source_hashes["episodes"]
    ):
        raise ValueError("Verification or training bundle changed")
    selected = {f"episode-{row['episode_index']:06d}": row for row in selection["episodes"]}
    bundle = {row["id"]: row for row in exported}
    records = {row["episode_id"]: row for row in ledger["episodes"]}
    if (
        len(selected) != len(selection["episodes"])
        or len(bundle) != len(exported)
        or len(records) != len(ledger["episodes"])
        or set(selected) != set(bundle)
        or set(selected) != set(records)
        or set(selected) != set(proof["episodes"])
    ):
        raise ValueError("Episode inventories do not match")
    if (
        len(selected) != coverage["requested_episodes"]
        or coverage["excluded_episodes"]
        or len(selected) != bundle_manifest["episode_count"]
    ):
        raise ValueError("Incomplete training ledger")
    if proof["stopped_episodes"] != coverage["stopped_episodes"]:
        raise ValueError("Stopped-episode evidence changed")
    episodes = []
    for eid, record in sorted(records.items()):
        source, media, evidence = selected[eid], bundle[eid], proof["episodes"][eid]
        checks = ledger["model_checks"][eid]
        frames = [p["frame"] for p in record["checkpoints"]]
        payload = {k: v for k, v in record.items() if k != "review"}
        if (
            record.get("review") != {"status": "draft"}
            or canonical(payload) != evidence["record_sha256"]
            or checks["payload_sha256"] != evidence["record_sha256"]
            or checks["bundle_sha256"] != ledger["bundle_sha256"]
            or checks["status"] != "model_checked"
            or evidence.get("student_input_protocol") != 2
            or checks["receipts_sha256"] != evidence["receipts_sha256"]
            or frames != evidence["checked_frames"]
            or frames != checks["checked_frames"]
            or frames != sorted(set(frames))
            or any(type(frame) is not int or not 0 <= frame < source["frame_count"] for frame in frames)
            or media["dataset_version_id"] != proof["source_fingerprint"]
            or media["annotation_sha256"] != source["annotation_sha256"]
            or media["episode_index"] != source["episode_index"]
            or media["frame_count"] != source["frame_count"]
            or media["fps"] != source["fps"]
            or media["cameras"] != ["image", "left_wrist_image", "right_wrist_image"]
            or media["lineage_group"] != source["lineage_group"]
            or record["task"] != source["task"]
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", source["run"])
        ):
            raise ValueError(f"Training record or source mismatch: {eid}")
        stopped = coverage["stopped_episodes"].get(eid)
        if evidence.get("complete") != (stopped is None):
            raise ValueError(f"Completion state differs: {eid}")
        if stopped and (
            stopped["accepted_checkpoints"] != len(frames)
            or (frames and stopped["stopped_at"] <= frames[-1])
            or not 0 <= stopped["stopped_at"] < source["frame_count"]
            or stopped["planned_checkpoints"] <= len(frames)
        ):
            raise ValueError(f"Invalid stopped prefix: {eid}")
        episodes.append(
            {
                "episode_index": source["episode_index"],
                "caption_run": source["run"],
                "annotation_sha256": source["annotation_sha256"],
                "frame_count": source["frame_count"],
                "fps": source["fps"],
                "split": source["split"],
                "task": record["task"],
                "accepted": len(frames),
                "complete": stopped is None,
                "stopped_at": stopped["stopped_at"] if stopped else None,
                "planned": stopped["planned_checkpoints"] if stopped else len(frames),
                "issues": stopped["issues"] if stopped else None,
                "checkpoints": record["checkpoints"],
                "facts": record["facts"],
            }
        )
    if sum(e["accepted"] for e in episodes) != coverage["included_checkpoints"]:
        raise ValueError("Checkpoint totals differ")
    if sum(e["complete"] for e in episodes) != coverage["complete_episodes"]:
        raise ValueError("Complete episode totals differ")
    keys = {row["dataset_key"] for row in bundle.values()}
    if len(keys) != 1:
        raise ValueError("Expected one dataset")
    return {
        "schema": "memory_training_review/v1",
        "dataset_key": keys.pop(),
        "source_fingerprint": proof["source_fingerprint"],
        "bundle_sha256": ledger["bundle_sha256"],
        "source_sha256": source_hashes,
        "quality": "model_only_experiment",
        "human_approvals": 0,
        "requested_episodes": len(episodes),
        "accepted_checkpoints": coverage["included_checkpoints"],
        "complete_episodes": coverage["complete_episodes"],
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("ledger", "coverage", "verification", "selection", "episodes", "bundle-manifest", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    raw = {
        name: getattr(args, name).read_bytes()
        for name in ("ledger", "coverage", "verification", "selection", "episodes", "bundle_manifest")
    }
    snapshot = build_snapshot(
        *(json.loads(raw[n]) for n in ("ledger", "coverage", "verification", "selection")),
        [json.loads(line) for line in raw["episodes"].splitlines()],
        json.loads(raw["bundle_manifest"]),
        {k: sha256(v) for k, v in raw.items()},
    )
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, prefix=".memory-review-", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump(snapshot, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_name, args.output)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
    print(f"Built {len(snapshot['episodes'])} episodes and {snapshot['accepted_checkpoints']} checkpoints")


if __name__ == "__main__":
    main()
