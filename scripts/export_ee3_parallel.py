#!/usr/bin/env python3
"""Bounded parallel EE3 export, preserving and verifying interrupted PNG output."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from lerobot.data_platform.artifacts.mem_export import (
    SCHEMA,
    contained,
    digest,
    inspect_selection,
    write_json,
)
from lerobot.data_platform.operators.implementations.caption.snapshot import caption_source_fingerprint
from lerobot.data_platform.runtime.supervisor import atomic_json


def export_episode(index, selection, staging, audit):
    item = inspect_selection(selection)
    eid = f"episode-{index:06d}"
    seal_path = audit / f"{eid}.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text())
        if seal["selection"] != selection or seal["source_hashes"] != item["hashes"]:
            raise ValueError("Completed export source changed")
        if any(digest(contained(staging, staging / p)) != h for p, h in seal["files"].items()):
            raise ValueError("Completed export files changed")
        return seal
    if item["videos"] or item["root"].name != "ee3_20260929_2_grasp_from_wm":
        raise ValueError("This operational exporter only accepts audited image-backed EE3")
    frames = [
        {"frame_index": i, "timestamp": float(row["timestamp"]), "images": []}
        for i, row in enumerate(item["rows"])
    ]
    media_hash = hashlib.sha256()
    files, reused = {}, 0
    for camera_index, camera in enumerate(item["cameras"]):
        for frame_index, row in enumerate(item["rows"]):
            value = row[camera]
            source = (
                io.BytesIO(value["bytes"])
                if value.get("bytes")
                else contained(item["root"], item["root"] / value["path"])
            )
            with Image.open(source) as original:
                image = original.convert("RGB")
            pixels = image.tobytes()
            relative = f"media/{eid}/camera-{camera_index:02d}/{frame_index:06d}.png"
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                # Never trust partial output merely because its filename exists.
                try:
                    with Image.open(path) as prior:
                        same = prior.mode == "RGB" and prior.size == image.size and prior.tobytes() == pixels
                except OSError:
                    # Preserve an incomplete write left by interruption, then reconstruct it from raw RGB.
                    interrupted = audit / "interrupted" / relative
                    interrupted.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(path, interrupted)
                else:
                    if not same:
                        raise ValueError(f"Interrupted image differs from raw source: {relative}")
                    reused += 1
            if not path.exists():
                if shutil.disk_usage(staging).free < 2 * 1024**3:
                    raise OSError("Insufficient data-disk space")
                temporary = path.with_suffix(".png.partial")
                image.save(temporary, format="PNG")
                os.replace(temporary, path)
            files[relative] = digest(path)
            media_hash.update(f"{camera}:{frame_index}:{image.size}".encode())
            media_hash.update(pixels)
            frames[frame_index]["images"].append(
                {
                    "camera": camera,
                    "path": relative,
                    "width": image.width,
                    "height": image.height,
                    "source_timestamp": float(row["timestamp"]),
                    "decode_error_s": 0.0,
                }
            )
    frames_path = f"media/{eid}/frames.jsonl"
    (staging / frames_path).write_text("".join(json.dumps(row) + "\n" for row in frames))
    original, normalized = f"annotations/{eid}.original.json", f"annotations/{eid}.json"
    (staging / "annotations").mkdir(exist_ok=True)
    shutil.copyfile(item["annotation_path"], staging / original)
    write_json(
        staging / normalized,
        {
            "interval_convention": "half-open",
            "result": item["artifact"]["result"],
            "model_review": item["artifact"].get("review"),
            "human_review": None,
            "annotation_binding": "metadata_checked_requires_human_source_confirmation",
        },
    )
    episode = {
        "id": eid,
        "dataset_key": selection["dataset_key"],
        "episode_index": selection["episode_index"],
        **item["identity"],
        "group_id": item["identity"]["lineage_group"] or media_hash.hexdigest(),
        "group_confirmed": bool(item["identity"]["lineage_group"]),
        "content_sha256": media_hash.hexdigest(),
        "frame_count": len(frames),
        "fps": item["fps"],
        "cameras": item["cameras"],
        "frames_file": frames_path,
        "task": selection["task"],
        "annotation": normalized,
        "original_annotation": original,
        "annotation_sha256": item["annotation_sha256"],
        "source_files_sha256": item["hashes"],
    }
    if (
        any(digest(item["root"] / p) != h for p, h in item["hashes"].items())
        or digest(item["annotation_path"]) != item["annotation_sha256"]
    ):
        raise ValueError("Source changed during export")
    for name in (frames_path, original, normalized):
        files[name] = digest(staging / name)
    seal = {
        "selection": selection,
        "source_hashes": item["hashes"],
        "episode": episode,
        "files": files,
        "reused_images": reused,
    }
    atomic_json(seal_path, seal)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 32:
        raise ValueError("Output must be new and workers bounded")
    output, staging = args.output.resolve(), args.staging.resolve()
    if staging.parent != output.parent or not staging.name.startswith(f".{output.name}-"):
        raise ValueError("Staging must be an export-owned sibling of the final output")
    selection = json.loads(args.manifest.read_text())
    rows = selection["episodes"]
    if selection["schema"] != "mem_export_selection/v1" or [s["episode_index"] for s in rows] != list(
        range(219)
    ):
        raise ValueError("Require the fixed complete 219-episode EE3 selection in original order")
    roots = {str(Path(s["root"]).resolve()) for s in rows}
    if len(roots) != 1 or any(staging.is_relative_to(Path(root)) for root in roots):
        raise ValueError("One immutable source and separate output required")
    root = Path(next(iter(roots)))
    args.audit.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=True)
    binding = {"selection_sha256": digest(args.manifest), "source_fingerprint": args.fingerprint}
    binding_file = args.audit / "binding.json"
    if binding_file.exists() and json.loads(binding_file.read_text()) != binding:
        raise ValueError("Export binding changed")
    atomic_json(binding_file, binding)
    if caption_source_fingerprint(root, args.audit / "hashes") != args.fingerprint:
        raise ValueError("Source fingerprint changed")
    seals = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export_episode, i, s, staging, args.audit): i for i, s in enumerate(rows)}
        for future in as_completed(futures):
            index = futures[future]
            seals[index] = future.result()
            atomic_json(
                args.audit / "progress.json",
                {"complete": len(seals), "total": len(rows), "last_episode": index},
            )
            print(json.dumps({"complete": len(seals), "last_episode": index}), flush=True)
    episodes = [seals[i]["episode"] for i in range(len(rows))]
    (staging / "episodes.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes)
    )
    files = {p: h for s in seals.values() for p, h in s["files"].items()}
    files["episodes.jsonl"] = digest(staging / "episodes.jsonl")
    if {str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()} != set(files):
        raise ValueError("Unexpected or partial file in staging")
    if (
        caption_source_fingerprint(root, args.audit / "hashes") != args.fingerprint
        or digest(args.manifest) != binding["selection_sha256"]
    ):
        raise ValueError("Source/selection changed during export")
    write_json(
        staging / "manifest.json",
        {
            "schema": SCHEMA,
            "selection_sha256": binding["selection_sha256"],
            "episode_count": len(episodes),
            "files": files,
            "interval_convention": "half-open",
            "media": "lossless RGB PNG, no annotation overlays",
            "causal_decode": True,
        },
    )
    if output.exists():
        raise FileExistsError(output)
    os.rename(staging, output)
    atomic_json(
        args.audit / "complete.json",
        {
            "episodes": len(episodes),
            "frames": sum(e["frame_count"] for e in episodes),
            "reused_images": sum(s["reused_images"] for s in seals.values()),
            "source_unchanged": True,
            "manifest_sha256": digest(output / "manifest.json"),
        },
    )


if __name__ == "__main__":
    main()
