"""Single durable worker using the installed, immutable FLORA adapter and shared GPU locks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from services.flora_api.errors import public_error
from services.flora_api.store import Store


@dataclass(frozen=True)
class Video:
    frames: int
    fps: float
    width: int
    height: int


def inspect_video(path: Path) -> Video:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "mov",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_streams",
            "-show_entries",
            "stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    data = json.loads(probe.stdout)
    stream = data["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    frames = data["frames"]
    width, height = int(stream["width"]), int(stream["height"])
    if not math.isfinite(fps) or not 1 <= fps <= 60 or len(frames) < 2 or width < 16 or height < 16:
        raise ValueError("Video must have at least 2 frames, 1–60 FPS and dimensions of at least 16x16")
    start = float(frames[0]["best_effort_timestamp_time"])
    for i, frame in enumerate(frames):
        timestamp = float(frame["best_effort_timestamp_time"])
        if not math.isfinite(timestamp) or abs(timestamp - start - i / fps) > 0.2 / fps:
            raise ValueError("Only constant-frame-rate videos without missing frames are supported")
    return Video(len(frames), fps, width, height)


def prepare(directory: Path) -> Path:
    settings = json.loads((directory / "input.json").read_text())
    roles = settings["roles"]
    videos = [inspect_video(directory / f"{role}.mp4") for role in roles]
    first = videos[0]
    if any(video.frames != first.frames or abs(video.fps - first.fps) > 0.001 for video in videos):
        raise ValueError("Camera streams must have matching frame counts and FPS")
    if (
        "left" in roles
        and "right" in roles
        and videos[roles.index("left")].height != videos[roles.index("right")].height
    ):
        raise ValueError("Left and right video heights must match")
    bundle = directory / "bundle"
    camera_dir = bundle / "camera-videos"
    camera_dir.mkdir(parents=True)
    digest = hashlib.sha256()
    for i, role in enumerate(roles):
        source = directory / f"{role}.mp4"
        with source.open("rb") as video:
            for chunk in iter(lambda: video.read(1024 * 1024), b""):
                digest.update(chunk)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-protocol_whitelist",
                "file,pipe",
                "-threads",
                "2",
                "-f",
                "mov",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                "setpts=PTS-STARTPTS",
                "-vsync",
                "0",
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv444p",
                "-threads",
                "2",
                str(camera_dir / f"{i}.mp4"),
            ],
            check=True,
            capture_output=True,
        )
        normalized = inspect_video(camera_dir / f"{i}.mp4")
        if normalized != videos[i]:
            raise ValueError("Video normalization changed timing or dimensions")
    # Review artifact only; inference still receives every separate camera stream.
    os.link(camera_dir / "0.mp4", bundle / "video.mp4")
    (bundle / "episode.json").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": first.frames,
                "fps": first.fps,
                "include_signals": False,
                "task_hint": [settings["instruction"]],
                "cameras": [f"{role}_wrist" if role != "main" else "scene" for role in roles],
                "source_digest": digest.hexdigest(),
            }
        )
    )
    return bundle


def recover(store: Store, validate):
    for job in store.interrupted():
        result = store.root / "jobs" / job["id"] / "output" / "caption.json"
        try:
            receipt = json.loads((result.parent / "completed.json").read_text())
            if receipt["sha256"] != hashlib.sha256(result.read_bytes()).hexdigest():
                raise ValueError("Result receipt mismatch")
            validate(json.loads(result.read_text()))
        except Exception:
            store.update(
                job["id"],
                status="failed",
                stage="interrupted",
                error="Worker interrupted; partial outputs retained. No automatic inference retry.",
            )
        else:
            store.update(job["id"], status="completed", stage="completed")


def execute(store: Store, job: dict, infer, preparation_slots=None):
    directory = store.root / "jobs" / job["id"]
    source = None
    stage = "preparing"
    try:
        from contextlib import nullcontext

        if shutil.disk_usage(store.root).free < 100 * 1024**3:
            raise ValueError("Not started: less than 100 GiB free on the shared data disk")

        keys = json.loads(Path(os.environ["FLORA_API_KEYS"]).read_text())
        if not any(record["id"] == job["owner"] and record.get("enabled", True) for record in keys):
            raise ValueError("API key revoked before execution")
        fingerprint = None
        store.update(job["id"], status="running", stage=stage)
        with preparation_slots or nullcontext():
            if job["kind"] == "episode":
                from lerobot.data_platform.temporal_caption import prepare as prepare_episode
                from lerobot.data_platform.temporal_caption_snapshot import caption_source_fingerprint
                from services.flora_api.datasets import resolve_dataset

                settings = json.loads((store.root / "jobs" / job["parent_id"] / "input.json").read_text())
                catalog = json.loads(Path(os.environ["FLORA_API_DATASETS"]).read_text())
                source = resolve_dataset(store.root, settings["dataset_id"], job["owner"], keys, catalog)
                fingerprint = caption_source_fingerprint(source, store.root / "source-hashes", lambda _: None)
                directory.mkdir(parents=True, exist_ok=True)
                bundle = directory / "bundle"
                prepare_episode(source, job["episode_index"], bundle, export_camera_videos=True)
                meta_path = bundle / "episode.json"
                meta = json.loads(meta_path.read_text())
                meta.update(include_signals=False, signals=[], signal_features={})
                meta_path.write_text(json.dumps(meta))
            else:
                bundle = prepare(directory)

        def progress(message):
            nonlocal stage
            stage = message
            # Only adapter progress, never model logs or input instructions, reaches HTTP clients.
            store.update(job["id"], status="running", stage=message)

        stage = "inference"
        infer(bundle, directory / "output", f"api/{job['id']}", progress=progress)
        stage = "validating output"
        if (
            source is not None
            and caption_source_fingerprint(source, store.root / "source-hashes", lambda _: None)
            != fingerprint
        ):
            raise ValueError("Source dataset changed during inference")
        result = directory / "output" / "caption.json"
        receipt = result.parent / "completed.tmp"
        receipt.write_text(json.dumps({"sha256": hashlib.sha256(result.read_bytes()).hexdigest()}))
        receipt.replace(result.parent / "completed.json")
    except Exception as error:
        logging.exception("FLORA job %s failed", job["id"])
        store.update(job["id"], status="failed", stage=stage, error=public_error(error, source))
    else:
        store.update(job["id"], status="completed", stage="completed")


def main():
    from lerobot.data_platform.temporal_caption_flora import FloraRuntime, infer_flora, validate_flora

    store = Store(Path(os.environ["FLORA_API_STATE"]))
    with (store.root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        FloraRuntime.load()
        recover(store, validate_flora)

        def heartbeat():
            while True:
                (store.root / "worker-heartbeat").touch()
                time.sleep(10)

        threading.Thread(target=heartbeat, daemon=True).start()
        concurrency = min(8, max(1, int(os.environ.get("FLORA_API_CONCURRENCY", "8"))))
        preparation_slots = threading.Semaphore(2)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = set()
            while True:
                pending = {future for future in pending if not future.done()}
                if len(pending) < concurrency:
                    job = store.claim()
                    if job:
                        pending.add(pool.submit(execute, store, job, infer_flora, preparation_slots))
                        continue
                time.sleep(1)


if __name__ == "__main__":
    main()
