"""Prepare original camera streams for a mixed evaluation episode."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from lerobot.data_platform.precompute.dataset_io import V3DatasetMetadata
from lerobot.data_platform.precompute.mixed_episode_viewer import episode_viewer_spec
from lerobot.data_platform.temporal_caption import write_json
from lerobot.data_platform.temporal_caption_flora import camera_roles


def _asset(root: Path, path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("Mixed episode native asset is missing or escapes the dataset")
    return path


def _check_video(path: Path, frames: int, fps: float) -> tuple[int, int]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "mov",
            "-count_frames",
            "-show_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate,width,height:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    data = json.loads(probe.stdout)
    streams = data.get("streams", [])
    if (
        len(streams) != 1
        or int(streams[0]["nb_read_frames"]) != frames
        or not math.isclose(float(Fraction(streams[0]["avg_frame_rate"])), fps, abs_tol=0.001)
    ):
        raise ValueError("Mixed episode native video timing does not match its metadata")
    times = [float(frame["best_effort_timestamp_time"]) for frame in data.get("frames", [])]
    if len(times) != frames or any(
        not math.isfinite(value) or abs(value - times[0] - i / fps) > 0.2 / fps
        for i, value in enumerate(times)
    ):
        raise ValueError("Mixed episode native video has irregular timestamps")
    return int(streams[0]["width"]), int(streams[0]["height"])


def prepare_flora_mixed(root: Path, episode: int, output: Path) -> None:
    """Use the preserved native bundle, retaining source FPS and real camera identities."""
    root = Path(root).resolve(strict=True)
    output = Path(output)
    if output.resolve().is_relative_to(root):
        raise ValueError("Output must be outside the source dataset")
    meta = V3DatasetMetadata(f"local/{root.name}", root)
    if episode not in meta.episodes:
        raise ValueError("Episode is not present in dataset metadata")
    spec = episode_viewer_spec(meta.info, meta.episodes[episode])
    if spec is None:
        raise ValueError("Expected a mixed evaluation dataset")
    records = json.loads(_asset(root, root / "provenance/episodes.json").read_text())
    matches = [r for r in records if r.get("episode_index") == episode]
    if len(matches) != 1:
        raise ValueError("Mixed episode provenance is missing or duplicated")
    record = matches[0]
    native = root / "native" / f"episode_{episode:06d}"
    if record.get("native_relative") != native.relative_to(root).as_posix():
        raise ValueError("Mixed episode native path does not match provenance")
    original = json.loads(_asset(root, native / "episode.json").read_text())
    frames, fps = original.get("frame_count"), original.get("fps")
    repeat = record.get("viewer_repeat_factor")
    cameras = original.get("cameras")
    if (
        type(frames) is not int
        or frames < 1
        or type(fps) not in (int, float)
        or not math.isfinite(fps)
        or fps <= 0
        or type(repeat) is not int
        or repeat < 1
        or frames != record.get("source_frames")
        or fps != record.get("source_fps")
        or frames * repeat != meta.episodes[episode]["length"]
        or not math.isclose(fps * repeat, meta.fps, abs_tol=0.001)
        or original.get("episode_index") != record.get("source_episode_index")
        or original.get("sample_id") != record.get("sample_id")
        or not isinstance(cameras, list)
        or not all(isinstance(c, str) for c in cameras)
        or record.get("camera_keys") != list(spec.camera_keys)
        or cameras != [record.get("videos", {}).get(k, {}).get("source_camera") for k in spec.camera_keys]
        or original.get("task_hint") != meta.episodes[episode].get("tasks")
    ):
        raise ValueError("Mixed episode native metadata does not match dataset provenance")
    roles = camera_roles(cameras)
    videos = [_asset(root, native / "camera-videos" / f"{i}.mp4") for i in range(len(cameras))]
    if any(
        record["videos"][key].get("source_video")
        != (native / "camera-videos" / f"{i}.mp4").relative_to(root).as_posix()
        for i, key in enumerate(spec.camera_keys)
    ):
        raise ValueError("Mixed episode native video path does not match provenance")
    review = _asset(root, native / "video.mp4")
    sizes = [_check_video(path, frames, fps) for path in videos]
    _check_video(review, frames, fps)
    if "left" in roles and sizes[cameras.index(roles["left"])][1] != sizes[cameras.index(roles["right"])][1]:
        raise ValueError("Mixed episode wrist videos must have matching heights")
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "camera-videos").mkdir()
        for i, video in enumerate(videos):
            shutil.copyfile(video, output / "camera-videos" / f"{i}.mp4")
        shutil.copyfile(review, output / "video.mp4")
        write_json(
            output / "episode.json",
            {
                **original,
                "dataset_name": root.name,
                "episode_index": episode,
                "source_episode_index": record["source_episode_index"],
                "signals": [],
                "signal_features": {},
                "include_signals": False,
                "signal_policy": "Visual only. No action/state, pose, gripper or motion evidence supplied.",
            },
        )
    except Exception:
        shutil.rmtree(output)
        raise
