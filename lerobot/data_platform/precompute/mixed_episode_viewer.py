"""Per-episode camera and signal metadata for mixed, read-only evaluation datasets."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from lerobot.data_platform.camera_layout import camera_rank

PROTOCOL_KEY = "data_platform_mixed_episode_viewer"
ROBOT_TYPE = "mixed_episode_evaluation_v1"


@dataclass(frozen=True)
class EpisodeViewerSpec:
    camera_keys: tuple[str, ...]
    signal_features: dict
    robot_type: str


def is_mixed_episode_viewer(info: dict) -> bool:
    return info.get(PROTOCOL_KEY) == 1 or info.get("robot_type") == ROBOT_TYPE


def episode_viewer_spec(info: dict, episode) -> EpisodeViewerSpec | None:
    """Validate declared episode fields against the dataset-wide feature registry."""
    if not is_mixed_episode_viewer(info):
        return None
    try:
        cameras = json.loads(episode["viewer/cameras_json"])
        signals = json.loads(episode["viewer/signals_json"])
        robot_type = episode["viewer/robot_type"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid mixed episode viewer metadata") from exc
    features = info.get("features") or {}
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(key, str) and features.get(key, {}).get("dtype") == "video" for key in cameras)
        or len(cameras) != len(set(cameras))
        or any(camera_rank(key) > 2 for key in cameras)
        or len({camera_rank(key) for key in cameras}) != len(cameras)
        or not isinstance(signals, dict)
        or not isinstance(robot_type, str)
        or not robot_type
    ):
        raise ValueError("Invalid mixed episode cameras or signals")
    for key, feature in signals.items():
        if (
            not isinstance(key, str)
            or key in {"timestamp", "episode_index", "frame_index", "index", "task_index"}
            or not isinstance(feature, dict)
            or key not in features
            or not str(feature.get("dtype") or "").startswith(("float", "int", "uint"))
            or feature.get("dtype") != features[key].get("dtype")
            or not isinstance(feature.get("shape"), list)
            or len(feature["shape"]) != 1
            or not isinstance(feature["shape"][0], int)
            or feature["shape"][0] < 1
            or not isinstance(features[key].get("shape"), list)
            or len(features[key]["shape"]) != 1
            or not isinstance(features[key]["shape"][0], int)
            or feature["shape"][0] > features[key]["shape"][0]
        ):
            raise ValueError(f"Invalid mixed episode signal: {key}")
    return EpisodeViewerSpec(tuple(cameras), signals, robot_type)


def write_mixed_episode_csv(root, meta, episode_id, out_path, max_frames, downsample, overwrite):
    """Write only this episode's declared raw signals without profile conversion."""
    from lerobot.data_platform.precompute.dataset_io import read_episode_table
    from lerobot.data_platform.precompute.signal_columns import signal_columns
    from lerobot.data_platform.precompute.video import temporary_output_path

    spec = episode_viewer_spec(meta.info, meta.episodes[episode_id])
    if spec is None:
        raise ValueError("Expected a mixed episode viewer dataset")
    out_path = Path(out_path)
    if not overwrite and out_path.is_file() and out_path.stat().st_size:
        return True, None, []
    columns = signal_columns(spec.signal_features, qualify=spec.robot_type.lower().startswith("umi"))
    table = read_episode_table(root, meta, episode_id, columns=["timestamp", *spec.signal_features])
    if table.num_rows != int(meta.episodes[episode_id]["length"]):
        raise ValueError(f"Mixed episode row count mismatch: {episode_id}")
    records = table.to_pylist()
    if max_frames is not None:
        records = records[:max_frames]
    if downsample and downsample > 1:
        records = records[::downsample]
    temporary = temporary_output_path(out_path)
    try:
        opener = gzip.open if out_path.name.endswith(".csv.gz") else open
        with opener(temporary, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", *[label for col in columns for label in col["value"]]])
            for row in records:
                values = [row["timestamp"]]
                for col in columns:
                    value = row[col["key"]]
                    dim = len(col["value"])
                    parts = (
                        [None] * dim if value is None else list(value) if isinstance(value, list) else [value]
                    )
                    if len(parts) != dim:
                        raise ValueError(f"Mixed episode signal dimension mismatch: {col['key']}")
                    values.extend(parts)
                writer.writerow(values)
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True, None, []
