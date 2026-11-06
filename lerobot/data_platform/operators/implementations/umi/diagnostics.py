"""Separate recorded validity, rotation encoding and observed trajectory anomalies."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from lerobot.data_platform.operators import OperatorResult
from lerobot.data_platform.operators.implementations.dataset_io import load_episode_records

PARTS = ("left_arm", "right_arm", "head")
BLOCKS = {"left_arm": slice(0, 7), "right_arm": slice(8, 15), "head": slice(16, 23)}


def ranges(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [
        [int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    ]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_files(root, info):
    pattern = re.sub(r"\{[^}]+\}", "*", info["data_path"])
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("Dataset data paths must stay inside the source")
    paths = sorted(root.glob(pattern))
    if not paths or any(
        path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) for path in paths
    ):
        raise ValueError("Missing or unsafe dataset Parquet files")
    return paths


def source_fingerprint(root):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    metadata = sorted(p for p in (root / "meta").rglob("*") if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in metadata):
        raise ValueError("Dataset metadata must stay inside the source")
    return {str(p.relative_to(root)): file_hash(p) for p in [*metadata, *data_files(root, info)]}


def episode_table(root, record, columns, paths=None):
    info = json.loads((root / "meta/info.json").read_text())
    paths = paths or data_files(root, info)
    if "data/chunk_index" in record and "data/file_index" in record:
        indexed = (
            root
            / "data"
            / f"chunk-{record['data/chunk_index']:03d}"
            / f"file-{record['data/file_index']:03d}.parquet"
        )
        # A v3 episode may span shards: trust the indexed fast path only when it is complete.
        candidates = [indexed, *(p for p in paths if p != indexed)] if indexed in paths else paths
    else:
        candidates = paths
    pieces = []
    count = 0
    for path in candidates:
        ids = pq.read_table(
            path, columns=["frame_index"], filters=[("episode_index", "=", record["episode_index"])]
        )
        if not len(ids):
            continue
        available = set(pq.read_schema(path).names)
        selected = list(dict.fromkeys(["frame_index", "episode_index", *columns]))
        table = pq.read_table(
            path,
            columns=[c for c in selected if c in available],
            filters=[("episode_index", "=", record["episode_index"])],
        )
        pieces.append(table)
        count += len(table)
    if not pieces or count != record["length"]:
        raise ValueError(f"Episode {record['episode_index']} row count differs from metadata")
    table = pa.concat_tables(pieces).sort_by("frame_index")
    if table["frame_index"].to_pylist() != list(range(len(table))):
        raise ValueError("Episode frames must be unique and contiguous")
    return table


def pose_arrays(table, info, part):
    name = part + "_quaternion_pose"
    if name in table.column_names:
        feature = info["features"][name]
        names = feature.get("names")
        expected = ("x", "y", "z", "qw", "qx", "qy", "qz")
        if not isinstance(names, list) or len(names) != 7 or set(names) != set(expected):
            raise ValueError("UMI quaternion fields require explicit xyz and quaternion names")
        values = np.asarray(table[name].to_pylist(), dtype=float)
        return values[:, [names.index(x) for x in expected]]
    if "state" in table.column_names:
        feature = info["features"]["state"]
        expected = [
            f"{p}_{v}"
            for p in ("left", "right", "head")
            for v in ("x", "y", "z", "qw", "qx", "qy", "qz", *(() if p == "head" else ("gripper",)))
        ]
        if feature.get("names") != expected:
            raise ValueError("Packed UMI state requires the declared 23D named layout")
        return np.asarray(table["state"].to_pylist(), dtype=float)[:, BLOCKS[part]]
    raise ValueError("Unsupported UMI pose layout")


def diagnose_part(
    pose, timestamps, recorded=None, legacy=None, *, position_step_mm=100.0, rotation_step_deg=10.0
):
    n = len(pose)
    norm = np.linalg.norm(pose[:, 3:], axis=1)
    finite = np.isfinite(pose).all(axis=1)
    valid_q = np.isfinite(pose[:, 3:]).all(axis=1) & (norm > 1e-8)
    safe = pose[:, 3:].copy()
    safe[~valid_q] = [1, 0, 0, 0]
    rotations = Rotation.from_quat(safe[:, [1, 2, 3, 0]])
    dt = np.diff(timestamps)
    edge_valid = np.isfinite(dt) & (dt > 0) & valid_q[:-1] & valid_q[1:] & finite[:-1] & finite[1:]
    pos = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) * 1000
    angle = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())
    pos_bad = edge_valid & (pos > position_step_mm)
    rot_bad = edge_valid & (angle > rotation_step_deg)
    bad = np.r_[False, pos_bad | rot_bad]
    all_good = np.ones(n, dtype=bool)
    recorded_values = None if recorded is None else np.asarray(recorded, dtype=object).reshape(-1)
    if recorded_values is not None and any(
        v is not None and not isinstance(v, (bool, np.bool_)) for v in recorded_values
    ):
        raise ValueError("Recorded validity must contain booleans or null")
    known = (
        np.zeros(n, dtype=bool)
        if recorded_values is None
        else np.array([v is not None for v in recorded_values])
    )
    flag = None if recorded_values is None else np.array([bool(v) for v in recorded_values])
    if flag is not None and len(flag) != n:
        raise ValueError("Recorded validity length mismatch")
    encoding = {"unknown": n}
    per_flag = {}
    if legacy is not None:
        legacy = np.asarray(legacy, dtype=float)
        comparable = valid_q & np.isfinite(legacy).all(axis=1)
        rv = np.zeros(n, dtype=bool)
        eu = rv.copy()
        for kind in ("rotvec", "euler_xyz"):
            rot = (
                Rotation.from_rotvec(legacy[comparable, 3:])
                if kind == "rotvec"
                else Rotation.from_euler("xyz", legacy[comparable, 3:])
            )
            match = np.rad2deg((rot.inv() * rotations[comparable]).magnitude()) < 0.1
            (rv if kind == "rotvec" else eu)[comparable] = match
        encoding = {
            "rotvec": int(rv.sum()),
            "euler_xyz": int(eu.sum()),
            "neither": int((~rv & ~eu).sum()),
            "both": int((rv & eu).sum()),
        }
        if flag is not None:
            per_flag = {
                str(value).lower(): {
                    "frames": int(((flag == value) & known).sum()),
                    "rotvec": int(((flag == value) & known & rv).sum()),
                    "euler_xyz": int(((flag == value) & known & eu).sum()),
                }
                for value in (True, False)
            }
    groups = {"all": all_good}
    if flag is not None:
        groups.update(
            recorded_true=flag & known,
            recorded_false=~flag & known,
            flag_transition=np.r_[False, (flag[1:] != flag[:-1]) & known[1:] & known[:-1]],
        )
    edge_stats = {}
    for name, mask in groups.items():
        edges = mask[1:] & edge_valid
        vals = pos[edges]
        edge_stats[name] = {
            "edges": int(edges.sum()),
            "position_jumps": int((edges & pos_bad).sum()),
            "rotation_jumps": int((edges & rot_bad).sum()),
            "median_position_step_mm": float(np.median(vals)) if len(vals) else None,
            "p99_position_step_mm": float(np.quantile(vals, 0.99)) if len(vals) else None,
        }
    frozen = np.r_[False, np.all(pose[1:] == pose[:-1], axis=1)]
    frozen_ranges = [] if frozen[1:].all() else [r for r in ranges(frozen) if r[1] - r[0] >= 4]
    return {
        "frames": n,
        "recorded_flag_present": flag is not None,
        "recorded_false_frames": int((~flag & known).sum()) if flag is not None else None,
        "recorded_unknown_frames": int((~known).sum()),
        "recorded_false_ranges": ranges(~flag & known) if flag is not None else [],
        "encoding": encoding,
        "encoding_by_recorded_flag": per_flag,
        "nonfinite_ranges": ranges(~finite),
        "invalid_quaternion_ranges": ranges(~valid_q | (np.abs(norm - 1) > 0.001)),
        "frozen_ranges": frozen_ranges,
        "bad_timestamp_edges": (np.flatnonzero(~np.isfinite(dt) | (dt <= 0)) + 1).tolist(),
        "position_jump_frames": (np.flatnonzero(pos_bad) + 1).tolist(),
        "rotation_jump_frames": (np.flatnonzero(rot_bad) + 1).tolist(),
        "observed_anomaly_ranges": ranges(bad),
        "edge_statistics": edge_stats,
        "max_abs_position_m": float(np.abs(pose[finite, :3]).max()) if finite.any() else None,
    }


def diagnose(
    root, destination, *, episodes=None, progress=None, position_step_mm=100.0, rotation_step_deg=10.0
):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if root == destination or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("UMI reports must be outside source datasets")
    info = json.loads((root / "meta/info.json").read_text())
    records = load_episode_records(root)
    selected = {r["episode_index"] for r in records} if episodes is None else set(episodes)
    if not selected or selected - {r["episode_index"] for r in records}:
        raise ValueError("Select existing episode indices")
    paths = data_files(root, info)
    fingerprint = source_fingerprint(root)
    rows = []
    columns = [
        "timestamp",
        "state",
        *[p + suffix for p in PARTS for suffix in ("_pose", "_quaternion_pose", "_pose_is_valid")],
    ]
    for record in records:
        if record["episode_index"] not in selected:
            continue
        table = episode_table(root, record, columns, paths)
        ts = np.asarray(table["timestamp"].to_pylist(), dtype=float).reshape(-1)
        parts = {}
        for part in PARTS:
            pose = pose_arrays(table, info, part)
            flag = (
                table[part + "_pose_is_valid"].to_pylist()
                if part + "_pose_is_valid" in table.column_names
                else None
            )
            legacy = table[part + "_pose"].to_pylist() if part + "_pose" in table.column_names else None
            parts[part] = diagnose_part(
                pose, ts, flag, legacy, position_step_mm=position_step_mm, rotation_step_deg=rotation_step_deg
            )
        rows.append(
            {
                "episode_index": record["episode_index"],
                "frames": len(table),
                "cosmos_window_eligible": len(table) >= 49 and info.get("fps") == 30,
                "parts": parts,
            }
        )
        if progress:
            progress(
                {
                    "current": len(rows),
                    "total": len(selected),
                    "message": f"Diagnosed UMI episode {record['episode_index']}",
                }
            )
    if source_fingerprint(root) != fingerprint:
        raise ValueError("Source changed during UMI diagnosis")
    report = {
        "schema_version": 1,
        "source": str(root),
        "source_fingerprint": fingerprint,
        "frame_semantics": "half_open",
        "config": {"position_step_mm": position_step_mm, "rotation_step_deg": rotation_step_deg},
        "episodes": rows,
        "summary": {
            "episodes": len(rows),
            "frames": sum(r["frames"] for r in rows),
            "short_episodes": [r["episode_index"] for r in rows if not r["cosmos_window_eligible"]],
            "recorded_flag_semantics": "unconfirmed",
            "physical_quality_approved": False,
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "umi_diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    lines = [
        "<!doctype html><meta charset=utf-8><title>UMI diagnosis</title><h1>UMI 采集标记与轨迹诊断</h1><p>采集标记含义尚未确认；编码匹配和轨迹跳变分别报告。</p><table border=1 cellpadding=8><tr><th>Episode</th><th>Tracker</th><th>Frames</th><th>Recorded false</th><th>Position jumps</th><th>Rotation jumps</th><th>Rotvec / Euler matches</th></tr>"
    ]
    for row in rows:
        for part, value in row["parts"].items():
            stats = value["edge_statistics"]["all"]
            values = [
                row["episode_index"],
                part,
                row["frames"],
                value["recorded_false_frames"],
                stats["position_jumps"],
                stats["rotation_jumps"],
                f"{value['encoding'].get('rotvec', 0)} / {value['encoding'].get('euler_xyz', 0)}",
            ]
            lines.append("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in values) + "</tr>")
    lines.append("</table><p><a href=umi_diagnostics.json>完整逐帧区间与来源指纹</a></p>")
    (destination / "umi_diagnostics.html").write_text("\n".join(lines))
    return report


def run_diagnostics(context, parameters):
    report = diagnose(context.source_root, context.work_dir, progress=context.emit, **parameters)
    return OperatorResult(
        summary=report["summary"],
        artifacts=(
            {"path": "umi_diagnostics.json", "kind": "umi_diagnostics"},
            {"path": "umi_diagnostics.html", "kind": "report"},
        ),
    )
