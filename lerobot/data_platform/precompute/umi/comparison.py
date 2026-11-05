#!/usr/bin/env python3
"""Build a read-only, offline review of existing UMI Tracker repair results."""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from lerobot.data_platform.operators import OperatorResult
from lerobot.data_platform.precompute.dataset_io import load_episode_records
from lerobot.data_platform.precompute.umi.diagnostics import episode_table, pose_arrays, source_fingerprint

PARTS = ("left_arm", "right_arm", "head")
COLORS = ((67, 146, 249), (248, 155, 57), (183, 124, 246))
IMAGES = ("left_arm_image", "head_image", "right_arm_image")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SIZE = (1280, 960)
VIEW = np.array([[0.8660254, 0, -0.5], [-0.25, 0.8660254, -0.4330127]])


def write_js(path, variable, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"window.{variable}=" + json.dumps(value, separators=(",", ":"), allow_nan=False) + ";\n")


def read_episode(root, metadata, columns):
    return episode_table(root, metadata, columns)


def rounded(x):
    return np.round(x, 6).tolist()


def motion(p, times):
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Nonincreasing timestamps")
    r = Rotation.from_quat(p[:, [4, 5, 6, 3]])
    return (
        np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1) / dt,
        np.rad2deg((r[:-1].inv() * r[1:]).magnitude()) / dt,
        r,
    )


def render_video(path, table, raw, repaired, times, flags, stats, fps, episode, label="existing repair"):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.mp4")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{SIZE[0]}x{SIZE[1]}",
        "-r",
        str(fps),
        "-i",
        "pipe:",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    fonts = {s: ImageFont.truetype(FONT, s) for s in (14, 16, 19, 24)}
    bg = Image.new("RGB", SIZE, (15, 22, 35))
    d = ImageDraw.Draw(bg)
    d.text(
        (22, 15),
        f"UMI | Episode {episode:02d} | Raw vs {label}",
        font=fonts[24],
        fill=(235, 241, 251),
    )
    for x, name in zip((22, 447, 872), ("Left wrist", "Head camera", "Right wrist"), strict=True):
        d.text((x, 58), name, font=fonts[19], fill=(209, 220, 238))
    for box, name in [((18, 331, 629, 680), "RAW TRACKER"), ((651, 331, 1262, 680), "REPAIRED TRACKER")]:
        d.rounded_rectangle(box, radius=12, fill=(24, 34, 51))
        d.text(
            (box[0] + 15, box[1] + 10),
            name + " | shared world coordinates",
            font=fonts[19],
            fill=(224, 232, 248),
        )
    all_xyz = np.concatenate([p[:, :3] for p in [*raw, *repaired]])
    projected = all_xyz @ VIEW.T
    lo = projected.min(axis=0)
    hi = projected.max(axis=0)
    center = (lo + hi) / 2
    span = max(float((hi - lo).max()), 0.5) * 1.25
    scale = 270 / span
    paths = [[p[:, :3] @ VIEW.T for p in group] for group in (raw, repaired)]
    mats = [[Rotation.from_quat(p[:, [4, 5, 6, 3]]).as_matrix() for p in group] for group in (raw, repaired)]

    def project(p, side):
        return (int(323 + side * 633 + (p[0] - center[0]) * scale), int(520 - (p[1] - center[1]) * scale))

    for side in range(2):
        for k in range(3):
            xy = [project(p, side) for p in paths[side][k]]
            if len(xy) > 1:
                d.line(xy, fill=tuple(int(c * 0.38) for c in COLORS[k]), width=1)
    vels = [[motion(p, times)[0] for p in group] for group in (raw, repaired)]
    y_positions = (728, 802, 876)
    for k, top in enumerate(y_positions):
        d.text(
            (22, top - 24),
            f"{PARTS[k]} | speed m/s | raw solid / repair dashed",
            font=fonts[14],
            fill=COLORS[k],
        )
        ymax = max(float(max(vels[0][k].max(initial=0), vels[1][k].max(initial=0))), 0.1)
        for j in range(2):
            vals = vels[j][k]
            xy = [
                (80 + i / max(len(times) - 1, 1) * 1160, top + 43 - float(v) / ymax * 42)
                for i, v in enumerate(vals, 1)
            ]
            if len(xy) > 1:
                if j == 0:
                    d.line(xy, fill=COLORS[k], width=2)
                else:
                    for a in range(0, len(xy) - 1, 6):
                        d.line(xy[a : a + 4], fill=(234, 240, 248), width=2)
        d.text((24, top + 5), f"{ymax:.1f}", font=fonts[14], fill=(156, 172, 198))
        d.line((80, top + 44, 1240, top + 44), fill=(60, 77, 100))
    d.text(
        (22, 940),
        "Same frames and camera bytes | numerical checks are not physical validation",
        font=fonts[14],
        fill=(158, 175, 201),
    )
    log = path.with_suffix(".ffmpeg.log")
    with log.open("wb") as errors:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in range(len(times)):
                img = bg.copy()
                draw = ImageDraw.Draw(img)
                for j, name in enumerate(IMAGES):
                    source = table[name][frame].as_py()
                    im = Image.open(BytesIO(source["bytes"])).convert("RGB")
                    im.thumbnail((408, 230))
                    img.paste(im, (22 + j * 425 + (408 - im.width) // 2, 88 + (230 - im.height) // 2))
                draw.text(
                    (904, 20),
                    f"frame {frame:04d}/{len(times) - 1}  {times[frame]:.2f}s",
                    font=fonts[16],
                    fill=(210, 221, 240),
                )
                for side in range(2):
                    for k in range(3):
                        points = paths[side][k]
                        tail = [project(p, side) for p in points[max(0, frame - 90) : frame + 1]]
                        if len(tail) > 1:
                            draw.line(tail, fill=COLORS[k], width=3)
                        x, y = project(points[frame], side)
                        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLORS[k])
                        p = (raw if side == 0 else repaired)[k][frame, :3]
                        for axis, color in zip(
                            range(3), ((245, 95, 95), (103, 213, 138), (99, 163, 253)), strict=True
                        ):
                            end = (p + mats[side][k][frame, :, axis] * 0.12) @ VIEW.T
                            draw.line(((x, y), project(end, side)), fill=color, width=2)
                        value = flags[k][frame]
                        flag_label = "unknown" if value is None else "true" if value else "false"
                        draw.text(
                            (34 + side * 633, 600 + k * 21),
                            f"{PARTS[k]}: recorded flag {flag_label}",
                            font=fonts[14],
                            fill=COLORS[k],
                        )
                cursor = 80 + frame / max(len(times) - 1, 1) * 1160
                for top in y_positions:
                    draw.line((cursor, top - 2, cursor, top + 45), fill=(255, 230, 137), width=2)
                process.stdin.write(img.tobytes())
        finally:
            process.stdin.close()
            rc = process.wait()
        if rc:
            raise RuntimeError(f"ffmpeg failed for episode {episode}: {log.read_text()}")
    partial.replace(path)
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,avg_frame_rate,width,height:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    if int(probe["streams"][0]["nb_read_frames"]) != len(times):
        raise ValueError(f"Video frame count mismatch {episode}")
    return probe


def build_episode(arguments):
    src, dst, out, meta, report, fps = arguments
    ep = meta["episode_index"]
    n = meta["length"]
    fields = [
        "timestamp",
        *IMAGES,
        *[p + s for p in PARTS for s in ("_quaternion_pose", "_pose_is_valid")],
    ]
    a = read_episode(src, meta, fields)
    b = read_episode(dst, meta, fields)
    if any(not a[name].combine_chunks().equals(b[name].combine_chunks()) for name in IMAGES):
        raise ValueError(f"Episode {ep} camera bytes differ; frame correspondence is unproven")
    times = np.asarray(a["timestamp"].to_pylist(), dtype=float).reshape(-1)
    if not np.array_equal(times, np.asarray(b["timestamp"].to_pylist()).reshape(-1)):
        raise ValueError("Timestamp mismatch")
    raw_info = json.loads((src / "meta/info.json").read_text())
    repaired_info = json.loads((dst / "meta/info.json").read_text())
    raw = [pose_arrays(a, raw_info, p) for p in PARTS]
    repaired = [pose_arrays(b, repaired_info, p) for p in PARTS]
    flags = [
        np.asarray(a[p + "_pose_is_valid"].to_pylist(), dtype=object).reshape(-1)
        if p + "_pose_is_valid" in a.column_names
        else np.full(n, None)
        for p in PARTS
    ]
    payload = {
        "episode": ep,
        "frames": n,
        "fps": fps,
        "times": rounded(times),
        "parts": {},
        "tasks": meta.get("tasks", []),
    }
    regression = False
    failed = False
    modified = 0
    for k, part in enumerate(PARTS):
        detail = report["parts"][part] if report else {}

        def metrics(p):
            _, _, rot = motion(p, times)
            step = np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1) * 1000
            angle = np.rad2deg((rot[:-1].inv() * rot[1:]).magnitude())
            return {
                "max_step_mm": float(step.max(initial=0)),
                "max_step_deg": float(angle.max(initial=0)),
                "steps_gt_100mm": int((step > 100).sum()),
                "steps_gt_10deg": int((angle > 10).sum()),
            }

        before = metrics(raw[k])
        after = metrics(repaired[k])
        regression |= any(after[x] > before[x] for x in ("steps_gt_100mm", "steps_gt_10deg"))
        failed |= after["steps_gt_100mm"] > 0 or after["steps_gt_10deg"] > 0
        changed = np.any(raw[k].astype(np.float32) != repaired[k].astype(np.float32), axis=1)
        modified += int(changed.sum())
        va, wa, ra = motion(raw[k], times)
        vb, wb, rb = motion(repaired[k], times)
        reference = ra[0]
        payload["parts"][part] = {
            "raw": rounded(raw[k]),
            "repaired": rounded(repaired[k]),
            "valid": [None if v is None else int(v) for v in flags[k]],
            "changed": changed.astype(int).tolist(),
            "raw_rotation": rounded(np.rad2deg((reference.inv() * ra).as_rotvec())),
            "repaired_rotation": rounded(np.rad2deg((reference.inv() * rb).as_rotvec())),
            "raw_speed": [None, *rounded(va)],
            "repaired_speed": [None, *rounded(vb)],
            "raw_angular_speed": [None, *rounded(wa)],
            "repaired_angular_speed": [None, *rounded(wb)],
            "before": before,
            "after": after,
            "segments": [
                {
                    key: v
                    for key, v in s.items()
                    if key in ("component", "start", "end_exclusive", "method", "confidence", "status")
                }
                for s in detail.get("segments", [])
            ],
        }
    write_js(out / "episodes" / f"episode_{ep:06d}.js", "UMI_EPISODE", payload)
    video = out / "videos" / f"episode_{ep:06d}.mp4"
    probe = render_video(video, a, raw, repaired, times, flags, {}, fps, ep)
    result = {
        "episode": ep,
        "frames": n,
        "duration": round(n / fps, 3),
        "regression": bool(regression),
        "residual": bool(failed),
        "modified_part_frames": modified,
        "video": f"videos/episode_{ep:06d}.mp4",
        "data": f"episodes/episode_{ep:06d}.js",
        "video_probe": probe,
    }
    print(
        json.dumps(
            {"completed_episode": ep, "frames": n, "residual": bool(failed), "regression": bool(regression)}
        ),
        flush=True,
    )
    return result


def run_comparison(context, parameters):
    roots = tuple(Path(p).resolve() for p in context.source_roots)
    if len(roots) != 2 or roots[0] != context.source_root.resolve() or roots[0] == roots[1]:
        raise ValueError("UMI comparison requires two distinct registered source locations")
    out = Path(context.work_dir).resolve()
    if any(out == r or out.is_relative_to(r) or r.is_relative_to(out) for r in roots):
        raise ValueError("Comparison artifacts must not overlap either source")
    infos = [json.loads((r / "meta/info.json").read_text()) for r in roots]
    if infos[0]["fps"] != infos[1]["fps"]:
        raise ValueError("Comparison requires the same frame rate")
    records = [load_episode_records(r) for r in roots]
    identities = [[(r["episode_index"], r["length"], r.get("tasks")) for r in rows] for rows in records]
    if identities[0] != identities[1]:
        raise ValueError("Comparison requires matching episode indices, frame counts and tasks")
    # This first integration deliberately accepts native Tracker columns; packed EE data
    # remains eligible for replay/ID through the isolated evaluation adapter.
    for info in infos:
        if not all(p + "_quaternion_pose" in info.get("features", {}) for p in PARTS):
            raise ValueError("UMI comparison currently requires native Tracker pose columns")
        if not all(info["features"].get(i, {}).get("dtype") == "image" for i in IMAGES):
            raise ValueError("UMI comparison currently requires embedded three-view images")
    selected = parameters.get("episodes")
    if selected is not None and (not selected or set(selected) - {r["episode_index"] for r in records[0]}):
        raise ValueError("Select existing episode indices")
    rows = [r for r in records[0] if selected is None or r["episode_index"] in selected]
    fingerprints = [source_fingerprint(r) for r in roots]
    reports = {}
    report_path = roots[1] / "pose_repair_report.json"
    if report_path.is_file():
        reports = {r["episode_index"]: r for r in json.loads(report_path.read_text())["episodes"]}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.js").exists() or (out / "videos").exists():
        raise ValueError("Use a fresh comparison output directory")
    for asset in (Path(__file__).parent / "assets").iterdir():
        shutil.copy2(asset, out / asset.name)
    manifest = {
        "title": "UMI 轨迹版本对比",
        "fps": infos[0]["fps"],
        "total_frames": sum(r["length"] for r in rows),
        "total_episodes": len(rows),
        "source": str(roots[0]),
        "repaired": str(roots[1]),
        "alignment": "episode_index_and_frame_with_identical_camera_bytes",
        "complete": False,
        "episodes": [],
    }
    workers = context.resources.workers(parameters.get("workers", 2)) if context.resources else 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [
            pool.submit(build_episode, (*roots, out, r, reports.get(r["episode_index"]), infos[0]["fps"]))
            for r in rows
        ]
        try:
            for future in as_completed(pending):
                value = future.result()
                manifest["episodes"].append(value)
                manifest["episodes"].sort(key=lambda x: x["episode"])
                write_js(out / "manifest.js", "UMI_MANIFEST", manifest)
                context.emit(
                    {
                        "current": len(manifest["episodes"]),
                        "total": len(rows),
                        "message": f"Compared UMI episode {value['episode']}",
                    }
                )
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    if [source_fingerprint(r) for r in roots] != fingerprints:
        raise ValueError("Source changed during UMI comparison")
    (out / "source_fingerprints.json").write_text(
        json.dumps(dict(zip(map(str, roots), fingerprints, strict=True)), indent=2) + "\n"
    )
    manifest["complete"] = True
    write_js(out / "manifest.js", "UMI_MANIFEST", manifest)
    artifacts = tuple(
        {"path": str(p.relative_to(out)), "kind": "report" if p.name == "index.html" else "comparison"}
        for p in sorted(out.rglob("*"))
        if p.is_file() and not p.name.endswith(".log")
    )
    return OperatorResult(
        summary={
            "episodes": len(rows),
            "frames": manifest["total_frames"],
            "physical_quality_approved": False,
        },
        artifacts=artifacts,
    )
