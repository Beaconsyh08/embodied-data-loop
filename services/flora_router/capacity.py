"""Read-only, fail-closed GPU and API capacity observations."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Capacity:
    free_gpus: tuple[int, ...]
    active_job_ids: tuple[str, ...]
    pending_gpu_job_ids: tuple[str, ...]
    concurrency: int


def _locked_inodes(text: str) -> set[tuple[int, int, int]]:
    result = set()
    for line in text.splitlines():
        match = re.search(r"\b([0-9a-fA-F]+):([0-9a-fA-F]+):(\d+)\b", line)
        if match:
            result.add((int(match[1], 16), int(match[2], 16), int(match[3])))
    return result


def _mib(value: str | None) -> int:
    match = re.fullmatch(r"\s*(\d+)\s+MiB\s*", value or "")
    if not match:
        raise ValueError("GPU memory information is unavailable")
    return int(match[1])


def eligible_gpus(xml: str, pool: list[int], lock_dir: Path, locks: str) -> tuple[int, ...]:
    """Report candidates without taking locks or starting GPU work."""
    if (
        not pool
        or len(pool) > 8
        or len(set(pool)) != len(pool)
        or any(type(x) is not int or x < 0 for x in pool)
    ):
        raise ValueError("Invalid GPU pool")
    locked = _locked_inodes(locks)
    devices = {}
    for gpu in ET.fromstring(xml).findall("gpu"):
        index = gpu.findtext("minor_number")
        if index is None or not index.isdigit() or int(index) in devices:
            raise ValueError("GPU identity is unavailable")
        devices[int(index)] = gpu
    result = []
    for index in pool:
        gpu = devices.get(index)
        if gpu is None:
            raise ValueError("A configured GPU is unavailable")
        processes = gpu.find("processes")
        if (
            processes is None
            or (processes.text or "").strip()
            or any(p.tag != "process_info" for p in processes)
        ):
            raise ValueError("GPU process occupancy is unavailable")
        path = lock_dir / f"gpu-{index}.lock"
        if path.is_symlink():
            raise ValueError("GPU lock is a symlink")
        try:
            stat = path.stat()
        except FileNotFoundError:
            lock_busy = False
        else:
            if not path.is_file():
                raise ValueError("GPU lock is not a regular file")
            lock_busy = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino) in locked
        if (
            not lock_busy
            and not processes.findall("process_info")
            and _mib(gpu.findtext("fb_memory_usage/total")) >= 75000
            and _mib(gpu.findtext("fb_memory_usage/free")) >= 73000
        ):
            result.append(index)
    return tuple(result)


def read_capacity(
    runtime: Path, state: Path, concurrency: int, *, extra_states: tuple[Path, ...] = (), timeout: int = 10
) -> Capacity:
    if not 1 <= concurrency <= 8:
        raise ValueError("Invalid API concurrency")
    config = json.loads(runtime.read_text())
    pool = config.get("gpus") or [config["gpu"]]
    locks = Path("/proc/locks").read_text()
    xml = subprocess.run(
        ["nvidia-smi", "-q", "-x"], capture_output=True, check=True, text=True, timeout=timeout
    ).stdout
    free = eligible_gpus(xml, pool, Path(config["lock_dir"]), locks)
    heartbeat = state / "worker-heartbeat"
    if not heartbeat.is_file() or time.time() - heartbeat.stat().st_mtime >= 90:
        raise RuntimeError("Backend worker is not ready")
    rows = []
    for path in (state, *extra_states):
        with sqlite3.connect(f"file:{path / 'jobs.sqlite3'}?mode=ro", uri=True, timeout=3) as db:
            rows.extend(
                db.execute(
                    "SELECT id,stage FROM jobs WHERE status IN ('uploading','queued','running') AND kind!='dataset'"
                )
            )
    gpu_stages = (
        "FLORA acquired GPU ",
        "FLORA segment ",
        "FLORA caption ",
        "FLORA terminal ",
        "FLORA dense ",
    )
    active = tuple(row[0] for row in rows)
    pending = tuple(job_id for job_id, stage in rows if not (stage or "").startswith(gpu_stages))
    return Capacity(free, active, pending, concurrency)
