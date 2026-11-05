"""Remove the temporary recovery guard after a bounded legacy bridge finishes."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path


def outstanding(database: Path, parents: tuple[str, ...]) -> bool:
    if not parents:
        raise ValueError("At least one explicitly selected parent is required")
    placeholders = ",".join("?" for _ in parents)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30) as db:
        running = db.execute(
            "SELECT 1 FROM jobs j JOIN legacy_bridge_claims b ON b.job_id=j.id "
            "WHERE j.status='running' LIMIT 1"
        ).fetchone()
        queued = db.execute(
            "SELECT 1 FROM jobs WHERE status='queued' AND kind='episode' "
            f"AND parent_id IN ({placeholders}) LIMIT 1",
            parents,
        ).fetchone()
    return bool(running or queued)


def main():
    if os.geteuid() != 0:
        raise RuntimeError("Root is required to remove the systemd recovery guard")
    root = Path(os.environ["FLORA_API_STATE"])
    parents = tuple(value for value in os.environ["FLORA_LEGACY_PARENTS"].split(",") if value)
    if outstanding(root / "jobs.sqlite3", parents):
        raise RuntimeError("Selected episodes remain queued or bridge-owned; guard retained")
    state = subprocess.check_output(
        ["systemctl", "show", "flora-legacy-h100-bridge.service", "-p", "ActiveState", "--value"], text=True
    ).strip()
    if state != "inactive":
        raise RuntimeError("Bridge is not inactive; guard retained")
    backup = Path(Path("/opt/flora-router/legacy-bridge/current-backup").read_text().strip())
    deployment = json.loads((backup / "deployment.json").read_text())
    guard = Path("/etc/systemd/system/flora-api-worker.service.d/90-flora-legacy-bridge.conf")
    if not guard.is_file() or hashlib.sha256(guard.read_bytes()).hexdigest() != deployment["guard_sha256"]:
        raise RuntimeError("Recovery guard was changed; leaving it untouched")
    guard.unlink()
    subprocess.run(["systemctl", "disable", "flora-legacy-h100-bridge.service"], check=True)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    receipt = backup / "cleanup.json"
    receipt.write_text(json.dumps({"guard_removed": True, "bridge_disabled": True, "at": time.time()}))
    receipt.chmod(0o600)


if __name__ == "__main__":
    main()
