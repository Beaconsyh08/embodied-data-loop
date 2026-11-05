"""Preserve bridge-owned remote attempts when the original API worker restarts."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


class RecoveryView:
    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):
        return getattr(self.store, name)

    def interrupted(self):
        with sqlite3.connect(self.store.root / "jobs.sqlite3", timeout=30) as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='legacy_bridge_claims'"
            ).fetchone()
            claimed = (
                set(db.execute("SELECT job_id FROM legacy_bridge_claims").fetchall()) if exists else set()
            )
        return [job for job in self.store.interrupted() if (job["id"],) not in claimed]


def main():
    # The systemd unit retains the immutable original release as its working directory.
    sys.path.insert(0, str(Path.cwd()))
    from services.flora_api import worker

    original = worker.recover
    worker.recover = lambda store, validate: original(RecoveryView(store), validate)
    worker.main()


if __name__ == "__main__":
    main()
