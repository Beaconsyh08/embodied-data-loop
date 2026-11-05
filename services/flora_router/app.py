"""Standalone gateway using the existing FLORA API and a live key catalog."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from flask import jsonify

from services.flora_api.app import create_app as create_flora_app


class KeyCatalog:
    """Combine existing and gateway-only identities without caching revocations."""

    def __init__(self, *paths: Path):
        self.paths = paths

    def read_text(self) -> str:
        records = []
        owners = set()
        digests = set()
        for path in self.paths:
            for record in json.loads(path.read_text()):
                owner, digest = record["id"], record["sha256"]
                if owner in owners or digest in digests:
                    raise ValueError("Duplicate API identity")
                owners.add(owner)
                digests.add(digest)
                records.append(record)
        return json.dumps(records)


def create_app():
    catalog = KeyCatalog(Path(os.environ["FLORA_API_KEYS"]), Path(os.environ["FLORA_ROUTER_KEYS"]))
    app = create_flora_app(keys=catalog)
    state = Path(os.environ["FLORA_API_STATE"])

    def health():
        heartbeat = state / "worker-heartbeat"
        ready = heartbeat.is_file() and time.time() - heartbeat.stat().st_mtime < 90
        return jsonify(service="flora-router", version="0.1.0", worker_ready=ready), 200 if ready else 503

    app.view_functions["health"] = health
    return app
