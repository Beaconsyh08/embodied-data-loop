"""Persist upstream verification receipts while retaining every upstream identity/path check."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path


def tuples(value):
    return tuple(tuples(item) for item in value) if isinstance(value, list) else value


def enable(namespace: dict, directory: str | None):
    """Wrap, never bypass, the upstream validator that checks manifest and stat signatures."""
    if not directory:
        return
    original = namespace["validate_manifest"]
    verified = namespace["_VERIFIED_MANIFESTS"]
    cache = Path(directory)
    cache.mkdir(parents=True, exist_ok=True)

    def validate(path):
        key = str(path.resolve())
        filename = hashlib.sha256(key.encode()).hexdigest()
        receipt = cache / f"{filename}.json"
        with (cache / f"{filename}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                saved = json.loads(receipt.read_text())
                if saved["schema"] == 1 and saved["root"] == key:
                    verified[key] = tuples(saved["verified"])
            except (OSError, ValueError, KeyError, TypeError):
                pass
            try:
                # The original validator rechecks containment, manifest hash, resolved paths,
                # device/inode/size/mtime/ctime before accepting its cached verification tuple.
                original(path)
            except Exception:
                verified.pop(key, None)
                receipt.unlink(missing_ok=True)
                raise
            if key in verified:
                payload = {"schema": 1, "root": key, "verified": verified[key]}
                with tempfile.NamedTemporaryFile(mode="w", dir=cache, delete=False) as output:
                    temporary = Path(output.name)
                    json.dump(payload, output)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(receipt)

    namespace["validate_manifest"] = validate
