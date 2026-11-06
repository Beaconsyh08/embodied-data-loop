"""Node-owned, fingerprint-bound reuse of previously computed native viewer evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .diagnostics import file_hash, source_fingerprint


def identity(roots, parameters, operation, config):
    assets = {}
    if operation == "umi.cosmos_id":
        for name in ("checkpoint_path", "processor_path", "vae_path", "framework_root"):
            root = Path(config[name])
            files = [root] if root.is_file() else []
            if root.is_dir():
                for directory, dirs, names in os.walk(root):
                    dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
                    files.extend(Path(directory) / name for name in sorted(names))
            assets[name] = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    value = {
        "schema_version": 1,
        "operator": operation,
        "operator_version": "1.1.0",
        "inputs": [source_fingerprint(Path(root)) for root in roots],
        "parameters": parameters,
        "laundry_commit": config["laundry_commit"],
        "calibration_profile": config.get("calibration_profile"),
        "urdf_sha256": file_hash(Path(config["urdf_path"])),
        "runtime_assets": assets,
        "adapter_sha256": {
            name: file_hash(Path(__file__).with_name(name)) for name in ("worker.py", "native_reports.py")
        },
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), value


def lookup(roots, parameters, operation, config):
    configured = config.get("native_result_cache_root")
    if not configured:
        return None
    cache = Path(configured)
    if not cache.is_absolute() or not cache.is_dir():
        raise ValueError("Native result cache must be an administrator-configured directory")
    key, expected = identity(roots, parameters, operation, config)
    entry = cache / key
    if not entry.exists():
        return None
    if entry.is_symlink():
        raise ValueError("Native cache entries cannot be symlinks")
    manifest = json.loads((entry / "cache.json").read_text())
    if manifest["identity"] != expected or not manifest.get("provenance"):
        raise ValueError("Native cache provenance differs from this operation")
    for relative, digest in manifest["files"].items():
        path = entry / "evidence" / relative
        if not path.resolve().is_relative_to((entry / "evidence").resolve()) or path.is_symlink():
            raise ValueError("Unsafe native cache artifact")
        if file_hash(path) != digest:
            raise ValueError("Native cache evidence changed")
    actual = {str(p.relative_to(entry / "evidence")) for p in (entry / "evidence").rglob("*") if p.is_file()}
    if actual != set(manifest["files"]):
        raise ValueError("Native cache file inventory changed")
    return entry / "evidence", {"key": key, "provenance": manifest["provenance"], "reused": True}
