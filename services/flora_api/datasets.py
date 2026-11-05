"""Scoped server datasets and resumable, checksummed dataset-file uploads."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath

from flask import g, jsonify, request


def safe_path(value):
    if not isinstance(value, str) or "\\" in value or len(value) > 512:
        raise ValueError("Invalid dataset file path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or len(path.parts) < 2:
        raise ValueError("Invalid dataset file path")
    if path.parts[0] not in {"meta", "data", "videos", "images"} or any(
        p.startswith(".") for p in path.parts
    ):
        raise ValueError("Only meta/, data/, videos/, images/ files are accepted")
    return value


def owned_upload(state, dataset_id, owner):
    if not re.fullmatch(r"[0-9a-f]{32}", dataset_id):
        raise ValueError("Unknown dataset")
    root = state / "datasets" / dataset_id
    if not (root / "manifest.json").is_file():
        raise ValueError("Unknown dataset")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["owner"] != owner:
        raise ValueError("Unknown dataset")
    return root, manifest


def resolve_dataset(state, dataset_id, owner, key_records, catalog):
    if dataset_id.startswith("server:"):
        alias = dataset_id.removeprefix("server:")
        record = next((r for r in key_records if r["id"] == owner and r.get("enabled", True)), {})
        if alias not in record.get("datasets", []) or alias not in catalog:
            raise ValueError("Unknown or unauthorized dataset")
        return Path(catalog[alias]).resolve(strict=True)
    root, _ = owned_upload(state, dataset_id, owner)
    if not (root / "sealed").exists():
        raise ValueError("Complete the dataset upload first")
    return root / "source"


def register(app, state, keys, store, public):
    def catalog():
        filename = os.environ.get("FLORA_API_DATASETS")
        return json.loads(Path(filename).read_text()) if filename else {}

    @app.get("/v1/flora/datasets")
    def list_datasets():
        records = json.loads(keys.read_text())
        record = next(r for r in records if r["id"] == g.owner)
        available = catalog()
        return jsonify(
            datasets=[
                {"id": f"server:{name}", "name": name}
                for name in record.get("datasets", [])
                if name in available
            ]
        )

    @app.post("/v1/flora/datasets")
    def begin():
        body = request.get_json()
        files = body.get("files") if isinstance(body, dict) else None
        if not isinstance(files, list) or not 1 <= len(files) <= 100000:
            return jsonify(error="Provide a file manifest with 1–100000 entries"), 400
        entries = {}
        try:
            for item in files:
                name = safe_path(item["path"])
                if name in entries or type(item["size"]) is not int or item["size"] < 0:
                    raise ValueError("Duplicate file or invalid file size")
                if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                    raise ValueError("Invalid SHA256")
                entries[name] = {"size": item["size"], "sha256": item["sha256"]}
            if "meta/info.json" not in entries:
                raise ValueError("Dataset must contain meta/info.json")
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400
        if sum(v["size"] for v in entries.values()) + 100 * 1024**3 > shutil.disk_usage(state).free:
            return jsonify(error="Insufficient dataset storage"), 507
        dataset_id = uuid.uuid4().hex
        root = state / "datasets" / dataset_id
        root.mkdir(parents=True)
        (root / "source").mkdir()
        (root / "manifest.json").write_text(json.dumps({"owner": g.owner, "files": entries}))
        return jsonify(id=dataset_id), 201

    @app.put("/v1/flora/datasets/<dataset_id>/files/<path:name>")
    def upload(dataset_id, name):
        try:
            name = safe_path(name)
            root, manifest = owned_upload(state, dataset_id, g.owner)
            expected = manifest["files"].get(name)
            if expected is None:
                raise ValueError("File is not in the manifest")
        except ValueError as error:
            return jsonify(error=str(error)), 400
        # One dataset upload at a time prevents sealing while a file is being replaced.
        with (root / "upload.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (root / "sealed").exists():
                return jsonify(error="Dataset is immutable after completion"), 409
            if request.content_length != expected["size"]:
                return jsonify(error="Content-Length must match the manifest"), 400
            if expected["size"] + 100 * 1024**3 > shutil.disk_usage(state).free:
                return jsonify(error="Insufficient storage"), 507
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = root / "upload.partial"
            digest = hashlib.sha256()
            try:
                size = 0
                with temporary.open("wb") as output:
                    while chunk := request.stream.read(1024 * 1024):
                        size += len(chunk)
                        if size > expected["size"]:
                            raise ValueError("Upload exceeds declared size")
                        digest.update(chunk)
                        output.write(chunk)
                if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                    raise ValueError("Upload size or SHA256 mismatch")
                temporary.replace(target)
                receipts = root / "receipts"
                receipts.mkdir(exist_ok=True)
                (receipts / hashlib.sha256(name.encode()).hexdigest()).touch()
            except ValueError as error:
                return jsonify(error=str(error)), 400
            finally:
                temporary.unlink(missing_ok=True)
        return jsonify(uploaded=name)

    @app.get("/v1/flora/datasets/<dataset_id>")
    def upload_status(dataset_id):
        try:
            root, manifest = owned_upload(state, dataset_id, g.owner)
        except ValueError:
            return jsonify(error="Not found"), 404
        missing = [
            name
            for name in manifest["files"]
            if not (root / "receipts" / hashlib.sha256(name.encode()).hexdigest()).exists()
        ]
        return jsonify(id=dataset_id, missing=missing, sealed=(root / "sealed").exists())

    @app.post("/v1/flora/datasets/<dataset_id>/complete")
    def seal(dataset_id):
        try:
            root, manifest = owned_upload(state, dataset_id, g.owner)
            with (root / "upload.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                for name, expected in manifest["files"].items():
                    path = root / "source" / name
                    receipt = root / "receipts" / hashlib.sha256(name.encode()).hexdigest()
                    if not receipt.exists() or not path.is_file() or path.stat().st_size != expected["size"]:
                        raise ValueError("Dataset upload is incomplete")
                (root / "sealed").touch()
        except ValueError as error:
            return jsonify(error=str(error)), 409
        return jsonify(id=dataset_id, sealed=True)

    @app.post("/v1/flora/dataset-jobs")
    def submit_dataset():
        from services.flora_api.store import QueueFullError

        body = request.get_json()
        if not isinstance(body, dict) or not isinstance(body.get("dataset_id"), str):
            return jsonify(error="dataset_id is required"), 400
        try:
            root = resolve_dataset(
                state, body["dataset_id"], g.owner, json.loads(keys.read_text()), catalog()
            )
            info = json.loads((root / "meta/info.json").read_text())
            if info.get("codebase_version") not in {"v2.1", "v3.0"}:
                raise ValueError("Only LeRobot v2.1/v3.0 is supported")
            total = info.get("total_episodes")
            if type(total) is not int or not 1 <= total <= 100000:
                raise ValueError("Invalid episode count")
            episodes = body.get("episodes", "all")
            if episodes == "all":
                episodes = list(range(total))
            if (
                not isinstance(episodes, list)
                or not episodes
                or any(type(i) is not int or not 0 <= i < total for i in episodes)
                or len(set(episodes)) != len(episodes)
            ):
                raise ValueError("episodes must be 'all' or unique episode indices")
            idempotency = request.headers.get("Idempotency-Key")
            if not idempotency or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", idempotency):
                raise ValueError("Idempotency-Key is required")
            parent, created = store.reserve(g.owner, idempotency, "dataset")
            if not created:
                return jsonify(public(store.get(parent["id"]))), 200
            directory = state / "jobs" / parent["id"]
            directory.mkdir(parents=True)
            (directory / "input.json").write_text(
                json.dumps(
                    {
                        "dataset_id": body["dataset_id"],
                        "source": str(root),
                    }
                )
            )
            store.add_children(parent, sorted(episodes))
        except QueueFullError:
            return jsonify(error="Queue full"), 429
        except (ValueError, OSError) as error:
            return jsonify(error=str(error) if isinstance(error, ValueError) else "Dataset unavailable"), 400
        return jsonify(public(store.get(parent["id"]))), 202

    @app.get("/v1/flora/jobs/<job_id>/episodes")
    def children(job_id):
        job = store.get(job_id, g.owner)
        if not job or job["kind"] != "dataset":
            return jsonify(error="Not found"), 404
        try:
            after = int(request.args.get("after", -1))
        except ValueError:
            return jsonify(error="Invalid after cursor"), 400
        jobs = store.children(job_id, after)
        return jsonify(jobs=[public(j) for j in jobs], next_after=jobs[-1]["episode_index"] if jobs else None)
