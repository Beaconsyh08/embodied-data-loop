"""Authenticated upload/status/result API. GPU work runs in a separate service."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import threading
from pathlib import Path

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from services.flora_api import __version__
from services.flora_api.store import QueueFullError, Store


def create_app(state: Path | None = None, keys: Path | None = None):
    state = state or Path(os.environ["FLORA_API_STATE"])
    keys = keys or Path(os.environ["FLORA_API_KEYS"])
    store = Store(state)
    app = Flask(__name__)
    app.config.update(MAX_FORM_MEMORY_SIZE=1024 * 1024, MAX_FORM_PARTS=8)

    @app.before_request
    def authenticate():
        if request.path == "/healthz" and request.method == "GET":
            return None
        if request.method in {"POST", "PUT"}:
            if request.content_length is None:
                return jsonify(error="Content-Length is required"), 413
            if (
                request.method == "POST"
                and request.path != "/v1/flora/jobs"
                and request.content_length > 32 * 1024**2
            ):
                return jsonify(error="Request body too large"), 413
        authorization = request.headers.get("Authorization", "")
        token = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        if not 32 <= len(token) <= 256:
            return jsonify(error="API key required"), 401
        digest = hashlib.sha256(token.encode()).hexdigest()
        records = json.loads(keys.read_text())
        for record in records:
            if record.get("enabled", True) and hmac.compare_digest(record["sha256"], digest):
                g.owner = record["id"]
                return None
        return jsonify(error="Invalid API key"), 401

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.name), error.code

    @app.get("/healthz")
    def health():
        heartbeat = state / "worker-heartbeat"
        import time

        ready = heartbeat.is_file() and time.time() - heartbeat.stat().st_mtime < 90
        return jsonify(service="flora-api", version=__version__, worker_ready=ready), 200 if ready else 503

    def public(job):
        return {
            key: job[key]
            for key in (
                "id",
                "kind",
                "parent_id",
                "episode_index",
                "status",
                "stage",
                "error",
                "created_at",
                "updated_at",
                "counts",
            )
            if key in job
        }

    @app.post("/v1/flora/jobs")
    def submit():
        idempotency = request.headers.get("Idempotency-Key")
        if idempotency is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", idempotency):
            return jsonify(error="Invalid Idempotency-Key"), 400
        if shutil.disk_usage(state).free < 100 * 1024**3:
            return jsonify(error="Storage capacity temporarily unavailable"), 503
        try:
            job, created = store.reserve(g.owner, idempotency)
        except QueueFullError:
            return jsonify(error="Queue full; retry later"), 429, {"Retry-After": "60"}
        if not created:
            return jsonify(public(job)), 200
        directory = state / "jobs" / job["id"]
        stopped = threading.Event()

        def keep_upload_alive():
            while not stopped.wait(60):
                store.touch_upload(job["id"])

        heartbeat = threading.Thread(target=keep_upload_alive, daemon=True)
        heartbeat.start()
        try:
            directory.mkdir(parents=True)
            instruction = request.form.get("instruction", "").strip()
            if not instruction or len(instruction) > 4000:
                raise ValueError("instruction must contain 1–4000 characters")
            if set(request.files) - {"left_video", "right_video", "main_video"}:
                raise ValueError("Unknown video field")
            fields = set(request.files)
            if fields not in (
                {"main_video"},
                {"left_video", "right_video"},
                {"left_video", "right_video", "main_video"},
            ):
                raise ValueError(
                    "Provide main_video alone, or both left_video and right_video with optional main_video"
                )
            roles = []
            for role in ("left", "right", "main"):
                uploads = request.files.getlist(f"{role}_video")
                if not uploads:
                    continue
                if len(uploads) != 1:
                    raise ValueError(f"Provide exactly one {role}_video")
                target = directory / f"{role}.mp4"
                uploads[0].save(target)
                with target.open("rb") as video:
                    header = video.read(12)
                if len(header) < 12 or header[4:8] != b"ftyp":
                    raise ValueError("Videos must be MP4 files")
                roles.append(role)
            (directory / "input.json").write_text(json.dumps({"instruction": instruction, "roles": roles}))
            store.update(job["id"], status="queued", stage="queued")
        except (ValueError, HTTPException) as error:
            shutil.rmtree(directory, ignore_errors=True)
            detail = str(error) if isinstance(error, ValueError) else error.name
            store.update(job["id"], status="failed", stage="upload", error=detail)
            return jsonify(error=detail, id=job["id"]), error.code if isinstance(
                error, HTTPException
            ) else 400
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            store.update(job["id"], status="failed", stage="upload", error="Upload failed")
            app.logger.exception("Upload failed for job %s", job["id"])
            return jsonify(error="Upload failed", id=job["id"]), 500
        finally:
            stopped.set()
            heartbeat.join()
        return jsonify(public(store.get(job["id"]))), 202

    @app.get("/v1/flora/jobs/<job_id>")
    def status(job_id):
        job = store.get(job_id, g.owner)
        return (jsonify(public(job)), 200) if job else (jsonify(error="Not found"), 404)

    @app.get("/v1/flora/jobs/<job_id>/result")
    def result(job_id):
        job = store.get(job_id, g.owner)
        if not job:
            return jsonify(error="Not found"), 404
        if job["kind"] == "dataset":
            return jsonify(
                {**public(job), "error": "Use /episodes and download each completed episode result"}
            ), 409
        if job["status"] != "completed":
            return jsonify(error="Result is not ready", status=job["status"]), 409
        return jsonify(json.loads((state / "jobs" / job_id / "output" / "caption.json").read_text()))

    from services.flora_api.datasets import register

    register(app, state, keys, store, public)
    return app
