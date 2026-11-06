"""Authenticated Text memory preparation, Jobs, revisioned Review and data-disk exports."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import uuid
from functools import wraps

from flask import abort, jsonify, render_template, request, send_file

from lerobot.data_platform.artifacts.mem_export import digest as file_digest
from lerobot.data_platform.memory.batch import episode_run
from lerobot.data_platform.memory.contracts import AnnotationConfig, digest
from lerobot.data_platform.memory.review import MemoryReviewStore
from lerobot.data_platform.memory.sources import select_pilot
from lerobot.data_platform.processing.service import PipelineManager
from lerobot.data_platform.routes.control_plane import _current_user
from lerobot.data_platform.runtime.jobs import JobConflictError
from lerobot.data_platform.runtime.supervisor import atomic_json


def register_memory_routes(app, ctx):
    control = ctx.control_plane_store
    if control is None:
        return
    ledger = ctx.lifecycle_store() if callable(ctx.lifecycle_store) else ctx.lifecycle_store
    manager = PipelineManager(control, ledger)
    reviews = MemoryReviewStore(ledger.repository)
    root = ledger.root / "operator-results"

    def record_audit(location_id, data, status, request_id=None):
        batch = data.get("audit_batch")
        episode = data.get("episode_index")
        if (
            not isinstance(batch, str)
            or not re.fullmatch(r"[a-f0-9-]{36}", batch)
            or type(episode) is not int
        ):
            return
        ledger.repository.put(
            "memory_audits",
            digest([location_id, batch, episode]),
            {
                "location_id": location_id,
                "batch": batch,
                "episode_index": episode,
                "status": status,
                "request_id": request_id,
                "owner": _current_user()["user_id"],
            },
        )

    def checked(handler):
        @wraps(handler)
        def wrapped(location_id, *args, **kwargs):
            actor = _current_user()
            if not actor:
                return jsonify(error="Authentication required"), 401
            if request.content_length and request.content_length > 1024 * 1024:
                return jsonify(error="Request too large"), 413
            try:
                with control.sessions() as session:
                    manager._location(
                        session, location_id, actor["user_id"], write=request.method not in {"GET", "HEAD"}
                    )
                response = handler(location_id, *args, **kwargs)
                if request.method not in {"GET", "HEAD"}:
                    with control.sessions.begin() as session:
                        manager._audit(
                            session, actor["user_id"], "memory." + handler.__name__, location_id=location_id
                        )
                return response
            except KeyError:
                return jsonify(error="Memory record not found"), 404
            except PermissionError as exc:
                return jsonify(error=str(exc)), 403
            except JobConflictError as exc:
                return jsonify(error=str(exc)), 409
            except TimeoutError as exc:
                return jsonify(error=str(exc)), 503
            except (ValueError, TypeError) as exc:
                if handler.__name__ == "prepare":
                    data = request.get_json(silent=True)
                    if isinstance(data, dict):
                        record_audit(location_id, data, "ineligible")
                return jsonify(error=str(exc)), 400

        wrapped.__name__ = "memory_" + handler.__name__
        return wrapped

    def body(allowed):
        value = request.get_json(silent=True)
        if not isinstance(value, dict) or set(value) - set(allowed):
            raise ValueError("Invalid request fields")
        return value

    def readiness(location_id, operation="memory.annotate"):
        with control.sessions() as session:
            location, node = manager._location(session, location_id, _current_user()["user_id"], write=False)
        caps = node.get("capabilities") or {}
        if node["status"] != "online":
            raise JobConflictError("Dataset Agent is offline")
        if (
            caps.get("memory_annotation_protocol") != 1
            or caps.get("job_protocol") != 2
            or operation not in caps.get("operations", [])
        ):
            raise JobConflictError("Upgrade the Agent to memory_annotation_protocol=1")
        if operation == "memory.annotate" and not caps.get("memory_qwen_configured"):
            raise JobConflictError("Configure Qwen on the dataset Agent")
        return location, node

    def stage_request(location_id, value, artifacts):
        request_id = str(uuid.uuid4())
        value = {
            **value,
            "request_id": request_id,
            "location_id": location_id,
            "dataset_key": control.get_location(location_id)["dataset_key"],
        }
        folder = root / request_id / "static"
        folder.mkdir(parents=True)
        atomic_json(folder / "input.json", value)
        for name, artifact in artifacts.items():
            if isinstance(artifact, bytes):
                (folder / name).write_bytes(artifact)
                (folder / name).chmod(0o600)
            else:
                atomic_json(folder / name, artifact)
        files = {
            name: {
                "storage": "operator",
                "run": request_id,
                "path": "static/" + name,
                "sha256": file_digest(folder / name),
                "size": (folder / name).stat().st_size,
            }
            for name in ["input.json", *artifacts]
        }
        ledger.repository.put(
            "memory_requests",
            request_id,
            {
                "location_id": location_id,
                "owner": _current_user()["user_id"],
                "value": value,
                "input_files": files,
            },
            immutable=True,
        )
        return request_id, value

    def prepared(location_id, request_id):
        value = ledger.repository.get("memory_requests", request_id)
        if not value or value["location_id"] != location_id:
            raise KeyError(request_id)
        return value

    def review_folder(location_id, run):
        child = ledger.repository.get("memory_batch_episodes", run)
        job = control.get_job(child["job_id"] if child else run)
        if (
            job["location_id"] != location_id
            or job["operation"] != "memory.annotate"
            or job["status"] != "done"
        ):
            raise KeyError(run)
        if child and child["location_id"] != location_id:
            raise KeyError(run)
        return root / job["job_id"] / "static" / "episodes" / run if child else root / run / "static"

    def current_review(location_id, run):
        folder = review_folder(location_id, run)
        try:
            return reviews.get(run)
        except KeyError:
            if not (folder / "review.json").is_file():
                raise KeyError(run) from None
            state = json.loads((folder / "review.json").read_text())
            return reviews.create(run, location_id, state)

    def enqueue(location_id, request_id):
        prepared_record = prepared(location_id, request_id)
        if prepared_record["owner"] != _current_user()["user_id"]:
            raise PermissionError("Only the preparing user may submit this request")
        value = prepared_record["value"]
        operation = value["operation"]
        location, node = readiness(location_id, operation)
        if operation == "memory.annotate":
            if "episodes" in value and node["capabilities"].get("memory_batch_protocol") != 1:
                raise JobConflictError("Upgrade the Agent to memory_batch_protocol=1")
            for item in value.get("episodes", [value]):
                if node["capabilities"]["memory_model"] != item["pin"]["config"]["model"]:
                    raise JobConflictError("Configured model changed; prepare a new request")
        for revision in value.get("revisions", []):
            if current_review(location_id, revision["run"])["revision"] != revision["revision"]:
                raise JobConflictError("Review changed after preparation")
        options = {
            "parameters": {"request_id": request_id},
            "target": {
                "dataset_key": location["dataset_key"],
                "location_id": location_id,
                "dataset_version_id": None,
            },
            "input_files": prepared_record["input_files"],
            "operator_snapshot": {
                "operator_id": operation,
                "version": "1.0.0",
                "release": os.environ.get("DATA_PLATFORM_RELEASE", "legacy"),
                "data_format": manager._info(location).get("codebase_version", "unknown"),
            },
        }
        delivery = request.headers.get("Idempotency-Key", "")
        if not delivery or len(delivery) > 128:
            raise ValueError("An Idempotency-Key of at most 128 characters is required")
        job = control.create_job(
            location_id=location_id,
            requested_by=_current_user()["user_id"],
            operation=operation,
            options=options,
            idempotency_key=digest(
                [
                    _current_user()["user_id"],
                    location_id,
                    request_id,
                    "dataset-batch" if "episodes" in value else delivery,
                ]
            ),
        )
        for item in value.get("episodes", []):
            ledger.repository.put(
                "memory_batch_episodes",
                episode_run(request_id, item["request_id"]),
                {"job_id": job["job_id"], "location_id": location_id},
                immutable=True,
            )
        return jsonify(job=control.job_manager.decorate(job, _current_user())), 202

    base = "/api/control/locations/<location_id>/memory"

    @app.get("/remote/<location_id>/memory")
    @checked
    def page(location_id):
        snapshot = ledger.root / "memory-training-reviews" / location_id / "snapshot.json"
        return render_template(
            "memory.html",
            location_id=location_id,
            can_edit=_current_user()["role"] != "viewer",
            training_review_available=snapshot.is_file() and not snapshot.is_symlink(),
        )

    def training_snapshot(location_id):
        path = ledger.root / "memory-training-reviews" / location_id / "snapshot.json"
        if path.is_symlink() or not path.is_file():
            raise KeyError(location_id)
        value = json.loads(path.read_text())
        if (
            value.get("schema") != "memory_training_review/v1"
            or value.get("dataset_key") != control.get_location(location_id)["dataset_key"]
        ):
            raise ValueError("Training review does not match this dataset")
        return value

    @app.get("/remote/<location_id>/memory/training-review")
    @checked
    def training_review_page(location_id):
        training_snapshot(location_id)
        return render_template("memory_training_review.html", location_id=location_id)

    @app.get(base + "/training-review")
    @checked
    def training_review_index(location_id):
        snapshot = training_snapshot(location_id)
        response = jsonify(
            **{k: v for k, v in snapshot.items() if k != "episodes"},
            episodes=[
                {k: v for k, v in item.items() if k not in {"checkpoints", "facts", "issues"}}
                for item in snapshot["episodes"]
            ],
        )
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get(base + "/training-review/<int:episode_index>")
    @checked
    def training_review_episode(location_id, episode_index):
        snapshot = training_snapshot(location_id)
        item = next((e for e in snapshot["episodes"] if e["episode_index"] == episode_index), None)
        if item is None:
            raise KeyError(episode_index)
        artifact = app.extensions["data_platform_memory_caption"](location_id, item["caption_run"])
        raw = artifact.pop("_memory_raw_bytes", None)
        if (
            not raw
            or hashlib.sha256(raw).hexdigest() != item["annotation_sha256"]
            or artifact.get("source_fingerprint") != snapshot["source_fingerprint"]
            or artifact.get("episode_index") != episode_index
            or artifact.get("frame_count") != item["frame_count"]
            or artifact.get("fps") != item["fps"]
        ):
            raise JobConflictError("Source video annotation does not match the frozen training input")
        cameras = app.extensions["data_platform_memory_video_cameras"](location_id, item["caption_run"])
        response = jsonify(**item, video_cameras=cameras, source_fingerprint=snapshot["source_fingerprint"])
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get(base)
    @checked
    def status(location_id):
        reason = None
        model = None
        parallel = 1
        try:
            _, node = readiness(location_id)
            model = node["capabilities"]["memory_model"]
            parallel = node["capabilities"].get("memory_batch_max_parallel", 1)
            if node["capabilities"].get("memory_batch_protocol") != 1:
                reason = "Upgrade the Agent to memory_batch_protocol=1 for dataset batches"
        except JobConflictError as exc:
            reason = str(exc)
        return jsonify(
            can_submit=reason is None and _current_user()["role"] != "viewer",
            unavailable_reason=reason,
            model=model,
            max_parallel_episodes=parallel,
            total_episodes=(control.get_location(location_id).get("metadata") or {}).get("total_episodes"),
            runs=[
                {k: j[k] for k in ("job_id", "operation", "status", "created_at", "result")}
                for j in control.list_jobs(limit=500, filters={"location_id": location_id})
                if j["location_id"] == location_id and j["operation"].startswith("memory.")
            ],
        )

    @app.post(base + "/prepare")
    @checked
    def prepare(location_id):
        data = body({"episode_index", "flora_run", "task", "lineage_group", "audit_batch"})
        location, node = readiness(location_id)
        index = data.get("episode_index")
        if type(index) is not int or index < 0:
            raise ValueError("Select an episode")
        run = data.get("flora_run", "")
        if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run):
            raise ValueError("Select a FLORA run")
        if not isinstance(data.get("task"), str) or not data["task"].strip() or len(data["task"]) > 4096:
            raise ValueError("An explicit total task is required")
        artifact = app.extensions["data_platform_memory_caption"](location_id, run)
        artifact_bytes = artifact.pop("_memory_raw_bytes", None) or json.dumps(artifact).encode()
        if artifact["episode_index"] != index:
            raise ValueError("FLORA run belongs to another episode")
        from lerobot.data_platform.operators.implementations.caption.flora import camera_roles

        info = location.get("metadata") or {}
        roles = camera_roles([k for k, v in info["features"].items() if v.get("dtype") in {"image", "video"}])
        if set(roles) != {"main", "left", "right"}:
            raise ValueError("Three camera views are required")
        selection = {
            "episode_index": index,
            "flora_run": run,
            "cameras": [roles[k] for k in ("main", "left", "right")],
            "task": data["task"].strip(),
            "lineage_group": data.get("lineage_group") or None,
        }
        config = AnnotationConfig(node["capabilities"]["memory_model"])
        reply = ctx.caption_read_relay.request(
            node["node_id"],
            {
                "action": "memory-preview",
                "root": location["root"],
                "dataset_key": location["dataset_key"],
                "selection": selection,
                "artifact_base64": base64.b64encode(artifact_bytes).decode(),
                "config": config.to_dict(),
            },
            timeout=210,
        )
        if reply.status != 200 or reply.binary:
            raise ValueError(
                "Source/FLORA preflight failed on Agent; inspect the source binding and coverage"
            )
        pin = json.loads(reply.body)
        request_id, value = stage_request(
            location_id, {"operation": "memory.annotate", "pin": pin}, {"flora.json": artifact_bytes}
        )
        record_audit(location_id, data, "eligible", request_id)
        return jsonify(request_id=request_id, pin=value["pin"])

    @app.post(base + "/jobs")
    @checked
    def submit(location_id):
        return enqueue(location_id, body({"request_id"})["request_id"])

    @app.post(base + "/batch")
    @checked
    def prepare_batch(location_id):
        data = body({"request_ids", "pilot_id"})
        _, node = readiness(location_id)
        if node["capabilities"].get("memory_batch_protocol") != 1:
            raise JobConflictError("Upgrade the Agent to memory_batch_protocol=1")
        pilot = ledger.repository.get("memory_pilots", data.get("pilot_id"))
        if not pilot or pilot["location_id"] != location_id:
            raise KeyError(data.get("pilot_id"))
        ids = data.get("request_ids")
        if not isinstance(ids, list) or not ids or len(ids) > 10000 or len(set(ids)) != len(ids):
            raise ValueError("Select distinct prepared episode requests")
        records = [prepared(location_id, key) for key in ids]
        if any(record["owner"] != _current_user()["user_id"] for record in records):
            raise PermissionError("Only the preparing user may batch episode requests")
        values = [record["value"] for record in records]
        if any(v["operation"] != "memory.annotate" or "pin" not in v for v in values):
            raise ValueError("A batch requires prepared episode annotations")
        if sorted(digest(v["pin"]) for v in values) != sorted(digest(p) for p in pilot["selected"]):
            raise ValueError("Batch must match exactly the frozen pilot")
        artifacts = {
            f"flora-{i}.json": (root / v["request_id"] / "static/flora.json").read_bytes()
            for i, v in enumerate(values)
        }
        request_id, _ = stage_request(
            location_id,
            {
                "operation": "memory.annotate",
                "episodes": [{"request_id": v["request_id"], "pin": v["pin"]} for v in values],
                "pilot_id": pilot["digest"],
            },
            artifacts,
        )
        return jsonify(
            request_id=request_id,
            episodes=len(values),
            max_parallel_episodes=node["capabilities"].get("memory_batch_max_parallel", 1),
        )

    @app.get(base + "/reviews/<run>")
    @checked
    def read_review(location_id, run):
        value = current_review(location_id, run)
        return jsonify(value)

    @app.post(base + "/reviews/<run>/<action>")
    @checked
    def change_review(location_id, run, action):
        data = body(
            {
                "expected_revision",
                "frame",
                "facts",
                "checkpoint",
                "reason",
                "source_confirmed",
                "goal_confirmed",
                "causal_evidence_confirmed",
                "independent",
                "adjudicated",
            }
        )
        current = current_review(location_id, run)
        if action == "regenerate":
            if data.get("expected_revision") != current["revision"]:
                raise JobConflictError("Review changed; reload")
            original = (review_folder(location_id, run) / "flora.json").read_bytes()
            state = current["state"]
            request_id, _ = stage_request(
                location_id,
                {
                    "operation": "memory.annotate",
                    "episodes": [{"request_id": str(uuid.uuid4()), "pin": state["pin"], "previous": state}],
                    "revisions": [{"run": run, "revision": current["revision"]}],
                },
                {"flora-0.json": original},
            )
            return jsonify(request_id=request_id)
        return jsonify(
            reviews.change(run, data.get("expected_revision"), _current_user()["user_id"], action, data)
        )

    @app.get(base + "/reviews/<run>/frames/<int:frame>.png")
    @checked
    def frame_image(location_id, run, frame):
        value = current_review(location_id, run)
        cutoff = request.args.get("cutoff", type=int)
        entry = next((e for e in value["state"]["entries"] if e["frame"] == cutoff), None)
        if not entry or frame not in {cutoff, *entry["history_frames"]} or frame > cutoff:
            abort(404)
        response = send_file(review_folder(location_id, run) / "frames" / f"{frame}.png")
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get(base + "/reviews/<run>/flora")
    @checked
    def flora_evidence(location_id, run):
        current_review(location_id, run)
        cutoff = request.args.get("cutoff", type=int)
        if cutoff is None:
            raise ValueError("A review cutoff is required")
        artifact = json.loads((review_folder(location_id, run) / "flora.json").read_text())
        # Caption spans ending in the future are excluded even in the auxiliary panel.
        return jsonify(
            label="FLORA auxiliary candidates, not confirmed facts",
            segments=[s for s in artifact["result"]["segments"] if s["end_frame"] - 1 <= cutoff],
        )

    @app.post(base + "/pilot")
    @checked
    def freeze_pilot(location_id):
        data = body({"request_ids", "seed", "audit_batch"})
        ids = data.get("request_ids")
        if not isinstance(ids, list) or len(ids) > 10000:
            raise ValueError("Supply all audited eligible episode requests")
        total = (control.get_location(location_id).get("metadata") or {}).get("total_episodes")
        if type(total) is not int or total <= 0:
            raise ValueError("Refresh the registered episode inventory before pilot selection")
        audit = [
            ledger.repository.get("memory_audits", digest([location_id, data.get("audit_batch"), episode]))
            for episode in range(total)
        ]
        if any(not row or row["owner"] != _current_user()["user_id"] for row in audit):
            raise ValueError("Audit every episode in one batch before freezing the pilot")
        if set(ids) != {row["request_id"] for row in audit if row["status"] == "eligible"}:
            raise ValueError("Pilot selection must include all eligible audited episodes")
        pins = [prepared(location_id, key)["value"]["pin"] for key in ids]
        pilot = select_pilot(pins, seed=data.get("seed", 42))
        pilot["location_id"] = location_id
        pilot["audit"] = audit
        pilot["digest"] = digest({k: v for k, v in pilot.items() if k != "digest"})
        ledger.repository.put("memory_pilots", pilot["digest"], pilot, immutable=True)
        return jsonify(pilot)

    @app.post(base + "/export")
    @checked
    def prepare_export(location_id):
        data = body({"runs", "pilot_id"})
        readiness(location_id, "memory.export")
        pilot = ledger.repository.get("memory_pilots", data["pilot_id"])
        if not pilot or pilot["location_id"] != location_id:
            raise KeyError(data["pilot_id"])
        runs = data.get("runs")
        if not isinstance(runs, list) or not 1 <= len(runs) <= 10 or len(set(runs)) != len(runs):
            raise ValueError("Select distinct reviewed episodes")
        values = [current_review(location_id, run) for run in runs]
        pins = {str(p["episode_index"]): p for p in pilot["selected"]}
        if {str(v["state"]["pin"]["episode_index"]) for v in values} != set(pins):
            raise ValueError("Export must cover exactly the frozen pilot episodes")
        for value in values:
            state = value["state"]
            pin = state["pin"]
            original = pins[str(pin["episode_index"])]
            if any(pin[k] != original[k] for k in original if k not in {"checkpoints", "expected_requests"}):
                raise ValueError("Episode source/config differs from the frozen pilot")
            review = state["record"]["review"]
            if review.get("status") != "approved":
                raise ValueError("All pilot episodes require human approval")
            if pilot["assignments"][str(pin["episode_index"])] == "test" and (
                not review.get("adjudicated")
                or not review.get("second_reviewer")
                or review["second_reviewer"] == review["reviewer"]
            ):
                raise ValueError("Test episodes require independent review and adjudication")
        artifacts = {
            f"flora-{i}.json": (review_folder(location_id, value["run"]) / "flora.json").read_bytes()
            for i, value in enumerate(values)
        }
        request_id, _ = stage_request(
            location_id,
            {
                "operation": "memory.export",
                "states": [v["state"] for v in values],
                "revisions": [{"run": v["run"], "revision": v["revision"]} for v in values],
                "assignments": pilot["assignments"],
                "seed": pilot["seed"],
            },
            artifacts,
        )
        return jsonify(request_id=request_id)
