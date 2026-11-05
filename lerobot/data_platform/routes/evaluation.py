"""Dataset-scoped human annotation workbench and reproducible evaluation exports."""

from __future__ import annotations

import json
import re
from functools import wraps

from flask import jsonify, render_template, request, send_file

from lerobot.data_platform.camera_layout import camera_rank
from lerobot.data_platform.evaluation.metrics import edges, evaluate
from lerobot.data_platform.evaluation.review import EvaluationStore, digest, draft
from lerobot.data_platform.job_management import JobConflictError
from lerobot.data_platform.processing.service import PipelineManager
from lerobot.data_platform.routes.control_plane import _current_user
from lerobot.data_platform.temporal_caption import artifact_key


def register_evaluation_routes(app, ctx, resolve, read_artifact):
    if not ctx.control_plane_store or not getattr(ctx, "lifecycle_store", None):
        return
    ledger = ctx.lifecycle_store() if callable(ctx.lifecycle_store) else ctx.lifecycle_store
    reviews = EvaluationStore(ledger.repository)
    manager = PipelineManager(ctx.control_plane_store, ledger)

    def checked(fn):
        @wraps(fn)
        def wrapped(location_id, *args, **kwargs):
            actor = _current_user()
            if not actor:
                return jsonify(error="Authentication required"), 401
            if request.content_length and request.content_length > 2 * 1024 * 1024:
                return jsonify(error="Annotation payload too large"), 413
            try:
                with ctx.control_plane_store.sessions() as session:
                    manager._location(
                        session, location_id, actor["user_id"], write=request.method not in ("GET", "HEAD")
                    )
                result = fn(location_id, *args, **kwargs)
                if request.method not in ("GET", "HEAD"):
                    with ctx.control_plane_store.sessions.begin() as session:
                        manager._audit(
                            session, actor["user_id"], "evaluation." + fn.__name__, location_id=location_id
                        )
                return result
            except KeyError:
                return jsonify(error="Evaluation record not found"), 404
            except PermissionError as exc:
                return jsonify(error=str(exc)), 403
            except JobConflictError as exc:
                return jsonify(error=str(exc)), 409
            except (ValueError, TypeError) as exc:
                return jsonify(error=str(exc)), 400

        wrapped.__name__ = "evaluation_" + fn.__name__
        return wrapped

    def candidates(location_id):
        key, _, root, _, _ = resolve(location_id=location_id)
        latest = {}
        for folder in sorted(root.glob("*")):
            if (
                not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", folder.name)
                or not (folder / "caption.json").is_file()
            ):
                continue
            artifact = read_artifact(root, folder.name, "caption", key)
            if artifact.get("scheme_id") != "flora":
                continue
            metadata = (
                json.loads((folder / "episode.json").read_text())
                if (folder / "episode.json").is_file()
                else {}
            )
            ep = artifact["episode_index"]
            if ep not in latest or artifact.get("created_at", 0) > latest[ep][1].get("created_at", 0):
                latest[ep] = (folder.name, artifact, metadata)
        return latest

    def present(record):
        value = {**record, "reference_sha256": digest(record["reference"])}
        value["pairs"] = [
            {"key": f"{i}:{j}", "prediction": i, "reference": j}
            for i, j in edges(record["prediction"]["segments"], record["reference"]["segments"])
        ]
        return value

    @app.get("/remote/<location_id>/temporal-caption/evaluation")
    @checked
    def page(location_id):
        _, name, _, _, _ = resolve(location_id=location_id)
        return render_template(
            "evaluation_review.html",
            page_title=name,
            location_id=location_id,
            read_only=_current_user()["role"] == "viewer",
        )

    @app.get("/api/control/locations/<location_id>/evaluation")
    @checked
    def overview(location_id):
        pilot = reviews.pilot(location_id)
        if not pilot:
            return jsonify(pilot=None, episodes=[])
        rows = [reviews.get(location_id, ep) for ep in pilot["episodes"]]
        return jsonify(
            pilot=pilot,
            episodes=[
                {k: r[k] for k in ("episode_index", "instruction", "status", "revision")} for r in rows
            ],
        )

    @app.post("/api/control/locations/<location_id>/evaluation/pilot")
    @checked
    def create_pilot(location_id):
        body = request.get_json(silent=True) or {}
        if set(body) - {"episodes"}:
            raise ValueError("Unknown pilot fields")
        available = candidates(location_id)
        selected = body.get("episodes")
        if selected is None:
            # Deterministic task diversity, then FPS/camera diversity and long/short videos.
            remaining, selected, seen = set(available), [], set()
            while remaining and len(selected) < 10:

                def rank(ep):
                    _, a, m = available[ep]
                    features = {
                        ("task", str(m.get("task_hint", []))),
                        ("fps", a["fps"]),
                        ("cameras", len(m.get("cameras", []))),
                        ("duration", int(a["duration_s"] // 30)),
                    }
                    return len(features - seen), -ep

                ep = max(remaining, key=rank)
                _, a, m = available[ep]
                seen.update(
                    {
                        ("task", str(m.get("task_hint", []))),
                        ("fps", a["fps"]),
                        ("cameras", len(m.get("cameras", []))),
                        ("duration", int(a["duration_s"] // 30)),
                    }
                )
                selected.append(ep)
                remaining.remove(ep)
        if not isinstance(selected, list) or any(
            type(ep) is not int or ep not in available for ep in selected
        ):
            raise ValueError("Select episodes with completed FLORA results")
        seeds = []
        for ep in selected:
            run, artifact, metadata = available[ep]
            seed = draft(artifact, metadata)
            seed["source_run"] = run
            key, _, root, _, _ = resolve(location_id=location_id)
            media = ledger.root / "evaluation-media" / artifact_key(location_id) / str(ep)
            seed["cameras"] = [
                str(i)
                for i in range(len(metadata.get("cameras", [])))
                if (media / f"{i}.mp4").is_file() or (root / run / "camera-videos" / f"{i}.mp4").is_file()
            ]
            if not seed["cameras"]:
                raise ValueError(f"Episode {ep}: prepare original camera media before starting the pilot")
            seed["camera_names"] = metadata.get("cameras", [])
            seed["cameras"].sort(key=lambda i: camera_rank(seed["camera_names"][int(i)]))
            source = media / "source.json"
            seed["provenance"] = json.loads(source.read_text()) if source.is_file() else {}
            seeds.append(seed)
        return jsonify(pilot=reviews.create(location_id, seeds, _current_user()["user_id"])), 201

    @app.get("/api/control/locations/<location_id>/evaluation/episodes/<int:episode>")
    @checked
    def episode(location_id, episode):
        revision = request.args.get("revision", type=int)
        return jsonify(present(reviews.get(location_id, episode, revision)))

    @app.post("/api/control/locations/<location_id>/evaluation/episodes/<int:episode>")
    @checked
    def change(location_id, episode):
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) - {
            "revision",
            "action",
            "reference",
            "confirmed",
            "pair",
            "assessment",
            "oracle",
        }:
            raise ValueError("Invalid annotation request")
        value = reviews.change(
            location_id, episode, body.get("revision"), _current_user()["user_id"], body.get("action"), body
        )
        return jsonify(present(value))

    @app.get("/api/control/locations/<location_id>/evaluation/episodes/<int:episode>/video/<camera>")
    @checked
    def video(location_id, episode, camera):
        record = reviews.get(location_id, episode)
        _, _, root, _, _ = resolve(location_id=location_id)
        folder = root / record["source_run"]
        if camera not in record["cameras"]:
            return jsonify(error="Camera not available"), 404
        path = folder / "camera-videos" / f"{camera}.mp4"
        media = ledger.root / "evaluation-media" / artifact_key(location_id) / str(episode)
        if not path.is_file():
            folder, path = media, media / f"{camera}.mp4"
        if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file():
            return jsonify(error="Camera video unavailable"), 404
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.get("/api/control/locations/<location_id>/evaluation/metrics")
    @checked
    def metrics(location_id):
        pilot = reviews.pilot(location_id)
        rows = [reviews.get(location_id, ep) for ep in pilot["episodes"]] if pilot else []
        response = jsonify(evaluate(rows))
        response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.get("/api/control/locations/<location_id>/evaluation/export")
    @checked
    def export(location_id):
        pilot = reviews.pilot(location_id)
        rows = [reviews.get(location_id, ep) for ep in pilot["episodes"]] if pilot else []
        response = jsonify(pilot=pilot, references=rows, metrics=evaluate(rows))
        response.headers["Content-Disposition"] = 'attachment; filename="flora-evaluation-pilot.json"'
        response.headers["Cache-Control"] = "private, no-store"
        return response
