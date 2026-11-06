"""Authenticated access to original laundry reports and CPU Viser sessions."""

from __future__ import annotations

import json
import threading
import time
from contextlib import suppress

from flask import Response, jsonify, render_template, request

from lerobot.data_platform.datasets.curation import artifact_path
from lerobot.data_platform.routes.control_plane import _current_user
from lerobot.data_platform.web.umi_viewers import ViewerSessions


def register_umi_viewer_routes(app, manager, ledger, checked, owner):
    if ledger is None:
        return
    sessions = ViewerSessions(ledger.root / "umi-viewer-sessions")
    app.extensions["umi_viewer_sessions"] = sessions

    def authorize(job_id):
        job = manager.store.get_job(job_id)
        if job["requested_by"] != owner() or job["status"] != "done":
            raise KeyError(job_id)
        if job["operation"] not in {"umi.cosmos_id", "umi.replay"}:
            raise ValueError("This job has no native UMI result")
        with manager.store.sessions() as db:
            for location in {job["location_id"], *job["options"].get("source_location_ids", [])}:
                manager._location(db, location, owner(), write=False)
        return job

    def root(job_id):
        return ledger.root / "operator-results" / job_id / "static"

    def describe(session):
        data = session.describe()
        data.pop("port")
        data["view_url"] = f"/api/umi-viewer-sessions/{session.session_id}/view"
        data["socket_url"] = f"/api/umi-viewer-sessions/{session.session_id}/socket"
        if data["status"] == "error":
            data["error"] = "The replay viewer failed to start; inspect the server viewer log."
        return data

    def audit(job_id, operation, session_id):
        from lerobot.data_platform.control.store import RemoteJob

        with manager.store.sessions.begin() as db:
            manager.store.job_manager.audit(
                db,
                db.get(RemoteJob, job_id),
                operation,
                actor=_current_user(),
                details={"viewer_session": session_id},
            )

    @app.get("/umi/results/<job_id>")
    @checked
    def umi_results(job_id):
        job = authorize(job_id)
        views = []
        reports = []
        for artifact in job.get("result", {}).get("artifacts", []):
            if artifact.get("kind") == "umi_replay_viewer":
                manifest = json.loads(artifact_path(root(job_id), artifact["path"]).read_text())
                views.append({"artifact": artifact["path"], "manifest": manifest})
            elif artifact.get("kind") == "report":
                reports.append(artifact)
        return render_template("umi_results.html", job=job, views=views, reports=reports)

    @app.post("/api/umi-viewer-sessions")
    @checked
    def umi_viewer_start():
        value = request.get_json(silent=True) or {}
        job = authorize(str(value.get("job_id", "")))
        path = str(value.get("artifact", ""))
        if not any(
            a.get("path") == path and a.get("kind") == "umi_replay_viewer"
            for a in job.get("result", {}).get("artifacts", [])
        ):
            raise ValueError("Select a replay artifact from this job")
        bundle = artifact_path(root(job["job_id"]), path).parent
        manifest = json.loads((bundle / "viewer.json").read_text())
        if not manifest.get("episodes"):
            raise ValueError("No executed frames are available; see the episode failure report")
        session = sessions.start(owner(), job["job_id"], path, bundle)
        audit(job["job_id"], "umi.viewer.open", session.session_id)
        return jsonify(describe(session)), 202

    @app.route("/api/umi-viewer-sessions/<session_id>", methods=["GET", "DELETE"])
    @checked
    def umi_viewer_session(session_id):
        session = sessions.get(session_id, owner())
        authorize(session.job_id)
        if request.method == "DELETE":
            sessions.remove(session_id, owner())
            audit(session.job_id, "umi.viewer.close", session_id)
            return jsonify(status="closed")
        return jsonify(describe(session))

    @app.get("/api/umi-viewer-sessions/<session_id>/view")
    @checked
    def umi_viewer_page(session_id):
        import requests

        session = sessions.get(session_id, owner())
        authorize(session.job_id)
        state = session.describe()
        if state["status"] != "ready":
            return jsonify(error="Viewer is not ready"), 409
        with requests.Session() as client:
            client.trust_env = False
            response = client.get(f"http://127.0.0.1:{state['port']}/", timeout=10)
            response.raise_for_status()
        result = Response(response.content, content_type="text/html; charset=utf-8")
        result.headers["Cache-Control"] = "no-store"
        result.direct_passthrough = True
        return result

    @app.before_request
    def umi_socket_access():
        prefix = "/api/umi-viewer-sessions/"
        if not request.path.startswith(prefix) or not request.path.endswith("/socket"):
            return None
        if not _current_user():
            return jsonify(error="Authentication required"), 401
        if request.headers.get("Origin") != request.host_url.rstrip("/"):
            return jsonify(error="Invalid viewer origin"), 403
        try:
            session = sessions.get(request.path[len(prefix) :].split("/")[0], owner())
            authorize(session.job_id)
            if session.describe()["status"] != "ready":
                return jsonify(error="Viewer is not ready"), 409
        except (KeyError, PermissionError):
            return jsonify(error="Viewer not found"), 404
        return None

    # Optional in algorithm-only installations; included in every server release.
    try:
        from flask_sock import Sock
    except ImportError:
        return
    app.config["SOCK_SERVER_OPTIONS"] = {"subprotocols": ["viser-v1.1.0"], "ping_interval": 20}
    sock = Sock(app)

    @sock.route("/api/umi-viewer-sessions/<session_id>/socket")
    def umi_viewer_socket(ws, session_id):
        from lerobot.data_platform.ops.environment import session_cookie_name

        token = request.cookies.get(session_cookie_name())
        import websocket
        from simple_websocket import ConnectionClosed

        session = sessions.get(session_id, owner())
        state = session.describe()
        upstream = websocket.create_connection(
            f"ws://127.0.0.1:{state['port']}/",
            timeout=2,
            subprotocols=["viser-v1.1.0"],
            http_no_proxy=["127.0.0.1"],
        )
        stopped = threading.Event()

        def forward():
            try:
                while not stopped.is_set():
                    try:
                        data = upstream.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not data:
                        break
                    session.touched = time.monotonic()
                    ws.send(data)
            except (websocket.WebSocketException, ConnectionClosed, OSError):
                pass
            finally:
                stopped.set()
                with suppress(ConnectionClosed):
                    ws.close()

        sender = threading.Thread(target=forward, daemon=True)
        sender.start()
        last_check = time.monotonic()
        try:
            while not stopped.is_set():
                if time.monotonic() - last_check > 30:
                    if token:
                        current = manager.store.verify_session(token)
                        if current is None or current["user_id"] != session.owner:
                            raise PermissionError("Viewer login expired")
                    authorize(session.job_id)
                    last_check = time.monotonic()
                data = ws.receive(timeout=1)
                if data is not None:
                    session.touched = time.monotonic()
                    upstream.send(
                        data,
                        opcode=websocket.ABNF.OPCODE_BINARY
                        if isinstance(data, bytes)
                        else websocket.ABNF.OPCODE_TEXT,
                    )
        except (websocket.WebSocketException, ConnectionClosed, OSError, KeyError, PermissionError):
            pass
        finally:
            stopped.set()
            upstream.close()
            sender.join(timeout=3)
