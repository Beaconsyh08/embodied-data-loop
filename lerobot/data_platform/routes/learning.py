"""Session authenticated learning controls; deliberately absent from model tools and MCP."""

from functools import wraps

from flask import jsonify, render_template, request

from lerobot.data_platform.ai_tools.learning import cases
from lerobot.data_platform.ai_tools.learning.service import Learning
from lerobot.data_platform.job_management import JobConflictError
from lerobot.data_platform.routes.control_plane import _current_user


def register_learning_routes(app, ctx):
    def checked(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            actor = _current_user()
            if not actor:
                return jsonify(error="Authentication required"), 401
            if request.method == "POST" and not request.content_length:
                return jsonify(error="Content-Length required"), 411
            if request.content_length and request.content_length > 16384:
                return jsonify(error="Learning request too large"), 413
            learning = Learning(app.extensions["data_platform_assistant"])
            try:
                return function(learning, actor["user_id"], *args, **kwargs)
            except PermissionError:
                return jsonify(error="Learning access denied"), 403
            except KeyError:
                return jsonify(error="Owned learning resource not found"), 404
            except JobConflictError as exc:
                return jsonify(error=str(exc)), 409
            except (ValueError, TypeError):
                return jsonify(error="Invalid learning request"), 400

        return wrapped

    @app.get("/learning")
    @checked
    def page(learning, owner):
        learning.state(owner)
        return render_template("data_platform_learning.html", admin=False)

    @app.get("/admin/learning")
    @checked
    def admin_page(learning, owner):
        learning.dashboard(owner)
        return render_template("data_platform_learning.html", admin=True)

    @app.get("/api/ai-assistant/learning")
    @checked
    def state(learning, owner):
        return jsonify(learning.state(owner))

    @app.get("/api/ai-assistant/learning/<resource>")
    @checked
    def read(learning, owner, resource):
        if resource == "feedback":
            return jsonify(
                learning.feedback_state(
                    owner, {"kind": request.args.get("kind"), "id": request.args.get("id")}
                )
            )
        if resource == "cases":
            return jsonify(cases.list_cases(learning, owner))
        method = {"materials": learning.materials, "admin": learning.dashboard}.get(resource)
        if not method:
            raise KeyError("Unknown learning resource")
        return jsonify(method(owner))

    @app.post("/api/ai-assistant/learning/<action>")
    @checked
    def write(learning, owner, action):
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        method = {
            "settings": learning.settings,
            "events": learning.event,
            "feedback": learning.feedback,
            "revoke": learning.revoke,
            "accept-preference": learning.accept_preference,
            "configure": learning.configure,
            "backfill": learning.backfill,
        }.get(action)
        if method:
            return jsonify(method(owner, value))
        if action in {"propose-cases", "review-case"}:
            return jsonify(
                (cases.propose if action == "propose-cases" else cases.review)(learning, owner, value)
            )
        raise KeyError("Unknown learning action")
