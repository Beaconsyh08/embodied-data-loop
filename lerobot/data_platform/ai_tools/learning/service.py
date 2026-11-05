"""Authenticated learning settings, weak feedback and revocable account-private projections."""

import time

from sqlalchemy import func, select
from sqlalchemy.orm import load_only

from lerobot.data_platform.ai_tools.learning import contracts, evidence
from lerobot.data_platform.ai_tools.learning.models import (
    LearningAnalysis,
    LearningCall,
    LearningControl,
    LearningEpisode,
    LearningEvent,
    LearningFeedback,
    LearningProfile,
)
from lerobot.data_platform.ai_tools.models import AssistantConversation, AssistantTurn
from lerobot.data_platform.control_plane import RemoteJob
from lerobot.data_platform.job_management import JobConflictError, scheduler_lock
from lerobot.data_platform.processing.models import Pipeline, PipelineRun
from lerobot.data_platform.task_catalog import content_digest


def now_ms():
    return int(time.time() * 1000)


def control(session):
    row = session.get(LearningControl, "environment")
    if row is None:
        row = LearningControl(
            key="environment",
            revision=0,
            enabled=False,
            calls=0,
            tokens=0,
            scan={},
            backfill={},
            lease_owner="",
            lease_until=0,
        )
        session.add(row)
        session.flush()
    return row


class Learning:
    def __init__(self, assistant):
        self.assistant, self.manager, self.store = assistant, assistant.manager, assistant.store

    def admin(self, session, owner):
        if self.manager._actor(session, owner, write=False).role != "admin":
            raise PermissionError("Administrator required")

    def target(self, session, owner, target):
        contracts.exact(target, {"kind", "id"})
        kind, key = target["kind"], target["id"]
        if kind == "page":
            self.manager._actor(session, owner, write=False)
            if key not in contracts.CONTROLS:
                raise ValueError("Unknown page")
            return None
        if kind == "operator":
            from lerobot.data_platform.operators import get_operator

            self.manager._actor(session, owner, write=False)
            return get_operator(key)
        self.manager._actor(session, owner, write=False)
        model = {"turn": AssistantTurn, "run": PipelineRun, "job": RemoteJob, "pipeline": Pipeline}.get(kind)
        if model is None:
            raise ValueError("Unknown learning target")
        contracts.identifier(key)
        row = (
            session.scalar(
                select(RemoteJob)
                .options(load_only(RemoteJob.job_id, RemoteJob.requested_by, RemoteJob.location_id))
                .where(RemoteJob.job_id == key)
            )
            if kind == "job"
            else session.get(model, key)
        )
        if row is None or (row.requested_by if kind == "job" else row.owner) != owner:
            raise KeyError("Owned learning target not found")
        if kind == "turn":
            conversation = session.get(AssistantConversation, row.conversation_id)
            if not conversation or conversation.owner != owner:
                raise KeyError("Conversation unavailable")
            self.assistant._target(session, owner, conversation.location_id, conversation.input_identity)
        elif kind in {"run", "job"}:
            self.manager._location(session, row.location_id, owner, write=False)
        return row

    def state(self, owner):
        with self.store.sessions() as session:
            self.manager._actor(session, owner, write=False)
            settings = session.get(LearningControl, "environment")
            profile = session.get(LearningProfile, owner)
            return {
                "collection_enabled": bool(settings and settings.enabled),
                "revision": profile.revision if profile else 0,
                "chat": bool(profile and profile.chat),
                "experience": bool(profile and profile.experience),
            }

    def configure(self, owner, value):
        contracts.exact(value, {"expected_revision", "enabled", "calls", "tokens"})
        if type(value["enabled"]) is not bool or any(
            type(value[k]) is not int or not 0 <= value[k] <= maximum
            for k, maximum in (("calls", 1000), ("tokens", 1000000))
        ):
            raise ValueError("Invalid learning budget")
        with self.store.sessions.begin() as session:
            scheduler_lock(session)
            self.admin(session, owner)
            row = control(session)
            contracts.revision(value["expected_revision"], row.revision)
            if value["enabled"] and not row.enabled:
                row.scan = {"since_ms": now_ms(), "run": "", "job": "", "until_ms": now_ms()}
            row.enabled, row.calls, row.tokens = value["enabled"], value["calls"], value["tokens"]
            row.administrator = owner
            row.revision += 1
            from lerobot.data_platform.ai_tools.learning.cases import invalidate

            invalidate(self, session)
            self.manager._audit(session, owner, "learning.configured", revision=row.revision)
        return self.dashboard(owner)

    def settings(self, owner, value):
        contracts.exact(value, {"expected_revision", "chat", "experience"})
        if type(value["chat"]) is not bool or type(value["experience"]) is not bool:
            raise ValueError("Boolean learning settings required")
        with self.store.sessions.begin() as session:
            scheduler_lock(session)
            self.manager._actor(session, owner, write=False)
            row = session.get(LearningProfile, owner)
            contracts.revision(value["expected_revision"], row.revision if row else 0)
            if row is None:
                row = LearningProfile(
                    owner=owner, revision=0, chat=False, chat_epoch=0, experience=False, since_ms=0
                )
                session.add(row)
            consent_changed = value["chat"] != row.chat
            if consent_changed:
                row.since_ms = now_ms()
                row.chat_epoch += 1
            row.chat, row.experience = value["chat"], value["experience"]
            row.revision += 1
            # Only a chat consent change invalidates private extraction; experience is independent.
            if consent_changed:
                for analysis in session.scalars(
                    select(LearningAnalysis).where(LearningAnalysis.owner == owner)
                ):
                    analysis.status, analysis.items = "revoked", []
                    analysis.revision += 1
                for call in session.scalars(select(LearningCall).where(LearningCall.owner == owner)):
                    call.result = {}
            self.manager._audit(session, owner, "learning.settings", revision=row.revision)
        return self.state(owner)

    def event(self, owner, value):
        contracts.exact(value, {"event_id", "kind", "control", "target"})
        contracts.identifier(value["event_id"])
        if value["kind"] not in contracts.EVENTS or value["control"] not in contracts.CONTROLS:
            raise ValueError("Unsupported interaction")
        key, digest = content_digest([owner, value["event_id"]]), content_digest(value)
        with self.store.sessions.begin() as session:
            scheduler_lock(session)
            self.target(session, owner, value["target"])
            row = session.get(LearningEvent, key)
            if row:
                if row.digest != digest:
                    raise JobConflictError("Event identity reused")
                return {"accepted": row.valid, "event_id": value["event_id"]}
            setting = session.get(LearningControl, "environment")
            if not setting or not setting.enabled:
                return {"accepted": False}
            count = session.scalar(
                select(func.count())
                .select_from(LearningEvent)
                .where(LearningEvent.owner == owner, LearningEvent.created_ms > now_ms() - 60000)
            )
            if count >= 120:
                return {"accepted": False, "reason": "rate_limit"}
            session.add(
                LearningEvent(
                    key=key,
                    owner=owner,
                    digest=digest,
                    kind=value["kind"],
                    control=value["control"],
                    target=value["target"],
                    created_ms=now_ms(),
                )
            )
            self.manager._audit(session, owner, "learning.event", event_key=key, kind=value["kind"])
        return {"accepted": True, "event_id": value["event_id"]}

    def feedback_state(self, owner, target):
        with self.store.sessions() as session:
            self.target(session, owner, target)
            row = session.get(LearningFeedback, content_digest([owner, target]))
            return {
                "revision": row.revision if row else 0,
                "verdict": row.verdict if row and row.valid else None,
            }

    def feedback(self, owner, value):
        contracts.exact(value, {"event_id", "expected_revision", "target", "verdict", "reason"})
        contracts.identifier(value["event_id"])
        if value["verdict"] not in contracts.VERDICTS or value["reason"] not in contracts.REASONS:
            raise ValueError("Unknown feedback")
        contracts.exact(value["target"], {"kind", "id"})
        if value["target"].get("kind") not in {"turn", "run", "job", "pipeline"}:
            raise ValueError("Feedback needs an owned task or response")
        key = content_digest([owner, value["target"]])
        with self.store.sessions.begin() as session:
            scheduler_lock(session)
            self.target(session, owner, value["target"])
            setting = session.get(LearningControl, "environment")
            if not setting or not setting.enabled:
                raise JobConflictError("Learning collection is disabled")
            row = session.get(LearningFeedback, key)
            if row and row.delivery == value["event_id"]:
                if not row.valid or row.verdict != value["verdict"] or row.reason != value["reason"]:
                    raise JobConflictError("Feedback identity reused")
                return {"revision": row.revision, "key": key}
            contracts.revision(value["expected_revision"], row.revision if row else 0)
            if row is None:
                row = LearningFeedback(key=key, owner=owner, target=value["target"], revision=0)
                session.add(row)
            row.delivery, row.verdict, row.reason = value["event_id"], value["verdict"], value["reason"]
            row.revision += 1
            row.created_ms = now_ms()
            row.valid = True
            self.manager._audit(session, owner, "learning.feedback", feedback_id=key, revision=row.revision)
            return {"revision": row.revision, "key": key}

    def materials(self, owner):
        from lerobot.data_platform.ai_tools.learning.chat import validate_analysis

        with self.store.sessions() as session:
            self.manager._actor(session, owner, write=False)
            episodes, analyses, feedback = [], [], []
            for row in session.scalars(
                select(LearningEpisode)
                .where(LearningEpisode.owner == owner, LearningEpisode.valid.is_(True))
                .order_by(LearningEpisode.created_ms.desc())
                .limit(100)
            ):
                try:
                    self.target(session, owner, {"kind": row.kind, "id": row.source_id})
                    if row.verified:
                        evidence.resolve(self.manager, session, evidence.reference(row), owner=owner)
                    episodes.append(
                        {
                            "key": row.key,
                            "revision": row.revision,
                            "kind": row.kind,
                            "source_id": row.source_id,
                            "verified": row.verified,
                            "projection": row.projection,
                        }
                    )
                except (PermissionError, KeyError, ValueError, JobConflictError):
                    continue
            for row in session.scalars(
                select(LearningAnalysis)
                .where(
                    LearningAnalysis.owner == owner,
                    LearningAnalysis.status.in_(["complete", "pending", "incomplete"]),
                )
                .order_by(LearningAnalysis.created_ms.desc())
                .limit(50)
            ):
                try:
                    validate_analysis(self, session, row)
                    analyses.append(
                        {"key": row.key, "revision": row.revision, "status": row.status, "items": row.items}
                    )
                except (PermissionError, KeyError, ValueError, JobConflictError):
                    continue
            for row in session.scalars(
                select(LearningFeedback).where(LearningFeedback.owner == owner).limit(100)
            ):
                try:
                    self.target(session, owner, row.target)
                    feedback.append(
                        {
                            "key": row.key,
                            "target": row.target,
                            "revision": row.revision,
                            "verdict": row.verdict,
                            "reason": row.reason,
                            "valid": row.valid,
                        }
                    )
                except (PermissionError, KeyError, ValueError, JobConflictError):
                    continue
            events = []
            for row in session.scalars(
                select(LearningEvent)
                .where(LearningEvent.owner == owner, LearningEvent.valid.is_(True))
                .order_by(LearningEvent.created_ms.desc())
                .limit(100)
            ):
                try:
                    self.target(session, owner, row.target)
                    events.append(
                        {"key": row.key, "revision": row.revision, "kind": row.kind, "target": row.target}
                    )
                except (PermissionError, KeyError, ValueError, JobConflictError):
                    continue
            return {"episodes": episodes, "analyses": analyses, "feedback": feedback, "events": events}

    def revoke(self, owner, value):
        contracts.exact(value, {"kind", "key", "expected_revision"})
        model = {
            "episode": LearningEpisode,
            "analysis": LearningAnalysis,
            "feedback": LearningFeedback,
            "event": LearningEvent,
        }.get(value["kind"])
        if model is None:
            raise ValueError("Unknown material")
        with self.store.sessions.begin() as session:
            scheduler_lock(session)
            self.manager._actor(session, owner, write=False)
            row = session.get(model, value["key"])
            if not row or row.owner != owner:
                raise KeyError("Owned material not found")
            contracts.revision(value["expected_revision"], row.revision)
            if model != LearningAnalysis:
                row.valid = False
            else:
                row.status, row.items = "revoked", []
                for call in session.scalars(
                    select(LearningCall).where(
                        LearningCall.key.in_([row.key + ":extract", row.key + ":verify"])
                    )
                ):
                    call.result = {}
            row.revision += 1
            from lerobot.data_platform.ai_tools.learning.cases import invalidate

            invalidate(self, session)
            self.manager._audit(session, owner, "learning.revoked", kind=value["kind"], key=row.key)
        return {"revoked": True}

    def accept_preference(self, owner, value):
        from lerobot.data_platform.ai_tools import memory
        from lerobot.data_platform.ai_tools.learning.chat import validate_analysis

        contracts.exact(value, {"key", "index", "expected_revision", "memory_revision"})
        if type(value["index"]) is not int or not 0 <= value["index"] < 8:
            raise ValueError("Invalid preference index")

        def checked(session):
            row = session.get(LearningAnalysis, value["key"])
            if not row or row.owner != owner:
                raise KeyError("Owned analysis not found")
            contracts.revision(value["expected_revision"], row.revision)
            validate_analysis(self, session, row)
            if value["index"] >= len(row.items) or row.items[value["index"]].get("accepted"):
                raise JobConflictError("Preference unavailable or already accepted")
            item = row.items[value["index"]]
            if item["kind"] != "preference" or item["label"] not in contracts.PREFERENCES:
                raise ValueError("Not a preference candidate")
            return row, item

        with self.store.sessions() as session:
            _, item = checked(session)
            text = contracts.PREFERENCES[item["label"]]

        def admit(session):
            row, _ = checked(session)
            row.revision += 1
            items = [dict(i) for i in row.items]
            items[value["index"]]["accepted"] = True
            row.items = items

        return memory.change(
            self.assistant,
            owner,
            {
                "operation": "save",
                "expected_revision": value["memory_revision"],
                "scope": "account",
                "location_id": None,
                "text": text,
            },
            admission_guard=admit,
        )

    def backfill(self, owner, value):
        contracts.exact(value, {"expected_revision", "action", "event_id"})
        contracts.identifier(value["event_id"])
        if value["action"] not in {"start", "pause", "resume"}:
            raise ValueError("Unknown backfill action")
        with self.store.sessions.begin() as session:
            scheduler_lock(session)
            self.admin(session, owner)
            row = control(session)
            if row.backfill.get("event_id") == value["event_id"]:
                if row.backfill.get("action") != value["action"]:
                    raise JobConflictError("Backfill identity reused")
                return {"backfill": row.backfill, "revision": row.revision}
            contracts.revision(value["expected_revision"], row.revision)
            if not row.enabled:
                raise JobConflictError("Enable collection before backfill")
            state = dict(row.backfill)
            if value["action"] == "start":
                if state.get("status") in {"running", "paused"}:
                    raise JobConflictError("Resume the existing backfill")
                state = {
                    "since_ms": now_ms() - 30 * 86400000,
                    "until_ms": now_ms(),
                    "run": "",
                    "job": "",
                    "run_done": False,
                    "job_done": False,
                    "count": 0,
                }
            elif state.get("status") not in {"running", "paused"}:
                raise JobConflictError("No resumable backfill")
            state.update(
                status="paused" if value["action"] == "pause" else "running",
                event_id=value["event_id"],
                action=value["action"],
            )
            row.backfill, row.administrator = state, owner
            row.revision += 1
            self.manager._audit(session, owner, "learning.backfill", action=value["action"])
            return {"backfill": state, "revision": row.revision}

    def dashboard(self, owner):
        with self.store.sessions() as session:
            self.admin(session, owner)
            row = session.get(LearningControl, "environment")
            counts = {
                "observed": 0,
                "verified": 0,
                "associated": 0,
                "material_changes": 0,
                "execution_failures": 0,
                "infrastructure_failures": 0,
                "cancelled": 0,
                "feedback_instances": 0,
            }
            labels, issues = {}, {}
            # One explicit cohort for every displayed rate; no global denominator with a sampled numerator.
            rows = session.scalars(
                select(LearningEpisode)
                .where(
                    LearningEpisode.valid.is_(True), LearningEpisode.created_ms >= now_ms() - 30 * 86400000
                )
                .order_by(LearningEpisode.created_ms.desc())
                .limit(200)
            ).all()
            for episode in rows:
                try:
                    evidence.project(self.manager, session, owner, episode.kind, episode.source_id)
                    if episode.verified:
                        evidence.resolve(self.manager, session, evidence.reference(episode))
                except (PermissionError, KeyError, ValueError, JobConflictError):
                    continue
                counts["observed"] += 1
                counts["verified"] += episode.verified
                counts["associated"] += bool(episode.projection.get("associated"))
                counts["material_changes"] += bool(episode.projection.get("material_change"))
                failures = [s.get("failure") for s in episode.projection["steps"]]
                infrastructure = any(
                    f in {"publication_failed", "process_lost", "provider_timeout"} for f in failures
                )
                counts["infrastructure_failures"] += infrastructure
                counts["execution_failures"] += (
                    episode.projection.get("status") == "error" and not infrastructure
                )
                counts["cancelled"] += episode.projection.get("status") == "cancelled"
                target = {"kind": episode.kind, "id": episode.source_id}
                feedback = session.get(LearningFeedback, content_digest([episode.owner, target]))
                signature = {
                    "format": episode.projection.get("format"),
                    "operators": [s["operator_id"] for s in episode.projection["steps"]],
                    "changes": episode.projection.get("changes", []),
                    "failures": sorted(set(failures) - {"none"}),
                }
                group = issues.setdefault(
                    content_digest(signature),
                    {**signature, "instances": 0, "verified": 0, "completed": 0, "feedback_reasons": {}},
                )
                group["instances"] += 1
                group["verified"] += episode.verified
                group["completed"] += episode.projection.get("status") == "done"
                if feedback and feedback.valid:
                    group["feedback_reasons"][feedback.reason] = (
                        group["feedback_reasons"].get(feedback.reason, 0) + 1
                    )
                    counts["feedback_instances"] += 1
                    labels[feedback.verdict] = labels.get(feedback.verdict, 0) + 1
            counts["window"] = "latest_200_accessible_collected_executions_within_30_days"
            counts["verifiable_ratio"] = (
                counts["verified"] / counts["observed"] if counts["observed"] else None
            )
            counts["material_change_ratio"] = (
                counts["material_changes"] / counts["associated"] if counts["associated"] else None
            )
            counts["feedback_coverage"] = (
                counts["feedback_instances"] / counts["observed"] if counts["observed"] else None
            )
            day = time.strftime("%Y-%m-%d", time.gmtime())
            calls = session.scalars(select(LearningCall).where(LearningCall.day == day)).all()
            return {
                "revision": row.revision if row else 0,
                "enabled": bool(row and row.enabled),
                "calls": row.calls if row else 0,
                "tokens": row.tokens if row else 0,
                "backfill": row.backfill if row else {},
                "coverage": {**counts, "collection": evidence.coverage(self.manager, session, owner)},
                "feedback": labels,
                "issues": list(issues.values()),
                "usage": {
                    "calls": len(calls),
                    "tokens": sum(c.reserved for c in calls),
                    "unknown_calls": sum(c.status == "unknown" for c in calls),
                    "reported_tokens": sum(
                        c.usage.get("input_tokens", 0) + c.usage.get("output_tokens", 0) for c in calls
                    ),
                },
                "planning_correctness": "See paired strategy evaluation; usage is not correctness",
            }
