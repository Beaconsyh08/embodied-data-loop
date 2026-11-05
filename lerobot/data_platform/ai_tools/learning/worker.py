"""Leased, bounded collection and private analysis. No task execution capability."""

import logging
import math
import signal
import threading
import time
import uuid
from datetime import datetime

from sqlalchemy import delete, func, select

from lerobot.data_platform.ai_tools.learning import chat, evidence
from lerobot.data_platform.ai_tools.learning.models import (
    LearningAnalysis,
    LearningCall,
    LearningControl,
    LearningEvent,
)
from lerobot.data_platform.ai_tools.learning.service import Learning, control, now_ms
from lerobot.data_platform.control_plane import RemoteJob
from lerobot.data_platform.job_management import JobConflictError, scheduler_lock
from lerobot.data_platform.processing.models import PipelineRun, PipelineStepRun

EXPECTED = (PermissionError, KeyError, ValueError, JobConflictError)


def scan(learning, session, settings, *, historical):
    """Freeze each window and page by identity. Live windows overlap legacy FLOAT timestamp rounding."""
    state = dict(settings.backfill if historical else settings.scan)
    if historical and state.get("status") != "running":
        return
    if not state:
        return
    remaining = 200
    for kind, model, pk in (("run", PipelineRun, PipelineRun.run_id), ("job", RemoteJob, RemoteJob.job_id)):
        if state.get(kind + "_done") or not remaining:
            continue
        low, high = state["since_ms"] / 1000, state["until_ms"] / 1000
        # Existing MySQL DATETIME(0) columns can round a fractional completion into the next second.
        # Freeze an inclusive whole-second boundary so a just-finished task cannot be omitted.
        low, high = datetime.utcfromtimestamp(low), datetime.utcfromtimestamp(math.ceil(high))
        timestamp = RemoteJob.finished_at
        if kind == "run":
            # Both legacy Run and Attempt clocks are MySQL FLOAT. The Job finalization timestamp
            # is a DATETIME, written by the same acknowledged completion transaction.
            timestamp = (
                select(func.max(RemoteJob.finished_at))
                .select_from(PipelineStepRun)
                .join(RemoteJob, RemoteJob.job_id == PipelineStepRun.job_id)
                .where(PipelineStepRun.run_id == PipelineRun.run_id)
                .correlate(PipelineRun)
                .scalar_subquery()
            )
            legacy_clock = (
                func.from_unixtime(PipelineRun.updated_at)
                if session.bind.dialect.name == "mysql"
                else func.datetime(PipelineRun.updated_at, "unixepoch")
            )
            timestamp = func.coalesce(timestamp, legacy_clock)
            if not historical:
                # A coordinator may finalize a Run long after its Jobs. Track that transition
                # on live scans, with a conservative envelope for the legacy FLOAT clock.
                timestamp = PipelineRun.updated_at
                low, high = state["since_ms"] / 1000 - 256, state["until_ms"] / 1000 + 256
        rows = session.scalars(
            select(pk)
            .where(
                timestamp >= low,
                timestamp <= high,
                pk > state.get(kind, ""),
                model.status.in_(["done", "error", "cancelled"]),
            )
            .order_by(pk)
            .limit(remaining)
        ).all()
        state[kind + "_done"] = len(rows) < remaining
        for identity in rows:
            try:
                with session.begin_nested():
                    evidence.capture(learning.manager, session, settings.administrator, kind, identity)
            except EXPECTED:
                pass
            state[kind] = identity
            state["count"] = state.get("count", 0) + 1
        remaining -= len(rows)
    if state.get("run_done") and state.get("job_done"):
        if historical:
            state["status"] = "complete"
        else:
            state = {
                "since_ms": max(settings.scan["since_ms"], state["until_ms"] - 256000),
                "until_ms": now_ms(),
                "run": "",
                "job": "",
            }
    if historical:
        settings.backfill = state
    else:
        settings.scan = state


def tick(assistant):
    learning, token = Learning(assistant), str(uuid.uuid4())
    with learning.store.sessions.begin() as session:
        scheduler_lock(session)
        settings = control(session)
        # Retention is independent of collection/consent; turning learning off cannot extend storage.
        session.execute(delete(LearningEvent).where(LearningEvent.created_ms < now_ms() - 30 * 86400000))
        for analysis in session.scalars(
            select(LearningAnalysis)
            .where(
                LearningAnalysis.created_ms < now_ms() - 30 * 86400000, LearningAnalysis.status != "revoked"
            )
            .limit(200)
        ):
            analysis.status, analysis.items = "revoked", []
            analysis.revision += 1
            for call in session.scalars(
                select(LearningCall).where(
                    LearningCall.key.in_([analysis.key + ":extract", analysis.key + ":verify"])
                )
            ):
                call.result = {}
        if not settings.enabled or settings.lease_until > time.time():
            return False
        learning.admin(session, settings.administrator)
        settings.lease_owner, settings.lease_until = token, time.time() + 120
        revision = settings.revision

    def guard(session):
        settings = session.get(LearningControl, "environment", populate_existing=True)
        if (
            not settings.enabled
            or settings.revision != revision
            or settings.lease_owner != token
            or settings.lease_until <= time.time()
        ):
            raise PermissionError("Learning lease or authorization changed")
        learning.admin(session, settings.administrator)

    key = None
    try:
        with learning.store.sessions.begin() as session:
            scheduler_lock(session)
            guard(session)
            settings = session.get(LearningControl, "environment")
            scan(learning, session, settings, historical=True)
            scan(learning, session, settings, historical=False)
            evidence.references(learning.manager, session)
            from lerobot.data_platform.ai_tools.learning.cases import invalidate

            invalidate(learning, session)
            session.execute(delete(LearningEvent).where(LearningEvent.created_ms < now_ms() - 30 * 86400000))
            for analysis in session.scalars(
                select(LearningAnalysis)
                .where(LearningAnalysis.status != "revoked")
                .order_by(LearningAnalysis.created_ms)
                .limit(200)
            ):
                try:
                    chat.validate_analysis(learning, session, analysis)
                    if analysis.created_ms < now_ms() - 30 * 86400000:
                        raise PermissionError("Learning retention elapsed")
                except EXPECTED:
                    analysis.status, analysis.items = "revoked", []
                    analysis.revision += 1
                    for call in session.scalars(
                        select(LearningCall).where(
                            LearningCall.key.in_([analysis.key + ":extract", analysis.key + ":verify"])
                        )
                    ):
                        call.result = {}
            guard(session)
            key = chat.prepare(learning, session)
        if key:
            chat.process(learning, key, guard)
    except chat.WaitingError:
        pass
    except Exception:
        # Never log provider text, conversation content, paths, or exception bodies.
        logging.warning("Learning batch incomplete; no unknown model call will be resent")
        if key:
            with learning.store.sessions.begin() as session:
                scheduler_lock(session)
                settings = session.get(LearningControl, "environment")
                if settings.lease_owner == token:
                    row = session.get(LearningAnalysis, key)
                    if row and row.status == "pending":
                        row.status, row.items = "incomplete", []
                        chat.advance(session, row)
    finally:
        with learning.store.sessions.begin() as session:
            scheduler_lock(session)
            settings = session.get(LearningControl, "environment")
            if settings.lease_owner == token:
                settings.lease_owner, settings.lease_until = "", 0
    return True


def main():
    from lerobot.data_platform.ai_tools.service import AssistantService
    from lerobot.data_platform.processing.coordinator import configured_manager

    assistant, stop = AssistantService(configured_manager()), threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        try:
            tick(assistant)
        except Exception:
            logging.warning("Learning storage unavailable; foreground requests are unaffected")
        stop.wait(300)


if __name__ == "__main__":
    main()
