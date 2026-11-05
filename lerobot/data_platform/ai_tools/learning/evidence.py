"""Server-derived links and evidence. Browser assertions never establish task success."""

import logging
import time
from dataclasses import asdict

from sqlalchemy import select
from sqlalchemy.orm import load_only

from lerobot.data_platform.ai_tools.learning.contracts import EvidenceRef
from lerobot.data_platform.ai_tools.learning.models import (
    LearningControl,
    LearningEpisode,
    LearningJobReceipt,
    LearningOrigin,
)
from lerobot.data_platform.ai_tools.models import AssistantConversation, AssistantTurn
from lerobot.data_platform.control_plane import RemoteJob
from lerobot.data_platform.job_management import JobConflictError
from lerobot.data_platform.management_storage import JobAttempt, JobControl
from lerobot.data_platform.operators import get_operator
from lerobot.data_platform.processing.models import PipelineRun, PipelineStepRun, PipelineVersion
from lerobot.data_platform.processing.planner import PipelineSpec
from lerobot.data_platform.processing.service import location_identity
from lerobot.data_platform.task_catalog import content_digest


def changes(before, after):
    a, b = PipelineSpec.from_dict(before).steps, PipelineSpec.from_dict(after).steps
    result = []
    if [s.operator_id for s in a] != [s.operator_id for s in b]:
        result.append(
            "order" if sorted(s.operator_id for s in a) == sorted(s.operator_id for s in b) else "operators"
        )
    if len(a) != len(b):
        return result
    if "operators" not in result and any(x.version != y.version for x, y in zip(a, b, strict=True)):
        result.append("operators")
    positions = [{s.step_id: i for i, s in enumerate(rows)} for rows in (a, b)]
    if any(
        positions[0].get(x.input, "source") != positions[1].get(y.input, "source")
        for x, y in zip(a, b, strict=True)
    ):
        result.append("inputs")
    if any(x.parameters != y.parameters for x, y in zip(a, b, strict=True)):
        result.append("parameters")
    if any(x.review_before != y.review_before or x.options != y.options for x, y in zip(a, b, strict=True)):
        result.append("review")
    if any(
        [positions[0][r] for r in x.artifacts] != [positions[1][r] for r in y.artifacts]
        for x, y in zip(a, b, strict=True)
    ):
        result.append("evidence")
    return result


def origin(manager, session, owner, version, request_id, previous_id=None):
    """Optional projection in a savepoint; it cannot break ordinary Pipeline persistence."""
    try:
        with session.begin_nested():
            control = session.get(LearningControl, "environment")
            if not control or not control.enabled:
                return
            previous = session.get(LearningOrigin, previous_id) if previous_id else None
            request_id = request_id or (previous.request_id if previous and previous.owner == owner else None)
            if not request_id:
                return
            turn = session.get(AssistantTurn, request_id)
            if not turn or turn.owner != owner or turn.status != "done":
                return
            conversation = session.get(AssistantConversation, turn.conversation_id)
            if not conversation or conversation.owner != owner:
                return
            manager._location(session, conversation.location_id, owner, write=False)
            proposal = turn.result.get("proposal") or turn.result.get("action") or {}
            definition = proposal.get("definition")
            if not definition:
                return
            session.add(
                LearningOrigin(
                    version_id=version.version_id,
                    owner=owner,
                    request_id=request_id,
                    original_digest=content_digest(definition),
                    changes=changes(definition, version.spec),
                    created_ms=int(time.time() * 1000),
                )
            )
    except Exception:
        logging.warning("Learning origin unavailable; ordinary task saving is unaffected")


def unconfirmed_projection(manager, session, administrator, run, steps):
    """An actual terminal Run without final receipts is usage only, never an execution proof."""
    from lerobot.data_platform.ai_tools.strategy_contracts import OPERATORS

    if run.status not in {"done", "error", "cancelled"}:
        raise ValueError("Run is not terminal")
    location, _ = manager._location(session, run.location_id, administrator, write=False)
    manager._location(session, run.location_id, run.owner, write=False)
    if location_identity(location) != run.input_identity:
        raise PermissionError("Input identity changed")
    version = session.get(PipelineVersion, run.version_id)
    if not version or content_digest(version.spec) != version.digest:
        raise PermissionError("Immutable definition changed")
    plan = PipelineSpec.from_dict(version.spec)
    if not 1 <= len(plan.steps) <= 12 or any(s.operator_id not in OPERATORS for s in plan.steps):
        raise ValueError("Unsupported learning operations")
    for step in steps:
        for location_id in (step.input_location_id, step.output_location_id):
            if location_id:
                manager._location(session, location_id, administrator, write=False)
                manager._location(session, location_id, run.owner, write=False)
    positions = {s.step_id: i for i, s in enumerate(plan.steps)}
    actual = {s.step_id: s for s in steps}
    value = {
        "format": location["metadata"].get("codebase_version"),
        "status": run.status,
        "associated": False,
        "changes": [],
        "material_change": False,
        "output_validation": "unverified",
        "steps": [
            {
                "operator_id": s.operator_id,
                "version": s.version,
                "input": positions.get(s.input, "source"),
                "artifacts": [positions[k] for k in s.artifacts],
                "status": actual[s.step_id].status if s.step_id in actual else "not_started",
                "failure": "cancelled" if run.status == "cancelled" else "unconfirmed",
                "planning_failure": False,
                "seconds": None,
                "output_registered": False,
                "acknowledged": False,
            }
            for s in plan.steps
        ],
    }
    return run.owner, run.location_id, value, False, content_digest([value, version.digest, run.revision])


def project(manager, session, administrator, kind, source_id):
    from lerobot.data_platform.ai_tools.experience_evidence import collect, failure_code
    from lerobot.data_platform.ai_tools.strategy_contracts import OPERATORS
    from lerobot.data_platform.ai_tools.strategy_evidence import projection

    if manager._actor(session, administrator, write=False).role != "admin":
        raise PermissionError("Active learning administrator required")
    if kind == "run":
        run = session.get(PipelineRun, source_id)
        if not run:
            raise KeyError("Run unavailable")
        steps = session.scalars(
            select(PipelineStepRun)
            .where(PipelineStepRun.run_id == run.run_id)
            .order_by(PipelineStepRun.position)
        ).all()
        from lerobot.data_platform.ai_tools.followups import _acknowledged

        if not _acknowledged(session, run, steps):
            return unconfirmed_projection(manager, session, administrator, run, steps)
        value, digest = projection(manager, session, administrator, run)
        executed = collect(session, run, steps)
        verified = executed is not None and all(
            s["output_registered"]
            or get_operator(s["operator_id"]).output_type != "dataset"
            or s["status"] != "done"
            for s in value["steps"]
        )
        linked = session.get(LearningOrigin, run.version_id)
        differences, request_id = [], None
        if linked:
            turn = session.get(AssistantTurn, linked.request_id)
            if not turn or turn.owner != run.owner or turn.status != "done":
                raise PermissionError("Origin withdrawn")
            definition = (turn.result.get("proposal") or turn.result.get("action") or {}).get("definition")
            if content_digest(definition) != linked.original_digest:
                raise PermissionError("Origin changed")
            conversation = session.get(AssistantConversation, turn.conversation_id)
            if not conversation or conversation.owner != run.owner:
                raise PermissionError("Origin unavailable")
            if (
                conversation.location_id == run.location_id
                and conversation.input_identity == run.input_identity
            ):
                differences, request_id = linked.changes, linked.request_id
        value = {
            **value,
            "associated": bool(request_id),
            "changes": differences,
            "material_change": bool(differences),
            "output_validation": "platform_receipt" if verified else "unverified",
        }
        return run.owner, run.location_id, value, verified, content_digest([digest, value, request_id])
    if kind != "job":
        raise ValueError("Unknown execution source")
    if session.scalar(select(PipelineStepRun.row_id).where(PipelineStepRun.job_id == source_id)):
        raise ValueError("Pipeline jobs belong to their Run")
    job = session.scalar(
        select(RemoteJob)
        .options(
            load_only(
                RemoteJob.job_id,
                RemoteJob.requested_by,
                RemoteJob.location_id,
                RemoteJob.operation,
                RemoteJob.options,
                RemoteJob.status,
            )
        )
        .where(RemoteJob.job_id == source_id)
    )
    if not job or job.status not in {"done", "error", "cancelled"} or job.operation not in OPERATORS:
        raise ValueError("Unsupported terminal job")
    location, _ = manager._location(session, job.location_id, administrator, write=False)
    manager._location(session, job.location_id, job.requested_by, write=False)
    snapshot = job.options.get("operator_snapshot") or {}
    control = session.get(JobControl, job.job_id)
    attempt = (
        session.scalar(
            select(JobAttempt)
            .options(
                load_only(
                    JobAttempt.attempt_id,
                    JobAttempt.job_id,
                    JobAttempt.status,
                    JobAttempt.finished_at,
                    JobAttempt.started_at,
                )
            )
            .where(JobAttempt.attempt_id == control.attempt_id)
        )
        if control and control.attempt_id
        else None
    )
    output = session.scalar(
        select(RemoteJob.result["output_location_id"]).where(RemoteJob.job_id == source_id)
    )
    if output:
        manager._location(session, output, administrator, write=False)
        manager._location(session, output, job.requested_by, write=False)
    version = snapshot.get("version")
    receipt = session.get(LearningJobReceipt, job.job_id)
    verified = bool(
        control
        and control.stop_confirmed
        and attempt
        and attempt.job_id == source_id
        and attempt.finished_at
        and attempt.status == job.status
        and receipt
        and receipt.owner == job.requested_by
        and receipt.location_id == job.location_id
        and receipt.command_digest == content_digest(job.options)
        and receipt.input_identity == location_identity(location)
        and receipt.version == version
        and receipt.format in {"v2.1", "v3.0"}
        and snapshot.get("operator_id") == job.operation
        and not snapshot.get("pipeline_run")
    )
    if verified and job.status == "done":
        attempt_output = session.scalar(
            select(JobAttempt.result["output_location_id"]).where(JobAttempt.attempt_id == attempt.attempt_id)
        )
        verified = output == attempt_output
    if verified:
        op = get_operator(job.operation, version)
        if op.output_type == "dataset" and job.status == "done" and not output:
            verified = False
    failure = "cancelled" if job.status == "cancelled" else "none"
    if job.status == "error":
        from sqlalchemy import func

        error = session.scalar(
            select(func.substr(RemoteJob.error, 1, 1024)).where(RemoteJob.job_id == source_id)
        )
        failure = (
            "publication_failed"
            if str(error)
            .lower()
            .startswith(("oss upload failed", "result upload failed", "result publication failed"))
            else failure_code(error)
        )
    value = {
        "format": location["metadata"].get("codebase_version"),
        "status": job.status,
        "changes": [],
        "material_change": False,
        "output_validation": "platform_receipt" if verified else "unverified",
        "steps": [
            {
                "operator_id": job.operation,
                "version": version if verified else None,
                "input": "source",
                "artifacts": [],
                "status": job.status,
                "failure": failure,
                "planning_failure": failure == "empty_selection",
                "seconds": max(0, int(attempt.finished_at - attempt.started_at))
                if attempt and attempt.finished_at and attempt.started_at
                else None,
                "output_registered": bool(output),
                "acknowledged": verified,
            }
        ],
    }
    digest = content_digest([value, snapshot, job.options, attempt.attempt_id if attempt else None, output])
    return job.requested_by, job.location_id, value, verified, digest


def capture(manager, session, administrator, kind, source_id):
    owner, location, value, verified, digest = project(manager, session, administrator, kind, source_id)
    key = content_digest([kind, source_id])
    row = session.get(LearningEpisode, key)
    if row:
        if row.valid and not row.verified and verified:
            row.projection, row.digest, row.verified = value, digest, True
            row.revision += 1
            return row
        if row.digest != digest:
            row.valid = False
            row.revision += 1
        return row
    created_ms = int(time.time() * 1000)
    if kind == "run":
        from lerobot.data_platform.ai_tools.strategy_models import StrategyEvidence

        previous = session.get(StrategyEvidence, source_id)
        if previous:
            created_ms = min(created_ms, previous.created_at * 1000)
    row = LearningEpisode(
        key=key,
        owner=owner,
        location_id=location,
        kind=kind,
        source_id=source_id,
        digest=digest,
        projection=value,
        verified=verified,
        valid=True,
        revision=0,
        created_ms=created_ms,
    )
    session.add(row)
    return row


def reference(row):
    return asdict(EvidenceRef("learning_episode", row.key, row.revision, row.digest))


def resolve(manager, session, ref, *, owner=None):
    parsed = EvidenceRef.parse(ref)
    row = session.get(LearningEpisode, parsed.source_id)
    control = session.get(LearningControl, "environment")
    if not row or not row.valid or row.revision != parsed.revision or row.digest != parsed.digest:
        raise PermissionError("Learning evidence revoked")
    if owner is not None and row.owner != owner:
        raise PermissionError("Private learning evidence")
    if not control or not control.enabled or not control.administrator:
        raise PermissionError("Learning collection disabled")
    _, _, value, verified, digest = project(manager, session, control.administrator, row.kind, row.source_id)
    if not verified or not row.verified or digest != row.digest:
        raise PermissionError("Learning evidence changed")
    return value


def references(manager, session):
    control = session.get(LearningControl, "environment")
    if not control or not control.enabled:
        return []
    result = []
    rows = session.scalars(
        select(LearningEpisode)
        .where(LearningEpisode.valid.is_(True), LearningEpisode.verified.is_(True))
        .order_by(LearningEpisode.created_ms.desc())
        .limit(200)
    )
    for row in rows:
        try:
            resolve(manager, session, reference(row))
            result.append(row)
        except (PermissionError, KeyError, ValueError, JobConflictError):
            row.valid = False
            row.revision += 1
    return result


def coverage(manager, session, administrator):
    """A bounded, authorized source cohort gives collection coverage its own denominator."""
    from datetime import datetime

    from lerobot.data_platform.ai_tools.strategy_contracts import OPERATORS

    cutoff = time.time() - 30 * 86400
    runs = session.scalars(
        select(PipelineRun.run_id)
        .where(PipelineRun.status.in_(["done", "error", "cancelled"]), PipelineRun.updated_at >= cutoff)
        .order_by(PipelineRun.updated_at.desc(), PipelineRun.run_id)
        .limit(100)
    ).all()
    jobs = session.scalars(
        select(RemoteJob.job_id)
        .where(
            RemoteJob.status.in_(["done", "error", "cancelled"]),
            RemoteJob.operation.in_(OPERATORS),
            RemoteJob.finished_at >= datetime.utcfromtimestamp(cutoff),
            ~select(PipelineStepRun.row_id).where(PipelineStepRun.job_id == RemoteJob.job_id).exists(),
        )
        .order_by(RemoteJob.finished_at.desc(), RemoteJob.job_id)
        .limit(100)
    ).all()
    eligible, captured = 0, 0
    for kind, identities in (("run", runs), ("job", jobs)):
        for source_id in identities:
            try:
                *_, digest = project(manager, session, administrator, kind, source_id)
            except (PermissionError, KeyError, ValueError, JobConflictError):
                continue
            eligible += 1
            row = session.get(LearningEpisode, content_digest([kind, source_id]))
            captured += bool(row and row.valid and row.digest == digest)
    return {
        "eligible_sample": eligible,
        "collected_sample": captured,
        "ratio": captured / eligible if eligible else None,
        "window": "latest_100_terminal_runs_and_100_independent_jobs_within_30_days",
    }
