"""Additive parent/child migration for annotation jobs; never edits execution records."""

from sqlalchemy import func, insert, literal, select

from lerobot.data_platform.annotations.groups import GROUP_PREFIX
from lerobot.data_platform.control.store import CaptionDatasetJob, CaptionJobMember, RemoteJob


def attach_caption_child(session, job):
    """Called within the existing scheduler transaction, including its admission lock."""
    parent_id = GROUP_PREFIX + job.location_id
    if session.get(CaptionDatasetJob, parent_id) is None:
        session.add(
            CaptionDatasetJob(job_id=parent_id, location_id=job.location_id, created_at=job.created_at)
        )
        session.flush()
    if session.get(CaptionJobMember, job.job_id) is None:
        session.add(CaptionJobMember(child_job_id=job.job_id, parent_job_id=parent_id))


def backfill_caption_parents(store):
    """Idempotently attach historical children, also catching old writers after rollback."""
    from lerobot.data_platform.artifacts.audit import build_operation_event
    from lerobot.data_platform.control.storage import enqueue_event
    from lerobot.data_platform.runtime.jobs import scheduler_lock

    with store.sessions.begin() as session:
        scheduler_lock(session)
        session.execute(
            insert(CaptionDatasetJob).from_select(
                ["job_id", "location_id", "created_at"],
                select(
                    literal(GROUP_PREFIX) + RemoteJob.location_id,
                    RemoteJob.location_id,
                    func.min(RemoteJob.created_at),
                )
                .where(
                    RemoteJob.operation == "caption.annotate",
                    RemoteJob.location_id.not_in(select(CaptionDatasetJob.location_id)),
                )
                .group_by(RemoteJob.location_id),
            )
        )
        inserted = session.execute(
            insert(CaptionJobMember).from_select(
                ["child_job_id", "parent_job_id"],
                select(RemoteJob.job_id, literal(GROUP_PREFIX) + RemoteJob.location_id).where(
                    RemoteJob.operation == "caption.annotate",
                    RemoteJob.job_id.not_in(select(CaptionJobMember.child_job_id)),
                ),
            )
        ).rowcount
        if inserted:
            enqueue_event(
                session,
                build_operation_event(
                    "caption.dataset_jobs.migrated",
                    status="success",
                    actor={"kind": "system"},
                    source="control-plane",
                    details={"attached_children": inserted},
                ),
            )
    return inserted
