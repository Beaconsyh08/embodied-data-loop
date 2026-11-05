"""Dataset-level annotation monitoring, without changing execution or attempt identities."""

from __future__ import annotations

from sqlalchemy import case, func, select, union_all

from lerobot.data_platform.list_queries import job_projection, parse_time

GROUP_PREFIX = "caption-dataset:"


def list_grouped_jobs(store, session, authorized, *, page, filters, limit, summary):
    """Aggregate all authorized children before filtering or paging the visible jobs."""
    from lerobot.data_platform.control_plane import (
        JOB_STATES,
        CaptionDatasetJob,
        CaptionJobMember,
        RemoteJob,
        _iso,
    )
    from lerobot.data_platform.job_management import ACTIVE_STATES

    eligible = authorized.with_only_columns(RemoteJob.job_id).subquery()
    scope = RemoteJob.job_id.in_(select(eligible.c.job_id))
    caption = RemoteJob.operation == "caption.annotate"
    skipped = (RemoteJob.status.in_(["error", "cancelled"])) & (
        RemoteJob.result["operator_disposition"]["status"].as_string() == "skipped"
    )
    effective = case((skipped, "skipped"), else_=RemoteJob.status)
    states = ["running", "cancel_requested", "queued", "interrupted", "error", "cancelled", "done", "skipped"]
    counts = [func.sum(case((effective == status, 1), else_=0)).label(status) for status in states]
    grouped = (
        select(
            CaptionDatasetJob.job_id.label("job_id"),
            RemoteJob.location_id,
            func.min(RemoteJob.node_id).label("node_id"),
            func.max(RemoteJob.created_at).label("created_at"),
            func.min(RemoteJob.created_at).label("first_created_at"),
            func.max(RemoteJob.updated_at).label("updated_at"),
            func.count().label("total"),
            func.count(func.distinct(RemoteJob.options["episode_index"].as_integer())).label("episodes"),
            *counts,
        )
        .select_from(RemoteJob)
        .join(CaptionJobMember, CaptionJobMember.child_job_id == RemoteJob.job_id)
        .join(CaptionDatasetJob, CaptionDatasetJob.job_id == CaptionJobMember.parent_job_id)
        .where(scope, caption)
        .group_by(CaptionDatasetJob.job_id, RemoteJob.location_id)
        .subquery()
    )
    status = case(*[(grouped.c[state] > 0, state) for state in states[:6]], else_="done").label("status")
    combined = union_all(
        select(RemoteJob.job_id, RemoteJob.created_at, RemoteJob.status).where(scope, ~caption),
        select(grouped.c.job_id, grouped.c.created_at, status),
    ).subquery()
    query = select(combined)
    value = filters.get("status")
    if value:
        if value == "active":
            query = query.where(combined.c.status.in_({"queued"} | ACTIVE_STATES))
        elif value == "history":
            query = query.where(combined.c.status.not_in({"queued"} | ACTIVE_STATES))
        elif value in JOB_STATES:
            query = query.where(combined.c.status == value)
        else:
            raise ValueError("Invalid job status")
    for key, condition in (
        ("created_from", combined.c.created_at.__ge__),
        ("created_to", combined.c.created_at.__lt__),
    ):
        if filters.get(key):
            query = query.where(condition(parse_time(filters[key])))
    query = (
        page.apply(query, combined.c.created_at, combined.c.job_id, date_time=True)
        if page
        else query.order_by(combined.c.created_at.desc(), combined.c.job_id.desc()).limit(
            max(1, min(1000, limit))
        )
    )
    rows = session.execute(query).all()
    paging = {}
    if page:
        rows, paging = page.response(rows, timestamp=lambda r: r.created_at, identity=lambda r: r.job_id)
    ids = [row.job_id for row in rows]
    parents = {}
    for row in session.execute(select(grouped).where(grouped.c.job_id.in_(ids))).mappings():
        counts = {state: int(row[state]) for state in states}
        settled = counts["done"] + counts["skipped"]
        parent_status = next((state for state in states[:6] if counts[state]), "done")
        parents[row["job_id"]] = {
            "job_id": row["job_id"],
            "operation": "caption.annotate",
            "caption_group": True,
            "location_id": row["location_id"],
            "node_id": row["node_id"],
            "status": parent_status,
            "created_at": _iso(row["created_at"]),
            "updated_at": _iso(row["updated_at"]),
            "started_at": _iso(row["first_created_at"]),
            "finished_at": _iso(row["updated_at"]) if parent_status == "done" else None,
            "caption_counts": {**counts, "total": int(row["total"]), "episodes": int(row["episodes"])},
            "progress": {
                "current": settled,
                "total": int(row["total"]),
                "progress": 100 * settled / row["total"],
            },
            "options": {},
            "result": {},
            "available_actions": [],
            "events": [],
        }
    details = select(RemoteJob).where(RemoteJob.job_id.in_([key for key in ids if key not in parents]))
    if summary:
        details = details.options(*job_projection(RemoteJob))
    jobs = {job.job_id: store._job_dict(job) for job in session.scalars(details)}
    jobs.update(parents)
    result = [jobs[key] for key in ids]
    return (result, paging) if page else result
