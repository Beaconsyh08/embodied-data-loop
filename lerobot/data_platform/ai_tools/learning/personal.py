"""Private, deterministically verified history adapter; never schedules Hindsight billing."""

from sqlalchemy import select

from lerobot.data_platform.ai_tools.learning import evidence
from lerobot.data_platform.ai_tools.learning.models import LearningEpisode, LearningProfile
from lerobot.data_platform.job_management import JobConflictError
from lerobot.data_platform.task_catalog import content_digest


def snapshot(service, session, owner, location):
    profile = session.get(LearningProfile, owner)
    items = []
    if profile and profile.experience:
        for row in session.scalars(
            select(LearningEpisode)
            .where(
                LearningEpisode.owner == owner,
                LearningEpisode.location_id == location,
                LearningEpisode.valid.is_(True),
                LearningEpisode.verified.is_(True),
            )
            .order_by(LearningEpisode.created_ms.desc(), LearningEpisode.key)
            .limit(4)
        ):
            try:
                value = evidence.resolve(service.manager, session, evidence.reference(row), owner=owner)
            except (PermissionError, KeyError, ValueError, JobConflictError):
                continue
            items.append(
                {
                    "historical_only": True,
                    "source": evidence.reference(row),
                    "execution_kind": row.kind,
                    "observed_execution": value,
                }
            )
    return items, content_digest([profile.revision if profile else 0, items])


def context(service, owner, location):
    with service.store.sessions() as session:
        return snapshot(service, session, owner, location)[0]
