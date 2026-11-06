"""Best-effort admission projection; no provider or background execution on the request path."""

import logging

from lerobot.data_platform.ai_tools.learning.models import LearningControl, LearningProfile
from lerobot.data_platform.ai_tools.security import DisclosureGuard


def admitted(session, turn, message):
    try:
        with session.begin_nested():
            control = session.get(LearningControl, "environment")
            profile = session.get(LearningProfile, turn.owner)
            if control and control.enabled and profile and profile.chat:
                turn.learning_input = DisclosureGuard.from_environment().redact(message)
                turn.learning_epoch = profile.chat_epoch
    except Exception:
        logging.warning("Learning admission unavailable; assistant request is unaffected")


def job_admitted(store, session, job, location):
    """Snapshot a validated versioned manual command; leave legacy unversioned jobs as usage only."""
    import os

    from lerobot.data_platform.ai_tools.learning.models import LearningJobReceipt
    from lerobot.data_platform.ai_tools.strategy_contracts import OPERATORS
    from lerobot.data_platform.datasets.tasks.catalog import content_digest
    from lerobot.data_platform.operators import get_operator
    from lerobot.data_platform.processing.service import location_identity

    try:
        with session.begin_nested():
            setting = session.get(LearningControl, "environment")
            snapshot = job.options.get("operator_snapshot") or {}
            if (
                not setting
                or not setting.enabled
                or job.operation not in OPERATORS
                or snapshot.get("operator_id") != job.operation
                or not snapshot.get("version")
                or snapshot.get("pipeline_run")
            ):
                return
            get_operator(job.operation, snapshot["version"])
            info = store._location_dict(location)
            session.add(
                LearningJobReceipt(
                    job_id=job.job_id,
                    owner=job.requested_by,
                    location_id=job.location_id,
                    command_digest=content_digest(job.options),
                    input_identity=location_identity(info),
                    version=snapshot["version"],
                    release=os.environ.get("DATA_PLATFORM_RELEASE", "legacy"),
                    format=info["metadata"].get("codebase_version", "unknown"),
                )
            )
    except Exception:
        logging.warning("Learning receipt unavailable; task admission is unaffected")
