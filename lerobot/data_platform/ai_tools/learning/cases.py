"""Reviewed synthetic development cases and transitive revocation of shared consumers."""

import json

from sqlalchemy import select

from lerobot.data_platform.ai_tools.learning import contracts, evidence
from lerobot.data_platform.ai_tools.learning.models import LearningCase, LearningDelivery, LearningSuite
from lerobot.data_platform.ai_tools.strategy_evaluation import initial_cases
from lerobot.data_platform.ai_tools.strategy_models import (
    StrategyBinding,
    StrategyExperiment,
    StrategyVersion,
)
from lerobot.data_platform.datasets.tasks.catalog import content_digest
from lerobot.data_platform.runtime.jobs import JobConflictError, scheduler_lock


def sources(learning, session, row):
    for ref in row.sources:
        evidence.resolve(learning.manager, session, ref)


def suite_sources(manager, session, suite_id):
    link = session.get(LearningSuite, suite_id)
    if link:
        for ref in link.cases:
            case = session.get(LearningCase, ref["key"])
            if not case or case.status != "approved" or case.revision != ref["revision"]:
                raise PermissionError("Development learning case withdrawn")
            if manager._actor(session, case.reviewed_by, write=False).role != "admin":
                raise PermissionError("Case reviewer authority changed")
            for source in case.sources:
                evidence.resolve(manager, session, source)


def invalidate(learning, session):
    for row in session.scalars(select(LearningCase).where(LearningCase.status != "revoked")):
        try:
            sources(learning, session, row)
        except (PermissionError, KeyError, ValueError, JobConflictError):
            row.status, row.revision = "revoked", row.revision + 1
    from lerobot.data_platform.ai_tools.strategy_evidence import validate_sources

    binding = session.get(StrategyBinding, "shared")
    for row in session.scalars(select(StrategyVersion).where(StrategyVersion.revoked.is_(False))):
        try:
            validate_sources(learning.manager, session, row.evidence)
            experiment = session.get(StrategyExperiment, row.experiment_id)
            if experiment:
                suite_sources(learning.manager, session, experiment.suite_id)
        except (PermissionError, KeyError, ValueError, JobConflictError):
            row.revoked = True
            if binding and binding.active_id == row.version_id:
                binding.active_id, binding.revision = None, binding.revision + 1
            learning.manager._audit(
                session, None, "strategy.learning_source_revoked", version_id=row.version_id
            )


def list_cases(learning, owner):
    with learning.store.sessions() as session:
        learning.admin(session, owner)
        result = []
        for row in session.scalars(select(LearningCase).limit(200)):
            try:
                sources(learning, session, row)
            except (PermissionError, KeyError, ValueError, JobConflictError):
                continue
            result.append(
                {
                    "key": row.key,
                    "revision": row.revision,
                    "status": row.status,
                    "definition": row.definition,
                    "source_count": len(row.sources),
                }
            )
        return {"cases": result}


def propose(learning, owner, value):
    contracts.exact(value, {"event_id", "expected_revision"})
    contracts.identifier(value["event_id"])
    from lerobot.data_platform.ai_tools.learning.service import control

    with learning.store.sessions.begin() as session:
        scheduler_lock(session)
        learning.admin(session, owner)
        setting = control(session)
        delivery_id = content_digest([owner, "propose-cases", value["event_id"]])
        previous = session.get(LearningDelivery, delivery_id)
        if previous:
            if previous.digest != content_digest(value):
                raise JobConflictError("Case request identity reused")
            return {"replayed": True}
        contracts.revision(value["expected_revision"], setting.revision)
        session.add(LearningDelivery(key=delivery_id, digest=content_digest(value)))
        # Fixed public scenarios supply synthetic parameters; no historical values or prose are copied.
        catalog = json.loads(json.dumps(initial_cases()))
        for episode in evidence.references(learning.manager, session):
            ops = [
                s["operator_id"]
                for s in episode.projection["steps"]
                if s["operator_id"] != "curation.snapshot"
            ]
            for case in catalog["development"] + catalog["holdout"]:
                if case["format"] != episode.projection["format"] or case["operations"] != ops:
                    continue
                ref = evidence.reference(episode)
                key = content_digest([ref, case])
                if not session.get(LearningCase, key):
                    session.add(
                        LearningCase(key=key, sources=[ref], definition=case, revision=0, status="draft")
                    )
        learning.manager._audit(session, owner, "learning.cases.proposed")
    return list_cases(learning, owner)


def review(learning, owner, value):
    contracts.exact(value, {"key", "expected_revision", "approve"})
    if type(value["approve"]) is not bool:
        raise ValueError("Explicit case approval required")
    with learning.store.sessions.begin() as session:
        scheduler_lock(session)
        learning.admin(session, owner)
        row = session.get(LearningCase, value["key"])
        if not row:
            raise KeyError("Case unavailable")
        contracts.revision(value["expected_revision"], row.revision)
        sources(learning, session, row)
        if row.status != "draft":
            raise JobConflictError("Case is no longer a draft")
        row.status, row.reviewed_by = "approved" if value["approve"] else "rejected", owner
        row.revision += 1
        learning.manager._audit(
            session, owner, "learning.case.reviewed", key=row.key, approved=value["approve"]
        )
    return list_cases(learning, owner)


def link_suite(session, cases, suite_id):
    """Automatically preserve reviewed-case provenance when importing a StrategySuite."""
    refs = []

    def fingerprint(case):
        return content_digest({k: v for k, v in case.items() if k not in {"case_id", "request"}})

    development = {fingerprint(c) for c in cases["development"]}
    hidden = {fingerprint(c) for c in cases["holdout"]}
    candidates = {}
    for row in session.scalars(select(LearningCase)):
        identity = fingerprint(row.definition)
        if identity in hidden:
            raise ValueError("Learning cases belong to development, never holdout")
        if identity in development:
            candidates.setdefault(identity, []).append(row)
    for versions in candidates.values():
        approved = [row for row in versions if row.status == "approved"]
        if not approved:
            raise ValueError("Learning development case needs review")
        refs.extend({"key": row.key, "revision": row.revision} for row in approved)
    if refs:
        session.add(LearningSuite(suite_id=suite_id, cases=refs))
