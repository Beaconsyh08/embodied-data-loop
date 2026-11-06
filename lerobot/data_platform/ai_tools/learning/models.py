"""Additive control records for usage, private analysis and independently verified evidence."""

from sqlalchemy import JSON, BigInteger, Boolean, Double, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from lerobot.data_platform.control.storage import ManagementBase


class LearningControl(ManagementBase):
    __tablename__ = "dp_learning_control"
    key: Mapped[str] = mapped_column(String(16), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    administrator: Mapped[str | None] = mapped_column(String(36), nullable=True)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    scan: Mapped[dict] = mapped_column(JSON, default=dict)
    backfill: Mapped[dict] = mapped_column(JSON, default=dict)
    lease_owner: Mapped[str] = mapped_column(String(36), default="")
    lease_until: Mapped[float] = mapped_column(Double, default=0)


class LearningProfile(ManagementBase):
    __tablename__ = "dp_learning_profiles"
    owner: Mapped[str] = mapped_column(String(36), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    chat: Mapped[bool] = mapped_column(Boolean, default=False)
    chat_epoch: Mapped[int] = mapped_column(Integer, default=0)
    experience: Mapped[bool] = mapped_column(Boolean, default=False)
    since_ms: Mapped[int] = mapped_column(BigInteger, default=0)


class LearningEvent(ManagementBase):
    __tablename__ = "dp_learning_events"
    control: Mapped[str] = mapped_column(String(16))
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(36), index=True)
    digest: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))
    target: Mapped[dict] = mapped_column(JSON)
    created_ms: Mapped[int] = mapped_column(BigInteger, index=True)


class LearningOrigin(ManagementBase):
    __tablename__ = "dp_learning_origins"
    version_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    owner: Mapped[str] = mapped_column(String(36), index=True)
    request_id: Mapped[str] = mapped_column(String(36))
    original_digest: Mapped[str] = mapped_column(String(64))
    changes: Mapped[list] = mapped_column(JSON)
    created_ms: Mapped[int] = mapped_column(BigInteger)


class LearningEpisode(ManagementBase):
    __tablename__ = "dp_learning_episodes"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(36), index=True)
    location_id: Mapped[str] = mapped_column(String(36))
    kind: Mapped[str] = mapped_column(String(16))
    source_id: Mapped[str] = mapped_column(String(80), index=True)
    digest: Mapped[str] = mapped_column(String(64))
    projection: Mapped[dict] = mapped_column(JSON)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    created_ms: Mapped[int] = mapped_column(BigInteger, index=True)


class LearningFeedback(ManagementBase):
    __tablename__ = "dp_learning_feedback"
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(36), index=True)
    target: Mapped[dict] = mapped_column(JSON)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    delivery: Mapped[str] = mapped_column(String(36))
    verdict: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(String(32))
    created_ms: Mapped[int] = mapped_column(BigInteger)


class LearningAnalysis(ManagementBase):
    __tablename__ = "dp_learning_analyses"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(36), index=True)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    epoch: Mapped[int] = mapped_column(Integer)
    sources: Mapped[list] = mapped_column(JSON)
    cursor: Mapped[dict] = mapped_column(JSON)
    items: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(24), default="pending")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    created_ms: Mapped[int] = mapped_column(BigInteger)


class LearningCall(ManagementBase):
    __tablename__ = "dp_learning_calls"
    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    owner: Mapped[str] = mapped_column(String(36))
    day: Mapped[str] = mapped_column(String(10), index=True)
    digest: Mapped[str] = mapped_column(String(64))
    reserved: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default="unknown")
    usage: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)


class LearningCase(ManagementBase):
    __tablename__ = "dp_learning_cases"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    sources: Mapped[list] = mapped_column(JSON)
    definition: Mapped[dict] = mapped_column(JSON)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="draft")
    reviewed_by: Mapped[str | None] = mapped_column(String(36), nullable=True)


class LearningSuite(ManagementBase):
    __tablename__ = "dp_learning_suites"
    suite_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cases: Mapped[list] = mapped_column(JSON)


class LearningChatCursor(ManagementBase):
    __tablename__ = "dp_learning_chat_cursors"
    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    offset: Mapped[int] = mapped_column(Integer, default=0)
    complete: Mapped[bool] = mapped_column(Boolean, default=False)


class LearningJobReceipt(ManagementBase):
    __tablename__ = "dp_learning_job_receipts"
    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner: Mapped[str] = mapped_column(String(36))
    location_id: Mapped[str] = mapped_column(String(36))
    command_digest: Mapped[str] = mapped_column(String(64))
    input_identity: Mapped[dict] = mapped_column(JSON)
    version: Mapped[str] = mapped_column(String(32))
    release: Mapped[str] = mapped_column(String(128))
    format: Mapped[str] = mapped_column(String(16))


class LearningDelivery(ManagementBase):
    __tablename__ = "dp_learning_deliveries"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    digest: Mapped[str] = mapped_column(String(64))
