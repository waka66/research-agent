from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ResearchRun(Base):
    """一项研究。首次启动和后续恢复共用同一个 run_id。"""

    __tablename__ = "research_runs"

    run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )

    question: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class RunExecution(Base):
    """一次启动或恢复，每次分配新的 execution_id。"""

    __tablename__ = "run_executions"

    __table_args__ = (
        CheckConstraint(
            "mode IN ('start', 'resume')",
            name="ck_run_executions_mode",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'interrupted')",
            name="ck_run_executions_status",
        ),
        Index(
            "idx_run_executions_run_created",
            "run_id",
            "created_at",
        ),
        Index(
            "uq_run_executions_active_run",
            "run_id",
            unique=True,
            postgresql_where=text("status IN ('pending', 'running')"),
        ),
    )

    execution_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
    )

    run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("research_runs.run_id"),
    )

    mode: Mapped[str] = mapped_column(String(8))

    status: Mapped[str] = mapped_column(
        String(16),
        default="pending",
        server_default=text("'pending'"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )

    records_complete: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
    )

    result_summary: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )


class ExecutionRecord(Base):
    """一次节点/工具调用；细节保留原始 record，检索字段单独成列。"""

    __tablename__ = "execution_records"
    __table_args__ = (
        CheckConstraint("kind IN ('node', 'tool')", name="ck_execution_records_kind"),
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'cancelled', 'skipped', "
            "'degraded', 'partial_success', 'interrupted', 'unknown')",
            name="ck_execution_records_status",
        ),
        CheckConstraint("duration_ms >= 0", name="ck_execution_records_duration"),
        CheckConstraint("record_id <> parent_record_id", name="ck_execution_records_parent"),
        UniqueConstraint("execution_id", "record_id", name="uq_execution_records_execution_id"),
        # 同时保证父记录存在，且属于同一次 execution。
        ForeignKeyConstraint(
            ["execution_id", "parent_record_id"],
            ["execution_records.execution_id", "execution_records.record_id"],
            name="fk_execution_records_parent",
        ),
        Index("idx_execution_records_execution_started", "execution_id", "started_at", "record_id"),
    )

    record_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    execution_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("run_executions.execution_id"),
    )
    parent_record_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    kind: Mapped[str] = mapped_column(String(8))
    name: Mapped[str] = mapped_column(String(128))
    task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
