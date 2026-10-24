from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
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
            "status IN ("
            "'pending', 'running', 'completed', "
            "'failed', 'cancelled', 'interrupted'"
            ")",
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
