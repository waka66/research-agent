"""Persist node and tool invocation records.

Revision ID: 3d619a8f2c44
Revises: 9a24e2d5b130
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "3d619a8f2c44"
down_revision = "9a24e2d5b130"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 旧 execution 没有过程记录，不能声称完整；仅新执行默认完整。
    op.add_column(
        "run_executions",
        sa.Column(
            "records_complete",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.alter_column("run_executions", "records_complete", server_default=sa.text("true"))
    op.create_table(
        "execution_records",
        sa.Column("record_id", sa.Uuid(), nullable=False),
        sa.Column("execution_id", sa.Uuid(), nullable=False),
        sa.Column("parent_record_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("task_id", sa.String(128), nullable=True),
        sa.Column("round", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("record_id"),
        sa.ForeignKeyConstraint(["execution_id"], ["run_executions.execution_id"]),
        sa.UniqueConstraint("execution_id", "record_id", name="uq_execution_records_execution_id"),
        sa.ForeignKeyConstraint(
            ["execution_id", "parent_record_id"],
            ["execution_records.execution_id", "execution_records.record_id"],
            name="fk_execution_records_parent",
        ),
        sa.CheckConstraint("kind IN ('node', 'tool')", name="ck_execution_records_kind"),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'cancelled', 'skipped', "
            "'degraded', 'partial_success', 'interrupted', 'unknown')",
            name="ck_execution_records_status",
        ),
        sa.CheckConstraint("duration_ms >= 0", name="ck_execution_records_duration"),
        sa.CheckConstraint("record_id <> parent_record_id", name="ck_execution_records_parent"),
    )
    op.create_index(
        "idx_execution_records_execution_started",
        "execution_records",
        ["execution_id", "started_at", "record_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_execution_records_execution_started", table_name="execution_records")
    op.drop_table("execution_records")
    op.drop_column("run_executions", "records_complete")
