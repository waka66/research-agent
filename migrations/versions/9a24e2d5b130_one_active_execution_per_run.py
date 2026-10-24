"""one active execution per run

Revision ID: 9a24e2d5b130
Revises: 5c459cf8bd77
"""

import sqlalchemy as sa
from alembic import op

revision = "9a24e2d5b130"
down_revision = "5c459cf8bd77"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 不自动改写历史数据；如果已有冲突，让迁移明确失败，先核实实际执行状态。
    duplicate = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT run_id FROM run_executions "
                "WHERE status IN ('pending', 'running') "
                "GROUP BY run_id HAVING COUNT(*) > 1 LIMIT 1"
            )
        )
        .first()
    )
    if duplicate is not None:
        raise RuntimeError("同一研究存在多条活跃 execution，请先核实历史状态后再迁移")
    op.create_index(
        "uq_run_executions_active_run",
        "run_executions",
        ["run_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("uq_run_executions_active_run", table_name="run_executions")
