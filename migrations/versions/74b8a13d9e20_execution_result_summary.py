"""Save result summaries at execution completion.

Revision ID: 74b8a13d9e20
Revises: 3d619a8f2c44
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "74b8a13d9e20"
down_revision = "3d619a8f2c44"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 旧执行没有当时的成果快照，保留 NULL，不用最新 checkpoint 回填。
    op.add_column(
        "run_executions",
        sa.Column(
            "result_summary",
            postgresql.JSONB(none_as_null=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("run_executions", "result_summary")
