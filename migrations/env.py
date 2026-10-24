import asyncio
import sys

# 仅在 Windows 上设置兼容的事件循环策略
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from research_agent.config import settings
from research_agent.db.models import Base


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# 继续使用现有连接配置，只为 SQLAlchemy 指定驱动。
# 不修改 settings.db_uri，LangGraph 仍然使用原来的连接串。
database_url = make_url(settings.db_uri).set(
    drivername="postgresql+psycopg"
)

# 只管理我们自己的业务表。
MANAGED_TABLES = {
    "research_runs",
    "run_executions",
    "execution_records",
}


def include_name(name, type_, parent_names):
    if type_ == "table":
        return name in MANAGED_TABLES
    return True


def run_migrations_offline():
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        include_name=include_name,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_name=include_name,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    engine = create_async_engine(
        database_url,
        poolclass=pool.NullPool,
    )

    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
