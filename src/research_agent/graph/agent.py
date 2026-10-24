import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from research_agent.config import settings
from research_agent.db.history import interrupt_active_executions
from research_agent.db.runtime import single_instance_runtime
from research_agent.db.session import engine
from research_agent.graph.main_workflow import agent_builder

DB_URI = settings.db_uri
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with single_instance_runtime(DB_URI):
            async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
                await checkpointer.setup()
                app.state.agent = agent_builder.compile(checkpointer=checkpointer)
                app.state.checkpointer = checkpointer
                # 获得锁说明旧服务已退出；只结束旧 execution，不自动恢复研究。
                interrupted = await interrupt_active_executions()
                if interrupted:
                    logger.warning("将 %s 条遗留执行标记为 interrupted", interrupted)
                try:
                    yield
                finally:
                    if shutdown := getattr(app.state, "shutdown_research", None):
                        await shutdown()
    finally:
        await engine.dispose()


def get_agent(request: Request):
    return request.app.state.agent
