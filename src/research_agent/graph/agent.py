from fastapi import FastAPI
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from fastapi import Request

from research_agent.config import settings
from research_agent.graph.main_workflow import agent_builder

DB_URI = (settings.db_uri)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        agent = agent_builder.compile(
            checkpointer=checkpointer,
        )

        app.state.agent = agent
        app.state.checkpointer = checkpointer

        yield


def get_agent(request: Request):
    return request.app.state.agent