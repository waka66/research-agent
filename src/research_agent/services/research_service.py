from uuid import UUID
import asyncio

from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

handlers = {
    "planner": "plan_ready",
    "research_node": "research_completed",
    "synthesis": "synthesis_started",
    "generator": "completed"
}
async def run_research(agent, run_id: UUID, question: str, queue: asyncio.Queue):

    try:
        config = {
            "configurable": {
                "thread_id": str(run_id)
            }
        }
        question = [HumanMessage(content=question)]
        async for chunk in agent.astream(
                {"question": question, "status": "running"},
                config=config,
                stream_mode="updates",
        ):
            print("STREAM:", chunk)
            node_name = next(iter(chunk))
            handler = handlers.get(node_name)
            data = chunk[node_name]
            await queue.put((handler,data))
    except asyncio.CancelledError:
        await queue.put(("cancelled", None))
        raise

async def research_status(agent, run_id: UUID):
    config = {
        "configurable": {
            "thread_id": str(run_id)
        }
    }

    return await agent.aget_state(config, subgraphs=False)


async def run_resume(agent, run_id: UUID, queue: asyncio.Queue):
    try:
        config = {
            "configurable": {
                "thread_id": str(run_id)
            }
        }

        async for chunk in agent.astream(
                None,
                config=config,
                stream_mode="updates",
        ):
            print("STREAM:", chunk)
            node_name = next(iter(chunk))
            handler = handlers.get(node_name)
            data = chunk[node_name]
            await queue.put((handler, data))
    except asyncio.CancelledError:
        await queue.put(("cancelled", None))
        raise



