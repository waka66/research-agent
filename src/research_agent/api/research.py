from fastapi import APIRouter, Depends
from uuid import uuid4
import asyncio
from uuid import UUID
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

from research_agent.schemas.research import ResearchRequest, ResearchRunResponse, ResearchStatusResponse
from research_agent.services.research_service import run_research, research_status, run_resume

from research_agent.graph.agent import get_agent


router = APIRouter(prefix="/research", tags=["Research"])

run_queues = {}
runs = {}


@router.post(
    "/",
    response_model=ResearchRunResponse,
    status_code=202
)
async def research_run(request: ResearchRequest, agent=Depends(get_agent)):
    run_id = uuid4()

    queue = asyncio.Queue()
    run_queues[run_id] = queue

    task = asyncio.create_task(
        run_research(agent, run_id=run_id, question=request.question, queue=queue)
    )
    runs[run_id] = task

    return ResearchRunResponse(
        run_id=run_id,
        status="pending",
    )


@router.get(
    "/{run_id}",
    response_model=ResearchStatusResponse,
    status_code=200
)
async def research_run_status(run_id: UUID, agent=Depends(get_agent)):

    Snapshot = await research_status(agent, run_id)
    return {
        "run_id": run_id,
        "question": Snapshot.values["question"],
        "status": Snapshot.values["status"],
        "plan": Snapshot.values["research_plan"],
        "results": Snapshot.values["research_results"],
        "final_answer": Snapshot.values["final_answer"],
        "error": Snapshot.error
    }


@router.get("/{run_id}/stream")
async def research_run_stream(run_id: UUID):
    queue = run_queues.get(run_id)
    # SSE 数据生成器
    async def event_generator():
        while True:
            # 等待新的事件
            event, data = await queue.get()

            # SSE 格式
            yield f"event: {event}\ndata: {data}\n\n"

            # 任务完成，结束 SSE
            if event == "completed" or event == "cancelled":
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.post("/{run_id}/cancel")
async def research_cancel(run_id: UUID):
    task = runs.get(run_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Research run not found",
        )

    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        return {
            "run_id": run_id,
            "status": "cancelled",
        }


@router.post("/{run_id}/resume")
async def research_resume(run_id: UUID, agent=Depends(get_agent)):

    queue = run_queues.get(run_id)
    if queue is None:
        raise HTTPException(
            status_code=404,
            detail="Research run not found",
        )

    task = asyncio.create_task(
        run_resume(agent, run_id=run_id, queue=queue)
    )
    runs[run_id] = task

    return ResearchRunResponse(
        run_id=run_id,
        status="running",
    )












