import asyncio
import json
import logging
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from research_agent.db.history import (
    create_start_execution,
    get_execution,
    get_latest_execution,
)
from research_agent.graph.agent import get_agent
from research_agent.schemas.research import (
    ResearchRequest,
    ResearchRunResponse,
    ResearchStatusResponse,
)
from research_agent.services.research_service import (
    TERMINAL_STATUSES,
    finalize_execution,
    research_status,
    run_research,
    run_resume,
    terminal_payload,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["Research"])
AgentDependency = Annotated[Any, Depends(get_agent)]
run_queues: dict[UUID, asyncio.Queue] = {}
runs: dict[UUID, asyncio.Task] = {}
run_execution_ids: dict[UUID, UUID] = {}


def observe_task(task):
    # 提取后台异常，避免 Task exception was never retrieved。
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("研究后台任务异常：%s", task.get_name(), exc_info=error)


@router.post("/", response_model=ResearchRunResponse, status_code=202)
async def research_run(request: ResearchRequest, agent: AgentDependency):
    run_id, execution_id = uuid4(), uuid4()
    # 必须先提交 pending 记录，后台任务才能标记 running。
    try:
        await create_start_execution(run_id, execution_id, request.question)
    except Exception as exc:
        logger.exception("无法创建研究历史")
        raise HTTPException(status_code=503, detail="无法保存研究请求") from exc
    queue = asyncio.Queue()
    coroutine = run_research(
        agent,
        run_id=run_id,
        execution_id=execution_id,
        question=request.question,
        queue=queue,
    )
    try:
        task = asyncio.create_task(coroutine, name=f"research-{execution_id}")
    except Exception as exc:
        coroutine.close()
        await finalize_execution(
            run_id,
            execution_id,
            queue,
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise HTTPException(status_code=500, detail="无法启动研究任务") from exc
    run_queues[run_id] = queue
    runs[run_id] = task
    run_execution_ids[run_id] = execution_id
    task.add_done_callback(observe_task)
    return ResearchRunResponse(
        run_id=run_id,
        execution_id=execution_id,
        status="pending",
    )


@router.get("/{run_id}", response_model=ResearchStatusResponse)
async def research_run_status(run_id: UUID, agent: AgentDependency):
    execution = await get_latest_execution(run_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Research run not found")
    checkpoint_available = True
    try:
        snapshot = await research_status(agent, run_id)
        values = snapshot.values or {}
    except Exception:
        logger.exception("检查点读取失败，返回已保存的执行状态：%s", run_id)
        values = {}
        checkpoint_available = False
    return {
        **execution,
        "plan": values.get("research_plan"),
        "results": values.get("research_results") or [],
        "final_answer": values.get("final_answer"),
        "checkpoint_available": checkpoint_available,
    }


def encode_event(event, data):
    payload = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get("/{run_id}/stream")
async def research_run_stream(run_id: UUID):
    execution = await get_latest_execution(run_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Research run not found")
    queue = run_queues.get(run_id)
    replay_terminal = (queue is None or queue.empty()) and execution["status"] in TERMINAL_STATUSES
    if queue is None and not replay_terminal:
        raise HTTPException(
            status_code=409,
            detail="当前进程没有该研究的事件流，请查询执行状态",
        )

    async def event_generator():
        if replay_terminal:
            status = execution["status"]
            event = "failed" if status == "interrupted" else status
            yield encode_event(
                event,
                terminal_payload(
                    run_id,
                    execution["execution_id"],
                    status,
                    error=execution["error"],
                ),
            )
            return
        while True:
            event, data = await queue.get()
            yield encode_event(event, data)
            if event in {"completed", "failed", "cancelled"}:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/{run_id}/cancel")
async def research_cancel(run_id: UUID):
    task = runs.get(run_id)
    if task is None:
        raise HTTPException(status_code=404, detail="当前进程没有该研究的后台任务")
    execution_id = run_execution_ids.get(run_id)
    if not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if execution_id is not None:
                execution = await get_execution(execution_id)
                # 协程尚未进入 run_research 就取消时，由接口补齐 pending 的收尾。
                if execution["status"] not in TERMINAL_STATUSES:
                    await finalize_execution(
                        run_id,
                        execution_id,
                        run_queues[run_id],
                        "cancelled",
                    )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="任务收尾失败，请查询执行状态") from exc
    if execution_id is not None:
        execution = await get_execution(execution_id)
        return {
            "run_id": run_id,
            "execution_id": execution_id,
            "status": execution["status"],
        }
    return {"run_id": run_id, "status": "cancelled"}


@router.post("/{run_id}/resume", response_model=ResearchRunResponse)
async def research_resume(run_id: UUID, agent: AgentDependency):
    # 此处保留旧恢复流程，resume execution 在下一阶段接入。
    queue = run_queues.get(run_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Research run not found")
    if (task := runs.get(run_id)) is not None and not task.done():
        raise HTTPException(status_code=409, detail="研究仍在执行，不能重复恢复")
    queue = asyncio.Queue()
    run_queues[run_id] = queue
    run_execution_ids.pop(run_id, None)
    task = asyncio.create_task(run_resume(agent, run_id=run_id, queue=queue))
    runs[run_id] = task
    task.add_done_callback(observe_task)
    return ResearchRunResponse(run_id=run_id, status="running")
