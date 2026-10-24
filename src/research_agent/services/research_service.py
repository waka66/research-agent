import asyncio
import logging
from uuid import UUID

from research_agent.db import records as record_store
from research_agent.db.history import finish_execution, mark_execution_running
from research_agent.graph.observability import recording_execution
from research_agent.services.result_snapshot import capture_result_summary

logger = logging.getLogger(__name__)

HANDLERS = {
    "Planner": "plan_ready",
    "research_node": "research_completed",
    "Synthesizer": "synthesis_completed",
}
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


def terminal_payload(run_id, execution_id, status, *, error=None, data=None):
    return {
        **(data or {}),
        "run_id": str(run_id),
        "execution_id": str(execution_id),
        "status": status,
        "error": error,
    }


async def finalize_execution(
    run_id: UUID,
    execution_id: UUID,
    queue: asyncio.Queue,
    status: str,
    *,
    error: dict | None = None,
    data: dict | None = None,
    records_complete: bool = True,
    agent=None,
):
    """提交终态后通知 SSE；取消不打断正在提交的收尾。"""

    async def persist_and_notify():
        result_summary = await capture_result_summary(agent, run_id)
        try:
            changed = await finish_execution(
                execution_id,
                status,
                error,
                records_complete=records_complete,
                result_summary=result_summary,
            )
        except Exception:
            logger.exception("无法保存 execution 终态：%s", execution_id)
            await queue.put(
                (
                    "failed",
                    {
                        **terminal_payload(run_id, execution_id, "unknown"),
                        "persisted": False,
                        "error": {
                            "type": "ExecutionPersistenceError",
                            "message": "执行结果未能保存，请稍后查询状态",
                        },
                        "execution_error": error,
                    },
                )
            )
            raise
        if changed:
            await queue.put(
                (
                    status,
                    terminal_payload(
                        run_id,
                        execution_id,
                        status,
                        error=error,
                        data=data,
                    ),
                )
            )

    finalizer = asyncio.create_task(persist_and_notify())
    cancelled = False
    while not finalizer.done():
        try:
            await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            if finalizer.cancelled():
                break
            cancelled = True
        except Exception:
            break
    finalizer.result()
    if cancelled:
        raise asyncio.CancelledError


async def _stream_updates(agent, inputs, config, queue):
    final_data = None
    async for chunk in agent.astream(inputs, config=config, stream_mode="updates"):
        for node_name, data in chunk.items():
            if node_name == "generator":
                # 暂存成果，数据库提交成功后才发送 completed。
                final_data = data
            elif handler := HANDLERS.get(node_name):
                await queue.put((handler, data))
    if final_data is None or final_data.get("final_answer") is None:
        raise RuntimeError("研究流程结束，但 generator 没有生成最终报告")
    return final_data


async def _run_execution(
    agent,
    run_id: UUID,
    execution_id: UUID,
    inputs: dict | None,
    queue: asyncio.Queue,
):
    config = {
        "configurable": {"thread_id": str(run_id)},
        "metadata": {"execution_id": str(execution_id)},
    }
    with recording_execution(execution_id, record_store) as recording:
        try:
            await mark_execution_running(execution_id)
            data = await _stream_updates(agent, inputs, config, queue)
        except asyncio.CancelledError:
            await finalize_execution(
                run_id,
                execution_id,
                queue,
                "cancelled",
                records_complete=recording.records_complete,
                agent=agent,
            )
            raise
        except Exception as exc:
            logger.exception("研究执行失败：run_id=%s execution_id=%s", run_id, execution_id)
            await finalize_execution(
                run_id,
                execution_id,
                queue,
                "failed",
                error={"type": type(exc).__name__, "message": str(exc)},
                records_complete=recording.records_complete,
                agent=agent,
            )
        else:
            await finalize_execution(
                run_id,
                execution_id,
                queue,
                "completed",
                data=data,
                records_complete=recording.records_complete,
                agent=agent,
            )


async def research_status(agent, run_id: UUID):
    return await agent.aget_state(
        {"configurable": {"thread_id": str(run_id)}},
        subgraphs=False,
    )


async def run_research(
    agent,
    run_id: UUID,
    execution_id: UUID,
    question: str,
    queue: asyncio.Queue,
):
    await _run_execution(
        agent,
        run_id,
        execution_id,
        {"question": question, "status": "running"},
        queue,
    )


async def run_resume(agent, run_id: UUID, execution_id: UUID, queue: asyncio.Queue):
    # None 表示从同一 thread 的未完成检查点继续，而非重新提交 question。
    await _run_execution(agent, run_id, execution_id, None, queue)
