from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update

from research_agent.db.errors import ExecutionConflictError
from research_agent.db.models import ResearchRun, RunExecution
from research_agent.db.records import close_open_records
from research_agent.db.session import SessionFactory


async def create_resume_execution(
    run_id: UUID,
    execution_id: UUID,
    expected_execution_id: UUID,
) -> None:
    """锁住研究行，在同一事务内检查旧执行并创建新的 pending execution。"""
    async with SessionFactory.begin() as session:
        research = await session.get(ResearchRun, run_id, with_for_update=True)
        if research is None:
            raise LookupError(f"研究不存在：{run_id}")
        latest = await session.scalar(
            select(RunExecution)
            .where(RunExecution.run_id == run_id)
            .order_by(RunExecution.created_at.desc(), RunExecution.execution_id.desc())
            .limit(1)
        )
        if latest is None or latest.execution_id != expected_execution_id:
            raise ExecutionConflictError("执行记录已变化，请重新查询后再恢复")
        active = await session.scalar(
            select(RunExecution.execution_id)
            .where(
                RunExecution.run_id == run_id,
                RunExecution.status.in_(("pending", "running")),
            )
            .limit(1)
        )
        if active is not None:
            raise ExecutionConflictError("研究仍在执行，不能重复恢复")
        if latest.status not in {"failed", "cancelled", "interrupted"}:
            raise ExecutionConflictError(f"当前状态 {latest.status} 不允许恢复")
        session.add(
            RunExecution(
                execution_id=execution_id,
                run_id=run_id,
                mode="resume",
                status="pending",
            )
        )


async def interrupt_active_executions(run_ids: set[UUID] | None = None) -> int:
    """仅在取得服务单实例锁后调用；旧进程已退出，其活跃记录不再代表运行中。"""
    statement = update(RunExecution).where(
        RunExecution.status.in_(("pending", "running")),
    )
    if run_ids is not None:
        statement = statement.where(RunExecution.run_id.in_(run_ids))
    async with SessionFactory.begin() as session:
        result = await session.execute(
            statement.values(
                status="interrupted",
                finished_at=datetime.now(UTC),
                error={
                    "type": "ProcessInterrupted",
                    "message": "上一次服务进程退出，执行未正常结束",
                },
                records_complete=False,
            ).returning(RunExecution.execution_id)
        )
        execution_ids = list(result.scalars().all())
        await close_open_records(session, execution_ids, interrupted=True)
        return len(execution_ids)


async def create_start_execution(
    run_id: UUID,
    execution_id: UUID,
    question: str,
) -> None:
    async with SessionFactory.begin() as session:
        research = ResearchRun(
            run_id=run_id,
            question=question,
        )
        session.add(research)

        # 先插入父记录，让后面的 execution 可以引用它。
        await session.flush()

        execution = RunExecution(
            execution_id=execution_id,
            run_id=run_id,
            mode="start",
            status="pending",
        )
        session.add(execution)


async def mark_execution_running(execution_id: UUID) -> None:
    async with SessionFactory.begin() as session:
        execution = await session.get(
            RunExecution,
            execution_id,
            with_for_update=True,
        )

        if execution is None:
            raise LookupError(f"execution 不存在：{execution_id}")

        if execution.status != "pending":
            raise ValueError(f"不能从 {execution.status} 开始执行")

        execution.status = "running"
        execution.started_at = datetime.now(UTC)


async def finish_execution(
    execution_id: UUID,
    status: str,
    error: dict[str, Any] | None = None,
    *,
    records_complete: bool = True,
) -> bool:
    if status not in {"completed", "failed", "cancelled"}:
        raise ValueError(f"不支持的结束状态：{status}")

    if status == "failed" and error is None:
        raise ValueError("失败时必须提供错误信息")

    async with SessionFactory.begin() as session:
        execution = await session.get(
            RunExecution,
            execution_id,
            with_for_update=True,
        )

        if execution is None:
            raise LookupError(f"execution 不存在：{execution_id}")

        if execution.status == status:
            execution.records_complete = execution.records_complete and records_complete
            # 后台取消处理与取消接口可能同时收尾，只允许第一次发送终态事件。
            return False

        if execution.status not in {"pending", "running"}:
            raise ValueError(f"execution 已结束，当前状态：{execution.status}")

        if status == "completed" and execution.status != "running":
            raise ValueError("尚未开始的 execution 不能标记完成")

        execution.status = status
        execution.finished_at = datetime.now(UTC)
        execution.error = error
        unclosed = await close_open_records(session, [execution_id], interrupted=False)
        execution.records_complete = (
            execution.records_complete and records_complete and not unclosed
        )
        return True


async def get_execution(execution_id: UUID) -> dict | None:
    async with SessionFactory() as session:
        execution = await session.get(
            RunExecution,
            execution_id,
        )

        if execution is None:
            return None

        return {
            "execution_id": execution.execution_id,
            "run_id": execution.run_id,
            "mode": execution.mode,
            "status": execution.status,
            "created_at": execution.created_at,
            "started_at": execution.started_at,
            "finished_at": execution.finished_at,
            "error": execution.error,
            "records_complete": execution.records_complete,
        }


async def get_latest_execution(run_id: UUID) -> dict | None:
    """按研究查询最近一次持久化执行，同时取得原始问题。"""
    async with SessionFactory() as session:
        result = await session.execute(
            select(ResearchRun.question, RunExecution)
            .join(RunExecution, RunExecution.run_id == ResearchRun.run_id)
            .where(ResearchRun.run_id == run_id)
            .order_by(RunExecution.created_at.desc(), RunExecution.execution_id.desc())
            .limit(1)
        )
        row = result.first()
        if row is None:
            return None
        question, execution = row
        return {
            "question": question,
            "execution_id": execution.execution_id,
            "run_id": execution.run_id,
            "mode": execution.mode,
            "status": execution.status,
            "created_at": execution.created_at,
            "started_at": execution.started_at,
            "finished_at": execution.finished_at,
            "error": execution.error,
            "records_complete": execution.records_complete,
        }
