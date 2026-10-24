"""研究执行历史与调用明细：只读 PostgreSQL，不读取图或进程内任务。"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from research_agent.api.pagination import decode_cursor, encode_cursor
from research_agent.db.history import get_execution, list_executions, load_execution_report
from research_agent.db.records import list_records
from research_agent.schemas.history import (
    ExecutionPage,
    ExecutionResponse,
    RecordKind,
    RecordPage,
    RecordResponse,
    RecordStatus,
)
from research_agent.schemas.report import ExecutionReport
from research_agent.services.execution_report import build_execution_report

router = APIRouter(tags=["Execution history"])
logger = logging.getLogger(__name__)
PageSize = Annotated[int, Query(ge=1, le=100)]
CursorQuery = Annotated[str | None, Query(min_length=1, max_length=2048)]


@router.get("/executions/{execution_id}/report", response_model=ExecutionReport)
async def execution_report(execution_id: UUID):
    try:
        data = await load_execution_report(execution_id)
    except Exception as exc:
        logger.exception("读取运行报告数据失败：%s", execution_id)
        raise HTTPException(status_code=503, detail="运行报告暂不可用，请稍后重试") from exc
    if data is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return build_execution_report(*data)


@router.get("/research/{run_id}/executions", response_model=ExecutionPage)
async def research_executions(run_id: UUID, limit: PageSize = 20, cursor: CursorQuery = None):
    scope = f"executions:{run_id}"
    position = decode_cursor(cursor, scope=scope, filters={})
    try:
        rows = await list_executions(
            run_id,
            limit=limit,
            before=(position.timestamp, position.item_id) if position else None,
        )
    except Exception as exc:
        logger.exception("查询执行历史失败：%s", run_id)
        raise HTTPException(status_code=503, detail="执行历史暂不可用，请稍后重试") from exc
    if rows is None:
        raise HTTPException(status_code=404, detail="Research run not found")
    items = [ExecutionResponse.model_validate(row) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = encode_cursor(
            last.created_at,
            last.execution_id,
            scope=scope,
            filters={},
        )
    return ExecutionPage(run_id=run_id, items=items, next_cursor=next_cursor)


@router.get("/executions/{execution_id}/records", response_model=RecordPage)
async def execution_records(
    execution_id: UUID,
    limit: PageSize = 50,
    cursor: CursorQuery = None,
    kind: RecordKind | None = None,
    status: RecordStatus | None = None,
    name: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    task_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    round_number: Annotated[int | None, Query(alias="round", ge=0)] = None,
):
    scope = f"records:{execution_id}"
    filters = {
        key: str(value)
        for key, value in {
            "kind": kind,
            "status": status,
            "name": name,
            "task_id": task_id,
            "round": round_number,
        }.items()
        if value is not None
    }
    position = decode_cursor(cursor, scope=scope, filters=filters)
    try:
        execution = await get_execution(execution_id)
        if execution is None:
            raise HTTPException(status_code=404, detail="Execution not found")
        rows = await list_records(
            execution_id,
            limit=limit,
            after=(position.timestamp, position.item_id) if position else None,
            kind=kind,
            status=status,
            name=name,
            task_id=task_id,
            round_number=round_number,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("查询执行记录失败：%s", execution_id)
        raise HTTPException(status_code=503, detail="执行记录暂不可用，请稍后重试") from exc
    items = [RecordResponse.model_validate(row) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = encode_cursor(
            last.started_at,
            last.record_id,
            scope=scope,
            filters=filters,
        )
    return RecordPage(execution=execution, items=items, next_cursor=next_cursor)
